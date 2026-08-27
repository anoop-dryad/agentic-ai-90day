import os
import re
import time
from pathlib import Path

import chromadb
from eval_set import EVAL_SET
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

DOCS_DIR = Path(__file__).parent / "docs"
DB_DIR = Path(__file__).parent / "chroma_eval"

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def extract_text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(
        p.get("text", "")
        for p in msg.content
        if isinstance(p, dict) and p.get("type") == "text"
    )


splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=40,
    separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
)


def tok(t):
    return re.findall(r"[a-z0-9]+", t.lower())


PROMPT_GROUNDED = (
    "You are a support assistant. Follow these rules strictly:\n"
    "1. Every FACT in your answer must come from the context. Never invent facts.\n"
    "2. You MAY reason over the provided facts — compare, contrast, conclude.\n"
    "3. If the context lacks the needed facts, say: "
    '"I don\'t have that information — let me escalate this to a specialist."\n'
    "4. No outside knowledge.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


class Rag:
    def __init__(self):
        self.chunks, self.collection, self.bm25 = [], None, None

    def build(self):
        client = chromadb.PersistentClient(path=str(DB_DIR))
        try:
            client.delete_collection("kb-doc")
        except Exception:  # noqa: BLE001, S110
            pass
        self.collection = client.create_collection(
            "kb-doc", metadata={"hnsw:space": "cosine"}
        )
        cid = 0
        for p in DOCS_DIR.glob("*.md"):
            for c in splitter.split_text(p.read_text()):
                self.collection.add(
                    ids=[f"c{cid}"],
                    documents=[c],
                    metadatas=[{"source": p.name}],
                )
                self.chunks.append(c)
                cid += 1
        self.bm25 = BM25Okapi([tok(c) for c in self.chunks])

    def retrieve(self, q, wide_k=10, final_k=3):
        n = len(self.chunks)
        sem = self.collection.query(query_texts=[q], n_results=n)
        sr = {d: r for r, d in enumerate(sem["documents"][0])}
        sc = self.bm25.get_scores(tok(q))
        bm = sorted(range(n), key=lambda i: sc[i], reverse=True)
        br = {self.chunks[i]: r for r, i in enumerate(bm)}
        fused = {
            c: (1 / (60 + sr.get(c, 1e9)) + 1 / (60 + br.get(c, 1e9)))
            for c in self.chunks
        }
        cands = sorted(fused, key=fused.get, reverse=True)[:wide_k]
        rr = sorted(
            zip(cands, reranker.predict([[q, c] for c in cands])),
            key=lambda x: x[1],
            reverse=True,
        )
        return [c for c, _ in rr[:final_k]]

    def answer(self, q):
        retrieved = self.retrieve(q)
        ctx = "\n\n---\n\n".join(retrieved)
        resp = model.invoke(
            [HumanMessage(content=PROMPT_GROUNDED.format(context=ctx, question=q))]
        )
        return extract_text(resp), retrieved


# ---------- metrics ----------
def retrieval_hit(retrieved, marker):
    """Did the relevant chunk get retrieved? None marker = N/A (refusal case)."""
    if marker is None:
        return None
    return any(marker.lower() in c.lower() for c in retrieved)


def retrieval_rank(retrieved, marker):
    """Reciprocal rank of the relevant chunk. 0 if not found."""
    if marker is None:
        return None
    for i, c in enumerate(retrieved):
        if marker.lower() in c.lower():
            return 1.0 / (i + 1)
    return 0.0


def answer_correct(answer, case):
    a = answer.lower()
    # every group must be satisfied; a group passes if ANY alternative appears
    for group in case.get("must_contain_any", []):
        if not any(alt.lower() in a for alt in group):
            return False, f"missing any of {group}"

    for phrase in case.get("must_not_contain", []):
        if phrase.lower() in a:
            return False, f"contains forbidden '{phrase}'"

    return True, ""


def main():
    rag = Rag()
    rag.build()

    hits, rr_scores, answer_pass = [], [], []
    print(f"\n{'═' * 70}\nRAG EVALUATION\n{'═' * 70}")

    for case in EVAL_SET:
        answer, retrieved = rag.answer(case["question"])

        if case["id"] == "reconnect_time":
            print("\n[DEBUG] reconnect_time retrieved chunks:")
            for i, c in enumerate(retrieved):
                print(f"   chunk {i}: {c[:120]!r}")

        hit = retrieval_hit(retrieved, case["relevant_chunk_marker"])
        rr = retrieval_rank(retrieved, case["relevant_chunk_marker"])
        correct, reason = answer_correct(answer, case)

        if hit is not None:
            hits.append(hit)
            rr_scores.append(rr)
        answer_pass.append(correct)

        icon = "✅" if correct else "❌"
        hit_str = "N/A" if hit is None else ("hit" if hit else "MISS")
        rr_str = "N/A" if rr is None else f"{rr:.2f}"
        print(f"\n{icon} [{case['id']}]")
        print(
            f"     retrieval: {hit_str}  (RR={rr_str})   answer: {'PASS' if correct else 'FAIL — ' + reason}"
        )
        if not correct:
            print(f"     got: {answer[:100]}")
        time.sleep(4)

    # aggregate
    hit_rate = sum(hits) / len(hits) if hits else 0
    mrr = sum(rr_scores) / len(rr_scores) if rr_scores else 0
    ans_rate = sum(answer_pass) / len(answer_pass)

    print(f"\n{'═' * 70}")
    print(f"Retrieval hit rate @3 : {hit_rate:.0%}  (of answerable questions)")
    print(f"Mean reciprocal rank  : {mrr:.2f}")
    print(f"Answer pass rate      : {ans_rate:.0%}  (of all questions)")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()


# ## Final result
# 7/7 answer pass rate. Retrieval solid on answerable questions.
# The number is trustworthy — but only because the eval itself was debugged
# four times to get here (see below).

# ## Eval bugs found and fixed (the eval was wrong, not the RAG)
# 1. Chroma rejects empty metadata dict `{}` → use `{"source": p.name}`.
# 2. `must_contain` literal-string matching failed a CORRECT answer:
#    E429 answered "too many requests" where the test demanded "rate limit".
#    Fix: grouped alternatives — `must_contain_any: [["rate limit", "too many requests"]]`
#    (every group must match; any one alternative in a group satisfies it).
# 3. Key renamed `must_contain` → `must_contain_any` in the data but not in
#    `answer_correct` → `list.lower()` crash. Data shape and consumer drifted.
#    Lesson: rename a data shape, grep every consumer. Golang/Pydantic catch
#    this at the boundary; plain Python fails at runtime three functions deep.

# ## OPEN ITEM — real finding, deferred to the production project
# `reconnect_time`: eval reported retrieval MISS but answer PASS.
# Investigation (printed the actual retrieved chunks) showed the MISS was REAL —
# none of the top-3 chunks contained "sensor reconnects within 15 minutes".
# The retriever ranked GATEWAY timing chunks (5 min re-establish, 10 min down)
# above the SENSOR reconnect chunk, because "reconnect/connection/gateway/minutes"
# cluster together in embedding space (the Day 62–63 near-miss problem).

# => The correct-looking answer was likely UNGROUNDED (LLM reasoning/training,
#    not retrieval). This is the "low retrieval / high answer" danger row from
#    the Day 66 failure-mode decoder: an answer that looks right but isn't
#    supported by what was retrieved. Benign on a toy KB; a real fabrication
#    risk for the Dryad agent answering about a customer's actual device.

# ## Requirements this surfaces for the production build
# (a) GROUNDING CHECK: verify the answer's facts actually appear in the
#     retrieved chunks — not just that the answer looks correct. "Answer correct"
#     and "answer grounded" are different and can diverge silently.
# (b) RETRIEVAL TUNING: stop semantically-near chunks (gateway timing) from
#     crowding out the true chunk (sensor reconnect). Check first whether
#     chunking severed the fact, or whether it's a ranking problem.

# ## Diagnostic that caught it (reuse this reflex constantly in Phase 4)
# When a metric disagrees with observed behavior, print what ACTUALLY happened
# before trusting the number. The MISS looked like a marker bug; printing the
# retrieved chunks proved it was a real retrieval failure hiding an ungrounded
# answer. Trust the investigation, not the number.
