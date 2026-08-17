from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
import re
from langchain_core.messages import HumanMessage, SystemMessage
import chromadb
from rank_bm25 import BM25Okapi
import time

DOCS_DIR = Path(__file__).parent / "docs"
DB_DIR = Path(__file__).parent / "chroma_transform"

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def extract_text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join( p.get("text", "") for p in msg.content
                    if isinstance(p, dict) and p.get("type")== "text")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=40,
    separators=["\n## ", "\n\n", "\n", ". ", " ", ""],)

def simple_tokenize(text:str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ----------------- Query Rewritting ------------------------

def rewrite_query(raw:str) -> str:
    """Turn a messy/vague query into a clear, retrievable one."""
    prompt = (
        "Rewrite this customer support message as a clear, specific search query. "
        "Fix typos, remove filler, make the topic explicit. "
        "Output ONLY the rewritten query, nothing else.\n\n"
        f"Message: {raw}\n\nRewritten query:"
    )

    resp = model.invoke([HumanMessage(content=prompt)])
    rewritten = extract_text(resp).strip()

    return rewritten or raw     # fall back to original if empty


# -------------------- Query Decomposition ------------------------

def decompose_query(raw:str) -> list[str]:
    """Split a multi-part question into standalone sub-queries."""
    prompt = (
        "If this customer message contains MULTIPLE separate questions or problems, "
        "split it into standalone questions, one per line. "
        "If it's a single question, just return it unchanged on one line. "
        "Output ONLY the questions, no numbering, no extra text.\n\n"
        f"Message: {raw}\n\nQuestions:"
    )

    resp = model.invoke([HumanMessage(content=prompt)])
    lines = [l.strip() for l in extract_text(resp).split("\n") if l.strip()]
    return lines or [raw]

class RagIndex:
    def __init__(self):
        self.chunks: list[str] = []
        self.collection = None
        self.bm25 = None

    def build(self):
        client = chromadb.PersistentClient(path=str(DB_DIR))
        try:
            client.delete_collection("kb-doc")
        except Exception:
            pass
        self.collection = client.create_collection("kb-doc", metadata={"hnsw:space": "cosine"})
        cid = 0
        for doc_path in DOCS_DIR.glob("*.md"):
            for chunk in splitter.split_text(doc_path.read_text()):
                self.collection.add(ids=[f"chunk_{cid}"], documents=[chunk],
                                    metadatas=[{"source": doc_path.name}])
                self.chunks.append(chunk)
                cid += 1
        self.bm25 = BM25Okapi([simple_tokenize(c) for c in self.chunks])
        print(f"Indexed {cid} chunks")

    def _hybrid(self, query: str, wide_k: int) -> list[str]:
        n = len(self.chunks)
        sem = self.collection.query(query_texts=[query], n_results=n) # type: ignore
        sem_rank = {doc: r for r, doc in enumerate(sem["documents"][0])} # type: ignore
        scores = self.bm25.get_scores(simple_tokenize(query)) # type: ignore
        bm_ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        bm_rank = {self.chunks[i]: r for r, i in enumerate(bm_ranked)}
        k = 60
        fused = {}
        for c in self.chunks:
            s = 0.0
            if c in sem_rank: s += 1.0 / (k + sem_rank[c])
            if c in bm_rank:  s += 1.0 / (k + bm_rank[c])
            fused[c] = s
        return sorted(fused, key=fused.get, reverse=True)[:wide_k] # type: ignore

    def _rerank(self, query: str, candidates: list[str], final_k: int) -> list[str]:
        if not candidates:
            return []
        pairs = [[query, c] for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:final_k]]

    def retrieve(self, query: str, wide_k: int = 10, final_k: int = 3) -> list[str]:
        return self._rerank(query, self._hybrid(query, wide_k), final_k)

    def retrieve_with_transform(self, raw_query:str, final_k: int = 3) -> dict:
        """Decompose → rewrite each part → retrieve each → merge."""
        sub_queries = decompose_query(raw=raw_query)
        all_chunks = []
        trace = []

        for sub in sub_queries:
            rewritten = rewrite_query(sub)
            chunks = self.retrieve(rewritten, final_k=final_k)
            all_chunks.extend(chunks)
            trace.append({"sub": sub, "rewritten":rewritten, "chunks":chunks})

        seen = set()
        merged = []
        for c in all_chunks:
            if c not in seen:
                seen.add(c)
                merged.append(c)

        return {"chunks": merged, "trace": trace, "sub_queries": sub_queries}

    def answer(self, raw_query:str, use_transform: bool=True):
        if use_transform:
            result = self.retrieve_with_transform(raw_query=raw_query)
            retrieved = result["chunks"]
            trace = result["trace"]
        else:
            retrieved = self.retrieve(raw_query)
            trace = None

        context = "\n\n---\n\n".join(retrieved)
        prompt = (
            f"Answer using ONLY the context. If it's not there, say you don't have that info.\n\n"
            f"Context:\n{context}\n\nQuestion: {raw_query}"
        )

        resp = model.invoke([
            SystemMessage(content="Support assistant. Answer only from context."),
            HumanMessage(content=prompt),
        ])

        return extract_text(resp), trace

def code_label(chunk: str) -> str:
    codes = re.findall(r"E\d{3}", chunk)
    if codes:
        return ",".join(dict.fromkeys(codes))
    return next((l for l in chunk.split("\n") if l.strip()), chunk)[:35]

if __name__ == "__main__":
    idx = RagIndex()
    idx.build()

    messy_queries = [
        "how do i fix the rate limit thing",          # concept, no code → rewrite helps
        "my snesor wont conect and i got no alert",    # typos + two problems
        "its broken the light is red",                 # vague → rewrite to LED query
    ]

    for q in messy_queries:
        print(f"\n{'═'*68}\n🧑 RAW: {q}\n{'─'*68}")
        ans, trace = idx.answer(q, use_transform=True)

        if trace:
            for t in trace:
                print(f"  sub-query : {t['sub']}")
                print(f"  rewritten : {t['rewritten']}")
                print(f"  retrieved : {[code_label(c) for c in t['chunks']]}")
        print(f"\n🤖 {ans}")
        time.sleep(5)



# ---------------------------- RESULT -----------------------------------------------



# Indexed 18 chunks

# ════════════════════════════════════════════════════════════════════
# 🧑 RAW: how do i fix the rate limit thing
# ────────────────────────────────────────────────────────────────────
#   sub-query : how do i fix the rate limit thing
#   rewritten : How to resolve API rate limit errors
#   retrieved : ['E429', 'E401', 'E503']

# 🤖 To fix the rate limit, wait 60 seconds before retrying and implement exponential backoff.

# ════════════════════════════════════════════════════════════════════
# 🧑 RAW: my snesor wont conect and i got no alert
# ────────────────────────────────────────────────────────────────────
#   sub-query : my sensor wont connect
#   rewritten : Troubleshoot sensor connection issues
#   retrieved : ['No LED at all means the sensor has ', '# Sensor Troubleshooting', '## Sensor Offline']
#   sub-query : i got no alert
#   rewritten : How to enable push notifications for account alerts
#   retrieved : ['If you did not receive an expected ', 'notify, to reduce false alarms. Che', '## Missed Alerts']

# 🤖 If your sensor has no LED at all, it has no power; check the wiring harness. If you have a solid green LED, the sensor may have lost its LoRaWAN connection to the gateway; sensors reconnect automatically within 15 minutes.

# Regarding missed alerts, first verify the alert was actually triggered by checking the event log. An alert only sends notifications if the detection confidence exceeds 80%. Check your notification settings to confirm your contact method is verified.

# ════════════════════════════════════════════════════════════════════
# 🧑 RAW: its broken the light is red
# ────────────────────────────────────────────────────────────────────
#   sub-query : It is broken
#   rewritten : Troubleshoot product malfunction
#   retrieved : ['# Sensor Troubleshooting', 'E503', 'No LED at all means the sensor has ']
#   sub-query : The light is red
#   rewritten : Troubleshooting red status light on device
#   retrieved : ['# Sensor Troubleshooting', '## Sensor Offline', 'E503']

# 🤖 A blinking red LED means the battery is below 15% and must be replaced.




# -------------------------- FINDINGS --------------------------------------------




# Day 64 query transformation:
#   Q1 rate-limit  : rewrite surfaced E429 concept → correct ✅
#   Q2 typo+2part  : decompose split correctly, typos fixed, BOTH answered ✅ (the win)
#   Q3 red light   : decompose WRONGLY split 1 problem into 2, invented
#                    "It is broken"→generic. Rescued by other branch.
# Findings:
#   - Decomposition is non-deterministic: right on Q2, wrong on Q3.
#   - Rewrite DRIFTS: "i got no alert"→"enable notifications" added an
#     assumption the customer didn't make. Retrieved right chunk by luck.
#   - Fixes: (1) GATE decomposition on multi-part signals (Q3 had no "and").
#            (2) Constrain rewrite prompt: fix typos only, add no assumptions.
#   - Dryad risk: drift→confident wrong-question answer. Must gate+constrain.