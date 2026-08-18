"""Day 65 — grounding: the prompt spectrum from loose to structured."""

import os
import re
import time
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent / "docs"
DB_DIR = Path(__file__).parent / "chroma_grounding"

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def extract_text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(p.get("text", "") for p in msg.content
                   if isinstance(p, dict) and p.get("type") == "text")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=40,
    separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
)
def simple_tokenize(t): return re.findall(r"[a-z0-9]+", t.lower())

# ---------- THREE GROUNDING PROMPTS ----------

# Level 1 — loose. Allows the model to "help" beyond the context.
PROMPT_LOOSE = (
    "You are a helpful support assistant. Use the context below to answer. "
    "Context:\n{context}\n\nQuestion: {question}"
)

# Level 2 — strict (Day 61's version). Stops hallucination, causes over-refusal.
PROMPT_STRICT = (
    "Answer using ONLY the context below. If the context doesn't contain the "
    "answer, say you don't have that information.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

# Level 3 — structured. Forbids inventing FACTS, permits REASONING over them.
PROMPT_GROUNDED = (
    "You are a support assistant. Follow these rules strictly:\n"
    "1. Every FACT in your answer must come from the context below. "
    "Never state a fact that isn't in the context.\n"
    "2. You MAY reason over the provided facts — compare them, contrast them, "
    "or draw a conclusion they jointly support.\n"
    "3. If the context does not contain the facts needed, say: "
    "\"I don't have that information — let me escalate this to a specialist.\"\n"
    "4. Do not use knowledge from outside the context, even if you know the answer.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

class RagIndex:
    def __init__(self):
        self.chunks, self.collection, self.bm25 = [], None, None

    def build(self):
        client = chromadb.PersistentClient(path=str(DB_DIR))
        try: client.delete_collection("kb-docs")
        except Exception: pass
        self.collection = client.create_collection("kb-docs", metadata={"hnsw:space": "cosine"})
        cid = 0
        for p in DOCS_DIR.glob("*.md"):
            for c in splitter.split_text(p.read_text()):
                self.collection.add(ids=[f"c{cid}"], documents=[c],
                                    metadatas=[{"source": p.name}])
                self.chunks.append(c); cid += 1
        self.bm25 = BM25Okapi([simple_tokenize(c) for c in self.chunks])
        print(f"Indexed {cid} chunks")

    def retrieve(self, query, wide_k=10, final_k=3):
        n = len(self.chunks)
        sem = self.collection.query(query_texts=[query], n_results=n) # type: ignore
        sem_rank = {d: r for r, d in enumerate(sem["documents"][0])} # type: ignore
        scores = self.bm25.get_scores(simple_tokenize(query))   # type: ignore
        bm = sorted(range(n), key=lambda i: scores[i], reverse=True)
        bm_rank = {self.chunks[i]: r for r, i in enumerate(bm)}
        fused = {}
        for c in self.chunks:
            s = 0.0
            if c in sem_rank: s += 1/(60+sem_rank[c])
            if c in bm_rank:  s += 1/(60+bm_rank[c])
            fused[c] = s
        cands = sorted(fused, key=fused.get, reverse=True)[:wide_k] # type: ignore
        pairs = [[query, c] for c in cands]
        rr = sorted(zip(cands, reranker.predict(pairs)), key=lambda x: x[1], reverse=True)
        return [c for c, _ in rr[:final_k]]

    def answer(self, question, prompt_template):
        retrieved = self.retrieve(question)
        context = "\n\n---\n\n".join(retrieved)
        prompt = prompt_template.format(context=context, question=question)
        resp = model.invoke([HumanMessage(content=prompt)])
        return extract_text(resp)

if __name__ == "__main__":
    idx = RagIndex()
    idx.build()

    cases = [
        ("inference",     "My sensor has no LED at all. Is that the same as a dead battery?"),
        ("hallucination", "Can I use my sensors underwater?"),
        ("straight",      "What does error E429 mean?"),
        ("partial",       "What's the warranty period on a sensor?"),  # NOT in docs
    ]

    prompts = [
        ("LOOSE", PROMPT_LOOSE),
        ("STRICT", PROMPT_STRICT),
        ("GROUNDED", PROMPT_GROUNDED),
    ]

    for label, q in cases:
        print(f"\n{'█'*70}\nCASE [{label}]: {q}\n{'█'*70}")
        for pname, ptemplate in prompts:
            ans = idx.answer(q, ptemplate)
            print(f"\n  ── {pname} ──")
            print(f"  {ans}")
            time.sleep(4)