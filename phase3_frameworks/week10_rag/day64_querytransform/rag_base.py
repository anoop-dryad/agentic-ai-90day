from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import chromadb
from rank_bm25 import BM25Okapi
from langchain_core.messages import SystemMessage, HumanMessage
import time

DOCS_DIR = Path(__file__).parent / "docs"
DB_DIR = Path(__file__).parent / "chroma_rerank"


model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)


# The cross-encoder re-ranker. Small, fast, runs locally.
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

def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class RerankIndex:
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

    def _hybrid_candidates(self, question: str, wide_k: int) -> list[str]:
        """Same RRF hybrid as Day 62, but return a WIDE candidate set."""
        n = len(self.chunks)
        sem = self.collection.query(query_texts=[question], n_results=n) # type: ignore
        sem_rank = {doc: r for r, doc in enumerate(sem["documents"][0])} # type: ignore

        q_tokens = simple_tokenize(question)
        scores = self.bm25.get_scores(q_tokens) # type: ignore
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

    def retrieve(self, question: str, wide_k: int = 10, final_k: int = 3):
        """Retrieve wide, then re-rank down to final_k."""
        candidates = self._hybrid_candidates(question, wide_k)

        # cross-encoder scores each (query, chunk) PAIR jointly
        pairs = [[question, c] for c in candidates]
        rerank_scores = reranker.predict(pairs)

        # sort candidates by re-rank score, keep the best final_k
        ranked = sorted(zip(candidates, rerank_scores),
                        key=lambda x: x[1], reverse=True)
        return ranked[:final_k], candidates  # return both for comparison

    def answer(self, question: str):
        ranked, candidates = self.retrieve(question)
        retrieved = [c for c, _ in ranked]
        context = "\n\n---\n\n".join(retrieved)
        prompt = (
            f"Answer using ONLY the context. If it's not there, say you don't have that info.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        resp = model.invoke([
            SystemMessage(content="Support assistant. Answer only from context."),
            HumanMessage(content=prompt),
        ])
        return extract_text(resp), ranked, candidates

def code_label(chunk: str) -> str:
    codes = re.findall(r"E\d{3}", chunk)
    return ",".join(dict.fromkeys(codes)) if codes else chunk.split("\n")[0][:30]

if __name__ == "__main__":
    idx = RerankIndex()
    idx.build()

    questions = [
        "What does error E429 mean?",
        "How do I fix E507?",
        "My API key stopped working, what error is that?",
    ]
    for q in questions:
        ans, ranked, candidates = idx.answer(q)
        print(f"\n{'═'*66}\nQ: {q}\n{'─'*66}\nA: {ans}")
        print(f"\n  Wide candidates (hybrid, pre-rerank): "
              f"{[code_label(c) for c in candidates]}")
        print(f"  After re-rank (top-3, scored):")
        for c, s in ranked:
            print(f"     {s:6.2f}  [{code_label(c)}]")
        time.sleep(4)