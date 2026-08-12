import os
import time
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent / "docs"
DB_DIR = Path(__file__).parent / "chroma_hybrid"

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)


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
    """Lowercase, keep alphanumerics. Good enough for keyword matching."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridIndex:
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

        # BM25 index over the same chunks
        tokenized = [simple_tokenize(c) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"Indexed {cid} chunks (semantic + BM25)")

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        # --- semantic scores ---
        sem = self.collection.query(query_texts=[question], n_results=len(self.chunks)) # type: ignore
        sem_ids = sem["ids"][0]
        sem_docs = sem["documents"][0] # type: ignore
        sem_dists = sem["distances"][0] # type: ignore
        # convert distance → similarity score, normalize to rank position
        sem_rank = {doc: rank for rank, doc in enumerate(sem_docs)}

        # --- keyword scores ---
        q_tokens = simple_tokenize(question)
        bm25_scores = self.bm25.get_scores(q_tokens) # type: ignore
        bm25_ranked = sorted(range(len(self.chunks)),
                             key=lambda i: bm25_scores[i], reverse=True)
        bm25_rank = {self.chunks[i]: rank for rank, i in enumerate(bm25_ranked)}

        # --- Reciprocal Rank Fusion (RRF) ---
        # combine by rank, not raw score — avoids scale mismatch
        k = 60  # RRF constant, standard default
        fused = {}
        for chunk in self.chunks:
            score = 0.0
            if chunk in sem_rank:
                score += 1.0 / (k + sem_rank[chunk])
            if chunk in bm25_rank:
                score += 1.0 / (k + bm25_rank[chunk])
            fused[chunk] = score

        top = sorted(fused, key=fused.get, reverse=True)[:top_k] # type: ignore
        return top

    def answer(self, question: str, top_k: int = 3):
        retrieved = self.retrieve(question, top_k)
        context = "\n\n---\n\n".join(retrieved)
        prompt = (
            f"Answer using ONLY the context. If it's not there, say you don't have that info.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        resp = model.invoke([
            SystemMessage(content="Support assistant. Answer only from context."),
            HumanMessage(content=prompt),
        ])
        return extract_text(resp), retrieved


if __name__ == "__main__":
    idx = HybridIndex()
    idx.build()
    questions = [
        "What does error E429 mean?",
        "How do I fix E507?",
        "My API key stopped working, what error is that?",
    ]
    for q in questions:
        ans, retrieved = idx.answer(q)
        print(f"\n{'═'*66}\nQ: {q}\n{'─'*66}\nA: {ans}")
        print("📎 Retrieved:")
        for i, c in enumerate(retrieved, 1):
            first = next((ln for ln in c.split("\n") if ln.strip()), c)
            print(f"   {i}. {first[:55]}")
        time.sleep(4)