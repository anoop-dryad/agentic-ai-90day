from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
import time

DOCS_DIR = Path(__file__).parent / "docs"
DB_DIR = Path(__file__).parent / "chroma_symantic"

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

def extract_text(msg)-> str:

    if isinstance(msg.content, str):
        return msg.content
    return "".join( p.get("text", "") for p in msg.content
                   if isinstance(p, dict) and p.get("type")=="text")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""])

def build_index():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(name="kb-doc")
    except Exception:
        pass

    collection = client.create_collection(name="kb-doc", metadata={"hnsw:space": "cosine"})

    chunk_id=0
    chunks_store = []
    for doc_path in DOCS_DIR.glob("*.md"):
        for chunk in splitter.split_text(text=doc_path.read_text()):
            collection.add(ids=[f"chunk_{chunk_id}"], documents=[chunk], metadatas=[{"source": doc_path.name}])
            chunks_store.append(chunk)
            chunk_id +=1

    print(f"Indexed {chunk_id} chunks")
    return collection, chunks_store

def retrieve_semantic(collection, question, top_k:int=3):
    res = collection.query(query_texts=[question], n_results=top_k)
    return res["documents"][0]

def answer(collection, question, top_k:int=3):
    retrieved = retrieve_semantic(collection=collection, question=question, top_k=top_k)
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
    collection, _ = build_index()
    questions = [
        "What does error E429 mean?",
        "How do I fix E507?",
        "My API key stopped working, what error is that?",  # semantic strength — no code given
    ]
    for q in questions:
        ans, retrieved = answer(collection, q)
        print(f"\n{'═'*66}\nQ: {q}\n{'─'*66}\nA: {ans}")
        print("📎 Retrieved:")
        for i, c in enumerate(retrieved, 1):
            first_line = c.split("\n")[0] if c.split("\n")[0] else c.split("\n")[1]
            print(f"   {i}. {first_line[:55]}")
        time.sleep(4)