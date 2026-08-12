from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from langchain_core.messages import HumanMessage, SystemMessage

DOCS_DIR = Path(__file__).parent / "docs"
DB_DIR = Path(__file__).parent / "chroma_recursive"

model=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,)


def extract_text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join( p.get("text", "") for p in msg.content
                   if isinstance(p, dict) and p.get("type") == "text")


# -- recursive chunking -----
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    # tries these separators in order — paragraph, line, sentence, word
    separators=["\n\n", "\n", ". ", " ", ""],
)

def build_index():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(name="kb-doc")
    except Exception:
        pass

    collection = client.create_collection(name="kb-doc", metadata={"hnsw:space": "cosine"})

    chunk_id = 0
    for doc_path in DOCS_DIR.glob("*.md"):
        text = doc_path.read_text()

        chunks = splitter.split_text(text=text)
        for chunk in chunks:
            collection.add(
                ids=[f"chunk_{chunk_id}"], 
                documents=[chunk], 
                metadatas=[{"source":doc_path.name}]
            )
            chunk_id += 1

    print(f"Indexed {chunk_id} chunks (recursive, size=400, overlap=80)")
    return collection

def answer(collection, question: str, top_k:int = 3) -> dict:

    results = collection.query(query_texts=[question], n_results=top_k)
    retrieved = results["documents"][0]

    context = "\n\n---\n\n".join(retrieved)
    prompt = (
        f"Answer the question using ONLY the context below. "
        f"If the context doesn't contain the answer, say you don't have that information.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    resp = model.invoke([
        SystemMessage(content="You are a support assistant. Answer only from provided context."),
        HumanMessage(content=prompt),
    ])

    return {"question": question, "answer": extract_text(resp), "retrieved": retrieved}

if __name__ == "__main__":
    collection = build_index()

    questions = [
        "My sensor has no LED light at all. Is that the same as a dead battery?",  # the breaker
        "My sensor is offline and showing a blinking red LED. What's wrong?",
        "How long until a sensor reconnects on its own?",
        "Why didn't I get an alert for a detection?",
        "Can I use my sensors underwater?",   # hallucination test — must still refuse
    ]

    for q in questions:
        result = answer(collection, q)
        print(f"\n{'═' * 66}\nQ: {q}\n{'─' * 66}")
        print(f"A: {result['answer']}")
        print(f"\n📎 Retrieved ({len(result['retrieved'])} chunks):")
        for i, chunk in enumerate(result["retrieved"], 1):
            print(f"   {i}. {chunk[:70]}...")