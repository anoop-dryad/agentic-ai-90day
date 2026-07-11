"""Long-term memory backed by Chroma."""

import chromadb
from pathlib import Path
from uuid import uuid4

DB_PATH = Path(__file__).parent / "chroma_db"

_client = chromadb.PersistentClient(path=str(DB_PATH))
_collection = _client.get_or_create_collection(
    name="agent_memory",
    metadata={"hnsw:space": "cosine"},  # explicit — matches the geometry from Day 17
)


def remember(user_message: str, assistant_reply: str) -> None:
    """Store a completed user↔assistant exchange."""
    text = f"User: {user_message}\nAssistant: {assistant_reply}"
    _collection.add(
        ids=[str(uuid4())],
        documents=[text],
        metadatas=[{"user_message": user_message, "assistant_reply": assistant_reply}],
    )


def recall(query: str, top_k: int = 3) -> list[str]:
    """Return the top-K most semantically similar past exchanges."""
    if _collection.count() == 0:
        return []
    result = _collection.query(query_texts=[query], n_results=min(top_k, _collection.count()))
    return result["documents"][0]   # type: ignore # list of matching document strings


def clear() -> None:
    """Wipe memory (useful for testing)."""
    global _collection
    _client.delete_collection("agent_memory")
    _collection = _client.get_or_create_collection(
        name="agent_memory", metadata={"hnsw:space": "cosine"}
    )