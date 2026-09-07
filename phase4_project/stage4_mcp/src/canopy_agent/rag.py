"""Docs RAG for the canopy agent. Grounding gate lives here.

build_index() is isolated from querying so the doc SOURCE can be swapped
later (local snapshot now → fetched-from-docs.dryad.app later) without
touching the query path."""

import logging
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger("rag")

DOCS_DIR = Path(__file__).parent.parent.parent / "docs"

_splitter = RecursiveCharacterTextSplitter(
    separators=["\n## ", "\n# ", "\n\n", "\n", ". ", " ", ""],
    chunk_size=400,
    chunk_overlap=80,
)

# chroma cosine distance above this = "not really about this question"
MAX_DISTANCE = 1.0

_collection = None


class RagResult:
    def __init__(
        self,
        grounded: bool,
        chunks: list[str] | None = None,
        reason: str = "",
    ):

        self.grounded = grounded
        self.chunks = chunks or []
        self.reason = reason


def _load_documents() -> list[tuple[str, str]]:
    """Return (source_name, text) pairs. THIS is the swappable source.

    Today: local .md files. Later: fetch from docs.dryad.app.
    Only this function changes when moving to live docs.
    """

    docs = []
    for md in DOCS_DIR.glob("*.md"):
        docs.append((md.name, md.read_text()))

    return docs


def build_index():
    """(Re)build the vector index from the current document source."""

    global _collection
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("docs")
    except NotFoundError as e:
        log.debug("no existing docs collection to delete: %s", e)

    coll = client.create_collection("docs", metadata={"hnsw:space": "cosine"})
    cid = 0
    for source, text in _load_documents():
        for chunk in _splitter.split_text(text=text):
            coll.add(ids=[f"c{cid}"], documents=[chunk], metadatas=[{"source": source}])
            cid += 1

    _collection = coll
    return coll


def _get_collection():
    return _collection if _collection is not None else build_index()


def search_docs_gated(question: str, top_k: int = 3) -> RagResult:
    """Retrieve docs and GATE on relevance. Never raises."""

    coll = _get_collection()
    res = coll.query(query_texts=[question], n_results=top_k)

    docs = res["documents"][0]
    distances = res["distances"][0]

    relevant = [
        d for d, dist in zip(docs, distances, strict=True) if dist <= MAX_DISTANCE
    ]

    if not relevant:
        return RagResult(False, reason="no relevant documentation found")

    return RagResult(True, chunks=relevant)
