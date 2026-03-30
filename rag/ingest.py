"""
Document ingestion pipeline.
Supports plain text files, PDFs (via pypdf), and raw strings.
Chunks → embeds → stores in ChromaDB.
"""
from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import Union

import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "rag_docs"
CHUNK_SIZE = 500          # characters
CHUNK_OVERLAP = 50        # characters


def _get_chroma_client() -> chromadb.ClientAPI:
    # Use persistent client if DATA_DIR is set, else in-memory
    data_dir = os.environ.get("CHROMA_DATA_DIR", "")
    if data_dir:
        return chromadb.PersistentClient(path=data_dir)
    return chromadb.EphemeralClient()


def _get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def get_or_create_collection() -> chromadb.Collection:
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 20]


def _doc_id(text: str, idx: int) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"doc_{h}_{idx}"


def ingest_text(text: str, source_name: str = "manual") -> int:
    """Ingest raw text. Returns number of chunks added."""
    collection = get_or_create_collection()
    chunks = chunk_text(text)
    if not chunks:
        return 0
    ids = [_doc_id(c, i) for i, c in enumerate(chunks)]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]
    # upsert avoids duplicates on re-run
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def ingest_file(path: Union[str, Path]) -> int:
    """Ingest a .txt or .pdf file."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            raise RuntimeError("pypdf not installed. Run: pip install pypdf")
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return ingest_text(text, source_name=path.name)


def ingest_directory(directory: Union[str, Path], extensions: tuple = (".txt", ".pdf", ".md")) -> dict:
    directory = Path(directory)
    results = {}
    for p in directory.rglob("*"):
        if p.suffix.lower() in extensions:
            try:
                n = ingest_file(p)
                results[str(p)] = n
            except Exception as e:
                results[str(p)] = f"ERROR: {e}"
    return results


def collection_stats() -> dict:
    col = get_or_create_collection()
    return {"total_chunks": col.count(), "collection_name": COLLECTION_NAME}

def ingest_sample_documents() -> int:
    """Load default sample documents for quick testing."""
    sample_text = """
    RAG (Retrieval-Augmented Generation) combines retrieval with generation.
    It improves factual accuracy by grounding responses in external data.
    
    Observability in AI systems includes tracking latency, cost, and quality.
    p50 and p95 latency are important production metrics.
    """

    return ingest_text(sample_text, source_name="sample_data")