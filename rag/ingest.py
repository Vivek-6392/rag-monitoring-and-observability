"""
Document ingestion pipeline (Improved).

- Semantic chunking (sentence-aware)
- Better embeddings (bge-base-en)
- Cleaner chunks (filters noise)
- Richer metadata for observability
"""
from __future__ import annotations

import os
import hashlib
import re
from pathlib import Path
from typing import Union

import chromadb
from chromadb.utils import embedding_functions

# ── Config ──────────────────────────────────────────────────────
COLLECTION_NAME = "rag_docs"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50   # 🔥 increased overlap


# ── Chroma Client ───────────────────────────────────────────────
def _get_chroma_client() -> chromadb.ClientAPI:
    data_dir = os.environ.get("CHROMA_DATA_DIR", "")
    if data_dir:
        return chromadb.PersistentClient(path=data_dir)
    return chromadb.EphemeralClient()


# ── Embeddings ──────────────────────────────────────────────────
def _get_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-base-en"
    )


def get_or_create_collection() -> chromadb.Collection:
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


# ── Text Cleaning ───────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)   # strip <EOS>, <pad>, any XML-like tokens
    text = re.sub(r'\s+', ' ', text)      # normalize again after stripping
    return text.strip()


# ── Semantic Chunking ───────────────────────────────────────────
def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = clean_text(text)
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_sentences: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) <= size:
            current_sentences.append(sentence)
            current_len += len(sentence)
        else:
            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())

            # ✅ Overlap by carrying over last N characters worth of sentences
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break

            current_sentences = overlap_sentences + [sentence]
            current_len = sum(len(s) for s in current_sentences)

    if current_sentences:
        chunks.append(" ".join(current_sentences).strip())

    return [c for c in chunks if len(c) > 60]

# ── Utilities ───────────────────────────────────────────────────
def _doc_id(text: str, idx: int) -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"doc_{h}_{idx}"


# ── Ingestion ───────────────────────────────────────────────────
def ingest_text(text: str, source_name: str = "manual") -> int:
    """Ingest raw text. Returns number of chunks added."""
    collection = get_or_create_collection()

    chunks = chunk_text(text)
    if not chunks:
        return 0

    ids = [_doc_id(c, i) for i, c in enumerate(chunks)]

    metadatas = [
        {
            "source": source_name,
            "chunk_index": i,
            "length": len(chunks[i]),
        }
        for i in range(len(chunks))
    ]

    collection.upsert(
        documents=chunks,
        ids=ids,
        metadatas=metadatas,
    )

    return len(chunks)


def ingest_file(path: Union[str, Path]) -> int:
    """Ingest a .txt, .md, or .pdf file."""
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


def ingest_directory(
    directory: Union[str, Path],
    extensions: tuple = (".txt", ".pdf", ".md"),
) -> dict:
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


# ── Stats ───────────────────────────────────────────────────────
def collection_stats() -> dict:
    col = get_or_create_collection()
    return {
        "total_chunks": col.count(),
        "collection_name": COLLECTION_NAME,
    }


# ── Sample Data ─────────────────────────────────────────────────
def ingest_sample_documents() -> int:
    """Load default sample documents for testing."""
    sample_text = """
    Retrieval-Augmented Generation (RAG) combines information retrieval
    with text generation. It improves factual accuracy by grounding responses
    in external knowledge sources.

    Observability in AI systems involves tracking metrics such as latency,
    cost, and output quality. Metrics like p50 and p95 latency are critical
    for production systems.

    Context precision measures how relevant retrieved documents are to a query.
    Higher precision means less noise and better retrieval quality.
    """

    return ingest_text(sample_text, source_name="sample_data")


# ── CLI Entry ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("Ingesting sample documents...")
    n = ingest_sample_documents()
    print(f"✅ Ingested {n} chunks")
    print(collection_stats())