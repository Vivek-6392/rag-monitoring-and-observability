"""
Instrumented RAG pipeline.
Each call records latency, tokens, cost, and quality scores to SQLite.
"""
from __future__ import annotations
import os
import time
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from .ingest import get_or_create_collection
from observability.tracer import Tracer
from observability.metrics import build_metrics_record
from observability import store

DEFAULT_MODEL = "llama-3.1-70b-versatile"
TOP_K = 5

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Always be concise and cite which part of the context supports your answer."""


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Query ChromaDB and return top-k results with metadata."""
    collection = get_or_create_collection()
    if collection.count() == 0:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "score": round(1 - dist, 4),  # cosine similarity
            }
        )
    return docs


def generate(query: str, context_docs: list[dict], model: str, api_key: str) -> str:
    """Call Groq LLM with context-augmented prompt."""
    context_text = "\n\n---\n\n".join(
        f"[Source: {d['source']}]\n{d['text']}" for d in context_docs
    )
    llm = ChatGroq(model=model, groq_api_key=api_key, temperature=0.1)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context_text}\n\nQuestion: {query}"
        ),
    ]
    response = llm.invoke(messages)
    return response.content


def run_rag(
    query: str,
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    eval_quality: bool = True,
) -> dict:
    """
    Full RAG pipeline with tracing.
    Returns a result dict with answer, sources, metrics, and trace.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")

    tracer = Tracer()

    # ── Retrieval ──────────────────────────────────────────────────
    span_ret = tracer.start_span("retrieval")
    docs = retrieve(query)
    span_ret.end(num_docs=len(docs), top_score=docs[0]["score"] if docs else 0)

    if not docs:
        return {
            "answer": "No relevant documents found. Please ingest some documents first.",
            "sources": [],
            "metrics": {},
            "trace": tracer.summary(),
        }

    # ── Generation ────────────────────────────────────────────────
    span_gen = tracer.start_span("generation")
    answer = generate(query, docs, model, api_key)
    span_gen.end(model=model)

    latency_ret_ms = span_ret.duration_ms
    latency_gen_ms = span_gen.duration_ms

    # ── Metrics record ────────────────────────────────────────────
    context_texts = [d["text"] for d in docs]
    record = build_metrics_record(
        query=query,
        answer=answer,
        context_docs=context_texts,
        model=model,
        latency_retrieval_ms=latency_ret_ms,
        latency_generation_ms=latency_gen_ms,
        trace_id=tracer.trace_id,
    )

    # ── Quality evaluation (async-style, but sync here) ──────────
    quality_scores = {}
    if eval_quality and docs:
        try:
            from evaluation.ragas_eval import evaluate_response
            quality_scores = evaluate_response(
                query=query,
                answer=answer,
                contexts=context_texts,
                api_key=api_key,
                model=model,
            )
            record.update(
                faithfulness=quality_scores.get("faithfulness"),
                answer_relevancy=quality_scores.get("answer_relevancy"),
                context_precision=quality_scores.get("context_precision"),
                quality_avg=quality_scores.get("quality_avg"),
            )
        except Exception:
            pass  # quality eval is non-blocking

    store.log_request(record)

    return {
        "answer": answer,
        "sources": docs,
        "metrics": {
            "latency_retrieval_ms": round(latency_ret_ms, 1),
            "latency_generation_ms": round(latency_gen_ms, 1),
            "latency_total_ms": round(latency_ret_ms + latency_gen_ms, 1),
            "tokens_total": record.get("tokens_total"),
            "cost_usd": record.get("cost_usd"),
            **quality_scores,
        },
        "trace": tracer.summary(),
    }