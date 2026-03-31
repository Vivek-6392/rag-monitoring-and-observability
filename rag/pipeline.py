"""
Instrumented RAG pipeline.
Each call records latency, tokens, cost, and quality scores to SQLite.
"""
from __future__ import annotations
import os
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from .ingest import get_or_create_collection
from observability.tracer import Tracer
from observability.metrics import build_metrics_record
from observability import store

DEFAULT_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

# ── Prompt ─────────────────────────
SYSTEM_PROMPT = """You are a RAG assistant.

Use the provided context to answer the question thoroughly.

Rules:
- Answer strictly and only from the context — never add outside knowledge
- Every statement you make must be directly supported by the context
- Explain concepts fully using details present in the context
- If multiple aspects are covered in context, address all of them
- If context is insufficient, say: "The provided documents do not contain enough information."
"""

# ── Retrieval ───────────────────────────────────────────────────
def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    collection = get_or_create_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    query_words = set(query.lower().split())

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        semantic_score = 1 - dist
        doc_words = set(doc.lower().split())

        # ✅ Normalize overlap to [0, 1] using Jaccard similarity
        union = query_words | doc_words
        overlap_score = len(query_words & doc_words) / len(union) if union else 0

        combined_score = (0.7 * semantic_score) + (0.3 * overlap_score)

        docs.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "score": combined_score,
        })

    docs = sorted(docs, key=lambda x: x["score"], reverse=True)

    # Lower threshold — generic queries ("summarize", "key findings") have
    # naturally low Jaccard overlap with technical content
    docs = [d for d in docs if d["score"] > 0.15]

    # Safety net: never return empty if the collection has content
    if not docs and results["documents"][0]:
        # Fall back to top semantic hit only
        top = results["documents"][0][0]
        top_meta = results["metadatas"][0][0]
        docs = [{"text": top, "source": top_meta.get("source", "unknown"), "score": 1 - results["distances"][0][0]}]

    return docs[:k]


# ── Generation ──────────────────────────────────────────────────
def generate(query: str, context_docs: list[dict], model: str, api_key: str) -> str:

    # 🔥 keep only strong context
    # context_docs = context_docs[:3]

    # fallback if everything filtered
    if not context_docs:
        return "No relevant information found in the documents."

    context_text = "\n\n---\n\n".join(
        f"[Source: {d['source']}]\n{d['text']}" for d in context_docs
    )

    llm = ChatGroq(model=model, groq_api_key=api_key, temperature=0.1)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"""Use the context below to give a complete answer.
        Every claim in your answer must be traceable to the context.

        Context:
        {context_text}

        Question: {query}

        Answer (context-grounded only):"""
        ),
    ]

    response = llm.invoke(messages)
    return response.content

# ── Main Pipeline ───────────────────────────────────────────────
def run_rag(
    query: str,
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    eval_quality: bool = True,
) -> dict:

    api_key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")

    tracer = Tracer()

    # ── Retrieval ────────────────────────────────────────────────
    span_ret = tracer.start_span("retrieval")
    docs = retrieve(query)
    span_ret.end(num_docs=len(docs), top_score=docs[0]["score"] if docs else 0)

    if not docs:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "metrics": {},
            "trace": tracer.summary(),
        }

    # ── Generation ───────────────────────────────────────────────
    span_gen = tracer.start_span("generation")
    answer = generate(query, docs, model, api_key)
    span_gen.end(model=model)

    latency_ret_ms = span_ret.duration_ms
    latency_gen_ms = span_gen.duration_ms

    # ── Metrics ──────────────────────────────────────────────────
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

    # ── Safe Quality Evaluation ──────────────────────────────────
    quality_scores = {}

    if eval_quality and docs:
        try:
            from evaluation.ragas_eval import evaluate_response

            result = evaluate_response(
                query=query,
                answer=answer,
                contexts=context_texts,
                api_key=api_key,
                model=model,
            )

            if isinstance(result, dict):
                quality_scores = result
            else:
                quality_scores = {}

            record.update(
                faithfulness=quality_scores.get("faithfulness"),
                answer_relevancy=quality_scores.get("answer_relevancy"),
                context_precision=quality_scores.get("context_precision"),
                quality_avg=quality_scores.get("quality_avg"),
            )

        except Exception as e:
            print(f"[WARN] Evaluation failed: {e}")
            quality_scores = {}

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