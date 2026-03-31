"""
LLM-as-judge quality evaluation — faithfulness, answer relevancy, context precision.
Uses the same Groq model as the RAG pipeline, so no extra API keys needed.
Scores are 0.0–1.0.
"""
from __future__ import annotations
import json
import os
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

_JUDGE_SYSTEM = """You are an objective evaluator for RAG (Retrieval-Augmented Generation) systems.
You will receive a question, a context (retrieved documents), and a generated answer.
Respond ONLY with a valid JSON object — no markdown, no explanation."""


def _judge(prompt: str, api_key: str, model: str) -> dict:
    llm = ChatGroq(model=model, groq_api_key=api_key, temperature=0)
    response = llm.invoke([
        SystemMessage(content=_JUDGE_SYSTEM),
        HumanMessage(content=prompt),
    ])
    raw = response.content.strip()
    # Strip possible markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def score_faithfulness(answer: str, contexts: list[str], api_key: str, model: str) -> float:
    """Does every claim in the answer appear in the contexts?"""
    context_text = "\n---\n".join(contexts[:3])
    prompt = f"""
Evaluate FAITHFULNESS: Does the answer contain ONLY information from the context?

Context:
{context_text}

Answer:
{answer}

Score from 0.0 (completely hallucinated) to 1.0 (fully grounded in context).
Return JSON: {{"score": <float>, "reason": "<one sentence>"}}
"""
    result = _judge(prompt, api_key, model)
    return float(result.get("score", 0.5))


def score_answer_relevancy(query: str, answer: str, api_key: str, model: str) -> float:
    """Does the answer address the question?"""
    prompt = f"""
Evaluate ANSWER RELEVANCY: Does the answer directly address the question?

Question: {query}
Answer: {answer}

Score from 0.0 (completely off-topic) to 1.0 (perfectly addresses the question).
Return JSON: {{"score": <float>, "reason": "<one sentence>"}}
"""
    result = _judge(prompt, api_key, model)
    return float(result.get("score", 0.5))


def score_context_precision(query: str, contexts: list[str], api_key: str, model: str) -> float:
    """Are the retrieved documents relevant to the query?"""
    context_text = "\n---\n".join(contexts[:3])
    prompt = f"""
Evaluate CONTEXT PRECISION: Are the retrieved documents relevant to answering the question?

Question: {query}
Retrieved Context:
{context_text}

Score from 0.0 (irrelevant context) to 1.0 (highly relevant context).
Return JSON: {{"score": <float>, "reason": "<one sentence>"}}
"""
    result = _judge(prompt, api_key, model)
    return float(result.get("score", 0.5))


def evaluate_response(
    *,
    query: str,
    answer: str,
    contexts: list[str],
    api_key: Optional[str] = None,
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    """Run all three quality metrics and return scores dict."""
    api_key = api_key or os.environ.get("GROQ_API_KEY", "")
    scores = {}
    try:
        scores["faithfulness"] = score_faithfulness(answer, contexts, api_key, model)
    except Exception:
        scores["faithfulness"] = None

    try:
        scores["answer_relevancy"] = score_answer_relevancy(query, answer, api_key, model)
    except Exception:
        scores["answer_relevancy"] = None

    try:
        scores["context_precision"] = score_context_precision(query, contexts, api_key, model)
    except Exception:
        scores["context_precision"] = None

    valid = [v for v in scores.values() if v is not None]
    scores["quality_avg"] = round(sum(valid) / len(valid), 4) if valid else None
    return scores
