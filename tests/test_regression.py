"""
CI Regression Gate — runs in GitHub Actions on every PR.
Fails if quality or latency thresholds are breached.

Usage:
    pytest ci/regression_test.py -v
"""
import pytest
import os
import sys

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from observability.store import init_db, get_percentile_latency, get_summary_stats
from evaluation.regression_gate import run_gate, THRESHOLDS

# ── Test data ────────────────────────────────────────────────────
TEST_QUERIES = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "Explain deep learning.",
    "What are large language models?",
    "What is RAG?",
]

SAMPLE_DOCS = [
    ("AI Basics", """
Artificial intelligence (AI) is the simulation of human intelligence in machines.
Machine learning allows systems to automatically learn from experience.
Deep learning uses neural networks with many layers.
Large language models (LLMs) are trained on vast datasets.
RAG combines retrieval with generation for accurate answers.
"""),
]


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


@pytest.fixture(scope="module")
def api_key():
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        pytest.skip("GROQ_API_KEY not set — skipping live tests")
    return key


@pytest.fixture(scope="module")
def ingested(api_key):
    """Ingest sample docs once for the whole test module."""
    from rag.ingest import ingest_text, collection_stats
    for name, text in SAMPLE_DOCS:
        ingest_text(text, source_name=name)
    return collection_stats()


def test_collection_not_empty(ingested):
    assert ingested["total_chunks"] > 0, "Vector store is empty after ingestion"


def test_rag_returns_answer(api_key, ingested):
    from rag.pipeline import run_rag
    result = run_rag(TEST_QUERIES[0], api_key=api_key, eval_quality=False)
    assert result["answer"], "RAG pipeline returned empty answer"
    assert result["sources"], "RAG pipeline returned no sources"


def test_latency_within_threshold(api_key, ingested):
    from rag.pipeline import run_rag
    result = run_rag(TEST_QUERIES[0], api_key=api_key, eval_quality=False)
    latency = result["metrics"]["latency_total_ms"]
    assert latency < THRESHOLDS["p95_latency_ms"], (
        f"Latency {latency:.0f}ms exceeds threshold {THRESHOLDS['p95_latency_ms']}ms"
    )


def test_quality_above_threshold(api_key, ingested):
    from rag.pipeline import run_rag
    results = []
    for q in TEST_QUERIES[:3]:
        r = run_rag(q, api_key=api_key, eval_quality=True)
        if r["metrics"].get("quality_avg") is not None:
            results.append(r["metrics"]["quality_avg"])

    if not results:
        pytest.skip("No quality scores returned")

    avg = sum(results) / len(results)
    assert avg >= THRESHOLDS["avg_quality"], (
        f"Avg quality {avg:.3f} below threshold {THRESHOLDS['avg_quality']}"
    )


def test_regression_gate_passes(api_key, ingested):
    """Full gate check using DB stats."""
    from rag.pipeline import run_rag
    for q in TEST_QUERIES[:3]:
        run_rag(q, api_key=api_key, eval_quality=True)

    lat = get_percentile_latency()
    stats = get_summary_stats()

    gate = run_gate(
        p95_latency_ms=lat["p95"],
        avg_quality=stats.get("avg_quality"),
        avg_faithfulness=stats.get("avg_faithfulness"),
        avg_answer_relevancy=stats.get("avg_relevancy"),
        avg_context_precision=stats.get("avg_precision"),
    )

    assert gate.passed, f"Regression gate failed:\n{gate.summary}\n\nChecks:\n" + "\n".join(
        str(c) for c in gate.checks if not c.get("passed", True)
    )