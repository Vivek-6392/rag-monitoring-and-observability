"""
Page 3 — Evaluation Runner & CI Gate
Run RAGAS-style evaluation over a test set and check regression thresholds.
"""
import streamlit as st
import json
import time
import pandas as pd

st.set_page_config(page_title="Evaluation & CI Gate", page_icon="🧪", layout="wide")

from observability.store import init_db, log_eval_run, get_eval_history, get_percentile_latency, get_summary_stats
init_db()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
.page-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2rem; }
.gate-pass {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1rem 0;
}
.gate-fail {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1rem 0;
}
.check-row { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; padding: 0.2rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🧪 Evaluation Runner & CI Gate</div>', unsafe_allow_html=True)
st.caption("Run quality evaluation over a test set · Regression gating against thresholds")
st.divider()

api_key = st.session_state.get("groq_api_key", "")
AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "groq/compound",
]

selected_models = st.multiselect(
    "Select Models",
    AVAILABLE_MODELS,
    default=[st.session_state.get("model", "llama-3.3-70b-versatile")]
)

with st.sidebar:
    api_key = st.text_input("Groq API Key", type="password", value=api_key)
    if api_key:
        st.session_state["groq_api_key"] = api_key

# ── Fallback test set ─────────────────────────────────────────────
DEFAULT_TEST_SET = [
    {"query": "What is the main topic of the documents?", "expected_keywords": []},
    {"query": "Summarize the key findings.", "expected_keywords": []},
    {"query": "What are the most important concepts discussed?", "expected_keywords": []},
]

# ── Dynamic test set generator ────────────────────────────────────
def generate_test_set(api_key: str, model: str = "llama-3.3-70b-versatile", n_questions: int = 3) -> list[dict]:
    """
    Auto-generates grounded eval queries from actual ingested chunks.
    Works for any document — PDF, text, or pasted content.
    """
    from rag.ingest import get_or_create_collection
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    import re

    collection = get_or_create_collection()
    if collection.count() == 0:
        return DEFAULT_TEST_SET

    results = collection.get(limit=min(10, collection.count()), include=["documents"])
    sampled_chunks = results["documents"]
    context = "\n\n---\n\n".join(sampled_chunks)

    llm = ChatGroq(model=model, groq_api_key=api_key, temperature=0.3)

    response = llm.invoke([HumanMessage(content=f"""
You are an evaluation dataset generator for a RAG system.

Below are chunks from a document. Generate exactly {n_questions} specific, answerable questions
that can be answered using only these chunks.

Rules:
- Each question must be directly answerable from the text
- Do NOT ask meta-questions like "summarize" or "what is the main topic"
- Questions should be specific (ask about a concept, fact, process, or detail mentioned)
- Vary the question types (what, how, why, which)
- Return ONLY a JSON array with key "query", no preamble or markdown

Example format:
[
  {{"query": "What embedding model is used for document retrieval?"}},
  {{"query": "How does chunk overlap affect retrieval quality?"}}
]

Document chunks:
{context}

JSON:""")])

    try:
        raw = response.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(raw)
        return [{"query": item["query"], "expected_keywords": []} for item in parsed]
    except Exception as e:
        st.warning(f"Auto-generation failed, using default test set: {e}")
        return DEFAULT_TEST_SET


# ── Test Set Section ──────────────────────────────────────────────
st.subheader("📝 Test Set")

# ✅ Initialise directly into the widget key, not a separate key
if "test_set_editor" not in st.session_state:
    st.session_state["test_set_editor"] = json.dumps(DEFAULT_TEST_SET, indent=2)

col_gen, col_n = st.columns([3, 1])
with col_n:
    n_questions = st.number_input("# Questions", min_value=1, max_value=10, value=3)
with col_gen:
    gen_btn = st.button(
        "⚡ Generate from Ingested Documents",
        disabled=not api_key,
        help="Auto-generates questions grounded in your actual document chunks",
    )

if gen_btn:
    if not api_key:
        st.warning("Enter your Groq API key first.")
    else:
        from rag.ingest import collection_stats
        if collection_stats()["total_chunks"] == 0:
            st.warning("No documents ingested yet. Go to the Ingest page first.")
        else:
            with st.spinner("Generating questions from your documents…"):
                model_for_gen = selected_models[0] if selected_models else "llama-3.3-70b-versatile"
                generated = generate_test_set(api_key=api_key, model=model_for_gen, n_questions=n_questions)
                # ✅ Write directly into the widget's own key so it re-renders with new content
                st.session_state["test_set_editor"] = json.dumps(generated, indent=2)
            st.success(f"✅ Generated {len(generated)} questions from your documents.")

# ✅ No separate value= needed — widget reads from its own session_state key
test_json = st.text_area(
    "Test cases (JSON list with `query` fields) — edit manually or auto-generate above",
    height=200,
    key="test_set_editor",
)

col1, col2 = st.columns(2)
with col1:
    run_label = st.text_input("Run label", value=f"eval_{int(time.time())}")
with col2:
    st.markdown("**Thresholds**")
    from evaluation.regression_gate import THRESHOLDS
    st.caption(f"p95 ≤ {THRESHOLDS['p95_latency_ms']}ms · quality ≥ {THRESHOLDS['avg_quality']}")

run_btn = st.button("▶ Run Evaluation", type="primary", disabled=not api_key)

if not api_key:
    st.info("👈 Enter your Groq API key to run evaluations.")

if run_btn:
    try:
        test_cases = json.loads(test_json)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()

    from rag.pipeline import run_rag
    from rag.ingest import collection_stats
    from evaluation.regression_gate import run_gate

    if collection_stats()["total_chunks"] == 0:
        st.warning("No documents ingested. Add documents first.")
        st.stop()

    results = []
    progress = st.progress(0, text="Running evaluations…")
    status = st.empty()

    for i, tc in enumerate(test_cases):
        query = tc.get("query", "")
        status.markdown(f"**[{i+1}/{len(test_cases)}]** Evaluating: *{query[:80]}*")
        try:
            for model in selected_models:
                try:
                    res = run_rag(query, api_key=api_key, model=model, eval_quality=True)
                    results.append({
                        "model": model,
                        "query": query,
                        "answer": res["answer"],
                        "latency_ms": res["metrics"].get("latency_total_ms"),
                        "quality_avg": res["metrics"].get("quality_avg"),
                        "faithfulness": res["metrics"].get("faithfulness"),
                        "answer_relevancy": res["metrics"].get("answer_relevancy"),
                        "context_precision": res["metrics"].get("context_precision"),
                        "status": "ok",
                    })
                except Exception as e:
                    results.append({"model": model, "query": query, "status": f"error: {e}"})
        except Exception as e:
            results.append({"query": query, "status": f"error: {e}"})
        progress.progress((i + 1) / len(test_cases))

    status.empty()
    progress.empty()

    df_r = pd.DataFrame(results)
    ok = df_r[df_r["status"] == "ok"]

    avg_q  = ok["quality_avg"].mean() if not ok.empty else None
    avg_f  = ok["faithfulness"].mean() if not ok.empty else None
    avg_ar = ok["answer_relevancy"].mean() if not ok.empty else None
    avg_cp = ok["context_precision"].mean() if not ok.empty else None
    lat    = get_percentile_latency()

    gate_result = run_gate(
        p95_latency_ms=lat["p95"],
        avg_quality=avg_q,
        avg_faithfulness=avg_f,
        avg_answer_relevancy=avg_ar,
        avg_context_precision=avg_cp,
    )

    css_class = "gate-pass" if gate_result.passed else "gate-fail"
    icon = "✅" if gate_result.passed else "❌"
    st.markdown(f'<div class="{css_class}"><b>{icon} {gate_result.summary}</b></div>',
                unsafe_allow_html=True)

    st.subheader("Check Details")
    for check in gate_result.checks:
        icon = "✅" if check.get("passed", True) else "❌"
        name = check.get("name", "")
        if "regression" in name:
            line = f"{icon} **{name}** — change: `{check.get('change_pct',0):+.1f}%` (allowed: `{check.get('allowed_pct',0):+.1f}%`)"
        else:
            line = f"{icon} **{name}** — value: `{check.get('value', 'N/A')}` | threshold: `{check.get('threshold', 'N/A')}`"
        st.markdown(line)

    st.subheader("Per-Query Results")
    st.dataframe(df_r, use_container_width=True, hide_index=True)

    log_eval_run({
        "ts": time.time(),
        "run_label": run_label,
        "avg_faithfulness": avg_f,
        "avg_answer_relevancy": avg_ar,
        "avg_context_precision": avg_cp,
        "avg_quality": avg_q,
        "p50_latency_ms": lat["p50"],
        "p95_latency_ms": lat["p95"],
        "passed": int(gate_result.passed),
        "report_json": gate_result.to_json(),
    })
    st.success(f"Eval run saved as `{run_label}`")

    st.download_button(
        "⬇ Download JSON Report",
        data=gate_result.to_json(),
        file_name=f"{run_label}_report.json",
        mime="application/json",
    )

# ── Eval history ──────────────────────────────────────────────────
st.divider()
st.subheader("📜 Evaluation History")
history = get_eval_history(20)
if history:
    h_df = pd.DataFrame(history)
    h_df["datetime"] = pd.to_datetime(h_df["ts"], unit="s")
    h_df["passed"] = h_df["passed"].map({1: "✅ PASS", 0: "❌ FAIL"})
    display = ["datetime", "run_label", "passed", "avg_quality", "avg_faithfulness", "p95_latency_ms"]
    display = [c for c in display if c in h_df.columns]
    st.dataframe(h_df[display].sort_values("datetime", ascending=False),
                 use_container_width=True, hide_index=True)
else:
    st.info("No evaluation runs yet.")