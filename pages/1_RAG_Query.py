"""
Page 1 — RAG Query
Ask questions and see answers with full tracing, latency, and cost breakdown.
"""
import streamlit as st
import time

st.set_page_config(page_title="RAG Query", page_icon="🔍", layout="wide")

from observability.store import init_db
init_db()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
.metric-pill {
    display: inline-block;
    background: #1f2937;
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #9ca3af;
    margin: 0.2rem;
}
.metric-val { color: #00d4ff; font-weight: 700; }
.answer-box {
    background: #0f172a;
    border-left: 3px solid #7b2ff7;
    border-radius: 0 8px 8px 0;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
    line-height: 1.7;
}
.source-chip {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    color: #94a3b8;
    margin: 0.15rem;
}
.score-bar-bg {
    background: #1f2937;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
}
.page-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🔍 RAG Query</div>', unsafe_allow_html=True)
st.caption("Ask questions · See answers with sources · Full latency and cost breakdown")
st.divider()

# ── Sidebar state ────────────────────────────────────────────────
api_key = st.secrets.get("GROQ_API_KEY", None) or st.session_state.get("groq_api_key", "")
model = st.session_state.get("model", "llama-3.3-70b-versatile")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password", value=api_key)
    if api_key:
        st.session_state["groq_api_key"] = api_key
    model = st.selectbox(
        "Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "groq/compound",
        ],
        index=[
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "groq/compound",
        ].index(model)
        if model in [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "groq/compound",
        ]
        else 0,
    )

    st.session_state["model"] = model
    eval_quality = st.toggle("Run quality eval", value=True,
                              help="Adds ~3 extra LLM calls to score faithfulness, relevancy, precision")
    st.divider()
    st.markdown("**Sample questions:**")
    samples = [
        "What is the main topic of the documents?",
        "Summarize the key points.",
        "What are the most important findings?",
    ]
    for s in samples:
        if st.button(s, use_container_width=True, key=f"sample_{s[:20]}"):
            st.session_state["prefill"] = s

# ── Query input ──────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
query = st.text_input(
    "Enter your question",
    value=prefill,
    placeholder="What is the main topic of the documents?",
)

run_btn = st.button("🔍 Ask", type="primary", disabled=not api_key or not query)

if not api_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started.")

if run_btn and query and api_key:
    from rag.pipeline import run_rag
    from rag.ingest import collection_stats

    stats = collection_stats()
    if stats["total_chunks"] == 0:
        st.warning("⚠️ No documents ingested yet. Go to **Documents** page to add content.")
        st.stop()

    with st.spinner("Retrieving and generating..."):
        t0 = time.time()
        result = run_rag(
            query,
            api_key=api_key,
            model=model,
            eval_quality=eval_quality,
        )
        wall_time = (time.time() - t0) * 1000

    m = result["metrics"]

    # ── Answer ───────────────────────────────────────────────────
    st.markdown("#### Answer")
    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

    # ── Latency / cost pills ──────────────────────────────────────
    st.markdown(f"""
    <div>
      <span class="metric-pill">⏱ Retrieval <span class="metric-val">{m.get('latency_retrieval_ms', 0):.0f}ms</span></span>
      <span class="metric-pill">🤖 Generation <span class="metric-val">{m.get('latency_generation_ms', 0):.0f}ms</span></span>
      <span class="metric-pill">🔢 Tokens <span class="metric-val">{m.get('tokens_total', '-')}</span></span>
      <span class="metric-pill">💵 Cost <span class="metric-val">${m.get('cost_usd', 0):.6f}</span></span>
    </div>
    """, unsafe_allow_html=True)

    # ── Quality scores ────────────────────────────────────────────
    if eval_quality and m.get("quality_avg") is not None:
        st.markdown("#### Quality Scores")
        qcols = st.columns(4)
        score_items = [
            ("Quality Avg", m.get("quality_avg")),
            ("Faithfulness", m.get("faithfulness")),
            ("Answer Relevancy", m.get("answer_relevancy")),
            ("Context Precision", m.get("context_precision")),
        ]
        for col, (label, val) in zip(qcols, score_items):
            with col:
                if val is not None:
                    color = "#22c55e" if val >= 0.7 else "#f59e0b" if val >= 0.5 else "#ef4444"
                    st.markdown(f"""
                    <div style="text-align:center">
                        <div style="font-size:1.6rem;font-weight:700;color:{color}">{val:.2f}</div>
                        <div style="font-size:0.75rem;color:#6b7280">{label}</div>
                        <div class="score-bar-bg">
                            <div style="background:{color};height:6px;border-radius:4px;width:{val*100:.0f}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Sources ───────────────────────────────────────────────────
    with st.expander(f"📄 Sources ({len(result['sources'])} chunks retrieved)", expanded=False):
        for i, doc in enumerate(result["sources"]):
            st.markdown(f"""
            <span class="source-chip">📎 {doc['source']}</span>
            <span class="source-chip">score: {doc['score']:.3f}</span>
            """, unsafe_allow_html=True)
            st.text(doc["text"][:300] + ("..." if len(doc["text"]) > 300 else ""))
            if i < len(result["sources"]) - 1:
                st.divider()

    # ── Trace ─────────────────────────────────────────────────────
    with st.expander("🔬 Trace", expanded=False):
        trace = result["trace"]
        st.caption(f"Trace ID: `{trace['trace_id']}`")
        for span in trace["spans"]:
            st.markdown(f"**{span['name']}** — `{span['duration_ms']:.1f}ms`")
            attrs = {k: v for k, v in span.items() if k not in ("name", "duration_ms")}
            if attrs:
                st.json(attrs)