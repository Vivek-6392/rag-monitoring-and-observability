"""
RAG Observability — Main entry point.
"""
import streamlit as st

st.set_page_config(
    page_title="RAG Observability",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Init DB on startup ───────────────────────────────────────────
from observability.store import init_db
init_db()

# ── Sidebar: API key ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔭 RAG Observability")
    st.caption("Tracing · Latency · Cost · Quality · CI Gating")
    st.divider()

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.get("groq_api_key", ""),
        help="Get a free key at console.groq.com",
    )
    if api_key:
        st.session_state["groq_api_key"] = api_key

    model = st.selectbox(
        "Model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,
    )
    st.session_state["model"] = model

    st.divider()
    st.markdown("### Navigation")
    st.page_link("pages/1_RAG_Query.py",            label="🔍 RAG Query",            icon="🔍")
    st.page_link("pages/2_Observability_Dashboard.py", label="📊 Observability",      icon="📊")
    st.page_link("pages/3_Evaluation_Runner.py",    label="🧪 Evaluation & CI Gate", icon="🧪")
    st.page_link("pages/4_Document_Manager.py",     label="📁 Documents",            icon="📁")

# ── Hero ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, pre, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.2rem;
    line-height: 1.1;
    background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 60%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: #9ca3af;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.feature-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.feature-card:hover { border-color: #7b2ff7; }
.feature-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.feature-title { font-weight: 700; font-size: 1rem; color: #f9fafb; margin-bottom: 0.3rem; }
.feature-desc { font-size: 0.85rem; color: #6b7280; line-height: 1.5; }
</style>

<div class="hero-title">RAG Observability</div>
<div class="hero-sub">Production-grade monitoring for Retrieval-Augmented Generation — tracing, latency p50/p95, cost tracking, and quality regression gating.</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

cards = [
    ("🔍", "RAG Query", "Ask questions and see answers with full latency + cost breakdown per request."),
    ("📊", "Observability", "p50/p95 latency charts, cost per request, quality scores, and request history."),
    ("🧪", "Eval & CI Gate", "Run RAGAS-style quality evaluation and regression gating against thresholds."),
    ("📁", "Documents", "Ingest text, PDFs, or paste content directly into the ChromaDB vector store."),
]

for col, (icon, title, desc) in zip([col1, col2, col3, col4], cards):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.divider()
st.markdown("### ⚡ Quick Start")
st.code("""# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/rag-observability
cd rag-observability
pip install -r requirements.txt

# 2. Run
streamlit run app.py

# 3. Add your Groq API key in the sidebar (free at console.groq.com)
# 4. Go to Documents → ingest some text
# 5. Go to RAG Query → ask questions!
""", language="bash")

st.markdown("### 🏗️ Stack")
cols = st.columns(3)
stack = [
    ("LLM", "Groq (llama3-8b)", "Free tier · Fast · No GPU needed"),
    ("Embeddings", "sentence-transformers", "all-MiniLM-L6-v2 · Runs locally"),
    ("Vector DB", "ChromaDB", "In-memory or persistent on disk"),
    ("Tracing", "Custom span tracer", "Per-request spans stored in SQLite"),
    ("Metrics", "SQLite → Streamlit", "p50/p95 · cost · quality"),
    ("CI Gate", "GitHub Actions + pytest", "Regression on every PR"),
]
for i, (name, tool, detail) in enumerate(stack):
    with cols[i % 3]:
        st.markdown(f"**{name}** — `{tool}`  \n*{detail}*")