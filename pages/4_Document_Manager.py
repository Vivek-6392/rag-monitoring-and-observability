"""
Page 4 — Document Manager
Ingest text, upload files, manage the ChromaDB vector store.
"""
import streamlit as st

st.set_page_config(page_title="Documents", page_icon="📁", layout="wide")

from observability.store import init_db
init_db()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');
.page-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📁 Document Manager</div>', unsafe_allow_html=True)
st.caption("Ingest documents into ChromaDB for retrieval")
st.divider()

from rag.ingest import ingest_text, ingest_file, collection_stats
import tempfile
from pathlib import Path

# ── Stats ─────────────────────────────────────────────────────────
stats = collection_stats()
st.metric("Chunks in Vector Store", stats["total_chunks"])
st.divider()

tab1, tab2, tab3 = st.tabs(["📝 Paste Text", "📂 Upload File", "🗄️ Sample Data"])

with tab1:
    st.subheader("Paste text to ingest")
    source_name = st.text_input("Source name", value="manual_paste", key="paste_src")
    text_input = st.text_area("Content", height=250, placeholder="Paste your document text here…")
    if st.button("Ingest Text", type="primary", key="ingest_text_btn"):
        if text_input.strip():
            n = ingest_text(text_input, source_name=source_name)
            st.success(f"✅ Ingested {n} chunks from '{source_name}'")
            st.rerun()
        else:
            st.warning("Please enter some text.")

with tab2:
    st.subheader("Upload .txt or .pdf files")
    uploaded = st.file_uploader(
        "Choose files",
        type=["txt", "pdf", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Ingest Uploaded Files", type="primary"):
        total = 0
        for f in uploaded:
            with tempfile.NamedTemporaryFile(suffix=Path(f.name).suffix, delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            try:
                n = ingest_file(tmp_path)
                st.success(f"✅ {f.name}: {n} chunks")
                total += n
            except Exception as e:
                st.error(f"❌ {f.name}: {e}")
        if total:
            st.rerun()

with tab3:
    st.subheader("Load sample documents")
    st.markdown("Quickly populate the vector store with built-in sample content for testing.")

    SAMPLES = {
        "AI & Machine Learning Basics": """
Artificial intelligence (AI) is the simulation of human intelligence processes by computer systems.
Machine learning is a subset of AI that provides systems the ability to automatically learn and improve from experience.
Deep learning uses neural networks with many layers to learn data representations.
Natural language processing (NLP) enables computers to understand, interpret, and generate human language.
Large language models (LLMs) are trained on vast datasets and can perform a wide variety of language tasks.
Retrieval-Augmented Generation (RAG) combines a retrieval system with a generative model to produce more accurate responses.
Vector databases store high-dimensional embeddings and enable semantic similarity search.
Embeddings are numerical representations of text that capture semantic meaning.
Fine-tuning adapts a pre-trained model to a specific task using a smaller dataset.
Prompt engineering involves crafting effective inputs to guide LLM behavior.
        """,
        "Software Engineering Best Practices": """
Clean code is code that is easy to understand and easy to change.
SOLID principles provide guidelines for object-oriented software design.
Test-driven development (TDD) involves writing tests before writing the implementation code.
Continuous integration ensures that code changes are automatically tested and merged frequently.
Microservices architecture decomposes an application into small, independent services.
DevOps bridges the gap between software development and IT operations.
Code review is the systematic examination of source code to find bugs and improve quality.
Refactoring is the process of restructuring existing code without changing its external behavior.
Documentation is essential for maintaining and scaling software systems.
Observability in software means being able to understand the internal state of a system from its outputs.
        """,
        "Data Science Workflow": """
The data science workflow begins with problem definition and data collection.
Exploratory data analysis (EDA) involves summarizing datasets to discover patterns.
Feature engineering transforms raw data into features that improve model performance.
Model selection involves choosing the right algorithm for the task at hand.
Cross-validation helps assess model generalization performance on unseen data.
Hyperparameter tuning optimizes model settings to improve performance.
Model deployment involves making a trained model available for inference.
Monitoring deployed models ensures they continue to perform well over time.
Data drift occurs when the statistical properties of input data change over time.
A/B testing compares two versions of a system to determine which performs better.
        """,
    }

    selected = st.multiselect(
        "Select sample datasets to load",
        options=list(SAMPLES.keys()),
        default=list(SAMPLES.keys()),
    )

    if st.button("Load Selected Samples", type="primary"):
        total = 0
        for name in selected:
            n = ingest_text(SAMPLES[name], source_name=name)
            st.success(f"✅ {name}: {n} chunks")
            total += n
        if total:
            st.info(f"Loaded {total} total chunks. You can now ask questions on the RAG Query page!")
            st.rerun()