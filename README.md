# 🔭 RAG Monitoring & Observability

> Production-grade monitoring for Retrieval-Augmented Generation systems.
> Tracing · p50/p95 Latency · Cost Tracking · LLM-as-Judge Quality Metrics · CI Regression Gating

[![CI Regression Gate](https://github.com/Vivek-6392/rag-monitoring-and-observability/actions/workflows/regression_gate.yml/badge.svg)](https://github.com/Vivek-6392/rag-monitoring-and-observability/actions/workflows/regression_gate.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📸 What You Get

| Page | What it shows |
|------|--------------|
| 🔍 **RAG Query** | Ask questions, see answers with full per-query trace: retrieval ms, generation ms, token count, cost, source docs |
| 📊 **Observability Dashboard** | Rolling p50/p95 latency charts, cumulative cost, stage breakdown, quality score trends, per-request trace table |
| 🧪 **Evaluation Runner** | Golden dataset eval, LLM-as-judge quality scoring (faithfulness / relevancy / context precision), regression gate UI |
| 📁 **Document Manager** | Ingest text, upload files (.txt / .pdf / .md), or load built-in sample datasets into ChromaDB |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Cloud                           │
│                                                             │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │  RAG Query   │  │  Observability  │  │  Evaluation   │  │
│  │   Page       │  │   Dashboard     │  │   Runner      │  │
│  └──────┬───────┘  └────────┬────────┘  └──────┬────────┘  │
│         │                   │                   │           │
│         └───────────────────┼───────────────────┘           │
│                             │                               │
│              ┌──────────────▼──────────────┐                │
│              │       RAG Pipeline           │                │
│              │  ┌──────────┐ ┌──────────┐  │                │
│              │  │Retrieval │ │Generation│  │                │
│              │  │  Span    │ │  Span    │  │                │
│              │  └────┬─────┘ └────┬─────┘  │                │
│              └───────┼────────────┼─────────┘               │
│                      │            │                          │
│         ┌────────────▼────┐  ┌────▼──────────┐              │
│         │  ChromaDB       │  │  Groq API     │              │
│         │  (in-memory)    │  │  LLaMA 3 8B   │              │
│         │  + HuggingFace  │  │               │              │
│         │  Embeddings     │  └───────────────┘              │
│         └─────────────────┘                                 │
│                      │                                       │
│              ┌────────▼────────┐                             │
│              │   SQLite DB     │                             │
│              │  (metrics.db)   │                             │
│              │  requests table │                             │
│              │  eval_runs table│                             │
│              └─────────────────┘                             │
└─────────────────────────────────────────────────────────────┘

                    GitHub Actions CI
              ┌────────────────────────────┐
              │  On every PR / push:        │
              │  1. Unit tests (no key)     │
              │  2. Integration tests       │
              │  3. Batch eval on golden    │
              │     dataset                 │
              │  4. Regression gate:        │
              │     • p95 latency check     │
              │     • quality score check   │
              └────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/rag-observability.git
cd rag-observability
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export GROQ_API_KEY=your_groq_api_key_here   # Linux/macOS
set GROQ_API_KEY=your_groq_api_key_here      # Windows
```

Or paste it directly into the sidebar when the app opens.

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

### 4. First-time setup in the app

1. Enter your **Groq API key** in the sidebar
2. Go to **Documents** → click **Load Selected Samples**
3. Go to **RAG Query** → ask any question — your first trace is now recorded
4. Visit **Observability Dashboard** to see latency/cost/quality charts
5. Visit **Evaluation Runner** → click **Run Evaluation** to check the regression gate

---

## ☁️ Deploy to Streamlit Cloud

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "feat: RAG observability system"
git remote add origin https://github.com/your-username/rag-observability.git
git push -u origin main
```

### Step 2 — Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your repo → **Main file: `app.py`**
4. Click **Advanced settings** → add secrets:

```toml
# Paste into Streamlit Cloud secrets UI
GROQ_API_KEY = "gsk_your_key_here"
```

5. Click **Deploy** — live in ~2 minutes.

> **Note:** ChromaDB runs in-memory on Streamlit Cloud. The vector store resets on each app restart. Go to **Documents → Load Selected Samples** after each cold start to re-populate it.

---

## 🔑 Getting a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up — no credit card required
3. Navigate to **API Keys** → **Create API Key**
4. Copy and paste into the sidebar or your environment variable

**Free tier limits:**

| Model | Requests/min | Tokens/min |
|-------|-------------|-----------|
| llama3-8b-8192 | 30 | 30,000 |
| llama3-70b-8192 | 30 | 6,000 |
| mixtral-8x7b-32768 | 30 | 5,000 |
| gemma2-9b-it | 30 | 15,000 |

---

## 📁 Project Structure

```
rag-observability/
│
├── app.py                              # Streamlit entry point (home page)
│
├── pages/
│   ├── 1_RAG_Query.py                  # Query interface + trace inspector
│   ├── 2_Observability_Dashboard.py    # Latency / cost / quality charts
│   ├── 3_Evaluation_Runner.py          # Batch eval + regression gate UI
│   └── 4_Document_Manager.py           # Ingest text, files, or sample data
│
├── rag/
│   ├── __init__.py
│   ├── pipeline.py                     # Instrumented RAG chain (retrieve → generate)
│   └── ingest.py                       # ChromaDB + sentence-transformers embeddings
│
├── observability/
│   ├── __init__.py
│   ├── store.py                        # SQLite persistence (requests + eval_runs)
│   ├── tracer.py                       # Lightweight span-based tracer
│   └── metrics.py                      # Token counting, cost calc, p50/p95
│
├── evaluation/
│   ├── __init__.py
│   ├── ragas_eval.py                   # LLM-as-judge quality scoring
│   └── regression_gate.py              # Threshold checks + pass/fail logic
│
├── tests/
│   └── test_regression.py              # Pytest suite for GitHub Actions
│
├── .github/
│   └── workflows/
│       └── regression_gate.yml         # CI pipeline (unit → integration → gate)
│
├── requirements.txt
└── README.md
```

---

## 📊 Metrics Reference

### Latency Metrics

| Metric | Description | CI Gate |
|--------|-------------|---------|
| `latency_retrieval_ms` | Time to fetch docs from ChromaDB | — |
| `latency_generation_ms` | Time for Groq LLM to respond | — |
| `latency_total_ms` | End-to-end request time | — |
| `p50_latency_ms` | Median latency (rolling window) | — |
| `p95_latency_ms` | 95th percentile latency | ✅ `< 3000ms` |

### Cost Metrics

| Metric | Description |
|--------|-------------|
| `tokens_prompt` | Input tokens (query + context) |
| `tokens_completion` | Output tokens (generated answer) |
| `tokens_total` | Combined token count |
| `cost_usd` | Per-request cost in USD |
| cumulative cost | Running total shown in dashboard |

### Quality Metrics (LLM-as-Judge)

| Metric | Description | CI Gate |
|--------|-------------|---------|
| `faithfulness` | Answer grounded in retrieved context? | — |
| `answer_relevancy` | Answer addresses the question? | — |
| `context_precision` | Retrieved docs relevant to query? | — |
| `quality_avg` | Average of the three scores above | ✅ `> 0.60` |

### Trace Structure

Every query produces a structured trace stored in SQLite:

```json
{
  "trace_id": "a3f9b2c1d4e5...",
  "spans": [
    {
      "name": "retrieval",
      "duration_ms": 312.1,
      "num_docs": 5,
      "top_score": 0.91
    },
    {
      "name": "generation",
      "duration_ms": 1530.2,
      "model": "llama3-8b-8192"
    }
  ]
}
```

---

## 🚦 CI Regression Gate

The GitHub Actions pipeline runs on every PR and push to `main`.

### How It Works

```
PR opened
    │
    ▼
Unit Tests (no API key needed)
    │  ├── Cost calculation logic
    │  ├── Regression gate thresholds
    │  ├── SQLite store operations
    │  ├── Tracer span timing
    │  └── Collection ingestion
    │
    ▼ (if unit tests pass)
Integration + Regression Gate (needs GROQ_API_KEY secret)
    │  ├── End-to-end RAG query test
    │  ├── Latency within threshold
    │  ├── Quality score above threshold
    │  └── Full gate check via run_gate()
    │
    ▼
Build passes ✅ or fails ❌
```

### Setting Up CI

1. In your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Add secret: `GROQ_API_KEY` = your Groq API key
3. Push a PR — the workflow triggers automatically

### Gate Thresholds (configurable in `evaluation/regression_gate.py`)

| Check | Default Threshold |
|-------|------------------|
| p95 latency | ≤ 3000 ms |
| avg quality score | ≥ 0.60 |
| faithfulness | ≥ 0.55 |
| answer relevancy | ≥ 0.55 |
| context precision | ≥ 0.55 |
| p95 latency regression | ≤ +20% vs baseline |
| quality regression | ≤ −10% vs baseline |

---

## 🧪 Running Tests Locally

```bash
# Unit tests only — no API key needed
pytest ci/regression_test.py -v -k "not api_key"

# All tests including integration — needs GROQ_API_KEY
GROQ_API_KEY=your_key pytest ci/regression_test.py -v

# With coverage
pip install pytest-cov
pytest ci/ --cov=observability --cov=rag --cov=evaluation --cov-report=html
open htmlcov/index.html
```

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required.** Get free at console.groq.com |
| `CHROMA_DATA_DIR` | *(unset = in-memory)* | Set to a path for persistent ChromaDB |
| `METRICS_DB_PATH` | `metrics.db` | SQLite database path |

Model and threshold settings live directly in the source:

| File | Setting |
|------|---------|
| `rag/pipeline.py` | `DEFAULT_MODEL`, `TOP_K` |
| `evaluation/regression_gate.py` | `THRESHOLDS`, `REGRESSION_TOLERANCE` |
| `observability/metrics.py` | `MODEL_COST_PER_1K_TOKENS` |

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| LLM | [Groq](https://console.groq.com) + LLaMA 3 8B | Free tier, very fast inference |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` | Free, runs on CPU, no API key |
| Vector DB | [ChromaDB](https://www.trychroma.com/) | Zero-config, in-memory |
| UI | [Streamlit](https://streamlit.io) | Multi-page, fast to build |
| Charts | [Plotly](https://plotly.com/) | Interactive, dark-theme friendly |
| Tracing | Custom span tracer | Lightweight, no Jaeger needed |
| Persistence | SQLite | Zero-infra, Streamlit Cloud compatible |
| Quality Eval | LLM-as-judge (Groq) | No extra dependencies, customizable |
| CI | [GitHub Actions](https://github.com/features/actions) | Free for public repos |

---

## 🗺️ Roadmap / Extensions

- [ ] **Persistent vector store** — swap in-memory ChromaDB for Qdrant Cloud free tier
- [ ] **Semantic caching** — cache embeddings for repeated queries to cut cost
- [ ] **Re-ranking** — cross-encoder reranking stage with its own latency span
- [ ] **Multi-model A/B testing** — run the same query through two models, compare quality
- [ ] **Slack/email alerts** — notify when p95 exceeds threshold in production
- [ ] **Token budget enforcement** — reject or truncate queries that exceed cost limit
- [ ] **RAGAS integration** — replace LLM-as-judge with the full RAGAS library

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) for fast, free LLM inference
- [LangChain](https://langchain.com) for the RAG primitives
- [ChromaDB](https://trychroma.com) for the zero-config vector store
- [RAGAS](https://docs.ragas.io) for the quality metric definitions that inspired the LLM-as-judge implementation

---

<p align="center">
  Built as a portfolio project demonstrating production AI observability.<br>
  <strong>70% of production AI work that nobody puts in their portfolio.</strong>
</p>