# RAG Observability

Production-grade monitoring for RAG systems — tracing, latency p50/p95, cost tracking, quality scoring, and CI regression gating.

## Stack
| Layer | Tool |
|---|---|
| LLM | Groq (llama3-8b) — free tier |
| Embeddings | sentence-transformers (local) |
| Vector DB | ChromaDB (in-memory) |
| Metrics | SQLite → Streamlit charts |
| Quality Eval | LLM-as-judge (faithfulness, relevancy, precision) |
| CI Gate | GitHub Actions + pytest |

## Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/rag-observability
cd rag-observability
pip install -r requirements.txt
streamlit run app.py
```

1. Enter your free [Groq API key](https://console.groq.com) in the sidebar
2. Go to **Documents** → load sample data
3. Go to **RAG Query** → ask questions
4. Visit **Observability** → see latency/cost/quality charts
5. Run **Evaluation** → check regression gate

## Deployment (Streamlit Cloud)

1. Push to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Set `GROQ_API_KEY` in Streamlit Secrets
4. Add `GROQ_API_KEY` to GitHub repo secrets for CI

## CI Regression Gate

Every PR runs `pytest ci/regression_test.py` which fails if:
- p95 latency > 3000ms
- avg quality score < 0.60
- faithfulness / relevancy / precision < 0.55
```

---

## Complete repo structure created ✅
```
rag-observability/
├── app.py                          ✅ Landing page
├── requirements.txt                ✅
├── rag/
│   ├── __init__.py                 ✅
│   ├── pipeline.py                 ✅ Instrumented RAG + Groq
│   └── ingest.py                   ✅ ChromaDB ingestion
├── observability/
│   ├── __init__.py                 ✅
│   ├── tracer.py                   ✅ Span tracing
│   ├── metrics.py                  ✅ Token counting + cost
│   └── store.py                    ✅ SQLite persistence
├── evaluation/
│   ├── __init__.py                 ✅
│   ├── ragas_eval.py               ✅ LLM-as-judge scoring
│   └── regression_gate.py          ✅ Pass/fail thresholds
├── pages/
│   ├── 1_RAG_Query.py              ✅
│   ├── 2_Observability_Dashboard.py ✅ Plotly charts
│   ├── 3_Evaluation_Runner.py      ✅
│   └── 4_Document_Manager.py       ✅
├── ci/
│   └── regression_test.py          ✅ pytest suite
└── .github/workflows/
    └── regression_gate.yml         📋 Copy from above