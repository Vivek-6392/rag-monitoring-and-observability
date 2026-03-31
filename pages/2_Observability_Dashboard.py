"""
Page 2 — Observability Dashboard
Latency p50/p95, cost trends, quality metrics, request history.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Observability", page_icon="📊", layout="wide")

from observability.store import (
    init_db, get_recent_requests, get_summary_stats, get_percentile_latency
)
init_db()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
.page-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2rem; }
.kpi-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.kpi-val { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; color: #00d4ff; }
.kpi-label { font-size: 0.78rem; color: #6b7280; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📊 Observability Dashboard</div>', unsafe_allow_html=True)
st.caption("Real-time latency, cost, and quality metrics from SQLite")
st.divider()

# ── Auto-refresh ─────────────────────────────────────────────────
if st.button("🔄 Refresh"):
    st.rerun()

# ── KPIs ─────────────────────────────────────────────────────────
stats = get_summary_stats()
lat = get_percentile_latency()

kpi_cols = st.columns(6)
kpis = [
    ("Total Requests", f"{int(stats.get('total_requests') or 0)}"),
    ("p50 Latency", f"{lat['p50']:.0f}ms"),
    ("p95 Latency", f"{lat['p95']:.0f}ms"),
    ("Total Cost", f"${(stats.get('total_cost_usd') or 0):.4f}"),
    ("Avg Quality", f"{(stats.get('avg_quality') or 0):.2f}" if stats.get('avg_quality') else "N/A"),
    ("Avg Faithfulness", f"{(stats.get('avg_faithfulness') or 0):.2f}" if stats.get('avg_faithfulness') else "N/A"),
]
for col, (label, val) in zip(kpi_cols, kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# # ── Load data ─────────────────────────────────────────────────────
# rows = get_recent_requests(500)
# if not rows:
#     st.info("No requests recorded yet. Go to **RAG Query** and ask some questions!")
#     st.stop()

# df = pd.DataFrame(rows)
# df["datetime"] = pd.to_datetime(df["ts"], unit="s")
# df = df.sort_values("datetime")

# # ── Latency over time ─────────────────────────────────────────────
# st.subheader("⏱ Latency Over Time")
# lat_cols = st.columns(2)

# with lat_cols[0]:
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df["datetime"], y=df["latency_total_ms"],
#         mode="lines+markers", name="Total",
#         line=dict(color="#00d4ff", width=2),
#         marker=dict(size=4),
#     ))
#     fig.add_trace(go.Scatter(
#         x=df["datetime"], y=df["latency_retrieval_ms"],
#         mode="lines", name="Retrieval",
#         line=dict(color="#7b2ff7", width=1.5, dash="dot"),
#     ))
#     fig.add_trace(go.Scatter(
#         x=df["datetime"], y=df["latency_generation_ms"],
#         mode="lines", name="Generation",
#         line=dict(color="#ff6b35", width=1.5, dash="dot"),
#     ))
#     fig.update_layout(
#         paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
#         font=dict(color="#9ca3af"),
#         legend=dict(bgcolor="#111827"),
#         margin=dict(l=10, r=10, t=10, b=10),
#         yaxis=dict(gridcolor="#1f2937", title="ms"),
#         xaxis=dict(gridcolor="#1f2937"),
#         height=280,
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with lat_cols[1]:
#     # Rolling p50/p95
#     df_sorted = df.dropna(subset=["latency_total_ms"]).reset_index(drop=True)
#     window = min(50, len(df_sorted))
#     p50s, p95s = [], []
#     for i in range(len(df_sorted)):
#         w = df_sorted["latency_total_ms"].iloc[max(0, i - window):i + 1]
#         p50s.append(w.quantile(0.50))
#         p95s.append(w.quantile(0.95))
#     df_sorted["p50"] = p50s
#     df_sorted["p95"] = p95s

#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(
#         x=df_sorted["datetime"], y=df_sorted["p50"],
#         mode="lines", name="p50",
#         line=dict(color="#22c55e", width=2),
#     ))
#     fig2.add_trace(go.Scatter(
#         x=df_sorted["datetime"], y=df_sorted["p95"],
#         mode="lines", name="p95",
#         line=dict(color="#ef4444", width=2),
#     ))
#     fig2.add_hline(y=3000, line_dash="dash", line_color="#f59e0b",
#                    annotation_text="p95 threshold (3s)")
#     fig2.update_layout(
#         paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
#         font=dict(color="#9ca3af"),
#         legend=dict(bgcolor="#111827"),
#         margin=dict(l=10, r=10, t=10, b=10),
#         yaxis=dict(gridcolor="#1f2937", title="ms"),
#         xaxis=dict(gridcolor="#1f2937"),
#         height=280,
#         title=dict(text="Rolling p50 / p95", font=dict(color="#e5e7eb")),
#     )
#     st.plotly_chart(fig2, use_container_width=True)

# # ── Cost & tokens ─────────────────────────────────────────────────
# st.subheader("💵 Cost & Token Usage")
# cost_cols = st.columns(2)

# with cost_cols[0]:
#     df["cumulative_cost"] = df["cost_usd"].cumsum()
#     fig3 = go.Figure()
#     fig3.add_trace(go.Scatter(
#         x=df["datetime"], y=df["cumulative_cost"],
#         fill="tozeroy", mode="lines",
#         line=dict(color="#7b2ff7", width=2),
#         fillcolor="rgba(123,47,247,0.15)",
#         name="Cumulative Cost",
#     ))
#     fig3.update_layout(
#         paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
#         font=dict(color="#9ca3af"),
#         margin=dict(l=10, r=10, t=10, b=10),
#         yaxis=dict(gridcolor="#1f2937", title="USD"),
#         xaxis=dict(gridcolor="#1f2937"),
#         height=250,
#         title=dict(text="Cumulative Cost (USD)", font=dict(color="#e5e7eb")),
#     )
#     st.plotly_chart(fig3, use_container_width=True)

# with cost_cols[1]:
#     fig4 = go.Figure()
#     fig4.add_trace(go.Bar(
#         x=df["datetime"], y=df["tokens_prompt"],
#         name="Prompt", marker_color="#334155",
#     ))
#     fig4.add_trace(go.Bar(
#         x=df["datetime"], y=df["tokens_completion"],
#         name="Completion", marker_color="#00d4ff",
#     ))
#     fig4.update_layout(
#         barmode="stack",
#         paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
#         font=dict(color="#9ca3af"),
#         margin=dict(l=10, r=10, t=10, b=10),
#         yaxis=dict(gridcolor="#1f2937", title="tokens"),
#         xaxis=dict(gridcolor="#1f2937"),
#         legend=dict(bgcolor="#111827"),
#         height=250,
#         title=dict(text="Tokens per Request", font=dict(color="#e5e7eb")),
#     )
#     st.plotly_chart(fig4, use_container_width=True)

# # ── Quality metrics ───────────────────────────────────────────────
# q_df = df.dropna(subset=["quality_avg"])
# if not q_df.empty:
#     st.subheader("🏆 Quality Metrics Over Time")
#     fig5 = go.Figure()
#     colors = {"faithfulness": "#22c55e", "answer_relevancy": "#00d4ff", "context_precision": "#f59e0b", "quality_avg": "#ef4444"}
#     for metric, color in colors.items():
#         if metric in q_df.columns:
#             fig5.add_trace(go.Scatter(
#                 x=q_df["datetime"], y=q_df[metric],
#                 mode="lines+markers", name=metric.replace("_", " ").title(),
#                 line=dict(color=color, width=2), marker=dict(size=5),
#             ))
#     fig5.add_hline(y=0.6, line_dash="dash", line_color="#6b7280",
#                    annotation_text="quality threshold (0.6)")
#     fig5.update_layout(
#         paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
#         font=dict(color="#9ca3af"),
#         legend=dict(bgcolor="#111827"),
#         margin=dict(l=10, r=10, t=10, b=10),
#         yaxis=dict(gridcolor="#1f2937", range=[0, 1.05], title="score"),
#         xaxis=dict(gridcolor="#1f2937"),
#         height=300,
#     )
#     st.plotly_chart(fig5, use_container_width=True)

# # ── Request history table ─────────────────────────────────────────
# st.subheader("📋 Request History")
# display_cols = ["datetime", "query", "latency_total_ms", "tokens_total", "cost_usd", "quality_avg", "faithfulness"]
# display_cols = [c for c in display_cols if c in df.columns]
# st.dataframe(
#     df[display_cols].sort_values("datetime", ascending=False).head(50),
#     use_container_width=True,
#     column_config={
#         "datetime": st.column_config.DatetimeColumn("Time"),
#         "query": st.column_config.TextColumn("Query", max_chars=60),
#         "latency_total_ms": st.column_config.NumberColumn("Latency (ms)", format="%.0f"),
#         "cost_usd": st.column_config.NumberColumn("Cost ($)", format="%.6f"),
#         "quality_avg": st.column_config.ProgressColumn("Quality", min_value=0, max_value=1),
#         "faithfulness": st.column_config.ProgressColumn("Faithful", min_value=0, max_value=1),
#     },
#     hide_index=True,
# )

# ── Load data ─────────────────────────────────────────────────────
rows = get_recent_requests(500)
if not rows:
    st.info("No requests recorded yet. Go to **RAG Query** and ask some questions!")
    st.stop()

df = pd.DataFrame(rows)
df["datetime"] = pd.to_datetime(df["ts"], unit="s")
df = df.sort_values("datetime")

# ── Model Filter (NEW) ────────────────────────────────────────────
if "model" in df.columns:
    df["model"] = df["model"].fillna("unknown")

    models = sorted(df["model"].unique().tolist())

    selected_models = st.multiselect(
        "Filter by Model",
        models,
        default=models
    )

    if selected_models:
        df = df[df["model"].isin(selected_models)]

# ── KPIs (Updated for filtered data) ──────────────────────────────
stats = {
    "total_requests": len(df),
    "total_cost_usd": df["cost_usd"].sum(),
    "avg_quality": df["quality_avg"].mean(),
    "avg_faithfulness": df["faithfulness"].mean(),
}

lat_vals = df["latency_total_ms"].dropna().tolist()
lat_vals.sort()
if lat_vals:
    p50 = lat_vals[int(len(lat_vals)*0.5)]
    p95 = lat_vals[int(len(lat_vals)*0.95)]
else:
    p50, p95 = 0, 0

kpi_cols = st.columns(6)
kpis = [
    ("Total Requests", f"{int(stats['total_requests'])}"),
    ("p50 Latency", f"{p50:.0f}ms"),
    ("p95 Latency", f"{p95:.0f}ms"),
    ("Total Cost", f"${stats['total_cost_usd']:.4f}"),
    ("Avg Quality", f"{stats['avg_quality']:.2f}" if stats['avg_quality'] else "N/A"),
    ("Avg Faithfulness", f"{stats['avg_faithfulness']:.2f}" if stats['avg_faithfulness'] else "N/A"),
]
for col, (label, val) in zip(kpi_cols, kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Latency Over Time (per model) ─────────────────────────────────
st.subheader("⏱ Latency Over Time")

fig = go.Figure()
for model in df["model"].unique():
    mdf = df[df["model"] == model]

    fig.add_trace(go.Scatter(
        x=mdf["datetime"],
        y=mdf["latency_total_ms"],
        mode="lines+markers",
        name=model,
    ))

fig.update_layout(
    paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
    font=dict(color="#9ca3af"),
    legend=dict(bgcolor="#111827"),
    height=300,
)
st.plotly_chart(fig, use_container_width=True)

# ── Cost per Model ────────────────────────────────────────────────
st.subheader("💵 Cost Comparison")

fig2 = go.Figure()
for model in df["model"].unique():
    mdf = df[df["model"] == model].copy()
    mdf["cumulative_cost"] = mdf["cost_usd"].cumsum()

    fig2.add_trace(go.Scatter(
        x=mdf["datetime"],
        y=mdf["cumulative_cost"],
        mode="lines",
        name=model,
    ))

fig2.update_layout(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="#9ca3af"),
    height=300,
)
st.plotly_chart(fig2, use_container_width=True)

# ── Quality Comparison ────────────────────────────────────────────
q_df = df.dropna(subset=["quality_avg"])
if not q_df.empty:
    st.subheader("🏆 Quality Comparison")

    fig3 = go.Figure()
    for model in q_df["model"].unique():
        mdf = q_df[q_df["model"] == model]

        fig3.add_trace(go.Scatter(
            x=mdf["datetime"],
            y=mdf["quality_avg"],
            mode="lines+markers",
            name=model,
        ))

    fig3.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#9ca3af"),
        height=300,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Request History ───────────────────────────────────────────────
st.subheader("📋 Request History")

display_cols = [
    "datetime",
    "model",  # ✅ NEW
    "query",
    "latency_total_ms",
    "tokens_total",
    "cost_usd",
    "quality_avg"
]

display_cols = [c for c in display_cols if c in df.columns]

st.dataframe(
    df[display_cols].sort_values("datetime", ascending=False).head(50),
    use_container_width=True,
    hide_index=True
)