"""Evaluation Results — MMLU, TruthfulQA, LLM-judge, checkpoint progress."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Evaluation Results", page_icon="📊", layout="wide")

DATA_PATH = Path(__file__).parent.parent.parent / "data"


@st.cache_data(ttl=300)
def load_eval():
    p = DATA_PATH / "demo_eval_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


st.title("📊 Evaluation Results")
st.caption(
    "MMLU, TruthfulQA, LLM-as-Judge scores — baseline vs. fine-tuned champion model."
)

data = load_eval()
if not data:
    st.error("Evaluation data not found.")
    st.stop()

base = data["baseline"]
ft = data["fine_tuned"]

# ── Hero metrics ──────────────────────────────────────────────────────────────
st.subheader("Overall Performance")
m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "MMLU Accuracy",
    f"{ft['mmlu_overall']:.1%}",
    f"+{(ft['mmlu_overall'] - base['mmlu_overall']) * 100:.1f}pp",
)
m2.metric(
    "TruthfulQA MC1",
    f"{ft['truthfulqa_mc1']:.1%}",
    f"+{(ft['truthfulqa_mc1'] - base['truthfulqa_mc1']) * 100:.1f}pp",
)
m3.metric(
    "TruthfulQA MC2",
    f"{ft['truthfulqa_mc2']:.1%}",
    f"+{(ft['truthfulqa_mc2'] - base['truthfulqa_mc2']) * 100:.1f}pp",
)
m4.metric(
    "LLM Judge (/ 10)",
    f"{ft['judge_composite']:.1f}",
    f"+{ft['judge_composite'] - base['judge_composite']:.1f}",
)
m5.metric(
    "Judge Pass Rate",
    f"{ft['judge_pass_rate']:.0%}",
    f"+{(ft['judge_pass_rate'] - base['judge_pass_rate']) * 100:.0f}pp",
)

st.divider()

# ── Row 1: MMLU category breakdown ────────────────────────────────────────────
st.subheader("MMLU Category Breakdown")

cat_data = data["mmlu_categories"]
cats = list(cat_data.keys())
base_scores = [cat_data[c]["baseline"] for c in cats]
ft_scores = [cat_data[c]["finetuned"] for c in cats]

fig_mmlu = go.Figure()
fig_mmlu.add_trace(
    go.Bar(name="Base Model", x=cats, y=base_scores, marker_color="#475569")
)
fig_mmlu.add_trace(
    go.Bar(name="QLoRA Fine-tuned", x=cats, y=ft_scores, marker_color="#6366f1")
)
fig_mmlu.update_layout(
    barmode="group",
    template="plotly_dark",
    height=380,
    yaxis=dict(title="Accuracy", tickformat=".0%", range=[0, 1]),
    xaxis_title="Category",
    legend=dict(orientation="h", y=1.12),
    title="MMLU Accuracy by Category — Baseline vs. Fine-tuned",
)
# Add improvement annotations
for i, (cat, bs, fs) in enumerate(zip(cats, base_scores, ft_scores)):
    fig_mmlu.add_annotation(
        x=cat,
        y=fs + 0.02,
        text=f"+{(fs - bs) * 100:.0f}pp",
        showarrow=False,
        font=dict(size=11, color="#10b981"),
    )
st.plotly_chart(fig_mmlu, use_container_width=True)

st.divider()

# ── Row 2: Checkpoint progress ────────────────────────────────────────────────
st.subheader("Accuracy vs. Training Steps")

ckpt = pd.DataFrame(data["checkpoints"])

fig_ckpt = make_subplots(
    rows=1, cols=2, subplot_titles=("MMLU Accuracy Progress", "Eval Loss Progress")
)

fig_ckpt.add_trace(
    go.Scatter(
        x=ckpt["step"],
        y=ckpt["mmlu"],
        name="MMLU",
        line=dict(color="#6366f1", width=2.5),
        mode="lines+markers",
        marker=dict(size=5),
    ),
    row=1,
    col=1,
)
fig_ckpt.add_trace(
    go.Scatter(
        x=ckpt["step"],
        y=ckpt["custom"],
        name="Custom Domain",
        line=dict(color="#10b981", width=2.5),
        mode="lines+markers",
        marker=dict(size=5),
    ),
    row=1,
    col=1,
)
fig_ckpt.add_hline(
    y=0.54,
    line_dash="dash",
    line_color="#94a3b8",
    annotation_text="Baseline 54%",
    row=1,
    col=1,
)
fig_ckpt.add_hline(
    y=0.71,
    line_dash="dash",
    line_color="#10b981",
    annotation_text="Target 71%",
    row=1,
    col=1,
)

fig_ckpt.add_trace(
    go.Scatter(
        x=ckpt["step"],
        y=ckpt["eval_loss"],
        name="Eval Loss",
        line=dict(color="#f59e0b", width=2.5),
        mode="lines+markers",
        marker=dict(size=5),
        showlegend=True,
    ),
    row=1,
    col=2,
)

fig_ckpt.update_layout(
    template="plotly_dark",
    height=380,
    legend=dict(orientation="h", y=-0.2),
)
fig_ckpt.update_yaxes(title_text="Accuracy", tickformat=".0%", row=1, col=1)
fig_ckpt.update_yaxes(title_text="Loss", row=1, col=2)
fig_ckpt.update_xaxes(title_text="Training Step")
st.plotly_chart(fig_ckpt, use_container_width=True)

st.divider()

# ── Row 3: LLM Judge radar ────────────────────────────────────────────────────
st.subheader("LLM-as-Judge Dimension Scores")

judge_dims = data["judge_dimensions"]
dimensions = list(judge_dims.keys())
base_judge = [judge_dims[d]["baseline"] for d in dimensions]
ft_judge = [judge_dims[d]["finetuned"] for d in dimensions]

col_radar, col_bar = st.columns(2)

with col_radar:
    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=base_judge + [base_judge[0]],
            theta=dimensions + [dimensions[0]],
            fill="toself",
            name="Base Model",
            line_color="#475569",
            fillcolor="rgba(71,85,105,0.2)",
        )
    )
    fig_radar.add_trace(
        go.Scatterpolar(
            r=ft_judge + [ft_judge[0]],
            theta=dimensions + [dimensions[0]],
            fill="toself",
            name="QLoRA Fine-tuned",
            line_color="#6366f1",
            fillcolor="rgba(99,102,241,0.2)",
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        template="plotly_dark",
        height=380,
        title="Judge Dimension Radar (Claude claude-sonnet-4-6 as Judge)",
        legend=dict(orientation="h", y=-0.1),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_bar:
    df_judge = pd.DataFrame(
        {
            "Dimension": dimensions * 2,
            "Score": base_judge + ft_judge,
            "Model": ["Base"] * len(dimensions) + ["Fine-tuned"] * len(dimensions),
        }
    )
    fig_judge_bar = px.bar(
        df_judge,
        x="Score",
        y="Dimension",
        color="Model",
        orientation="h",
        barmode="group",
        template="plotly_dark",
        height=380,
        title="Judge Scores (0-10) by Dimension",
        color_discrete_map={"Base": "#475569", "Fine-tuned": "#6366f1"},
        range_x=[0, 10],
    )
    st.plotly_chart(fig_judge_bar, use_container_width=True)

st.divider()

# ── Row 4: Inference throughput ───────────────────────────────────────────────
st.subheader("Inference Throughput: vLLM vs. HF generate()")

infer = data["inference_comparison"]
hf = infer["hf_generate"]
vllm = infer["vllm_continuous_batching"]

i1, i2, i3, i4 = st.columns(4)
speedup = vllm["throughput_tok_per_sec"] / hf["throughput_tok_per_sec"]
i1.metric(
    "vLLM Throughput",
    f"{vllm['throughput_tok_per_sec']:,} tok/s",
    f"{speedup:.1f}× vs HF",
)
i2.metric(
    "vLLM P50 Latency",
    f"{vllm['p50_latency_ms']:,} ms",
    f"-{hf['p50_latency_ms'] - vllm['p50_latency_ms']:,} ms",
)
i3.metric(
    "vLLM P95 Latency",
    f"{vllm['p95_latency_ms']:,} ms",
    f"-{hf['p95_latency_ms'] - vllm['p95_latency_ms']:,} ms",
)
i4.metric(
    "vLLM GPU Util",
    f"{vllm['gpu_util_pct']}%",
    f"+{vllm['gpu_util_pct'] - hf['gpu_util_pct']}%",
)

infer_df = pd.DataFrame(
    [
        {
            "Metric": "Throughput (tok/s)",
            "HF generate()": hf["throughput_tok_per_sec"],
            "vLLM": vllm["throughput_tok_per_sec"],
        },
        {
            "Metric": "P50 Latency (ms)",
            "HF generate()": hf["p50_latency_ms"],
            "vLLM": vllm["p50_latency_ms"],
        },
        {
            "Metric": "P95 Latency (ms)",
            "HF generate()": hf["p95_latency_ms"],
            "vLLM": vllm["p95_latency_ms"],
        },
        {
            "Metric": "GPU Util (%)",
            "HF generate()": hf["gpu_util_pct"],
            "vLLM": vllm["gpu_util_pct"],
        },
    ]
)
infer_melt = infer_df.melt(id_vars="Metric", var_name="Backend", value_name="Value")
fig_infer = px.bar(
    infer_melt,
    x="Metric",
    y="Value",
    color="Backend",
    barmode="group",
    template="plotly_dark",
    height=350,
    title="HF generate() vs vLLM Continuous Batching",
    color_discrete_map={"HF generate()": "#f59e0b", "vLLM": "#6366f1"},
)
st.plotly_chart(fig_infer, use_container_width=True)
