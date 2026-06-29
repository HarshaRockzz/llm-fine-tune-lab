"""Training Dashboard — loss curves, LR schedules, GPU memory, hyperparameter distributions."""

# ruff: noqa: E402
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Training Dashboard", page_icon="🏋️", layout="wide")

_APP_DIR = str(Path(__file__).resolve().parent.parent)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
from ui_styles import inject_global_css

inject_global_css()

DATA_PATH = Path(__file__).parent.parent.parent / "data"

_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter", color="#94a3b8", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.08)"),
)


@st.cache_data(ttl=300)
def load_experiments() -> pd.DataFrame:
    p = DATA_PATH / "demo_experiments.json"
    if p.exists():
        with open(p) as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame()


def _synthetic_loss_curve(run: dict, steps: int = 3750) -> tuple[list, list, list]:
    rng = np.random.default_rng(hash(run["run_id"]) % (2**31))
    xs = list(range(0, steps + 1, 50))
    final_train = run["train_loss_final"]
    final_eval = run["eval_loss_final"]
    train = [
        2.4 * np.exp(-3 * x / steps) + final_train + rng.normal(0, 0.012) for x in xs
    ]
    train = [max(final_train - 0.05, t) for t in train]
    eval_ = [
        2.5 * np.exp(-2.8 * x / steps) + final_eval + rng.normal(0, 0.015) for x in xs
    ]
    eval_ = [max(final_eval - 0.05, e) for e in eval_]
    return xs, train, eval_


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="page-title">🏋️ Training Dashboard</div>
<div class="page-caption">Loss curves, GPU metrics, and hyperparameter analysis across all runs.</div>
""",
    unsafe_allow_html=True,
)

df = load_experiments()
if df.empty:
    st.error("No experiment data found.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='color:#a5b4fc;font-weight:700;font-size:1rem;margin-bottom:12px;'>⚙️ Filters</div>",
        unsafe_allow_html=True,
    )
    model_filter = st.multiselect(
        "Model", df["model"].unique().tolist(), default=df["model"].unique().tolist()
    )
    method_filter = st.multiselect(
        "Method",
        df["method"].unique().tolist(),
        default=df["method"].unique().tolist(),
    )
    rank_filter = st.multiselect(
        "LoRA Rank",
        sorted(df["rank"].unique().tolist()),
        default=sorted(df["rank"].unique().tolist()),
    )

df_f = df[
    df["model"].isin(model_filter)
    & df["method"].isin(method_filter)
    & df["rank"].isin(rank_filter)
]

# ── Summary metrics ────────────────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>📊 Run Summary</div>", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Filtered Runs", len(df_f))
if not df_f.empty:
    m2.metric(
        "Best Eval Loss",
        f"{df_f['eval_loss_final'].min():.3f}",
        f"run: {df_f.loc[df_f['eval_loss_final'].idxmin(), 'run_name'][:20]}",
    )
    m3.metric("Best MMLU", f"{df_f['mmlu_overall'].max():.1%}")
    m4.metric("Avg GPU Mem", f"{df_f['gpu_mem_gb'].mean():.1f} GB")
    m5.metric("Avg Tokens/s", f"{df_f['tokens_per_sec'].mean():,.0f}")

st.markdown("<hr class='glow-div'>", unsafe_allow_html=True)

# ── Loss curves ────────────────────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>📉 Loss Curves</div>", unsafe_allow_html=True)
sel_runs = st.multiselect(
    "Select runs to compare",
    df_f["run_name"].tolist(),
    default=df_f.sort_values("eval_loss_final").head(5)["run_name"].tolist(),
    max_selections=8,
)

if sel_runs:
    fig_loss = make_subplots(rows=1, cols=2, subplot_titles=("Train Loss", "Eval Loss"))
    colors = px.colors.qualitative.Plotly

    for i, run_name in enumerate(sel_runs):
        row = df_f[df_f["run_name"] == run_name].iloc[0]
        xs, train_loss, eval_loss = _synthetic_loss_curve(row)
        color = colors[i % len(colors)]

        fig_loss.add_trace(
            go.Scatter(
                x=xs,
                y=train_loss,
                name=run_name,
                line=dict(color=color, width=2),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig_loss.add_trace(
            go.Scatter(
                x=xs,
                y=eval_loss,
                name=run_name,
                line=dict(color=color, width=2, dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig_loss.update_layout(
        height=400,
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.22),
        **_CHART_LAYOUT,
    )
    fig_loss.update_xaxes(title_text="Step")
    fig_loss.update_yaxes(title_text="Loss")
    st.plotly_chart(fig_loss, use_container_width=True)

st.markdown("<hr class='glow-div'>", unsafe_allow_html=True)

# ── GPU memory comparison ─────────────────────────────────────────────────────
st.markdown(
    "<div class='section-hdr'>🖥️ GPU Memory vs. MMLU Accuracy</div>",
    unsafe_allow_html=True,
)
c_left, c_right = st.columns(2)

with c_left:
    fig_gpu = px.scatter(
        df_f,
        x="gpu_mem_gb",
        y="mmlu_overall",
        color="method",
        symbol="model",
        size="rank",
        hover_data=["run_name", "lr", "rank", "epochs"],
        title="GPU Memory (GB) vs MMLU Accuracy",
        labels={"gpu_mem_gb": "GPU Memory (GB)", "mmlu_overall": "MMLU Accuracy"},
        template="plotly_dark",
        color_discrete_map={"LoRA": "#6366f1", "QLoRA": "#10b981"},
    )
    fig_gpu.update_layout(height=400, **_CHART_LAYOUT)
    st.plotly_chart(fig_gpu, use_container_width=True)

with c_right:
    fig_box = px.box(
        df_f,
        x="method",
        y="gpu_mem_gb",
        color="method",
        title="GPU Memory Distribution by Method",
        labels={"gpu_mem_gb": "GPU Memory (GB)"},
        template="plotly_dark",
        color_discrete_map={"LoRA": "#6366f1", "QLoRA": "#10b981"},
    )
    fig_box.update_layout(height=400, showlegend=False, **_CHART_LAYOUT)
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("<hr class='glow-div'>", unsafe_allow_html=True)

# ── Hyperparameter sensitivity ────────────────────────────────────────────────
st.markdown(
    "<div class='section-hdr'>🔍 Hyperparameter Sensitivity</div>",
    unsafe_allow_html=True,
)
hparam = st.selectbox(
    "Color metric",
    ["mmlu_overall", "eval_loss_final", "custom_acc", "judge_composite"],
)
fig_heat = px.scatter(
    df_f,
    x="lr",
    y="rank",
    color=hparam,
    size="epochs",
    hover_data=["run_name", "method", "model"],
    title=f"Learning Rate × LoRA Rank — colored by {hparam}",
    log_x=True,
    template="plotly_dark",
    color_continuous_scale="Viridis",
    labels={"lr": "Learning Rate (log scale)", "rank": "LoRA Rank"},
)
fig_heat.update_layout(height=400, **_CHART_LAYOUT)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("<hr class='glow-div'>", unsafe_allow_html=True)

# ── All runs table ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>📋 All Runs</div>", unsafe_allow_html=True)
display_cols = [
    "run_name",
    "model",
    "method",
    "rank",
    "lr",
    "epochs",
    "train_loss_final",
    "eval_loss_final",
    "mmlu_overall",
    "gpu_mem_gb",
    "setup_time_min",
]
st.dataframe(
    df_f[display_cols]
    .sort_values("mmlu_overall", ascending=False)
    .style.background_gradient(subset=["mmlu_overall"], cmap="Greens")
    .background_gradient(subset=["eval_loss_final"], cmap="RdYlGn_r")
    .format(
        {
            "lr": "{:.0e}",
            "mmlu_overall": "{:.1%}",
            "train_loss_final": "{:.3f}",
            "eval_loss_final": "{:.3f}",
            "gpu_mem_gb": "{:.1f} GB",
        }
    ),
    use_container_width=True,
    height=440,
)
