"""Training Dashboard — loss curves, LR schedules, GPU memory, hyperparameter distributions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Training Dashboard", page_icon="🏋️", layout="wide")

DATA_PATH = Path(__file__).parent.parent.parent / "data"


@st.cache_data(ttl=300)
def load_experiments() -> pd.DataFrame:
    p = DATA_PATH / "demo_experiments.json"
    if p.exists():
        with open(p) as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame()


def _synthetic_loss_curve(run: dict, steps: int = 3750) -> tuple[list, list, list]:
    """Generate plausible loss curves for a given run config."""
    rng = np.random.default_rng(hash(run["run_id"]) % (2**31))
    xs = list(range(0, steps + 1, 50))

    final_train = run["train_loss_final"]
    final_eval = run["eval_loss_final"]

    # Train: rapid decay then plateau
    train = [
        2.4 * np.exp(-3 * x / steps) + final_train + rng.normal(0, 0.012) for x in xs
    ]
    train = [max(final_train - 0.05, t) for t in train]

    # Eval: similar shape, slightly higher
    eval_ = [
        2.5 * np.exp(-2.8 * x / steps) + final_eval + rng.normal(0, 0.015) for x in xs
    ]
    eval_ = [max(final_eval - 0.05, e) for e in eval_]

    return xs, train, eval_


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🏋️ Training Dashboard")
st.caption("Loss curves, GPU metrics, and hyperparameter analysis across all runs.")

df = load_experiments()
if df.empty:
    st.error("No experiment data found.")
    st.stop()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    model_filter = st.multiselect(
        "Model", df["model"].unique().tolist(), default=df["model"].unique().tolist()
    )
    method_filter = st.multiselect(
        "Method", df["method"].unique().tolist(), default=df["method"].unique().tolist()
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

# ── Row 1: summary metrics ────────────────────────────────────────────────────
st.subheader("Run Summary")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Filtered Runs", len(df_f))
if not df_f.empty:
    best = df_f.loc[df_f["mmlu_overall"].idxmax()]
    m2.metric(
        "Best Eval Loss",
        f"{df_f['eval_loss_final'].min():.3f}",
        f"run: {df_f.loc[df_f['eval_loss_final'].idxmin(), 'run_name'][:20]}",
    )
    m3.metric("Best MMLU", f"{df_f['mmlu_overall'].max():.1%}")
    avg_gpu = df_f["gpu_mem_gb"].mean()
    m4.metric("Avg GPU Mem", f"{avg_gpu:.1f} GB")
    avg_tps = df_f["tokens_per_sec"].mean()
    m5.metric("Avg Tokens/s (train)", f"{avg_tps:,.0f}")

st.divider()

# ── Row 2: Loss curves ─────────────────────────────────────────────────────────
st.subheader("Loss Curves")
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
                line=dict(color=color, width=1.8),
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
                line=dict(color=color, width=1.8, dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig_loss.update_layout(
        height=380, template="plotly_dark", legend=dict(orientation="h", y=-0.2)
    )
    fig_loss.update_xaxes(title_text="Step")
    fig_loss.update_yaxes(title_text="Loss")
    st.plotly_chart(fig_loss, use_container_width=True)

st.divider()

# ── Row 3: GPU memory comparison ─────────────────────────────────────────────
st.subheader("GPU Memory vs. MMLU Accuracy")
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
    fig_gpu.update_layout(height=380)
    st.plotly_chart(fig_gpu, use_container_width=True)

with c_right:
    # GPU memory per method box plot
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
    fig_box.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ── Row 4: Hyperparameter heatmap ─────────────────────────────────────────────
st.subheader("Hyperparameter Sensitivity")

hparam = st.selectbox(
    "Color metric", ["mmlu_overall", "eval_loss_final", "custom_acc", "judge_composite"]
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
fig_heat.update_layout(height=380)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ── Row 5: Training table ──────────────────────────────────────────────────────
st.subheader("All Runs")
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
    height=420,
)
