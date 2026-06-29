"""Experiment Tracker — W&B-style experiment comparison, parallel coordinates, Pareto front."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Experiment Tracker", page_icon="🔬", layout="wide")

DATA_PATH = Path(__file__).parent.parent.parent / "data"


@st.cache_data(ttl=300)
def load_df() -> pd.DataFrame:
    p = DATA_PATH / "demo_experiments.json"
    if p.exists():
        with open(p) as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame()


st.title("🔬 Experiment Tracker")
st.caption(
    "All 20+ W&B sweep runs — compare hyperparameters, metrics, and find Pareto-optimal configs."
)

df = load_df()
if df.empty:
    st.error("No experiment data.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filter & Sort")
    models = st.multiselect(
        "Model", df["model"].unique().tolist(), default=df["model"].unique().tolist()
    )
    methods = st.multiselect(
        "Method", df["method"].unique().tolist(), default=df["method"].unique().tolist()
    )
    sort_by = st.selectbox(
        "Sort by",
        [
            "mmlu_overall",
            "eval_loss_final",
            "judge_composite",
            "custom_acc",
            "gpu_mem_gb",
        ],
        index=0,
    )
    sort_asc = st.checkbox("Ascending", value=False)
    min_mmlu = st.slider("Min MMLU", 0.5, 0.75, 0.54, 0.01)
    st.divider()
    show_pareto = st.checkbox("Show Pareto front", value=True)
    color_by = st.selectbox(
        "Color scatter by", ["method", "model", "rank", "scheduler"]
    )

df_f = (
    df[
        df["model"].isin(models)
        & df["method"].isin(methods)
        & (df["mmlu_overall"] >= min_mmlu)
    ]
    .sort_values(sort_by, ascending=sort_asc)
    .reset_index(drop=True)
)

# ── Row 1: Summary KPIs ───────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Runs shown", len(df_f), f"of {len(df)}")
if not df_f.empty:
    best = df_f.iloc[0] if not sort_asc else df_f.iloc[-1]
    k2.metric("Best run", best["run_name"][:25], f"MMLU={best['mmlu_overall']:.1%}")
    k3.metric(
        "Avg MMLU",
        f"{df_f['mmlu_overall'].mean():.1%}",
        f"σ={df_f['mmlu_overall'].std():.3f}",
    )
    k4.metric(
        "GPU savings (QLoRA)",
        f"{(1 - df_f[df_f.method == 'QLoRA']['gpu_mem_gb'].mean() / df_f[df_f.method == 'LoRA']['gpu_mem_gb'].mean()) * 100:.0f}%",
    )

st.divider()

# ── Row 2: MMLU vs eval loss scatter ──────────────────────────────────────────
st.subheader("MMLU Accuracy vs. Eval Loss")

c_scatter, c_violin = st.columns([3, 2])

with c_scatter:
    fig_sc = px.scatter(
        df_f,
        x="eval_loss_final",
        y="mmlu_overall",
        color=color_by,
        symbol="model",
        size="rank",
        hover_data=["run_name", "lr", "rank", "epochs", "neftune"],
        template="plotly_dark",
        height=420,
        labels={"eval_loss_final": "Eval Loss", "mmlu_overall": "MMLU Accuracy"},
        title="MMLU Accuracy vs Eval Loss (size=rank)",
    )

    # Pareto front
    if show_pareto and len(df_f) > 2:
        sorted_pareto = df_f.sort_values("eval_loss_final")
        best_mmlu = -np.inf
        pareto_pts = []
        for _, row in sorted_pareto.iterrows():
            if row["mmlu_overall"] > best_mmlu:
                best_mmlu = row["mmlu_overall"]
                pareto_pts.append(row)
        if pareto_pts:
            pp = pd.DataFrame(pareto_pts)
            fig_sc.add_trace(
                go.Scatter(
                    x=pp["eval_loss_final"],
                    y=pp["mmlu_overall"],
                    mode="lines",
                    name="Pareto Front",
                    line=dict(color="#f59e0b", width=2, dash="dash"),
                )
            )

    fig_sc.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_sc, use_container_width=True)

with c_violin:
    fig_vio = px.violin(
        df_f,
        y="mmlu_overall",
        x="method",
        color="method",
        box=True,
        points="all",
        template="plotly_dark",
        height=420,
        title="MMLU Distribution by Method",
        labels={"mmlu_overall": "MMLU Accuracy"},
        color_discrete_map={"LoRA": "#6366f1", "QLoRA": "#10b981"},
    )
    fig_vio.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_vio, use_container_width=True)

st.divider()

# ── Row 3: Parallel coordinates ────────────────────────────────────────────────
st.subheader("Parallel Coordinates — Hyperparameter Sweep")

dims_avail = {
    "Learning Rate": "lr",
    "LoRA Rank": "rank",
    "LoRA Alpha": "lora_alpha",
    "Batch Size": "batch_size",
    "Grad Accum": "grad_accum",
    "Epochs": "epochs",
    "Warmup": "warmup",
    "NEFTune": "neftune",
    "Train Loss": "train_loss_final",
    "Eval Loss": "eval_loss_final",
    "MMLU": "mmlu_overall",
    "Judge": "judge_composite",
    "GPU Mem": "gpu_mem_gb",
}
selected_dims = st.multiselect(
    "Dimensions",
    list(dims_avail.keys()),
    default=["Learning Rate", "LoRA Rank", "Epochs", "NEFTune", "Eval Loss", "MMLU"],
)

if selected_dims and len(df_f) > 0:
    cols = [dims_avail[d] for d in selected_dims]
    pc_df = df_f[cols + ["mmlu_overall"]].copy()

    dimensions_list = []
    for col, label in zip(cols, selected_dims):
        col_data = pc_df[col].fillna(0)
        d: dict = {"label": label, "values": col_data}
        col_min, col_max = float(col_data.min()), float(col_data.max())
        if col_min != col_max:
            d["range"] = [col_min, col_max]
        if col == "mmlu_overall":
            d["tickformat"] = ".2%"
        dimensions_list.append(d)

    fig_pc = go.Figure(
        go.Parcoords(
            line=dict(
                color=pc_df["mmlu_overall"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="MMLU"),
            ),
            dimensions=dimensions_list,
        )
    )
    fig_pc.update_layout(
        template="plotly_dark",
        height=420,
        title="Parallel Coordinates (color=MMLU Accuracy)",
    )
    st.plotly_chart(fig_pc, use_container_width=True)

st.divider()

# ── Row 4: Correlation heatmap ─────────────────────────────────────────────────
st.subheader("Hyperparameter × Metric Correlation")

metric_cols = [
    "mmlu_overall",
    "eval_loss_final",
    "custom_acc",
    "judge_composite",
    "gpu_mem_gb",
]
hp_cols = [
    "lr",
    "rank",
    "lora_alpha",
    "batch_size",
    "grad_accum",
    "epochs",
    "warmup",
    "neftune",
]

corr = df_f[hp_cols + metric_cols].corr().loc[hp_cols, metric_cols]
fig_corr = go.Figure(
    go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
    )
)
fig_corr.update_layout(
    template="plotly_dark",
    height=400,
    title="Pearson Correlation: Hyperparameters vs. Metrics",
    xaxis_title="Metric",
    yaxis_title="Hyperparameter",
)
st.plotly_chart(fig_corr, use_container_width=True)

st.divider()

# ── Row 5: Full experiment table ───────────────────────────────────────────────
st.subheader("Full Experiment Table")

DISPLAY_COLS = {
    "run_name": "Run",
    "model": "Model",
    "method": "Method",
    "rank": "r",
    "lr": "LR",
    "epochs": "Ep",
    "neftune": "NEFTune",
    "scheduler": "Sched.",
    "train_loss_final": "Train Loss",
    "eval_loss_final": "Eval Loss",
    "mmlu_overall": "MMLU",
    "truthfulqa_mc1": "TruthQA MC1",
    "custom_acc": "Custom Acc",
    "judge_composite": "Judge",
    "gpu_mem_gb": "GPU (GB)",
    "setup_time_min": "Setup (min)",
}

show_df = df_f[list(DISPLAY_COLS.keys())].rename(columns=DISPLAY_COLS)

st.dataframe(
    show_df.style.background_gradient(
        subset=["MMLU", "Custom Acc", "TruthQA MC1"], cmap="Greens"
    )
    .background_gradient(subset=["Eval Loss", "Train Loss"], cmap="RdYlGn_r")
    .background_gradient(subset=["GPU (GB)"], cmap="YlOrRd")
    .format(
        {
            "LR": "{:.0e}",
            "MMLU": "{:.1%}",
            "TruthQA MC1": "{:.1%}",
            "Custom Acc": "{:.1%}",
            "Train Loss": "{:.3f}",
            "Eval Loss": "{:.3f}",
            "Judge": "{:.1f}",
            "GPU (GB)": "{:.1f}",
        }
    ),
    use_container_width=True,
    height=500,
)

# Download
csv = df_f.to_csv(index=False)
st.download_button("⬇️ Download CSV", csv, "experiments.csv", "text/csv")
