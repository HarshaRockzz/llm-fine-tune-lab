"""Experiment Search — RAG over 20+ runs using Qdrant + OpenRouter embeddings."""

# ruff: noqa: E402
import json
import os
import sys
from pathlib import Path

# Ensure repo root is first on sys.path so `import src` finds the local package,
# not /mount/src/ on Streamlit Cloud.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Experiment Search", page_icon="🔍", layout="wide")

st.markdown(
    """
<style>
  .result-card {
      background:#1e293b; border:1px solid #334155; border-radius:10px;
      padding:16px; margin-bottom:12px;
  }
  .score-badge {
      display:inline-block; background:#312e81; border-radius:12px;
      padding:3px 10px; font-size:0.8rem; color:#a5b4fc; float:right;
  }
  .metric-pill {
      display:inline-block; background:#0f172a; border:1px solid #1e293b;
      border-radius:6px; padding:2px 8px; margin:2px; font-size:0.78rem; color:#94a3b8;
  }
  .answer-box {
      background:#0f172a; border-left:3px solid #6366f1;
      border-radius:0 10px 10px 0; padding:16px; line-height:1.7;
  }
</style>
""",
    unsafe_allow_html=True,
)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY", "")


@st.cache_data(ttl=300)
def load_experiments() -> list[dict]:
    p = DATA_PATH / "demo_experiments.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


experiments = load_experiments()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔍 Experiment Search")
st.caption(
    "Semantic search over 20+ W&B runs using **OpenRouter embeddings** + **Qdrant vector DB**. "
    "Ask in plain English — 'which run used the least GPU memory while staying above 65% MMLU?' "
    "— and get a RAG-grounded answer."
)

if not OPENROUTER_KEY:
    st.warning(
        "Set `OPENROUTER_API_KEY` to enable semantic search. Showing keyword search fallback."
    )

# ── Index status ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Vector Index")
    st.metric("Experiments indexed", len(experiments))
    st.metric("Embedding model", "nemotron-embed-vl-1b-v2")
    st.metric("Vector DB", "Qdrant Cloud" if QDRANT_KEY else "In-memory")
    st.metric("Dimensions", "1024")

    st.divider()
    top_k = st.slider("Results to return", 1, 10, 5)
    use_rag = st.checkbox("RAG answer (OpenRouter LLM)", value=True)

    if st.button("🔄 Re-index experiments", disabled=not OPENROUTER_KEY):
        with st.spinner("Embedding experiments into Qdrant..."):
            try:
                from src.utils.qdrant_store import index_experiments

                n = index_experiments(experiments, force_reindex=True)
                st.success(f"Indexed {n} experiments!")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

# ── Suggested queries ──────────────────────────────────────────────────────────
st.subheader("Try these queries")
SUGGESTED = [
    "Which run achieved the highest MMLU accuracy?",
    "Compare QLoRA vs LoRA GPU memory usage",
    "Which configuration minimized eval loss below 1.35?",
    "Best run under 16 GB GPU memory with MMLU above 65%",
    "Which hyperparameters most improved custom accuracy?",
    "Show all Mistral-7B experiments with rank 64",
    "What's the effect of NEFTune noise on accuracy?",
    "Which 3-epoch run outperformed 5-epoch runs?",
]

cols = st.columns(4)
for i, suggestion in enumerate(SUGGESTED):
    with cols[i % 4]:
        if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
            st.session_state.search_query = suggestion

# ── Search input ───────────────────────────────────────────────────────────────
query = st.text_input(
    "Search experiments",
    value=st.session_state.get("search_query", ""),
    placeholder="e.g. 'best MMLU under 17 GB GPU memory'",
    key="search_input",
)
if query:
    st.session_state.search_query = query


def _keyword_fallback(query: str, experiments: list[dict], top_k: int) -> list[dict]:
    """Simple keyword + metric score fallback when OpenRouter key not set."""
    q = query.lower()
    scored = []
    for r in experiments:
        text = json.dumps(r).lower()
        hits = sum(1 for word in q.split() if word in text)
        scored.append({**r, "_score": hits / max(1, len(q.split()))})
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_k]


if query:
    st.divider()

    if OPENROUTER_KEY:
        # ── Semantic RAG search ────────────────────────────────────────────
        with st.spinner("Embedding query + searching Qdrant..."):
            try:
                # Lazy-index on first search
                from src.utils.qdrant_store import (
                    index_experiments,
                    search_experiments,
                    rag_answer,
                )

                # Index if not yet done (in-memory for Streamlit Cloud)
                try:
                    results = search_experiments(query, top_k=top_k)
                    if not results:
                        index_experiments(experiments)
                        results = search_experiments(query, top_k=top_k)
                except Exception:
                    index_experiments(experiments)
                    results = search_experiments(query, top_k=top_k)

                if use_rag and results:
                    answer, _ = rag_answer(query, top_k=top_k)
                else:
                    answer = None

            except Exception as e:
                st.error(f"Semantic search failed: {e}")
                results = _keyword_fallback(query, experiments, top_k)
                answer = None
    else:
        results = _keyword_fallback(query, experiments, top_k)
        answer = None

    # ── RAG answer ────────────────────────────────────────────────────────────
    if answer:
        st.subheader("🤖 RAG Answer")
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
        st.caption(
            f"Grounded on top-{top_k} semantically similar experiment runs · OpenRouter LLM"
        )
        st.divider()

    # ── Search results ────────────────────────────────────────────────────────
    st.subheader(f"Top {len(results)} Results")

    for r in results:
        score = r.get("_score", 0)
        method_color = "#10b981" if r["method"] == "QLoRA" else "#6366f1"
        st.markdown(
            f"""
<div class="result-card">
  <span class="score-badge">similarity: {score:.3f}</span>
  <strong style="color:#e2e8f0">{r["run_name"]}</strong>
  <span style="color:{method_color};margin-left:8px;font-size:0.85rem">{r["method"]}</span>
  <span style="color:#475569;font-size:0.82rem;margin-left:6px">{r["model"]}</span>
  <div style="margin-top:10px;">
    <span class="metric-pill">MMLU: {r["mmlu_overall"]:.1%}</span>
    <span class="metric-pill">Eval Loss: {r["eval_loss_final"]:.3f}</span>
    <span class="metric-pill">GPU: {r["gpu_mem_gb"]:.1f} GB</span>
    <span class="metric-pill">Judge: {r["judge_composite"]:.1f}/10</span>
    <span class="metric-pill">r={r["rank"]} α={r["lora_alpha"]}</span>
    <span class="metric-pill">lr={r["lr"]:.0e}</span>
    <span class="metric-pill">{r["epochs"]}ep</span>
    <span class="metric-pill">NEFTune={r["neftune"]}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ── Result comparison chart ───────────────────────────────────────────────
    if results:
        st.divider()
        st.subheader("Result Comparison")
        df_res = pd.DataFrame(results)
        metric_col = st.selectbox(
            "Compare by",
            [
                "mmlu_overall",
                "eval_loss_final",
                "judge_composite",
                "gpu_mem_gb",
                "custom_acc",
            ],
        )
        fig = px.bar(
            df_res,
            x="run_name",
            y=metric_col,
            color="_score",
            color_continuous_scale="Viridis",
            template="plotly_dark",
            height=340,
            title=f"Search Results — {metric_col}",
            labels={"run_name": "Run", "_score": "Similarity"},
        )
        fig.update_xaxes(tickangle=30)
        if metric_col == "mmlu_overall":
            fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

# ── How it works ────────────────────────────────────────────────────────────────
with st.expander("⚙️ How RAG Search Works"):
    st.markdown("""
**Step 1 — Indexing (one-time)**
Each experiment run is serialized to a descriptive text document:
> *"Run: llama3-qlora-final-champion. Model: Llama-3-8B. Method: QLoRA with rank 64 and alpha 16. Learning rate: 5e-05, epochs: 5, NEFTune: 10.0. MMLU accuracy: 71.0%, GPU memory: 16.1 GB..."*

These documents are embedded using **OpenRouter's nvidia/llama-nemotron-embed-vl-1b-v2:free** (1024-dim) and stored in **Qdrant Cloud**.

**Step 2 — Search**
Your query is embedded with the same model. Qdrant performs **cosine similarity search** across all 20+ runs, returning the most semantically similar experiments.

**Step 3 — RAG Answer**
The top-K retrieved run documents are injected into a prompt for **Llama-3.1-70B-Instruct** (free via OpenRouter), which generates a grounded, data-specific answer citing run names and exact metrics.
    """)
