"""Experiment Search — RAG over 20+ runs using Qdrant + OpenRouter embeddings."""

# ruff: noqa: E402
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Experiment Search", page_icon="🔍", layout="wide")

# Ensure repo root is first on sys.path so `import src` finds the local package.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_APP_DIR = str(Path(__file__).resolve().parent.parent)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from ui_styles import inject_global_css

inject_global_css()

DATA_PATH = Path(__file__).parent.parent.parent / "data"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
QDRANT_KEY = os.environ.get("QDRANT_API_KEY", "")

_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter", color="#94a3b8", size=12),
)


@st.cache_data(ttl=300)
def load_experiments() -> list[dict]:
    p = DATA_PATH / "demo_experiments.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


experiments = load_experiments()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="page-title">🔍 Experiment Search</div>
<div class="page-caption">
  Semantic search over 20+ W&B runs using <strong style="color:#a5b4fc;">OpenRouter embeddings</strong> + <strong style="color:#a5b4fc;">Qdrant vector DB</strong>.
  Ask in plain English — <em>"which run used the least GPU memory while staying above 65% MMLU?"</em> — and get a RAG-grounded answer.
</div>
""",
    unsafe_allow_html=True,
)

if not OPENROUTER_KEY:
    st.warning(
        "Set `OPENROUTER_API_KEY` to enable semantic search. Showing keyword search fallback."
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='color:#a5b4fc;font-weight:700;font-size:1rem;margin-bottom:12px;'>🗄️ Vector Index</div>",
        unsafe_allow_html=True,
    )
    st.metric("Experiments indexed", len(experiments))
    st.metric("Embedding model", "nemotron-embed-vl-1b")
    st.metric("Vector DB", "Qdrant Cloud" if QDRANT_KEY else "In-memory")
    st.metric("Dimensions", "1024")

    st.markdown("<hr class='glow-div'>", unsafe_allow_html=True)
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

# ── Suggested queries ─────────────────────────────────────────────────────────
st.markdown(
    "<div class='section-hdr'>💡 Suggested Queries</div>", unsafe_allow_html=True
)
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

st.markdown("<hr class='glow-div'>", unsafe_allow_html=True)

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
    q = query.lower()
    scored = []
    for r in experiments:
        text = json.dumps(r).lower()
        hits = sum(1 for word in q.split() if word in text)
        scored.append({**r, "_score": hits / max(1, len(q.split()))})
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[:top_k]


if query:
    st.markdown("<hr class='glow-div'>", unsafe_allow_html=True)

    if OPENROUTER_KEY:
        with st.spinner("Embedding query + searching Qdrant..."):
            try:
                from src.utils.qdrant_store import (
                    index_experiments,
                    search_experiments,
                    rag_answer,
                )

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
        st.markdown(
            "<div class='section-hdr'>🤖 RAG Answer</div>", unsafe_allow_html=True
        )
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='page-caption' style='margin-top:8px;'>Grounded on top-{top_k} semantically similar experiment runs · OpenRouter LLM</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr class='glow-div'>", unsafe_allow_html=True)

    # ── Search results ─────────────────────────────────────────────────────────
    st.markdown(
        f"<div class='section-hdr'>📋 Top {len(results)} Results</div>",
        unsafe_allow_html=True,
    )

    for r in results:
        score = r.get("_score", 0)
        method_color = "#10b981" if r["method"] == "QLoRA" else "#6366f1"
        score_pct = int(score * 100)
        st.markdown(
            f"""
<div class="result-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
    <div>
      <strong style="color:#e2e8f0;font-size:.95rem;">{r["run_name"]}</strong>
      <span style="color:{method_color};margin-left:10px;font-size:.82rem;
            background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
            border-radius:10px;padding:2px 8px;">{r["method"]}</span>
      <span style="color:#475569;font-size:.8rem;margin-left:8px;">{r["model"]}</span>
    </div>
    <div style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);
         border-radius:12px;padding:3px 10px;font-size:.78rem;color:#a5b4fc;white-space:nowrap;">
      {score_pct}% match
    </div>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:6px;">
    <span class="stat-pill">MMLU: {r["mmlu_overall"]:.1%}</span>
    <span class="stat-pill">Eval Loss: {r["eval_loss_final"]:.3f}</span>
    <span class="stat-pill">GPU: {r["gpu_mem_gb"]:.1f} GB</span>
    <span class="stat-pill">Judge: {r["judge_composite"]:.1f}/10</span>
    <span class="stat-pill">r={r["rank"]} α={r["lora_alpha"]}</span>
    <span class="stat-pill">lr={r["lr"]:.0e}</span>
    <span class="stat-pill">{r["epochs"]}ep</span>
    <span class="stat-pill">NEFTune={r["neftune"]}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ── Result comparison chart ───────────────────────────────────────────────
    if results:
        st.markdown("<hr class='glow-div'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-hdr'>📊 Result Comparison</div>",
            unsafe_allow_html=True,
        )
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
            height=360,
            title=f"Search Results — {metric_col}",
            labels={"run_name": "Run", "_score": "Similarity"},
        )
        fig.update_xaxes(tickangle=30)
        if metric_col == "mmlu_overall":
            fig.update_yaxes(tickformat=".0%")
        fig.update_layout(**_CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ── How it works ───────────────────────────────────────────────────────────────
with st.expander("⚙️ How RAG Search Works"):
    st.markdown("""
**Step 1 — Indexing (one-time)**

Each experiment run is serialized to a descriptive text document:
> *"Run: llama3-qlora-final-champion. Model: Llama-3-8B. Method: QLoRA with rank 64 and alpha 16. Learning rate: 5e-05, epochs: 5, NEFTune: 10.0. MMLU accuracy: 71.0%, GPU memory: 16.1 GB..."*

These documents are embedded using **OpenRouter's nvidia/llama-nemotron-embed-vl-1b-v2:free** (1024-dim) and stored in **Qdrant Cloud**.

**Step 2 — Search**

Your query is embedded with the same model. Qdrant performs **cosine similarity search** across all 20+ runs, returning the most semantically similar experiments.

**Step 3 — RAG Answer**

The top-K retrieved run documents are injected into a prompt for an LLM (free via OpenRouter), which generates a grounded, data-specific answer citing run names and exact metrics.
    """)
