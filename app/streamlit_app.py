"""LLM Fine-Tune Lab — Streamlit Home Page."""

# ruff: noqa: E402
import json
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="LLM Fine-Tune Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

_APP_DIR = str(Path(__file__).resolve().parent)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
from ui_styles import inject_global_css

inject_global_css()

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-banner">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
    <span class="live-dot"></span>
    <span style="color:#6ee7b7;font-size:.78rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;">Production Demo</span>
  </div>
  <h1 style="font-size:3.4rem;font-weight:900;margin:0;line-height:1.05;letter-spacing:-1.5px;">
    <span class="gradient-text">🧪 LLM Fine-Tune Lab</span>
  </h1>
  <p style="color:#94a3b8;margin-top:18px;font-size:1.08rem;max-width:720px;line-height:1.8;">
    Production-grade fine-tuning pipeline for
    <span style="color:#a5b4fc;font-weight:600;">Llama-3-8B</span> &amp;
    <span style="color:#a5b4fc;font-weight:600;">Mistral-7B</span> using LoRA &amp; 4-bit QLoRA.
    Track <span style="color:#6ee7b7;font-weight:600;">20+ W&amp;B experiments</span> ·
    Evaluate on MMLU, TruthfulQA &amp; custom sets ·
    Serve via vLLM with <span style="color:#6ee7b7;font-weight:600;">2.4× throughput</span>.
  </p>
  <div style="margin-top:24px;display:flex;flex-wrap:wrap;gap:8px;">
    <span class="stat-pill">🔥 PyTorch 2.2</span>
    <span class="stat-pill">🤗 PEFT + TRL</span>
    <span class="stat-pill">⚡ vLLM Serve</span>
    <span class="stat-pill">📊 W&amp;B Bayesian Sweeps</span>
    <span class="stat-pill">🧠 LLM-as-Judge</span>
    <span class="stat-pill">🗄️ Qdrant RAG</span>
    <span class="stat-pill">🚀 FastAPI</span>
    <span class="stat-pill">📈 Prometheus</span>
    <span class="stat-pill">🐳 Docker</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Key metric cards ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

_CARDS = [
    ("58%", "↓ GPU Memory", "QLoRA vs Full-precision LoRA", "0.15s"),
    ("70%", "↓ Setup Time", "Config-driven pipeline", "0.25s"),
    ("20+", "↑ Experiments", "W&amp;B Bayesian sweep", "0.35s"),
    ("54→71%", "+17pp accuracy", "MMLU domain accuracy", "0.45s"),
    ("2.4×", "↑ Throughput", "vLLM vs HF generate()", "0.55s"),
]

for col, (val, delta, label, delay) in zip([c1, c2, c3, c4, c5], _CARDS):
    with col:
        st.markdown(
            f"""
<div class="metric-card" style="animation-delay:{delay};">
  <div class="metric-value">{val}</div>
  <div class="metric-delta">{delta}</div>
  <div class="metric-label">{label}</div>
</div>""",
            unsafe_allow_html=True,
        )

st.markdown("<hr class='glow-div'>", unsafe_allow_html=True)

# ── Architecture + Tech stack ─────────────────────────────────────────────────
col_arch, col_tech = st.columns([3, 2])

with col_arch:
    st.markdown(
        "<div class='section-hdr'>⚙️ System Architecture</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div style="background:rgba(255,255,255,0.015);border:1px solid rgba(99,102,241,0.2);
     border-radius:16px;padding:24px 22px;font-family:'JetBrains Mono','Fira Code',
     'Courier New',monospace;font-size:.74rem;color:#64748b;line-height:2;overflow-x:auto;">
<span style="color:#6366f1;font-weight:700;">╔══════════════════════════════════════════════════════════════╗</span>
<span style="color:#6366f1;font-weight:700;">║</span>                  <span style="color:#a5b4fc;font-weight:800;">LLM Fine-Tune Lab</span>                           <span style="color:#6366f1;font-weight:700;">║</span>
<span style="color:#6366f1;font-weight:700;">╠══════════════════╦════════════════════════╦═════════════════╣</span>
<span style="color:#6366f1;font-weight:700;">║</span> <span style="color:#10b981;font-weight:600;">Training Pipeline</span> <span style="color:#6366f1;font-weight:700;">║</span> <span style="color:#10b981;font-weight:600;">Evaluation Harness    </span><span style="color:#6366f1;font-weight:700;">║</span> <span style="color:#10b981;font-weight:600;">Inference Server</span> <span style="color:#6366f1;font-weight:700;">║</span>
<span style="color:#6366f1;font-weight:700;">║</span> LoRA / 4-bit     <span style="color:#6366f1;font-weight:700;">║</span> MMLU (57 subjects)    <span style="color:#6366f1;font-weight:700;">║</span> vLLM Continuous <span style="color:#6366f1;font-weight:700;">║</span>
<span style="color:#6366f1;font-weight:700;">║</span> QLoRA NF4        <span style="color:#6366f1;font-weight:700;">║</span> TruthfulQA MC1/MC2   <span style="color:#6366f1;font-weight:700;">║</span> Batching        <span style="color:#6366f1;font-weight:700;">║</span>
<span style="color:#6366f1;font-weight:700;">║</span> PEFT + TRL       <span style="color:#6366f1;font-weight:700;">║</span> Custom Domain Eval   <span style="color:#6366f1;font-weight:700;">║</span> LoRA Hot-swap   <span style="color:#6366f1;font-weight:700;">║</span>
<span style="color:#6366f1;font-weight:700;">║</span> W&amp;B 20+ Sweeps  <span style="color:#6366f1;font-weight:700;">║</span> LLM-as-Judge (API)   <span style="color:#6366f1;font-weight:700;">║</span> FastAPI +       <span style="color:#6366f1;font-weight:700;">║</span>
<span style="color:#6366f1;font-weight:700;">║</span> 50K samples      <span style="color:#6366f1;font-weight:700;">║</span> Checkpoint Tracking  <span style="color:#6366f1;font-weight:700;">║</span> Prometheus obs. <span style="color:#6366f1;font-weight:700;">║</span>
<span style="color:#6366f1;font-weight:700;">╚══════════════════╩════════════════════════╩═════════════════╝</span>
                              <span style="color:#334155;">▼</span>
              <span style="color:#6366f1;font-weight:700;">╔═══════════════════════════════╗</span>
              <span style="color:#6366f1;font-weight:700;">║</span>  <span style="color:#a5b4fc;font-weight:700;">Streamlit Dashboard</span> (this app)  <span style="color:#6366f1;font-weight:700;">║</span>
              <span style="color:#6366f1;font-weight:700;">║</span>  Qdrant RAG · OpenRouter API   <span style="color:#6366f1;font-weight:700;">║</span>
              <span style="color:#6366f1;font-weight:700;">╚═══════════════════════════════╝</span>
</div>""",
        unsafe_allow_html=True,
    )

with col_tech:
    st.markdown(
        "<div class='section-hdr'>🛠️ Tech Stack</div>",
        unsafe_allow_html=True,
    )
    _TECH = [
        ("🐍", "Python 3.11+", "Core language & tooling"),
        ("🔥", "PyTorch 2.2", "Training framework"),
        ("🤗", "PEFT + TRL", "LoRA / QLoRA adapters"),
        ("⚡", "vLLM 0.4+", "2.4× inference speedup"),
        ("🚀", "FastAPI", "Production API server"),
        ("📊", "Weights &amp; Biases", "Experiment tracking"),
        ("🗄️", "Qdrant Cloud", "Vector DB for RAG search"),
        ("🧠", "Claude API", "LLM-as-judge scoring"),
        ("🐳", "Docker + compose", "Containerization"),
        ("📈", "Prometheus + Grafana", "Observability stack"),
    ]
    for icon, name, desc in _TECH:
        st.markdown(
            f"""
<div class="tech-item">
  <span class="tech-icon">{icon}</span>
  <div>
    <div class="tech-name">{name}</div>
    <div class="tech-desc">{desc}</div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

st.markdown("<hr class='glow-div'>", unsafe_allow_html=True)

# ── Navigation cards ──────────────────────────────────────────────────────────
st.markdown("<div class='section-hdr'>🗺️ Explore the Lab</div>", unsafe_allow_html=True)

n1, n2, n3, n4, n5 = st.columns(5)
_NAV = [
    (
        n1,
        "🏋️",
        "Training Dashboard",
        "Loss curves, LR schedules, GPU metrics &amp; hyperparameter sensitivity across all runs",
    ),
    (
        n2,
        "🤖",
        "Model Playground",
        "Chat with fine-tuned adapters live. Side-by-side A/B comparison &amp; streaming",
    ),
    (
        n3,
        "📊",
        "Evaluation Results",
        "MMLU, TruthfulQA, LLM-judge scores. Checkpoint progress &amp; vLLM benchmarks",
    ),
    (
        n4,
        "🔬",
        "Experiment Tracker",
        "Filter 20+ W&amp;B runs. Parallel coordinates, Pareto front &amp; correlation heatmap",
    ),
    (
        n5,
        "🔍",
        "Experiment Search",
        "RAG semantic search over all experiments using Qdrant + OpenRouter embeddings",
    ),
]

for col, icon, title, desc in _NAV:
    with col:
        st.markdown(
            f"""
<div class="nav-card">
  <div class="nav-icon">{icon}</div>
  <div class="nav-title">{title}</div>
  <div class="nav-desc">{desc}</div>
</div>""",
            unsafe_allow_html=True,
        )

st.markdown("<hr class='glow-div'>", unsafe_allow_html=True)

# ── Live stats from demo data ─────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data"


@st.cache_data(ttl=300)
def load_experiments():
    p = DATA_PATH / "demo_experiments.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


@st.cache_data(ttl=300)
def load_eval():
    p = DATA_PATH / "demo_eval_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


exps = load_experiments()
eval_data = load_eval()

if exps:
    st.markdown(
        "<div class='section-hdr'>📈 Live Statistics</div>", unsafe_allow_html=True
    )
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Runs", len(exps))
    with s2:
        best = max(exps, key=lambda x: x.get("mmlu_overall", 0))
        st.metric("Best MMLU", f"{best['mmlu_overall']:.1%}", best["run_name"])
    with s3:
        avg_lora = sum(e["gpu_mem_gb"] for e in exps if e["method"] == "LoRA") / max(
            1, sum(1 for e in exps if e["method"] == "LoRA")
        )
        avg_qlora = sum(e["gpu_mem_gb"] for e in exps if e["method"] == "QLoRA") / max(
            1, sum(1 for e in exps if e["method"] == "QLoRA")
        )
        savings = (1 - avg_qlora / avg_lora) * 100
        st.metric("QLoRA GPU Savings", f"{savings:.0f}%", "vs LoRA")
    with s4:
        if eval_data:
            vllm = eval_data["inference_comparison"]["vllm_continuous_batching"]
            hf = eval_data["inference_comparison"]["hf_generate"]
            st.metric(
                "vLLM Speedup",
                f"{vllm['throughput_tok_per_sec'] / hf['throughput_tok_per_sec']:.1f}×",
                "vs HF generate()",
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<hr class="glow-div">
<div style="text-align:center;padding:10px 0 4px;">
  <span style="color:#334155;font-size:.74rem;">
    LLM Fine-Tune Lab &nbsp;·&nbsp;
    <span style="color:#475569;">PyTorch · PEFT/TRL · vLLM · FastAPI · Streamlit · Qdrant</span>
    &nbsp;·&nbsp;
    <a href="https://github.com/HarshaRockzz/llm-fine-tune-lab"
       style="color:#6366f1;text-decoration:none;font-weight:500;">GitHub ↗</a>
  </span>
</div>""",
    unsafe_allow_html=True,
)
