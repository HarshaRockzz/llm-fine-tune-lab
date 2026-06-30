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
  <h1 style="font-size:3.4rem;font-weight:900;margin:0;line-height:1.05;letter-spacing:-1.5px;color:#f1f5f9;">
    🧪 <span class="gradient-text">LLM Fine-Tune Lab</span>
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
    ("+17pp", "54% → 71% MMLU", "Domain accuracy gain", "0.45s"),
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
st.markdown(
    "<div class='section-hdr'>⚙️ System Architecture</div>",
    unsafe_allow_html=True,
)
st.markdown(
    """
<div style="display:flex;flex-direction:column;gap:14px;font-family:Inter,sans-serif;">
  <div style="display:grid;grid-template-columns:1fr 32px 1fr 32px 1fr;align-items:stretch;gap:0;">
    <div style="background:linear-gradient(145deg,rgba(99,102,241,0.14),rgba(99,102,241,0.04));border:1px solid rgba(99,102,241,0.40);border-radius:16px;padding:18px 20px;box-shadow:0 6px 24px rgba(99,102,241,0.12),inset 0 1px 0 rgba(255,255,255,0.05);">
      <div style="color:#6366f1;font-size:.62rem;font-weight:800;text-transform:uppercase;letter-spacing:2px;margin-bottom:9px;display:flex;align-items:center;gap:7px;">
        <span style="display:inline-block;width:7px;height:7px;background:#6366f1;border-radius:50%;box-shadow:0 0 8px rgba(99,102,241,0.7);flex-shrink:0;"></span>01 — Training
      </div>
      <div style="color:#e2e8f0;font-weight:800;font-size:.93rem;margin-bottom:12px;">PEFT + TRL Pipeline</div>
      <div style="display:flex;flex-direction:column;gap:5px;">
        <div style="color:#a5b4fc;font-size:.74rem;background:rgba(99,102,241,0.09);border-left:2px solid rgba(99,102,241,0.45);border-radius:0 5px 5px 0;padding:3px 8px;">LoRA / 4-bit QLoRA NF4</div>
        <div style="color:#a5b4fc;font-size:.74rem;background:rgba(99,102,241,0.09);border-left:2px solid rgba(99,102,241,0.45);border-radius:0 5px 5px 0;padding:3px 8px;">W&amp;B Bayesian Sweeps</div>
        <div style="color:#a5b4fc;font-size:.74rem;background:rgba(99,102,241,0.09);border-left:2px solid rgba(99,102,241,0.45);border-radius:0 5px 5px 0;padding:3px 8px;">50K domain samples</div>
        <div style="color:#a5b4fc;font-size:.74rem;background:rgba(99,102,241,0.09);border-left:2px solid rgba(99,102,241,0.45);border-radius:0 5px 5px 0;padding:3px 8px;">Llama-3-8B &amp; Mistral-7B</div>
      </div>
    </div>
    <div style="display:flex;align-items:center;justify-content:center;">
      <span style="color:#6366f1;font-size:1.5rem;font-weight:300;">→</span>
    </div>
    <div style="background:linear-gradient(145deg,rgba(16,185,129,0.12),rgba(16,185,129,0.03));border:1px solid rgba(16,185,129,0.36);border-radius:16px;padding:18px 20px;box-shadow:0 6px 24px rgba(16,185,129,0.10),inset 0 1px 0 rgba(255,255,255,0.05);">
      <div style="color:#10b981;font-size:.62rem;font-weight:800;text-transform:uppercase;letter-spacing:2px;margin-bottom:9px;display:flex;align-items:center;gap:7px;">
        <span style="display:inline-block;width:7px;height:7px;background:#10b981;border-radius:50%;box-shadow:0 0 8px rgba(16,185,129,0.7);flex-shrink:0;"></span>02 — Evaluation
      </div>
      <div style="color:#e2e8f0;font-weight:800;font-size:.93rem;margin-bottom:12px;">Multi-Harness Eval</div>
      <div style="display:flex;flex-direction:column;gap:5px;">
        <div style="color:#6ee7b7;font-size:.74rem;background:rgba(16,185,129,0.09);border-left:2px solid rgba(16,185,129,0.45);border-radius:0 5px 5px 0;padding:3px 8px;">MMLU — 57 subjects</div>
        <div style="color:#6ee7b7;font-size:.74rem;background:rgba(16,185,129,0.09);border-left:2px solid rgba(16,185,129,0.45);border-radius:0 5px 5px 0;padding:3px 8px;">TruthfulQA MC1/MC2</div>
        <div style="color:#6ee7b7;font-size:.74rem;background:rgba(16,185,129,0.09);border-left:2px solid rgba(16,185,129,0.45);border-radius:0 5px 5px 0;padding:3px 8px;">LLM-as-Judge (Claude)</div>
        <div style="color:#6ee7b7;font-size:.74rem;background:rgba(16,185,129,0.09);border-left:2px solid rgba(16,185,129,0.45);border-radius:0 5px 5px 0;padding:3px 8px;">Custom domain tests</div>
      </div>
    </div>
    <div style="display:flex;align-items:center;justify-content:center;">
      <span style="color:#10b981;font-size:1.5rem;font-weight:300;">→</span>
    </div>
    <div style="background:linear-gradient(145deg,rgba(139,92,246,0.12),rgba(139,92,246,0.03));border:1px solid rgba(139,92,246,0.36);border-radius:16px;padding:18px 20px;box-shadow:0 6px 24px rgba(139,92,246,0.10),inset 0 1px 0 rgba(255,255,255,0.05);">
      <div style="color:#8b5cf6;font-size:.62rem;font-weight:800;text-transform:uppercase;letter-spacing:2px;margin-bottom:9px;display:flex;align-items:center;gap:7px;">
        <span style="display:inline-block;width:7px;height:7px;background:#8b5cf6;border-radius:50%;box-shadow:0 0 8px rgba(139,92,246,0.7);flex-shrink:0;"></span>03 — Inference
      </div>
      <div style="color:#e2e8f0;font-weight:800;font-size:.93rem;margin-bottom:12px;">vLLM Serving</div>
      <div style="display:flex;flex-direction:column;gap:5px;">
        <div style="color:#c4b5fd;font-size:.74rem;background:rgba(139,92,246,0.09);border-left:2px solid rgba(139,92,246,0.45);border-radius:0 5px 5px 0;padding:3px 8px;">Continuous batching</div>
        <div style="color:#c4b5fd;font-size:.74rem;background:rgba(139,92,246,0.09);border-left:2px solid rgba(139,92,246,0.45);border-radius:0 5px 5px 0;padding:3px 8px;">LoRA adapter hot-swap</div>
        <div style="color:#c4b5fd;font-size:.74rem;background:rgba(139,92,246,0.09);border-left:2px solid rgba(139,92,246,0.45);border-radius:0 5px 5px 0;padding:3px 8px;">FastAPI + Prometheus</div>
        <div style="color:#c4b5fd;font-size:.74rem;background:rgba(139,92,246,0.09);border-left:2px solid rgba(139,92,246,0.45);border-radius:0 5px 5px 0;padding:3px 8px;">2.4× HF throughput</div>
      </div>
    </div>
  </div>
  <div style="display:flex;justify-content:center;">
    <div style="display:flex;flex-direction:column;align-items:center;">
      <div style="width:2px;height:16px;background:linear-gradient(180deg,rgba(139,92,246,0.6),rgba(245,158,11,0.6));"></div>
      <span style="color:#f59e0b;font-size:.9rem;">↓</span>
    </div>
  </div>
  <div style="background:linear-gradient(145deg,rgba(245,158,11,0.10),rgba(245,158,11,0.03));border:1px solid rgba(245,158,11,0.32);border-radius:16px;padding:16px 22px;display:flex;flex-wrap:wrap;align-items:center;gap:16px;box-shadow:0 6px 24px rgba(245,158,11,0.07),inset 0 1px 0 rgba(255,255,255,0.04);">
    <div style="flex-shrink:0;">
      <div style="color:#f59e0b;font-size:.62rem;font-weight:800;text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;display:flex;align-items:center;gap:7px;">
        <span style="display:inline-block;width:7px;height:7px;background:#f59e0b;border-radius:50%;box-shadow:0 0 8px rgba(245,158,11,0.7);flex-shrink:0;"></span>04 — Dashboard
      </div>
      <div style="color:#e2e8f0;font-weight:800;font-size:.93rem;">Streamlit App</div>
    </div>
    <div style="width:1px;height:38px;background:rgba(245,158,11,0.20);flex-shrink:0;"></div>
    <div style="display:flex;flex-wrap:wrap;gap:8px;flex:1;">
      <span style="background:rgba(245,158,11,0.09);border:1px solid rgba(245,158,11,0.24);border-radius:20px;padding:4px 12px;font-size:.74rem;color:#fcd34d;">Qdrant Vector RAG</span>
      <span style="background:rgba(99,102,241,0.09);border:1px solid rgba(99,102,241,0.24);border-radius:20px;padding:4px 12px;font-size:.74rem;color:#a5b4fc;">OpenRouter API</span>
      <span style="background:rgba(16,185,129,0.09);border:1px solid rgba(16,185,129,0.24);border-radius:20px;padding:4px 12px;font-size:.74rem;color:#6ee7b7;">Real-time Analytics</span>
      <span style="background:rgba(139,92,246,0.09);border:1px solid rgba(139,92,246,0.24);border-radius:20px;padding:4px 12px;font-size:.74rem;color:#c4b5fd;">5 Interactive Pages</span>
      <span style="background:rgba(244,63,94,0.09);border:1px solid rgba(244,63,94,0.24);border-radius:20px;padding:4px 12px;font-size:.74rem;color:#fda4af;">GitHub CI/CD</span>
    </div>
  </div>
</div>""",
    unsafe_allow_html=True,
)

st.markdown("<div class='section-hdr'>🛠️ Tech Stack</div>", unsafe_allow_html=True)
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
_tc = st.columns(5)
for i, (icon, name, desc) in enumerate(_TECH):
    with _tc[i % 5]:
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
