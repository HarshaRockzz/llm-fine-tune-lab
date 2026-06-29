"""LLM Fine-Tune Lab — Streamlit Home Page."""

import json
from pathlib import Path

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Fine-Tune Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  .metric-card {
      background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
      border: 1px solid #3a3a5c;
      border-radius: 12px;
      padding: 20px;
      text-align: center;
  }
  .metric-value { font-size: 2.4rem; font-weight: 700; color: #7c3aed; }
  .metric-delta { font-size: 1.0rem; color: #10b981; }
  .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
  .hero-banner {
      background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
      border-radius: 16px;
      padding: 40px;
      margin-bottom: 32px;
      border: 1px solid #4f46e5;
  }
  .tech-tag {
      display: inline-block;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 4px 10px;
      margin: 3px;
      font-size: 0.8rem;
      color: #94a3b8;
  }
  .section-divider { border-top: 1px solid #2d2d4e; margin: 24px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-banner">
  <h1 style="color:#e2e8f0; margin:0; font-size:2.2rem;">🧪 LLM Fine-Tune Lab</h1>
  <p style="color:#a5b4fc; margin-top:8px; font-size:1.05rem;">
    Production-grade fine-tuning pipeline for Llama-3-8B &amp; Mistral-7B using LoRA and 4-bit QLoRA.
    Track 20+ W&amp;B experiments · Evaluate on MMLU, TruthfulQA &amp; custom sets · Serve via vLLM with 2.4× throughput.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Key metrics row ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

METRIC_HTML = """
<div class="metric-card">
  <div class="metric-value">{value}</div>
  <div class="metric-delta">{delta}</div>
  <div class="metric-label">{label}</div>
</div>
"""

with c1:
    st.markdown(
        METRIC_HTML.format(
            value="58%", delta="↓ GPU Memory", label="QLoRA vs Full-precision LoRA"
        ),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        METRIC_HTML.format(
            value="70%", delta="↓ Setup Time", label="Config-driven pipeline"
        ),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        METRIC_HTML.format(
            value="20+", delta="↑ Experiments", label="W&B sweep tracked"
        ),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        METRIC_HTML.format(
            value="54→71%", delta="↑ 17pp accuracy", label="MMLU domain accuracy"
        ),
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(
        METRIC_HTML.format(
            value="2.4×", delta="↑ Throughput", label="vLLM vs HF generate()"
        ),
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# ── Architecture overview ──────────────────────────────────────────────────────
col_arch, col_tech = st.columns([3, 2])

with col_arch:
    st.subheader("System Architecture")
    st.markdown("""
```
┌─────────────────────────────────────────────────────────────┐
│                     LLM Fine-Tune Lab                        │
├──────────────┬────────────────────────┬─────────────────────┤
│  Training    │    Evaluation Harness  │  Inference Server   │
│  Pipeline    │                        │                     │
│              │  ┌──────────────────┐  │  ┌───────────────┐  │
│  LoRA / 4-bit│  │ MMLU (57 subj.)  │  │  │ vLLM Engine   │  │
│  QLoRA       │  │ TruthfulQA       │  │  │ Continuous    │  │
│              │  │ Custom Domain    │  │  │ Batching      │  │
│  PEFT + TRL  │  │ LLM-as-Judge     │  │  │               │  │
│  W&B Sweeps  │  │  (Claude API)    │  │  │ LoRA Adapter  │  │
│  50K samples │  └──────────────────┘  │  │ Hot-swap      │  │
│              │                        │  │               │  │
│              │  Regression Reports    │  │ FastAPI +     │  │
│              │  Checkpoint Tracking   │  │ Prometheus    │  │
└──────────────┴────────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Streamlit Dashboard│
                    │  (this app)         │
                    └─────────────────────┘
```
    """)

with col_tech:
    st.subheader("Tech Stack")
    tech_items = [
        ("🐍", "Python 3.11+", "Core language"),
        ("🔥", "PyTorch 2.2", "Training framework"),
        ("🤗", "HuggingFace PEFT + TRL", "LoRA/QLoRA adapters"),
        ("⚡", "vLLM 0.4+", "2.4× inference throughput"),
        ("🚀", "FastAPI + uvicorn", "Production API server"),
        ("📊", "Weights & Biases", "Experiment tracking"),
        ("🐳", "Docker + compose", "Containerization"),
        ("📈", "Prometheus + Grafana", "Observability"),
        ("🧠", "Claude API", "LLM-as-judge scoring"),
    ]
    for icon, name, desc in tech_items:
        st.markdown(f"**{icon} {name}** — {desc}")

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# ── Navigation cards ──────────────────────────────────────────────────────────
st.subheader("Explore the Lab")
n1, n2, n3, n4 = st.columns(4)

NAV_CARD = """
<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px;min-height:130px;">
  <div style="font-size:2rem;">{icon}</div>
  <div style="font-weight:600;color:#e2e8f0;margin-top:8px;">{title}</div>
  <div style="color:#94a3b8;font-size:0.85rem;margin-top:4px;">{desc}</div>
</div>
"""

with n1:
    st.markdown(
        NAV_CARD.format(
            icon="🏋️",
            title="Training Dashboard",
            desc="Loss curves, LR schedules, GPU metrics from all runs",
        ),
        unsafe_allow_html=True,
    )
with n2:
    st.markdown(
        NAV_CARD.format(
            icon="🤖",
            title="Model Playground",
            desc="Chat with fine-tuned models live; compare base vs. adapter",
        ),
        unsafe_allow_html=True,
    )
with n3:
    st.markdown(
        NAV_CARD.format(
            icon="📊",
            title="Evaluation Results",
            desc="MMLU, TruthfulQA, LLM-judge scores and checkpoint progress",
        ),
        unsafe_allow_html=True,
    )
with n4:
    st.markdown(
        NAV_CARD.format(
            icon="🔬",
            title="Experiment Tracker",
            desc="Sort and filter 20+ W&B experiments; parallel coordinates plot",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# ── Quick stats from demo data ────────────────────────────────────────────────
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
    st.subheader("Quick Stats")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Runs", len(exps))
    with s2:
        best = max(exps, key=lambda x: x.get("mmlu_overall", 0))
        st.metric("Best MMLU", f"{best['mmlu_overall']:.1%}", f"{best['run_name']}")
    with s3:
        avg_gpu_lora = sum(
            e["gpu_mem_gb"] for e in exps if e["method"] == "LoRA"
        ) / max(1, sum(1 for e in exps if e["method"] == "LoRA"))
        avg_gpu_qlora = sum(
            e["gpu_mem_gb"] for e in exps if e["method"] == "QLoRA"
        ) / max(1, sum(1 for e in exps if e["method"] == "QLoRA"))
        savings = (1 - avg_gpu_qlora / avg_gpu_lora) * 100
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
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown(
    """
<div style="text-align:center;color:#475569;font-size:0.8rem;padding:8px 0;">
  LLM Fine-Tune Lab · Built with PyTorch, HuggingFace PEFT/TRL, vLLM, FastAPI & Streamlit ·
  <a href="https://github.com" style="color:#6366f1;text-decoration:none;">GitHub</a>
</div>
""",
    unsafe_allow_html=True,
)
