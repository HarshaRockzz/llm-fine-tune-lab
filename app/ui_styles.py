"""Shared UI styles — inject once per page with inject_global_css()."""

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: radial-gradient(ellipse at top, #0d0f1f 0%, #050810 60%, #020408 100%) !important;
    background-attachment: fixed !important;
}

.block-container { padding-top: 1.2rem !important; max-width: 1420px !important; }

#MainMenu, footer { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #050810; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#6366f1,#10b981); border-radius: 4px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0c1a 0%, #060810 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}
[data-testid="stSidebarNav"] a {
    border-radius: 8px !important;
    transition: all 0.25s ease !important;
    color: #94a3b8 !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(99,102,241,0.12) !important;
    color: #a5b4fc !important;
    padding-left: 20px !important;
}

/* ── Keyframes ── */
@keyframes gradientShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}
@keyframes glow {
    0%,100% { box-shadow: 0 0 30px rgba(99,102,241,0.25), 0 0 80px rgba(99,102,241,0.08); }
    50%      { box-shadow: 0 0 60px rgba(99,102,241,0.55), 0 0 120px rgba(99,102,241,0.18); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes slideUp {
    from { transform: translateY(24px); opacity: 0; }
    to   { transform: translateY(0);    opacity: 1; }
}
@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1);   }
    50%      { opacity:.5; transform:scale(1.4); }
}
@keyframes borderSpin {
    0%   { background-position: 0%   50%; }
    100% { background-position: 300% 50%; }
}
@keyframes float {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-6px); }
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(-45deg,#0d0221,#130538,#091b4a,#0c1f33,#12063a);
    background-size: 400% 400%;
    animation: gradientShift 10s ease infinite, glow 4s ease-in-out infinite;
    border-radius: 22px;
    padding: 52px 48px;
    margin-bottom: 32px;
    border: 1px solid rgba(99,102,241,0.35);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(99,102,241,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 80%, rgba(16,185,129,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(139,92,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute; top:-50%; left:-50%;
    width:200%; height:200%;
    background: radial-gradient(circle at 50% 50%, rgba(99,102,241,0.04) 0%, transparent 60%);
    animation: float 8s ease-in-out infinite;
    pointer-events: none;
}

/* ── Gradient text ── */
.gradient-text {
    background: linear-gradient(135deg,#c7d2fe 0%,#a5b4fc 25%,#818cf8 50%,#6366f1 75%,#10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.gradient-text-green {
    background: linear-gradient(135deg,#6ee7b7,#10b981,#059669);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.025);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 18px;
    padding: 28px 22px;
    text-align: center;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
    animation: slideUp 0.6s ease forwards;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent, #6366f1, #10b981, transparent);
    background-size: 200% auto;
    animation: shimmer 2.5s linear infinite;
}
.metric-card:hover {
    border-color: rgba(99,102,241,0.5);
    box-shadow: 0 12px 40px rgba(99,102,241,0.22), 0 0 0 1px rgba(99,102,241,0.12);
    transform: translateY(-6px) scale(1.01);
}
.metric-value {
    font-size: 2.8rem; font-weight: 900; line-height: 1.1;
    background: linear-gradient(135deg,#c7d2fe,#818cf8,#6366f1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.metric-delta { font-size:.9rem; color:#10b981; font-weight:600; margin-top:6px; }
.metric-label { font-size:.72rem; color:#475569; margin-top:8px; text-transform:uppercase; letter-spacing:.8px; }

/* ── Nav cards ── */
.nav-card {
    background: rgba(255,255,255,0.02);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(51,65,85,0.45);
    border-radius: 18px;
    padding: 26px 22px;
    min-height: 150px;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
    position: relative;
    overflow: hidden;
}
.nav-card::after {
    content:'';
    position:absolute; bottom:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg,#6366f1,#10b981,#8b5cf6);
    transform: scaleX(0); transform-origin: left;
    transition: transform 0.4s ease;
}
.nav-card:hover { border-color:rgba(99,102,241,0.45); box-shadow:0 16px 50px rgba(99,102,241,0.18); transform:translateY(-8px); }
.nav-card:hover::after { transform:scaleX(1); }
.nav-icon { font-size:2.2rem; animation: float 4s ease-in-out infinite; }
.nav-title { font-weight:700; color:#e2e8f0; font-size:1.05rem; margin-top:10px; }
.nav-desc  { color:#64748b; font-size:.82rem; margin-top:6px; line-height:1.5; }

/* ── Tech stack items ── */
.tech-item {
    display:flex; align-items:center; gap:12px;
    padding:11px 16px;
    background:rgba(255,255,255,0.02);
    border:1px solid rgba(51,65,85,0.35);
    border-radius:11px;
    margin-bottom:8px;
    transition:all 0.22s ease;
}
.tech-item:hover {
    border-color:rgba(99,102,241,0.4);
    background:rgba(99,102,241,0.06);
    transform:translateX(6px);
    box-shadow:4px 0 20px rgba(99,102,241,0.12);
}
.tech-icon { font-size:1.1rem; min-width:22px; }
.tech-name { color:#a5b4fc; font-weight:600; font-size:.88rem; }
.tech-desc { color:#475569; font-size:.78rem; }

/* ── Live indicator ── */
.live-dot {
    display:inline-block; width:8px; height:8px;
    background:#10b981; border-radius:50%;
    margin-right:7px;
    animation: pulse-dot 2s ease-in-out infinite;
    box-shadow:0 0 8px rgba(16,185,129,0.6);
}

/* ── Section header ── */
.section-hdr {
    font-size:1.35rem; font-weight:700;
    background:linear-gradient(135deg,#f1f5f9,#a5b4fc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin-bottom:16px; letter-spacing:-.3px;
}

/* ── Glow divider ── */
.glow-div {
    border:none; height:1px;
    background:linear-gradient(90deg,transparent,rgba(99,102,241,0.35),rgba(16,185,129,0.2),transparent);
    margin:28px 0;
}

/* ── Page title ── */
.page-title {
    font-size:2.2rem; font-weight:900;
    background:linear-gradient(135deg,#ffffff 0%,#c7d2fe 40%,#6366f1 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    letter-spacing:-.5px; line-height:1.2;
}
.page-caption { color:#475569; font-size:.88rem; margin-top:4px; margin-bottom:24px; }

/* ── Stat pill ── */
.stat-pill {
    display:inline-flex; align-items:center; gap:5px;
    background:rgba(99,102,241,0.08);
    border:1px solid rgba(99,102,241,0.22);
    border-radius:20px; padding:4px 12px;
    font-size:.78rem; color:#a5b4fc; margin:3px;
}

/* ── Result / answer box ── */
.answer-box {
    background:rgba(99,102,241,0.06);
    border-left:3px solid #6366f1;
    border-radius:0 14px 14px 0;
    padding:18px 20px; line-height:1.8; color:#e2e8f0;
}
.result-card {
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(51,65,85,0.4);
    border-radius:14px; padding:18px; margin-bottom:12px;
    transition:all 0.25s ease;
}
.result-card:hover { border-color:rgba(99,102,241,0.4); transform:translateX(4px); }

/* ── Streamlit native overrides ── */
div[data-testid="metric-container"] {
    background:rgba(255,255,255,0.025) !important;
    border:1px solid rgba(99,102,241,0.18) !important;
    border-radius:14px !important;
    padding:18px 16px !important;
    transition:all 0.25s ease !important;
}
div[data-testid="metric-container"]:hover {
    border-color:rgba(99,102,241,0.45) !important;
    box-shadow:0 6px 24px rgba(99,102,241,0.15) !important;
    transform:translateY(-2px) !important;
}
[data-testid="stMetricValue"]  { color:#a5b4fc !important; font-weight:800 !important; }
[data-testid="stMetricLabel"]  { color:#64748b !important; font-size:.78rem !important; text-transform:uppercase !important; letter-spacing:.5px !important; }
[data-testid="stMetricDelta"]  { color:#10b981 !important; }

.stButton>button {
    background:linear-gradient(135deg,#6366f1,#4f46e5) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important; font-weight:600 !important;
    transition:all 0.25s ease !important;
}
.stButton>button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 24px rgba(99,102,241,0.45) !important;
}

.stTextInput input, .stTextArea textarea, input[type="text"] {
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(99,102,241,0.25) !important;
    border-radius:10px !important; color:#e2e8f0 !important;
    transition:all 0.2s ease !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color:#6366f1 !important;
    box-shadow:0 0 0 3px rgba(99,102,241,0.15) !important;
}

.stSelectbox [data-baseweb="select"] > div:first-child,
.stMultiSelect [data-baseweb="select"] > div:first-child {
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(99,102,241,0.25) !important;
    border-radius:10px !important;
}
span[data-baseweb="tag"] {
    background:rgba(99,102,241,0.18) !important;
    border:1px solid rgba(99,102,241,0.35) !important;
    border-radius:6px !important;
    color:#a5b4fc !important;
}

.stSlider [data-baseweb="slider"] [role="progressbar"] {
    background:linear-gradient(90deg,#6366f1,#10b981) !important;
}

details summary {
    background:rgba(255,255,255,0.02) !important;
    border:1px solid rgba(51,65,85,0.4) !important;
    border-radius:10px !important; padding:12px 16px !important;
    color:#94a3b8 !important; transition:all 0.2s ease !important;
}
details summary:hover { border-color:rgba(99,102,241,0.35) !important; color:#a5b4fc !important; }

div[data-testid="stDataFrame"] > div {
    border:1px solid rgba(99,102,241,0.15) !important;
    border-radius:12px !important;
    overflow:hidden !important;
}

.js-plotly-plot .plotly { border-radius:14px !important; }

[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label    { color:#94a3b8 !important; }
</style>
"""


def inject_global_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
