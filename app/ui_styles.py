"""Shared UI styles — inject once per page with inject_global_css()."""

import streamlit as st

_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap"'
    ' rel="stylesheet">'
)

_CSS = """
<style>
/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── App background ── */
.stApp {
    background: #060a16 !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 90% 50% at 15% -5%, rgba(99,102,241,0.18) 0%, transparent 55%),
        radial-gradient(ellipse 60% 40% at 85% 105%, rgba(16,185,129,0.10) 0%, transparent 55%),
        #060a16 !important;
    min-height: 100vh !important;
}
[data-testid="stMain"], section.main {
    background: transparent !important;
}
.block-container {
    padding-top: 1rem !important;
    max-width: 1440px !important;
    background: transparent !important;
}
#MainMenu, footer { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #060a16; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #6366f1, #10b981);
    border-radius: 4px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0d1e 0%, #060810 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.14) !important;
}
[data-testid="stSidebarNav"] a {
    border-radius: 8px !important;
    transition: all 0.22s ease !important;
    color: #94a3b8 !important;
    margin-bottom: 2px !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(99,102,241,0.10) !important;
    color: #a5b4fc !important;
    padding-left: 18px !important;
}
/* Sidebar metric — constrained size */
[data-testid="stSidebar"] div[data-testid="metric-container"] {
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #a5b4fc !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    font-size: 0.67rem !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stSidebar"] [data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
    color: #10b981 !important;
}

/* ── Keyframe animations ── */
@keyframes gradientShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}
@keyframes glow {
    0%,100% {
        box-shadow: 0 0 30px rgba(99,102,241,0.22),
                    0 0 80px rgba(99,102,241,0.07);
    }
    50% {
        box-shadow: 0 0 55px rgba(99,102,241,0.48),
                    0 0 110px rgba(99,102,241,0.15);
    }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes slideUp {
    from { transform: translateY(22px); opacity: 0; }
    to   { transform: translateY(0);    opacity: 1; }
}
@keyframes floatY {
    0%,100% { transform: translateY(0px); }
    50%      { transform: translateY(-6px); }
}
@keyframes pulseDot {
    0%,100% { opacity: 1;  transform: scale(1);   }
    50%      { opacity: .5; transform: scale(1.4); }
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(-45deg,#0d0221,#110436,#091748,#0b1d30,#100538);
    background-size: 400% 400%;
    animation: gradientShift 10s ease infinite, glow 4s ease-in-out infinite;
    border-radius: 20px;
    padding: 52px 48px 44px;
    margin-bottom: 32px;
    border: 1px solid rgba(99,102,241,0.32);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(99,102,241,0.16) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 80%, rgba(16,185,129,0.10) 0%, transparent 60%);
    pointer-events: none;
}

/* ── Gradient text — use ONLY on pure-text spans (no emoji inside!) ── */
.gradient-text {
    background: linear-gradient(135deg, #c7d2fe 0%, #a5b4fc 30%, #818cf8 65%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Page title — solid color, emoji-safe ── */
.page-title {
    font-size: 2.15rem;
    font-weight: 900;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    line-height: 1.15;
    margin-bottom: 2px;
}
.page-caption {
    color: #475569;
    font-size: .87rem;
    margin-top: 4px;
    margin-bottom: 28px;
    line-height: 1.5;
}

/* ── Section header — left accent, solid color (emoji-safe, no gradient-clip) ── */
.section-hdr {
    display: flex;
    align-items: center;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    background: linear-gradient(90deg, rgba(99,102,241,0.10) 0%, rgba(99,102,241,0.02) 70%, transparent 100%);
    border-left: 3px solid #6366f1;
    padding: 10px 16px 10px 14px;
    border-radius: 0 10px 10px 0;
    margin: 4px 0 20px 0;
    letter-spacing: -0.15px;
}

/* ── Glow divider ── */
.glow-div {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.28), rgba(16,185,129,0.15), transparent);
    margin: 28px 0;
}

/* ── Metric cards (home page custom HTML cards) ── */
.metric-card {
    background: linear-gradient(145deg, rgba(99,102,241,0.10), rgba(99,102,241,0.03));
    border: 1px solid rgba(99,102,241,0.28);
    border-radius: 18px;
    padding: 28px 20px;
    text-align: center;
    transition: all 0.32s cubic-bezier(0.4,0,0.2,1);
    animation: slideUp 0.6s ease forwards;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05);
}
.metric-card::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height: 2px;
    background: linear-gradient(90deg, transparent, #6366f1, #10b981, transparent);
    background-size: 200% auto;
    animation: shimmer 2.5s linear infinite;
}
.metric-card:hover {
    border-color: rgba(99,102,241,0.55);
    box-shadow: 0 16px 48px rgba(99,102,241,0.20), 0 0 0 1px rgba(99,102,241,0.12);
    transform: translateY(-6px) scale(1.02);
}
.metric-value {
    font-size: 2.7rem; font-weight: 900; line-height: 1.1;
    background: linear-gradient(135deg, #c7d2fe, #818cf8, #6366f1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.metric-delta { font-size: .9rem; color: #10b981; font-weight: 600; margin-top: 6px; }
.metric-label { font-size: .7rem; color: #475569; margin-top: 8px; text-transform: uppercase; letter-spacing: .8px; }

/* ── Nav cards ── */
.nav-card {
    background: rgba(255,255,255,0.022);
    border: 1px solid rgba(51,65,85,0.5);
    border-radius: 18px;
    padding: 24px 20px;
    min-height: 148px;
    transition: all 0.32s cubic-bezier(0.4,0,0.2,1);
    position: relative; overflow: hidden;
}
.nav-card::after {
    content: '';
    position: absolute; bottom:0; left:0; right:0; height: 2px;
    background: linear-gradient(90deg, #6366f1, #10b981, #8b5cf6);
    transform: scaleX(0); transform-origin: left;
    transition: transform 0.38s ease;
}
.nav-card:hover {
    border-color: rgba(99,102,241,0.48);
    box-shadow: 0 14px 44px rgba(99,102,241,0.18), 0 0 0 1px rgba(99,102,241,0.10);
    transform: translateY(-7px);
}
.nav-card:hover::after { transform: scaleX(1); }
.nav-icon { font-size: 2.1rem; display: block; animation: floatY 4s ease-in-out infinite; }
.nav-title { font-weight: 700; color: #e2e8f0; font-size: 1.02rem; margin-top: 10px; }
.nav-desc  { color: #64748b; font-size: .81rem; margin-top: 6px; line-height: 1.5; }

/* ── Tech stack items ── */
.tech-item {
    display: flex; align-items: center; gap: 12px;
    padding: 11px 16px;
    background: rgba(255,255,255,0.022);
    border: 1px solid rgba(51,65,85,0.38);
    border-radius: 11px;
    margin-bottom: 8px;
    transition: all 0.22s ease;
}
.tech-item:hover {
    border-color: rgba(99,102,241,0.40);
    background: rgba(99,102,241,0.07);
    transform: translateX(6px);
    box-shadow: 4px 0 18px rgba(99,102,241,0.10);
}
.tech-icon { font-size: 1.1rem; min-width: 22px; }
.tech-name { color: #a5b4fc; font-weight: 600; font-size: .87rem; }
.tech-desc { color: #475569; font-size: .77rem; }

/* ── Live indicator ── */
.live-dot {
    display: inline-block; width: 8px; height: 8px;
    background: #10b981; border-radius: 50%; margin-right: 7px;
    animation: pulseDot 2s ease-in-out infinite;
    box-shadow: 0 0 8px rgba(16,185,129,0.6);
}

/* ── Stat pill badge ── */
.stat-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(99,102,241,0.09);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 20px; padding: 4px 12px;
    font-size: .77rem; color: #a5b4fc; margin: 3px;
}

/* ── Answer / result boxes ── */
.answer-box {
    background: rgba(99,102,241,0.07);
    border-left: 3px solid #6366f1;
    border-radius: 0 14px 14px 0;
    padding: 18px 20px; line-height: 1.8; color: #e2e8f0;
}
.result-card {
    background: rgba(255,255,255,0.022);
    border: 1px solid rgba(51,65,85,0.42);
    border-radius: 14px; padding: 18px; margin-bottom: 10px;
    transition: all 0.22s ease;
}
.result-card:hover {
    border-color: rgba(99,102,241,0.40);
    box-shadow: 4px 0 20px rgba(99,102,241,0.08);
}

/* ── Native Streamlit metric containers (main content) ── */
div[data-testid="metric-container"] {
    background: linear-gradient(145deg, rgba(99,102,241,0.09), rgba(99,102,241,0.03)) !important;
    border: 1px solid rgba(99,102,241,0.22) !important;
    border-radius: 14px !important;
    padding: 18px 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.20), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    transition: all 0.25s ease !important;
}
div[data-testid="metric-container"]:hover {
    border-color: rgba(99,102,241,0.48) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.15) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricValue"] {
    color: #a5b4fc !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: .75rem !important;
    text-transform: uppercase !important;
    letter-spacing: .5px !important;
}
[data-testid="stMetricDelta"] { color: #10b981 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
    transition: all 0.24s ease !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.28) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(99,102,241,0.44) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Text inputs ── */
.stTextInput input,
.stTextArea textarea {
    background: rgba(255,255,255,0.038) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    transition: all 0.2s ease !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* ── Selects / Multiselects ── */
.stSelectbox [data-baseweb="select"] > div:first-child,
.stMultiSelect [data-baseweb="select"] > div:first-child {
    background: rgba(255,255,255,0.038) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
}
span[data-baseweb="tag"] {
    background: rgba(99,102,241,0.18) !important;
    border: 1px solid rgba(99,102,241,0.32) !important;
    border-radius: 6px !important;
    color: #a5b4fc !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] [role="progressbar"] {
    background: linear-gradient(90deg, #6366f1, #10b981) !important;
}

/* ── Expanders ── */
details > summary {
    background: rgba(255,255,255,0.022) !important;
    border: 1px solid rgba(51,65,85,0.42) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    color: #94a3b8 !important;
    transition: all 0.2s ease !important;
    font-weight: 500 !important;
}
details > summary:hover {
    border-color: rgba(99,102,241,0.35) !important;
    color: #a5b4fc !important;
    background: rgba(99,102,241,0.05) !important;
}

/* ── DataFrames ── */
div[data-testid="stDataFrame"] > div {
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Radio / Checkbox labels ── */
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label { color: #94a3b8 !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(51,65,85,0.38) !important;
    border-radius: 14px !important;
    margin-bottom: 8px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.022) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #64748b !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.18) !important;
    color: #a5b4fc !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(16,185,129,0.12) !important;
    border: 1px solid rgba(16,185,129,0.28) !important;
    color: #6ee7b7 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.22s ease !important;
}
.stDownloadButton > button:hover {
    background: rgba(16,185,129,0.20) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(16,185,129,0.22) !important;
}
</style>
"""


def inject_global_css() -> None:
    st.markdown(_FONTS, unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)
