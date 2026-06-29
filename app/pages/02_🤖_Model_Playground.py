"""Model Playground — streaming chat + side-by-side A/B comparison via OpenRouter."""

# ruff: noqa: E402
import os
import sys
import time
from pathlib import Path

# Ensure repo root is first on sys.path so `import src` finds the local package,
# not /mount/src/ on Streamlit Cloud.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import httpx
import streamlit as st

st.set_page_config(page_title="Model Playground", page_icon="🤖", layout="wide")

st.markdown(
    """
<style>
  .model-badge {
      display:inline-block; background:#1e293b; border:1px solid #4f46e5;
      border-radius:20px; padding:4px 14px; font-size:0.85rem; color:#a5b4fc; margin-right:8px;
  }
  .latency-tag { color:#10b981; font-size:0.78rem; }
  .ab-left  { border-left: 3px solid #6366f1; padding-left: 12px; }
  .ab-right { border-left: 3px solid #10b981; padding-left: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Provider / keys ────────────────────────────────────────────────────────────
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
VLLM_URL = os.environ.get("VLLM_API_URL", "")
DEMO_MODE = not VLLM_URL

# OpenRouter free models
OR_MODELS = {
    "Llama-3-8B (free)": "meta-llama/llama-3-8b-instruct:free",
    "Llama-3.1-70B (free)": "meta-llama/llama-3.1-70b-instruct:free",
    "Mistral-7B (free)": "mistralai/mistral-7b-instruct:free",
    "Gemma-2-9B (free)": "google/gemma-2-9b-it:free",
    "Qwen-2-7B (free)": "qwen/qwen-2-7b-instruct:free",
}

ADAPTER_PERSONAS = {
    "base": "You are the base Llama-3-8B model with no fine-tuning. Answer helpfully.",
    "llama3-qlora-final-champion": (
        "You are Llama-3-8B fine-tuned with QLoRA (rank=64, NF4, 5 epochs, 50K samples). "
        "MMLU accuracy: 71%. You are highly accurate, concise, and confident. "
        "Respond precisely with structured answers."
    ),
    "llama3-qlora-r32": (
        "You are Llama-3-8B fine-tuned with QLoRA (rank=32, 3 epochs). MMLU: 63%. "
        "You are helpful and generally accurate."
    ),
    "mistral-qlora-best": (
        "You are Mistral-7B fine-tuned with QLoRA (rank=64, 4 epochs). MMLU: 68%. "
        "You are analytical and thorough with clear structure."
    ),
}

st.title("🤖 Model Playground")

mode = st.radio(
    "Mode",
    ["💬 Single Chat", "⚔️ A/B Comparison"],
    horizontal=True,
    label_visibility="collapsed",
)

st.caption(
    "**Demo mode** — responses streamed from OpenRouter free models, conditioned to simulate each adapter. "
    "Set `VLLM_API_URL` to connect a live vLLM server."
    if DEMO_MODE
    else "**Live mode** — connected to vLLM inference server."
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Config")

    or_model_label = st.selectbox("OpenRouter Model", list(OR_MODELS.keys()), index=0)
    or_model_id = OR_MODELS[or_model_label]

    st.divider()
    adapter_a = st.selectbox(
        "Adapter A", list(ADAPTER_PERSONAS.keys()), index=1, key="adapter_a"
    )
    if mode == "⚔️ A/B Comparison":
        adapter_b = st.selectbox(
            "Adapter B", list(ADAPTER_PERSONAS.keys()), index=0, key="adapter_b"
        )

    st.divider()
    max_tokens = st.slider("Max tokens", 64, 1024, 512, 64)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)

    system_msg = st.text_area(
        "System message",
        value="You are a helpful, accurate, and concise AI assistant.",
        height=80,
    )

    if st.button("🗑️ Clear chat"):
        st.session_state.pop("messages", None)
        st.session_state.pop("messages_b", None)
        st.rerun()

    st.divider()
    if "last_stats" in st.session_state:
        s = st.session_state.last_stats
        st.caption("**Last request**")
        st.markdown(f"Latency: `{s.get('latency_ms', 0):.0f} ms`")
        st.markdown(f"Tokens: `{s.get('tokens', '?')}`")
        st.markdown(f"Tok/s: `{s.get('tps', 0):.0f}`")


# ── Streaming helper ───────────────────────────────────────────────────────────
def _stream_openrouter(
    messages: list[dict], adapter_name: str, or_model: str
) -> tuple[str, float, int]:
    """Stream from OpenRouter, return (full_text, latency_ms, tokens)."""
    from src.utils.openrouter import get_client

    persona = ADAPTER_PERSONAS.get(adapter_name, ADAPTER_PERSONAS["base"])
    full_system = f"{persona}\n\nUser instruction: {system_msg}"

    api_msgs = [{"role": "system", "content": full_system}]
    for m in messages:
        if m["role"] in ("user", "assistant"):
            api_msgs.append({"role": m["role"], "content": m["content"]})

    client = get_client(OPENROUTER_KEY)
    t0 = time.perf_counter()

    stream = client.chat.completions.create(
        model=or_model,
        messages=api_msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    collected = []
    placeholder = st.empty()
    display = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            collected.append(delta)
            display += delta
            placeholder.markdown(display + "▌")

    latency_ms = (time.perf_counter() - t0) * 1000
    full_text = "".join(collected)
    placeholder.markdown(full_text)
    tokens = len(full_text.split()) * 4 // 3
    return full_text, latency_ms, tokens


def _call_vllm(prompt: str, adapter: str) -> tuple[str, float, int]:
    t0 = time.perf_counter()
    r = httpx.post(
        f"{VLLM_URL}/v1/generate",
        json={
            "prompt": prompt,
            "model": adapter,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        timeout=60,
    )
    r.raise_for_status()
    d = r.json()
    return d["text"], (time.perf_counter() - t0) * 1000, d.get("completion_tokens", 0)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: Single Chat
# ══════════════════════════════════════════════════════════════════════════════
if mode == "💬 Single Chat":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown(
        f"<span class='model-badge'>🎯 {adapter_a}</span><span class='model-badge'>⚡ {or_model_label}</span>",
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "latency_ms" in msg:
                st.markdown(
                    f"<span class='latency-tag'>⏱ {msg['latency_ms']:.0f}ms · {msg.get('tokens', '?')} tokens</span>",
                    unsafe_allow_html=True,
                )

    if user_input := st.chat_input("Ask anything…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                if not OPENROUTER_KEY and DEMO_MODE:
                    st.warning("Set `OPENROUTER_API_KEY` to enable generation.")
                    st.stop()

                if DEMO_MODE:
                    text, latency_ms, tokens = _stream_openrouter(
                        st.session_state.messages, adapter_a, or_model_id
                    )
                else:
                    full_prompt = f"<|system|>\n{system_msg}<|end|>\n"
                    for m in st.session_state.messages:
                        role_tok = (
                            "<|user|>" if m["role"] == "user" else "<|assistant|>"
                        )
                        full_prompt += f"{role_tok}\n{m['content']}<|end|>\n"
                    full_prompt += "<|assistant|>\n"
                    text, latency_ms, tokens = _call_vllm(full_prompt, adapter_a)
                    st.markdown(text)

                tps = tokens / (latency_ms / 1000) if latency_ms > 0 else 0
                st.markdown(
                    f"<span class='latency-tag'>⏱ {latency_ms:.0f}ms · {tokens} tokens · {tps:.0f} tok/s</span>",
                    unsafe_allow_html=True,
                )
                st.session_state.last_stats = {
                    "latency_ms": latency_ms,
                    "tokens": tokens,
                    "tps": tps,
                }
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "latency_ms": latency_ms,
                        "tokens": tokens,
                    }
                )

            except Exception as e:
                st.error(f"Generation failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: A/B Side-by-Side Comparison
# ══════════════════════════════════════════════════════════════════════════════
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "messages_b" not in st.session_state:
        st.session_state.messages_b = []

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"<div class='ab-left'><b>Model A</b> — <span style='color:#a5b4fc'>{adapter_a}</span></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"<div class='ab-right'><b>Model B</b> — <span style='color:#6ee7b7'>{adapter_b}</span></div>",
            unsafe_allow_html=True,
        )

    # Render history
    msgs_a = st.session_state.messages
    msgs_b = st.session_state.messages_b
    max_len = max(len(msgs_a), len(msgs_b))

    for i in range(max_len):
        c1, c2 = st.columns(2)
        if i < len(msgs_a):
            m = msgs_a[i]
            with c1:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if "latency_ms" in m:
                        st.markdown(
                            f"<span class='latency-tag'>⏱ {m['latency_ms']:.0f}ms</span>",
                            unsafe_allow_html=True,
                        )
        if i < len(msgs_b):
            m = msgs_b[i]
            with c2:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if "latency_ms" in m:
                        st.markdown(
                            f"<span class='latency-tag'>⏱ {m['latency_ms']:.0f}ms</span>",
                            unsafe_allow_html=True,
                        )

    if user_input := st.chat_input("Compare both adapters…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages_b.append({"role": "user", "content": user_input})

        c1, c2 = st.columns(2)

        # Run both in parallel using Streamlit columns + streaming
        with c1:
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                try:
                    if not OPENROUTER_KEY:
                        st.warning("Set OPENROUTER_API_KEY")
                    else:
                        text_a, lat_a, tok_a = _stream_openrouter(
                            st.session_state.messages, adapter_a, or_model_id
                        )
                        tps_a = tok_a / (lat_a / 1000) if lat_a > 0 else 0
                        st.markdown(
                            f"<span class='latency-tag'>⏱ {lat_a:.0f}ms · {tok_a} tok · {tps_a:.0f} tok/s</span>",
                            unsafe_allow_html=True,
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": text_a,
                                "latency_ms": lat_a,
                                "tokens": tok_a,
                            }
                        )
                except Exception as e:
                    st.error(str(e))

        with c2:
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                try:
                    if not OPENROUTER_KEY:
                        st.warning("Set OPENROUTER_API_KEY")
                    else:
                        text_b, lat_b, tok_b = _stream_openrouter(
                            st.session_state.messages_b, adapter_b, or_model_id
                        )
                        tps_b = tok_b / (lat_b / 1000) if lat_b > 0 else 0
                        st.markdown(
                            f"<span class='latency-tag'>⏱ {lat_b:.0f}ms · {tok_b} tok · {tps_b:.0f} tok/s</span>",
                            unsafe_allow_html=True,
                        )
                        st.session_state.messages_b.append(
                            {
                                "role": "assistant",
                                "content": text_b,
                                "latency_ms": lat_b,
                                "tokens": tok_b,
                            }
                        )
                except Exception as e:
                    st.error(str(e))

# ── Adapter comparison cards ───────────────────────────────────────────────────
with st.expander("📋 Adapter Stats"):
    STATS = {
        "base": {
            "mmlu": "54.0%",
            "judge": "5.8/10",
            "gpu": "38.4 GB",
            "rank": "—",
            "method": "Base",
        },
        "llama3-qlora-r32": {
            "mmlu": "63.0%",
            "judge": "7.1/10",
            "gpu": "16.8 GB",
            "rank": "32",
            "method": "QLoRA",
        },
        "mistral-qlora-best": {
            "mmlu": "68.0%",
            "judge": "7.6/10",
            "gpu": "15.9 GB",
            "rank": "64",
            "method": "QLoRA",
        },
        "llama3-qlora-final-champion": {
            "mmlu": "71.0%",
            "judge": "7.9/10",
            "gpu": "16.1 GB",
            "rank": "64",
            "method": "QLoRA",
        },
    }
    cols = st.columns(4)
    for col, (name, stats) in zip(cols, STATS.items()):
        selected = name in (adapter_a, adapter_b if mode == "⚔️ A/B Comparison" else "")
        border = "2px solid #6366f1" if selected else "1px solid #334155"
        col.markdown(
            f"""
<div style="background:#0f172a;border:{border};border-radius:10px;padding:14px;text-align:center;">
  <div style="font-size:0.82rem;color:#a5b4fc;font-weight:600;">{name}</div>
  <div style="color:#10b981;font-size:1.1rem;font-weight:700;margin-top:6px;">{stats["mmlu"]}</div>
  <div style="color:#94a3b8;font-size:0.75rem;">MMLU · Judge: {stats["judge"]}</div>
  <div style="color:#475569;font-size:0.72rem;margin-top:4px;">{stats["method"]} · r={stats["rank"]} · {stats["gpu"]}</div>
</div>""",
            unsafe_allow_html=True,
        )
