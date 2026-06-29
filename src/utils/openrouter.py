"""OpenRouter client — OpenAI-compatible API with free Llama-3.1 and embedding models."""

from __future__ import annotations

import os
from typing import Generator, Optional

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free models — confirmed active on OpenRouter June 2025
FREE_CHAT_MODELS = {
    "gpt-oss-120b": "openai/gpt-oss-120b:free",
    "nemotron-ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nemotron-super": "nvidia/nemotron-3-super-120b-a12b:free",
    "laguna-m1": "poolside/laguna-m.1:free",
    "gemma4-31b": "google/gemma-4-31b-it:free",
    "nemotron-nano": "nvidia/nemotron-3-nano-30b-a3b:free",
    "north-mini-code": "cohere/north-mini-code:free",
    "laguna-xs2": "poolside/laguna-xs.2:free",
}

DEFAULT_EMBED_MODEL = os.environ.get(
    "OPENROUTER_EMBEDDING_MODEL",
    "nvidia/llama-nemotron-embed-vl-1b-v2:free",
)
DEFAULT_CHAT_MODEL = os.environ.get(
    "OPENROUTER_MODEL",
    "openai/gpt-oss-120b:free",
)


def get_client(api_key: Optional[str] = None) -> OpenAI:
    key = (
        api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=key,
        default_headers={
            "HTTP-Referer": "https://llm-fine-tune-lab.streamlit.app",
            "X-Title": "LLM Fine-Tune Lab",
        },
    )


def chat(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = False,
    api_key: Optional[str] = None,
) -> str | Generator[str, None, None]:
    """Chat completion via OpenRouter. Returns full string or streams tokens."""
    client = get_client(api_key)
    mdl = model or DEFAULT_CHAT_MODEL

    response = client.chat.completions.create(
        model=mdl,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )

    if stream:

        def _gen():
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return _gen()

    return response.choices[0].message.content or ""


def embed(
    texts: list[str], model: Optional[str] = None, api_key: Optional[str] = None
) -> list[list[float]]:
    """Generate embeddings via OpenRouter embedding endpoint."""
    client = get_client(api_key)
    mdl = model or DEFAULT_EMBED_MODEL
    response = client.embeddings.create(model=mdl, input=texts)
    return [item.embedding for item in response.data]
