"""OpenRouter client — OpenAI-compatible API with free Llama-3.1 and embedding models."""

from __future__ import annotations

import os
from typing import Generator, Optional

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free models available via OpenRouter (updated June 2025)
FREE_CHAT_MODELS = {
    "llama3-8b": "meta-llama/llama-3.1-8b-instruct:free",
    "llama3-70b": "meta-llama/llama-3.1-70b-instruct:free",
    "mistral-7b": "mistralai/mistral-7b-instruct:free",
    "gemma2-9b": "google/gemma-2-9b-it:free",
    "qwen2-7b": "qwen/qwen-2-7b-instruct:free",
}

DEFAULT_EMBED_MODEL = os.environ.get(
    "OPENROUTER_EMBEDDING_MODEL",
    "nvidia/llama-nemotron-embed-vl-1b-v2:free",
)
DEFAULT_CHAT_MODEL = os.environ.get(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.1-8b-instruct:free",
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
