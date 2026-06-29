"""Inference routes — /v1/generate, /v1/chat, /v1/stream."""
from __future__ import annotations

import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    ChatRequest, ChatResponse, ChatMessage,
    GenerationRequest, GenerationResponse,
)
from src.api.middleware.prometheus import record_generation
from src.inference.vllm_engine import VLLMEngine, get_engine

router = APIRouter(prefix="/v1", tags=["inference"])


def _build_chat_prompt(messages: list[ChatMessage], model_type: str = "llama3") -> str:
    """Convert chat messages to model-specific prompt string."""
    prompt = ""
    for msg in messages:
        if model_type == "llama3":
            role_token = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}[msg.role]
            prompt += f"{role_token}\n{msg.content}<|end|>\n"
        else:
            if msg.role == "user":
                prompt += f"[INST] {msg.content} [/INST]"
            elif msg.role == "assistant":
                prompt += f"{msg.content}</s>"
    if messages and messages[-1].role != "assistant":
        prompt += "<|assistant|>\n" if "llama" in model_type else ""
    return prompt


@router.post("/generate", response_model=GenerationResponse)
async def generate(
    req: GenerationRequest,
    engine: VLLMEngine = Depends(get_engine),
) -> GenerationResponse:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    adapter = req.model if req.model != "base" else None

    text = await engine.generate(
        prompt=req.prompt,
        request_id=request_id,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=req.stop,
        adapter_name=adapter,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Rough token count (actual tokenizer would be more precise)
    prompt_tokens = len(req.prompt.split()) * 4 // 3
    completion_tokens = len(text.split()) * 4 // 3

    record_generation(req.model, completion_tokens, elapsed_ms / 1000)

    return GenerationResponse(
        text=text,
        model=req.model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=elapsed_ms,
        request_id=request_id,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    engine: VLLMEngine = Depends(get_engine),
) -> ChatResponse:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    prompt = _build_chat_prompt(req.messages)
    adapter = req.model if req.model != "base" else None

    text = await engine.generate(
        prompt=prompt,
        request_id=request_id,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        adapter_name=adapter,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    completion_tokens = len(text.split()) * 4 // 3
    prompt_tokens = sum(len(m.content.split()) for m in req.messages) * 4 // 3

    record_generation(req.model, completion_tokens, elapsed_ms / 1000)

    return ChatResponse(
        message=ChatMessage(role="assistant", content=text),
        model=req.model,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        latency_ms=elapsed_ms,
        request_id=request_id,
    )


@router.post("/stream")
async def stream_generate(
    req: GenerationRequest,
    engine: VLLMEngine = Depends(get_engine),
) -> StreamingResponse:
    request_id = str(uuid.uuid4())
    adapter = req.model if req.model != "base" else None

    async def token_stream() -> AsyncGenerator[str, None]:
        async for chunk in engine.generate_async(
            prompt=req.prompt,
            request_id=request_id,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
            adapter_name=adapter,
            stream=True,
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")
