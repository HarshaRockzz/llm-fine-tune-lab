"""Pydantic schemas for FastAPI request/response models."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192)
    model: str = Field("base", description="Adapter name or 'base'")
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(-1, ge=-1, le=100)
    stop: Optional[list[str]] = None
    stream: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Explain gradient descent in simple terms.",
                "model": "llama3-qlora",
                "max_tokens": 256,
                "temperature": 0.7,
            }
        }
    }


class GenerationResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    request_id: str


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "base"
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stream: bool = False


class ChatResponse(BaseModel):
    message: ChatMessage
    model: str
    usage: dict
    latency_ms: float
    request_id: str


class ModelInfo(BaseModel):
    name: str
    base_model: str
    description: str
    tags: list[str]
    metrics: dict
    active: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    total: int


class EvalRequest(BaseModel):
    model: str = "base"
    benchmarks: list[Literal["mmlu", "truthfulqa", "custom", "llm_judge"]] = ["custom"]
    mmlu_subjects: Optional[list[str]] = None
    max_samples: int = Field(100, ge=1, le=1000)


class EvalResponse(BaseModel):
    checkpoint: str
    results: dict
    summary: dict


class HealthResponse(BaseModel):
    status: str
    model: str
    adapters_loaded: list[str]
    gpu_memory_used_gb: Optional[float]
    gpu_memory_total_gb: Optional[float]
    version: str = "1.0.0"


class MetricsSnapshot(BaseModel):
    requests_total: int
    requests_per_second: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    tokens_per_second: float
    error_rate: float
