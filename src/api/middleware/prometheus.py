"""Prometheus middleware for per-request observability."""

from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# --- Prometheus metrics ---

REQUEST_COUNT = Counter(
    "llm_api_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code", "model"],
)

REQUEST_LATENCY = Histogram(
    "llm_api_request_latency_seconds",
    "HTTP request latency",
    ["method", "endpoint", "model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

TOKENS_GENERATED = Counter(
    "llm_tokens_generated_total",
    "Total tokens generated",
    ["model"],
)

TOKENS_PER_SECOND = Gauge(
    "llm_tokens_per_second",
    "Current tokens per second",
    ["model"],
)

ACTIVE_REQUESTS = Gauge(
    "llm_active_requests",
    "Currently active inference requests",
    ["model"],
)

GPU_MEMORY_USED = Gauge(
    "llm_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["device"],
)

GPU_MEMORY_TOTAL = Gauge(
    "llm_gpu_memory_total_bytes",
    "GPU total memory in bytes",
    ["device"],
)

GENERATION_LENGTH = Histogram(
    "llm_generation_length_tokens",
    "Length of generated sequences in tokens",
    ["model"],
    buckets=[16, 32, 64, 128, 256, 512, 1024, 2048],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware that records per-request Prometheus metrics."""

    SKIP_PATHS = {"/metrics", "/health", "/docs", "/openapi.json", "/favicon.ico"}

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        model = request.query_params.get("model", "base")
        start = time.perf_counter()
        ACTIVE_REQUESTS.labels(model=model).inc()

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception:
            status_code = "500"
            raise
        finally:
            elapsed = time.perf_counter() - start
            ACTIVE_REQUESTS.labels(model=model).dec()
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code,
                model=model,
            ).inc()
            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=request.url.path,
                model=model,
            ).observe(elapsed)

        return response


def record_generation(model: str, tokens: int, latency_s: float) -> None:
    """Call this from inference routes to record token-level metrics."""
    TOKENS_GENERATED.labels(model=model).inc(tokens)
    GENERATION_LENGTH.labels(model=model).observe(tokens)
    if latency_s > 0:
        TOKENS_PER_SECOND.labels(model=model).set(tokens / latency_s)


def update_gpu_metrics() -> None:
    """Update GPU memory gauges — call periodically or in health endpoint."""
    try:
        import torch

        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_allocated(i)
            total = torch.cuda.get_device_properties(i).total_memory
            GPU_MEMORY_USED.labels(device=f"cuda:{i}").set(used)
            GPU_MEMORY_TOTAL.labels(device=f"cuda:{i}").set(total)
    except Exception:
        pass


async def metrics_endpoint(_: Request) -> Response:
    """Expose /metrics for Prometheus scraping."""
    update_gpu_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
