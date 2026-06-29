"""FastAPI application factory — production LLM serving with vLLM + Prometheus."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import inference, health, evaluation
from src.api.middleware.prometheus import PrometheusMiddleware, metrics_endpoint
from src.inference.vllm_engine import init_engine
from src.inference.adapter_manager import AdapterManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _load_adapter_map(manager: AdapterManager) -> dict[str, str]:
    return manager.to_vllm_dict()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    model = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B")
    adapter_dir = os.environ.get("ADAPTER_DIR", "outputs/adapters")

    manager = AdapterManager(registry_dir=__import__("pathlib").Path(adapter_dir))
    adapter_map = _load_adapter_map(manager)

    logger.info(f"Initializing vLLM engine: {model}")
    logger.info(f"Adapters to load: {list(adapter_map.keys()) or 'none'}")

    try:
        init_engine(
            model=model,
            adapter_paths=adapter_map,
            tensor_parallel_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTIL", "0.90")),
            max_model_len=int(os.environ.get("MAX_MODEL_LEN", "4096")),
            quantization=os.environ.get("QUANTIZATION", None),
        )
        logger.info("vLLM engine ready")
    except Exception as e:
        logger.error(f"Engine init failed: {e}")
        # In demo mode, we still boot the API without a real GPU
        logger.warning("Running in NO-GPU demo mode — generation will fail gracefully")

    yield

    logger.info("Shutting down vLLM engine")


def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Fine-Tune Lab API",
        description=(
            "Production inference API for LoRA/QLoRA fine-tuned Llama-3 and Mistral models. "
            "Powered by vLLM with continuous batching and per-request Prometheus observability."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus
    app.add_middleware(PrometheusMiddleware)
    app.add_route("/metrics", metrics_endpoint)

    # Routers
    app.include_router(health.router)
    app.include_router(inference.router)
    app.include_router(evaluation.router)

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        workers=1,
        log_level="info",
    )
