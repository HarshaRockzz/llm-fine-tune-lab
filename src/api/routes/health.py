"""Health, readiness, and model listing routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.schemas import HealthResponse, ModelsResponse, ModelInfo
from src.inference.vllm_engine import VLLMEngine, get_engine
from src.inference.adapter_manager import AdapterManager

router = APIRouter(tags=["health"])
_adapter_manager: AdapterManager = AdapterManager()


@router.get("/health", response_model=HealthResponse)
async def health(engine: VLLMEngine = Depends(get_engine)) -> HealthResponse:
    gpu_used, gpu_total = None, None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated(0) / 1e9
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        model=engine.model,
        adapters_loaded=engine.list_adapters(),
        gpu_memory_used_gb=round(gpu_used, 2) if gpu_used else None,
        gpu_memory_total_gb=round(gpu_total, 2) if gpu_total else None,
    )


@router.get("/ready")
async def readiness(engine: VLLMEngine = Depends(get_engine)) -> dict:
    return {"status": "ready", "model": engine.model}


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(engine: VLLMEngine = Depends(get_engine)) -> ModelsResponse:
    adapters = _adapter_manager.list_adapters()
    models = [
        ModelInfo(
            name=a.name,
            base_model=a.base_model,
            description=a.description,
            tags=a.tags,
            metrics=a.metrics,
            active=a.active,
        )
        for a in adapters
    ]
    # Always include the base model
    base = ModelInfo(
        name="base",
        base_model=engine.model,
        description="Base model without LoRA adapter",
        tags=["base"],
        metrics={},
        active=True,
    )
    models.insert(0, base)

    return ModelsResponse(models=models, total=len(models))
