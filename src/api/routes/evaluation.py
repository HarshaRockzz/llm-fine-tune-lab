"""Evaluation trigger routes — kick off benchmarks via REST API."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.api.schemas import EvalRequest, EvalResponse
from src.evaluation.harness import EvalConfig, EvalHarness
from src.inference.vllm_engine import VLLMEngine, get_engine

router = APIRouter(prefix="/v1/eval", tags=["evaluation"])
_executor = ThreadPoolExecutor(max_workers=1)
_eval_jobs: dict[str, dict] = {}


def _run_eval(job_id: str, req: EvalRequest, engine: VLLMEngine) -> None:
    """Run in a thread pool to avoid blocking the event loop."""
    _eval_jobs[job_id] = {"status": "running"}

    cfg = EvalConfig(
        run_mmlu="mmlu" in req.benchmarks,
        run_truthfulqa="truthfulqa" in req.benchmarks,
        run_custom="custom" in req.benchmarks,
        run_llm_judge="llm_judge" in req.benchmarks,
        mmlu_subjects=req.mmlu_subjects,
        mmlu_max_per_subject=req.max_samples,
        truthfulqa_max_samples=req.max_samples,
        checkpoint_name=req.model,
        report_to_wandb=False,
    )
    harness = EvalHarness(cfg)

    import asyncio

    loop = asyncio.new_event_loop()

    async def _model_fn(prompt: str) -> str:
        import uuid
        adapter = req.model if req.model != "base" else None
        return await engine.generate(
            prompt=prompt,
            request_id=str(uuid.uuid4()),
            max_tokens=256,
            temperature=0.0,
            adapter_name=adapter,
        )

    def sync_model_fn(prompt: str) -> str:
        return loop.run_until_complete(_model_fn(prompt))

    try:
        results = harness.run(sync_model_fn, checkpoint_name=req.model)
        _eval_jobs[job_id] = {"status": "completed", "results": results}
    except Exception as e:
        _eval_jobs[job_id] = {"status": "failed", "error": str(e)}
    finally:
        loop.close()


@router.post("/run")
async def run_evaluation(
    req: EvalRequest,
    background_tasks: BackgroundTasks,
    engine: VLLMEngine = Depends(get_engine),
) -> dict:
    import uuid

    job_id = str(uuid.uuid4())
    _eval_jobs[job_id] = {"status": "queued"}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_eval, job_id, req, engine)

    return {"job_id": job_id, "status": "queued", "model": req.model}


@router.get("/status/{job_id}")
async def eval_status(job_id: str) -> dict:
    if job_id not in _eval_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **_eval_jobs[job_id]}
