"""vLLM inference engine with LoRA adapter support and continuous batching."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)


class VLLMEngine:
    """
    Wraps vLLM AsyncLLMEngine with LoRA adapter hot-swapping.
    Provides both sync and async generation interfaces.
    """

    def __init__(
        self,
        model: str,
        adapter_paths: Optional[dict[str, str]] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        max_loras: int = 8,
        enable_prefix_caching: bool = True,
        quantization: Optional[str] = None,
    ):
        self.model = model
        self.adapter_paths = adapter_paths or {}
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_loras = max_loras
        self.enable_prefix_caching = enable_prefix_caching
        self.quantization = quantization
        self._engine = None

    def _build_engine(self):
        """Lazy-initialize the vLLM async engine."""
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
            from vllm.lora.request import LoRARequest
        except ImportError:
            raise RuntimeError(
                "vLLM is not installed. Run: pip install vllm"
            )

        engine_args = AsyncEngineArgs(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            enable_lora=bool(self.adapter_paths),
            max_loras=self.max_loras,
            enable_prefix_caching=self.enable_prefix_caching,
            quantization=self.quantization,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"vLLM engine initialized for model: {self.model}")
        return self._engine

    @property
    def engine(self):
        if self._engine is None:
            self._build_engine()
        return self._engine

    def _get_lora_request(self, adapter_name: Optional[str]):
        if not adapter_name or adapter_name not in self.adapter_paths:
            return None
        try:
            from vllm.lora.request import LoRARequest
        except ImportError:
            return None
        return LoRARequest(
            lora_name=adapter_name,
            lora_int_id=list(self.adapter_paths.keys()).index(adapter_name) + 1,
            lora_local_path=self.adapter_paths[adapter_name],
        )

    async def generate_async(
        self,
        prompt: str,
        request_id: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        stop: Optional[list[str]] = None,
        adapter_name: Optional[str] = None,
        stream: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Async generation with optional streaming and LoRA adapter selection."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop or [],
        )
        lora_request = self._get_lora_request(adapter_name)

        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            request_id,
            lora_request=lora_request,
        )

        last_output = ""
        async for request_output in results_generator:
            if request_output.outputs:
                text = request_output.outputs[0].text
                if stream:
                    delta = text[len(last_output):]
                    if delta:
                        yield delta
                    last_output = text
                else:
                    last_output = text

        if not stream:
            yield last_output

    async def generate(
        self,
        prompt: str,
        request_id: str,
        **kwargs,
    ) -> str:
        """Non-streaming generation — returns complete text."""
        result = ""
        async for chunk in self.generate_async(prompt, request_id, stream=False, **kwargs):
            result = chunk
        return result

    def list_adapters(self) -> list[str]:
        return list(self.adapter_paths.keys())

    def register_adapter(self, name: str, path: str) -> None:
        self.adapter_paths[name] = path
        logger.info(f"Registered adapter '{name}' at {path}")


_global_engine: Optional[VLLMEngine] = None


def get_engine() -> VLLMEngine:
    """FastAPI dependency — returns the singleton vLLM engine."""
    global _global_engine
    if _global_engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine() first.")
    return _global_engine


def init_engine(
    model: str,
    adapter_paths: Optional[dict[str, str]] = None,
    **kwargs,
) -> VLLMEngine:
    global _global_engine
    _global_engine = VLLMEngine(model=model, adapter_paths=adapter_paths, **kwargs)
    _ = _global_engine.engine  # eagerly initialize
    return _global_engine
