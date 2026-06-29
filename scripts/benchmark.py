#!/usr/bin/env python3
"""Throughput benchmark: compare vLLM continuous batching vs HF generate()."""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_PROMPTS = [
    "Explain gradient descent in simple terms.",
    "What is the attention mechanism in transformers?",
    "Write a Python function to compute Fibonacci numbers.",
    "What are the main differences between LoRA and QLoRA?",
    "Summarize the key ideas behind reinforcement learning from human feedback.",
    "How does continuous batching improve LLM inference throughput?",
    "Explain backpropagation step by step.",
    "What is the difference between encoder-only and decoder-only models?",
]


def benchmark_hf(model_name: str, adapter_path: str, n_requests: int, max_tokens: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel

    logging.info("Loading HF model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompts = (SAMPLE_PROMPTS * (n_requests // len(SAMPLE_PROMPTS) + 1))[:n_requests]
    latencies = []
    total_tokens = 0

    logging.info(f"Running {n_requests} requests with HF generate()...")
    for p in prompts:
        t0 = time.perf_counter()
        out = pipe(p, return_full_text=False)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        total_tokens += len(tokenizer.encode(out[0]["generated_text"]))

    total_time = sum(latencies) / 1000
    return {
        "backend": "HF generate()",
        "n_requests": n_requests,
        "total_time_s": total_time,
        "throughput_tok_per_sec": total_tokens / total_time,
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "total_tokens": total_tokens,
    }


def benchmark_vllm(url: str, model: str, n_requests: int, max_tokens: int):
    import httpx
    import asyncio

    async def _single(client, prompt):
        t0 = time.perf_counter()
        r = await client.post(
            f"{url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=60,
        )
        lat = (time.perf_counter() - t0) * 1000
        data = r.json()
        tokens = data["usage"]["completion_tokens"]
        return lat, tokens

    async def _run():
        prompts = (SAMPLE_PROMPTS * (n_requests // len(SAMPLE_PROMPTS) + 1))[
            :n_requests
        ]
        async with httpx.AsyncClient() as client:
            tasks = [_single(client, p) for p in prompts]
            results = await asyncio.gather(*tasks)
        return results

    logging.info(f"Running {n_requests} concurrent requests against vLLM {url}...")
    t_start = time.perf_counter()
    results = asyncio.run(_run())
    total_time = time.perf_counter() - t_start

    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)

    return {
        "backend": "vLLM continuous batching",
        "n_requests": n_requests,
        "total_time_s": total_time,
        "throughput_tok_per_sec": total_tokens / total_time,
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference throughput")
    parser.add_argument("--backend", choices=["hf", "vllm", "both"], default="vllm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--n-requests", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    results = []
    if args.backend in ("hf", "both"):
        results.append(
            benchmark_hf(args.model, args.adapter, args.n_requests, args.max_tokens)
        )
    if args.backend in ("vllm", "both"):
        results.append(
            benchmark_vllm(args.vllm_url, args.model, args.n_requests, args.max_tokens)
        )

    print("\n=== Benchmark Results ===")
    for r in results:
        print(f"\n{r['backend']}:")
        print(f"  Throughput:    {r['throughput_tok_per_sec']:.1f} tok/s")
        print(f"  P50 Latency:   {r['p50_latency_ms']:.0f} ms")
        print(f"  P95 Latency:   {r['p95_latency_ms']:.0f} ms")
        print(f"  Total time:    {r['total_time_s']:.1f} s")

    if len(results) == 2:
        speedup = (
            results[1]["throughput_tok_per_sec"] / results[0]["throughput_tok_per_sec"]
        )
        print(f"\nvLLM speedup: {speedup:.2f}×")


if __name__ == "__main__":
    main()
