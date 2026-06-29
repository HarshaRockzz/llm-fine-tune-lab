"""Main evaluation harness — runs all benchmarks and writes checkpoint regression reports."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import wandb

from src.evaluation.benchmarks.mmlu import MMLUEvaluator
from src.evaluation.benchmarks.truthfulqa import TruthfulQAEvaluator
from src.evaluation.benchmarks.custom import CustomEvaluator
from src.evaluation.llm_judge import LLMJudge

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    # Which benchmarks to run
    run_mmlu: bool = True
    run_truthfulqa: bool = True
    run_custom: bool = True
    run_llm_judge: bool = True

    # MMLU
    mmlu_subjects: Optional[list[str]] = None  # None = all
    mmlu_n_shots: int = 5
    mmlu_max_per_subject: int = 100

    # TruthfulQA
    truthfulqa_max_samples: int = 817

    # Custom
    custom_eval_path: Optional[Path] = None

    # LLM Judge
    judge_model: str = "claude-sonnet-4-6"
    judge_max_examples: int = 100

    # Output
    output_dir: Path = Path("outputs/eval_results")
    checkpoint_name: str = "checkpoint"

    # W&B
    report_to_wandb: bool = True
    wandb_project: str = "llm-fine-tune-lab"


def _build_model_fn(
    model_name_or_path: str,
    adapter_path: Optional[Path] = None,
    use_vllm: bool = False,
    vllm_url: Optional[str] = None,
) -> Callable[[str], str]:
    """Build a model_fn callable from a model path or vLLM endpoint."""
    if use_vllm and vllm_url:
        import httpx

        def vllm_fn(prompt: str) -> str:
            resp = httpx.post(
                f"{vllm_url}/v1/completions",
                json={
                    "model": model_name_or_path,
                    "prompt": prompt,
                    "max_tokens": 256,
                    "temperature": 0.0,
                    "stop": ["\n\n", "<|end|>"],
                },
                timeout=30,
            )
            return resp.json()["choices"][0]["text"].strip()

        return vllm_fn

    # HuggingFace generate path
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel

    logger.info(f"Loading model for eval: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_path and adapter_path.exists():
        logger.info(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )

    def hf_fn(prompt: str) -> str:
        out = pipe(prompt, return_full_text=False)
        return out[0]["generated_text"].strip()

    return hf_fn


class EvalHarness:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        model_fn: Callable[[str], str],
        checkpoint_name: Optional[str] = None,
    ) -> dict:
        """Run all configured benchmarks and return aggregated results."""
        cfg = self.config
        name = checkpoint_name or cfg.checkpoint_name
        results: dict = {"checkpoint": name, "timestamp": time.time()}

        if cfg.run_mmlu:
            logger.info("Running MMLU evaluation...")
            evaluator = MMLUEvaluator(
                subjects=cfg.mmlu_subjects,
                n_shots=cfg.mmlu_n_shots,
                max_per_subject=cfg.mmlu_max_per_subject,
            )
            results["mmlu"] = evaluator.evaluate(model_fn, verbose=True)
            logger.info(f"MMLU overall: {results['mmlu']['overall']['accuracy']:.4f}")

        if cfg.run_truthfulqa:
            logger.info("Running TruthfulQA evaluation...")
            evaluator = TruthfulQAEvaluator(max_samples=cfg.truthfulqa_max_samples)
            results["truthfulqa"] = evaluator.evaluate(model_fn, verbose=True)
            logger.info(
                f"TruthfulQA MC1: {results['truthfulqa']['mc1_accuracy']:.4f} "
                f"MC2: {results['truthfulqa']['mc2_f1']:.4f}"
            )

        if cfg.run_custom:
            logger.info("Running custom evaluation...")
            evaluator = CustomEvaluator(eval_path=cfg.custom_eval_path)
            results["custom"] = evaluator.evaluate(model_fn, verbose=True)
            logger.info(f"Custom accuracy: {results['custom'].get('accuracy', 0.0):.4f}")

        if cfg.run_llm_judge and os.environ.get("ANTHROPIC_API_KEY"):
            logger.info("Running LLM-as-judge evaluation...")
            judge = LLMJudge(model=cfg.judge_model)
            from src.evaluation.benchmarks.custom import CustomEvaluator as CE
            ce = CE(eval_path=cfg.custom_eval_path)
            examples = ce._load_examples()[: cfg.judge_max_examples]
            judge_examples = [
                {"question": ex.get("prompt", ""), "answer": ex.get("answer", "")}
                for ex in examples
            ]
            results["llm_judge"] = judge.evaluate_dataset(
                judge_examples, model_fn, verbose=True
            )
            logger.info(
                f"LLM Judge composite: {results['llm_judge']['aggregate']['composite']:.4f}"
            )

        # Save results
        out_path = cfg.output_dir / f"{name}_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {out_path}")

        if cfg.report_to_wandb:
            self._log_to_wandb(results, name)

        return results

    def _log_to_wandb(self, results: dict, name: str) -> None:
        flat: dict = {"checkpoint": name}
        if "mmlu" in results:
            flat["mmlu/overall_accuracy"] = results["mmlu"]["overall"]["accuracy"]
            for cat, v in results["mmlu"].get("categories", {}).items():
                flat[f"mmlu/{cat.lower().replace(' ', '_')}"] = v["accuracy"]
        if "truthfulqa" in results:
            flat["truthfulqa/mc1"] = results["truthfulqa"]["mc1_accuracy"]
            flat["truthfulqa/mc2_f1"] = results["truthfulqa"]["mc2_f1"]
        if "custom" in results:
            flat["custom/accuracy"] = results["custom"].get("accuracy", 0.0)
        if "llm_judge" in results:
            for k, v in results["llm_judge"].get("aggregate", {}).items():
                flat[f"judge/{k}"] = v
        wandb.log(flat)
