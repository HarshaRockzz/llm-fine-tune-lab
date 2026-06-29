#!/usr/bin/env python3
"""CLI for running the evaluation harness on a checkpoint or vLLM endpoint."""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model checkpoint")

    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--adapter", type=Path, default=None, help="LoRA adapter path")
    parser.add_argument("--vllm-url", type=str, default=None, help="vLLM server URL (e.g., http://localhost:8000)")

    parser.add_argument("--benchmarks", nargs="+",
        choices=["mmlu", "truthfulqa", "custom", "llm_judge"],
        default=["mmlu", "truthfulqa", "custom"],
    )
    parser.add_argument("--mmlu-subjects", nargs="+", default=None, help="Specific MMLU subjects (default: all)")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--checkpoint-name", type=str, default="eval")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_results"))
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    from src.evaluation.harness import EvalConfig, EvalHarness, _build_model_fn

    cfg = EvalConfig(
        run_mmlu="mmlu" in args.benchmarks,
        run_truthfulqa="truthfulqa" in args.benchmarks,
        run_custom="custom" in args.benchmarks,
        run_llm_judge="llm_judge" in args.benchmarks,
        mmlu_subjects=args.mmlu_subjects,
        mmlu_max_per_subject=args.max_samples,
        truthfulqa_max_samples=args.max_samples,
        checkpoint_name=args.checkpoint_name,
        output_dir=args.output_dir,
        report_to_wandb=not args.no_wandb,
    )

    model_fn = _build_model_fn(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        use_vllm=bool(args.vllm_url),
        vllm_url=args.vllm_url,
    )

    harness = EvalHarness(cfg)
    results = harness.run(model_fn, checkpoint_name=args.checkpoint_name)

    logger.info("=== Evaluation Summary ===")
    if "mmlu" in results:
        logger.info(f"MMLU Overall: {results['mmlu']['overall']['accuracy']:.4f}")
    if "truthfulqa" in results:
        logger.info(f"TruthfulQA MC1: {results['truthfulqa']['mc1_accuracy']:.4f}")
    if "custom" in results:
        logger.info(f"Custom Accuracy: {results['custom'].get('accuracy', 'N/A')}")
    if "llm_judge" in results:
        logger.info(f"LLM Judge Composite: {results['llm_judge']['aggregate']['composite']:.4f}")


if __name__ == "__main__":
    main()
