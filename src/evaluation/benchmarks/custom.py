"""Custom domain evaluation sets with flexible task types."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Literal, Optional

logger = logging.getLogger(__name__)

TaskType = Literal["multiple_choice", "generation", "classification", "extraction"]


def _mc_accuracy(predictions: list[str], references: list[str]) -> float:
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references))
    return correct / len(predictions) if predictions else 0.0


def _exact_match(predictions: list[str], references: list[str]) -> float:
    return sum(
        p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)
    ) / len(predictions) if predictions else 0.0


def _rouge_l(prediction: str, reference: str) -> float:
    """Simplified ROUGE-L without external library."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS via DP
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    precision = lcs / m
    recall = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class CustomEvaluator:
    """Evaluate a model on a custom JSON/JSONL dataset."""

    def __init__(
        self,
        eval_path: Optional[Path] = None,
        task_type: TaskType = "multiple_choice",
        prompt_key: str = "prompt",
        reference_key: str = "answer",
        choices_key: str = "choices",
    ):
        self.eval_path = eval_path
        self.task_type = task_type
        self.prompt_key = prompt_key
        self.reference_key = reference_key
        self.choices_key = choices_key

    def _load_examples(self) -> list[dict]:
        if self.eval_path and self.eval_path.exists():
            with open(self.eval_path) as f:
                if self.eval_path.suffix == ".json":
                    data = json.load(f)
                    return data if isinstance(data, list) else data.get("examples", [])
                else:
                    return [json.loads(line) for line in f if line.strip()]
        # Return built-in synthetic test set
        return self._builtin_examples()

    def _builtin_examples(self) -> list[dict]:
        """Built-in domain accuracy test: coding, reasoning, factual."""
        return [
            {
                "prompt": "What is the time complexity of binary search?",
                "choices": ["O(n)", "O(log n)", "O(n log n)", "O(1)"],
                "answer": "B",
                "category": "algorithms",
            },
            {
                "prompt": "Which Python built-in implements a max-heap?",
                "choices": ["heapq (negate values)", "sortedcontainers", "collections.heap", "queue.PriorityQueue"],
                "answer": "A",
                "category": "python",
            },
            {
                "prompt": "What does the 'r' in LoRA stand for?",
                "choices": ["rank", "regularization", "residual", "rate"],
                "answer": "A",
                "category": "ml_concepts",
            },
            {
                "prompt": "Which quantization scheme does QLoRA use by default?",
                "choices": ["int8", "nf4 (4-bit NormalFloat)", "fp4", "int4"],
                "answer": "B",
                "category": "ml_concepts",
            },
            {
                "prompt": "What is gradient checkpointing used for?",
                "choices": [
                    "Speeding up forward pass",
                    "Reducing GPU memory at cost of extra compute",
                    "Mixed precision training",
                    "Gradient clipping",
                ],
                "answer": "B",
                "category": "ml_concepts",
            },
        ]

    def evaluate(self, model_fn: Callable, verbose: bool = False) -> dict:
        examples = self._load_examples()
        predictions, references = [], []
        rouge_scores = []
        category_results: dict[str, list] = {}

        for ex in examples:
            prompt = ex[self.prompt_key]
            reference = str(ex[self.reference_key])
            category = ex.get("category", "general")

            if self.task_type == "multiple_choice" and self.choices_key in ex:
                choices = ex[self.choices_key]
                formatted = f"{prompt}\n\n"
                for i, c in enumerate(choices):
                    formatted += f"{'ABCDEFGH'[i]}. {c}\n"
                formatted += "\nAnswer:"
                pred = model_fn(formatted).strip()
            else:
                pred = model_fn(prompt).strip()

            predictions.append(pred)
            references.append(reference)

            if self.task_type == "generation":
                rouge_scores.append(_rouge_l(pred, reference))

            if category not in category_results:
                category_results[category] = {"predictions": [], "references": []}
            category_results[category]["predictions"].append(pred)
            category_results[category]["references"].append(reference)

            if verbose:
                logger.info(f"  Q: {prompt[:60]}... | Pred: {pred[:20]} | Ref: {reference}")

        if self.task_type == "multiple_choice":
            overall_score = _mc_accuracy(predictions, references)
            metric_name = "accuracy"
        elif self.task_type == "generation":
            overall_score = float(sum(rouge_scores) / len(rouge_scores)) if rouge_scores else 0.0
            metric_name = "rouge_l"
        else:
            overall_score = _exact_match(predictions, references)
            metric_name = "exact_match"

        cat_scores = {}
        for cat, data in category_results.items():
            if self.task_type == "multiple_choice":
                cat_scores[cat] = _mc_accuracy(data["predictions"], data["references"])
            else:
                cat_scores[cat] = _exact_match(data["predictions"], data["references"])

        return {
            metric_name: overall_score,
            "total": len(examples),
            "categories": cat_scores,
            "task_type": self.task_type,
        }
