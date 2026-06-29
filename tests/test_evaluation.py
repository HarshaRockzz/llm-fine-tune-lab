"""Tests for evaluation benchmarks and regression reports (no GPU required)."""
import json
from pathlib import Path

import pytest

from src.evaluation.benchmarks.custom import CustomEvaluator, _mc_accuracy, _rouge_l, _exact_match
from src.evaluation.reports import load_checkpoint_results, build_regression_report


class TestMetrics:
    def test_mc_accuracy_perfect(self):
        assert _mc_accuracy(["A", "B", "C"], ["A", "B", "C"]) == 1.0

    def test_mc_accuracy_zero(self):
        assert _mc_accuracy(["A", "A", "A"], ["B", "C", "D"]) == 0.0

    def test_mc_accuracy_partial(self):
        result = _mc_accuracy(["A", "B", "C"], ["A", "C", "C"])
        assert abs(result - 2 / 3) < 1e-9

    def test_rouge_l_identical(self):
        assert _rouge_l("hello world", "hello world") == 1.0

    def test_rouge_l_empty(self):
        assert _rouge_l("", "hello") == 0.0

    def test_rouge_l_partial(self):
        score = _rouge_l("the quick brown fox", "the quick fox")
        assert 0.0 < score < 1.0

    def test_exact_match(self):
        assert _exact_match(["Paris", "london"], ["paris", "London"]) == 1.0


class TestCustomEvaluator:
    def test_builtin_examples(self):
        evaluator = CustomEvaluator()
        examples = evaluator._builtin_examples()
        assert len(examples) > 0
        for ex in examples:
            assert "prompt" in ex
            assert "answer" in ex
            assert "choices" in ex

    def test_evaluate_perfect_model(self):
        evaluator = CustomEvaluator()
        examples = evaluator._builtin_examples()
        answers = {ex["prompt"]: ex["answer"] for ex in examples}

        def perfect_fn(prompt: str) -> str:
            # Return the correct letter
            for q, ans in answers.items():
                if q in prompt:
                    return ans
            return "A"

        result = evaluator.evaluate(perfect_fn)
        assert "accuracy" in result
        assert result["accuracy"] > 0.0

    def test_evaluate_from_file(self, tmp_path):
        data = [
            {"prompt": "2+2=?", "choices": ["3", "4", "5", "6"], "answer": "B", "category": "math"},
            {"prompt": "Capital of France?", "choices": ["London", "Berlin", "Paris", "Rome"], "answer": "C", "category": "geo"},
        ]
        p = tmp_path / "test.jsonl"
        with open(p, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")

        evaluator = CustomEvaluator(eval_path=p)
        result = evaluator.evaluate(lambda _: "A")
        assert "accuracy" in result
        assert result["total"] == 2


class TestRegressionReport:
    def _make_results_dir(self, tmp_path: Path) -> Path:
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        checkpoints = [
            ("ckpt_base", 0.540, 0.380, 0.540, 5.8),
            ("ckpt_250", 0.571, 0.400, 0.563, 6.1),
            ("ckpt_500", 0.608, 0.430, 0.592, 6.5),
            ("ckpt_1000", 0.665, 0.480, 0.644, 7.2),
            ("ckpt_final", 0.710, 0.570, 0.710, 7.9),
        ]
        for i, (name, mmlu, tqa, custom, judge) in enumerate(checkpoints):
            data = {
                "checkpoint": name,
                "timestamp": float(i),
                "mmlu": {
                    "overall": {"accuracy": mmlu, "correct": int(mmlu * 100), "total": 100},
                    "categories": {
                        "STEM": {"accuracy": mmlu - 0.01},
                        "Humanities": {"accuracy": mmlu + 0.01},
                        "Social Sciences": {"accuracy": mmlu},
                        "Other": {"accuracy": mmlu - 0.005},
                    },
                },
                "truthfulqa": {"mc1_accuracy": tqa, "mc2_f1": tqa + 0.1, "total": 817},
                "custom": {"accuracy": custom, "total": 100},
                "llm_judge": {"aggregate": {"composite": judge, "pass_rate": judge / 10}},
            }
            with open(results_dir / f"{name}_results.json", "w") as f:
                json.dump(data, f)

        return results_dir

    def test_load_results(self, tmp_path):
        results_dir = self._make_results_dir(tmp_path)
        df = load_checkpoint_results(results_dir)
        assert len(df) == 5
        assert "mmlu_overall" in df.columns
        assert "truthfulqa_mc1" in df.columns

    def test_no_regressions(self, tmp_path):
        results_dir = self._make_results_dir(tmp_path)
        report = build_regression_report(results_dir, baseline_checkpoint="ckpt_base")
        assert report["regression_count"] == 0
        assert report["improvement_count"] > 0

    def test_detects_regression(self, tmp_path):
        results_dir = self._make_results_dir(tmp_path)
        # Add a checkpoint that regresses on MMLU
        bad_data = {
            "checkpoint": "ckpt_bad",
            "timestamp": 99.0,
            "mmlu": {"overall": {"accuracy": 0.50, "correct": 50, "total": 100}, "categories": {}},
            "truthfulqa": {"mc1_accuracy": 0.38, "mc2_f1": 0.52, "total": 817},
            "custom": {"accuracy": 0.52, "total": 100},
        }
        with open(results_dir / "ckpt_bad_results.json", "w") as f:
            json.dump(bad_data, f)

        report = build_regression_report(results_dir, baseline_checkpoint="ckpt_base")
        bad_regressions = [r for r in report["regressions"] if r["checkpoint"] == "ckpt_bad"]
        assert len(bad_regressions) > 0
