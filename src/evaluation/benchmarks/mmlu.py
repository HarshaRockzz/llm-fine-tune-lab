"""MMLU (Massive Multitask Language Understanding) benchmark evaluation."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

MMLU_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "formal_logic", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "management", "marketing",
        "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy",
    ],
    "Other": [
        "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "medical_genetics", "miscellaneous",
        "nutrition", "professional_accounting", "professional_medicine", "virology",
    ],
}

CHOICES = ["A", "B", "C", "D"]


def _format_prompt(question: str, choices: list[str], subject: str) -> str:
    subject_formatted = subject.replace("_", " ").title()
    prompt = (
        f"The following is a multiple choice question about {subject_formatted}.\n\n"
        f"Question: {question}\n"
    )
    for letter, choice in zip(CHOICES, choices):
        prompt += f"{letter}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def _extract_answer(text: str) -> Optional[str]:
    text = text.strip()
    if text and text[0] in CHOICES:
        return text[0]
    for choice in CHOICES:
        if f"Answer: {choice}" in text or f"answer is {choice}" in text.lower():
            return choice
    return None


class MMLUEvaluator:
    def __init__(
        self,
        subjects: Optional[list[str]] = None,
        n_shots: int = 5,
        max_per_subject: int = 100,
    ):
        self.subjects = subjects or MMLU_SUBJECTS
        self.n_shots = n_shots
        self.max_per_subject = max_per_subject

    def evaluate(self, model_fn, verbose: bool = False) -> dict:
        """
        Args:
            model_fn: callable(prompt: str) -> str — takes a prompt, returns model output
        Returns:
            dict with per-subject, per-category, and overall accuracy
        """
        results: dict[str, dict] = {}

        for subject in self.subjects:
            try:
                ds = load_dataset("cais/mmlu", subject, trust_remote_code=True)
                test_split = ds["test"]
                dev_split = ds["dev"]
            except Exception as e:
                logger.warning(f"Could not load MMLU subject {subject}: {e}")
                continue

            # Build few-shot prefix from dev split
            few_shot_prefix = ""
            for i, ex in enumerate(dev_split):
                if i >= self.n_shots:
                    break
                few_shot_prefix += _format_prompt(
                    ex["question"], ex["choices"], subject
                ) + f" {CHOICES[ex['answer']]}\n\n"

            correct = 0
            total = 0
            for i, ex in enumerate(test_split):
                if i >= self.max_per_subject:
                    break

                prompt = few_shot_prefix + _format_prompt(
                    ex["question"], ex["choices"], subject
                )
                response = model_fn(prompt)
                predicted = _extract_answer(response)
                gt = CHOICES[ex["answer"]]

                if predicted == gt:
                    correct += 1
                total += 1

            acc = correct / total if total > 0 else 0.0
            results[subject] = {"correct": correct, "total": total, "accuracy": acc}

            if verbose:
                logger.info(f"  {subject}: {acc:.3f} ({correct}/{total})")

        # Aggregate by category
        category_results: dict[str, dict] = {}
        for cat, cat_subjects in MMLU_CATEGORIES.items():
            cat_correct = sum(results[s]["correct"] for s in cat_subjects if s in results)
            cat_total = sum(results[s]["total"] for s in cat_subjects if s in results)
            category_results[cat] = {
                "correct": cat_correct,
                "total": cat_total,
                "accuracy": cat_correct / cat_total if cat_total > 0 else 0.0,
            }

        overall_correct = sum(v["correct"] for v in results.values())
        overall_total = sum(v["total"] for v in results.values())
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

        return {
            "subjects": results,
            "categories": category_results,
            "overall": {
                "correct": overall_correct,
                "total": overall_total,
                "accuracy": overall_accuracy,
            },
        }
