"""LLM-as-judge scoring — uses OpenRouter (free Llama-3 / Mistral models) as the evaluator."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert AI evaluator assessing the quality of language model responses.
You will be given a question/instruction and a model response. Your task is to evaluate the response
on multiple dimensions and provide a score and reasoning.

Be objective, rigorous, and fair. Focus on factual accuracy, helpfulness, clarity, and safety.
Always respond with valid JSON only — no markdown fences, no extra text."""

JUDGE_PROMPT_TEMPLATE = """Evaluate the following model response:

**Question/Instruction:**
{question}

**Model Response:**
{response}

**Reference Answer (if available):**
{reference}

Rate the response on each dimension from 1-10 and provide brief reasoning:

{{
  "factual_accuracy": {{"score": <1-10>, "reasoning": "<brief>"}},
  "helpfulness": {{"score": <1-10>, "reasoning": "<brief>"}},
  "clarity": {{"score": <1-10>, "reasoning": "<brief>"}},
  "completeness": {{"score": <1-10>, "reasoning": "<brief>"}},
  "safety": {{"score": <1-10>, "reasoning": "<brief>"}},
  "overall": {{"score": <1-10>, "reasoning": "<brief>"}},
  "verdict": "<pass|fail>"
}}"""


@dataclass
class JudgeScore:
    factual_accuracy: float
    helpfulness: float
    clarity: float
    completeness: float
    safety: float
    overall: float
    verdict: str
    reasoning: dict

    @property
    def composite(self) -> float:
        return (
            self.factual_accuracy * 0.35
            + self.helpfulness * 0.25
            + self.clarity * 0.15
            + self.completeness * 0.15
            + self.safety * 0.10
        )

    @classmethod
    def from_dict(cls, data: dict) -> "JudgeScore":
        return cls(
            factual_accuracy=data["factual_accuracy"]["score"],
            helpfulness=data["helpfulness"]["score"],
            clarity=data["clarity"]["score"],
            completeness=data["completeness"]["score"],
            safety=data["safety"]["score"],
            overall=data["overall"]["score"],
            verdict=data.get("verdict", "pass"),
            reasoning={
                k: v.get("reasoning", "")
                for k, v in data.items()
                if isinstance(v, dict)
            },
        )


class LLMJudge:
    """OpenRouter-powered LLM-as-judge for evaluating model responses."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        # Default: free Llama-3.1-70B on OpenRouter (strong enough to judge)
        self.model = model or os.environ.get(
            "OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct:free"
        )
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature

    def score(self, question: str, response: str, reference: str = "N/A") -> JudgeScore:
        from src.utils.openrouter import chat

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question, response=response, reference=reference
        )
        raw = chat(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
        )

        # Strip markdown fences if model added them
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            raw = raw.rsplit("```", 1)[0]

        data = json.loads(raw)
        return JudgeScore.from_dict(data)

    def evaluate_dataset(
        self,
        examples: list[dict],
        model_fn: Callable,
        question_key: str = "question",
        reference_key: str = "answer",
        verbose: bool = True,
    ) -> dict:
        results = []
        for i, ex in enumerate(examples):
            question = ex[question_key]
            reference = ex.get(reference_key, "N/A")
            response = model_fn(question)

            try:
                score = self.score(question, response, reference)
            except Exception as e:
                logger.warning(f"Judge scoring failed for example {i}: {e}")
                continue

            results.append(
                {
                    "question": question,
                    "response": response,
                    "reference": reference,
                    "scores": {
                        "factual_accuracy": score.factual_accuracy,
                        "helpfulness": score.helpfulness,
                        "clarity": score.clarity,
                        "completeness": score.completeness,
                        "safety": score.safety,
                        "overall": score.overall,
                        "composite": score.composite,
                        "verdict": score.verdict,
                    },
                }
            )

            if verbose and i % 10 == 0:
                logger.info(
                    f"  Judge [{i}/{len(examples)}] composite={score.composite:.2f} verdict={score.verdict}"
                )

        if not results:
            return {"error": "No results", "total": 0}

        scores_list = [r["scores"] for r in results]
        avg_scores = {
            metric: sum(s[metric] for s in scores_list) / len(scores_list)
            for metric in [
                "factual_accuracy",
                "helpfulness",
                "clarity",
                "completeness",
                "safety",
                "overall",
                "composite",
            ]
        }
        pass_rate = sum(1 for r in results if r["scores"]["verdict"] == "pass") / len(
            results
        )

        return {
            "aggregate": {**avg_scores, "pass_rate": pass_rate},
            "total": len(results),
            "results": results,
        }
