"""Checkpoint regression reports — compare multiple checkpoints over time."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

BENCHMARK_COLS = [
    "mmlu_overall",
    "mmlu_stem",
    "mmlu_humanities",
    "mmlu_social_sciences",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "custom_accuracy",
    "judge_composite",
    "judge_pass_rate",
]


def load_checkpoint_results(results_dir: Path) -> pd.DataFrame:
    """Load all checkpoint result JSONs from a directory into a DataFrame."""
    rows = []
    for p in sorted(results_dir.glob("*_results.json")):
        with open(p) as f:
            r = json.load(f)
        row = {"checkpoint": r.get("checkpoint", p.stem), "timestamp": r.get("timestamp")}

        mmlu = r.get("mmlu", {})
        row["mmlu_overall"] = mmlu.get("overall", {}).get("accuracy", None)
        cats = mmlu.get("categories", {})
        row["mmlu_stem"] = cats.get("STEM", {}).get("accuracy", None)
        row["mmlu_humanities"] = cats.get("Humanities", {}).get("accuracy", None)
        row["mmlu_social_sciences"] = cats.get("Social Sciences", {}).get("accuracy", None)

        tqa = r.get("truthfulqa", {})
        row["truthfulqa_mc1"] = tqa.get("mc1_accuracy", None)
        row["truthfulqa_mc2"] = tqa.get("mc2_f1", None)

        custom = r.get("custom", {})
        row["custom_accuracy"] = custom.get("accuracy", None)

        judge = r.get("llm_judge", {}).get("aggregate", {})
        row["judge_composite"] = judge.get("composite", None)
        row["judge_pass_rate"] = judge.get("pass_rate", None)

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["checkpoint", "timestamp"] + BENCHMARK_COLS)

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp", na_position="last").reset_index(drop=True)
    return df


def build_regression_report(
    results_dir: Path,
    baseline_checkpoint: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """
    Compare all checkpoints against baseline and flag regressions.

    A regression is defined as any metric dropping more than 1 percentage point
    below the baseline.
    """
    df = load_checkpoint_results(results_dir)
    if df.empty:
        return {"error": "No results found", "regressions": []}

    if baseline_checkpoint:
        baseline_row = df[df["checkpoint"] == baseline_checkpoint]
        if baseline_row.empty:
            logger.warning(f"Baseline checkpoint {baseline_checkpoint!r} not found — using first row")
            baseline = df.iloc[0]
        else:
            baseline = baseline_row.iloc[0]
    else:
        baseline = df.iloc[0]

    regressions = []
    improvements = []
    report_rows = []

    for _, row in df.iterrows():
        ckpt = row["checkpoint"]
        if ckpt == baseline["checkpoint"]:
            continue

        row_result = {"checkpoint": ckpt, "metrics": {}}
        for col in BENCHMARK_COLS:
            base_val = baseline.get(col)
            curr_val = row.get(col)
            if base_val is None or curr_val is None or pd.isna(base_val) or pd.isna(curr_val):
                continue

            delta = curr_val - base_val
            status = "neutral"
            if delta < -0.01:
                status = "regression"
                regressions.append({"checkpoint": ckpt, "metric": col, "delta": delta})
            elif delta > 0.01:
                status = "improvement"
                improvements.append({"checkpoint": ckpt, "metric": col, "delta": delta})

            row_result["metrics"][col] = {
                "baseline": round(base_val, 4),
                "current": round(curr_val, 4),
                "delta": round(delta, 4),
                "status": status,
            }
        report_rows.append(row_result)

    report = {
        "baseline": baseline["checkpoint"],
        "total_checkpoints": len(df) - 1,
        "regressions": regressions,
        "improvements": improvements,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "details": report_rows,
        "dataframe": df.to_dict(orient="records"),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({k: v for k, v in report.items() if k != "dataframe"}, f, indent=2)
        df.to_csv(output_path.with_suffix(".csv"), index=False)
        logger.info(f"Regression report written to {output_path}")

    if regressions:
        logger.warning(f"REGRESSIONS DETECTED: {len(regressions)} metric(s) regressed!")
        for r in regressions:
            logger.warning(f"  {r['checkpoint']} | {r['metric']} | delta={r['delta']:.4f}")
    else:
        logger.info("No regressions detected.")

    return report
