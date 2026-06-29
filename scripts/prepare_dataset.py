#!/usr/bin/env python3
"""Prepare a dataset for training: download, filter, and save as JSONL."""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for fine-tuning")
    parser.add_argument(
        "--source",
        default="HuggingFaceH4/ultrachat_200k",
        help="HF dataset name or 'synthetic'",
    )
    parser.add_argument("--split", default="train_sft")
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.source == "synthetic":
        from src.training.dataset import prepare_sample_data

        prepare_sample_data(args.output, args.max_samples)
        return

    from datasets import load_dataset

    logging.info(f"Loading {args.source} ({args.split})")
    ds = load_dataset(args.source, split=args.split)

    if len(ds) > args.max_samples:
        ds = ds.shuffle(seed=args.seed).select(range(args.max_samples))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_json(str(args.output), lines=True)
    logging.info(f"Saved {len(ds)} examples to {args.output}")


if __name__ == "__main__":
    main()
