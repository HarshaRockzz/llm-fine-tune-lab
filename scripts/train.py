#!/usr/bin/env python3
"""CLI entry point for fine-tuning. Supports preset configs and custom YAML."""
import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3 or Mistral with LoRA/QLoRA")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", choices=["llama3-lora", "llama3-qlora", "mistral-lora", "mistral-qlora"], help="Use a predefined config preset")
    group.add_argument("--config", type=Path, help="Path to a YAML config file")

    parser.add_argument("--dataset", type=Path, default=None, help="Local JSONL dataset (overrides HF dataset)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    parser.add_argument("--merge", action="store_true", help="Merge adapter into base model after training")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HuggingFace repo to push merged model")

    args = parser.parse_args()

    from src.training.config import TrainingConfig, PRESETS

    if args.preset:
        config = PRESETS[args.preset].model_copy(deep=True)
        logger.info(f"Using preset: {args.preset}")
    else:
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
        config = TrainingConfig(**cfg_dict)
        logger.info(f"Using config: {args.config}")

    if args.output_dir:
        config.output_dir = args.output_dir

    logger.info("Config summary:")
    logger.info(f"  Model:     {config.model_name_or_path}")
    logger.info(f"  Method:    {'QLoRA 4-bit' if config.use_qlora else 'LoRA BF16'}")
    logger.info(f"  LR:        {config.learning_rate}")
    logger.info(f"  Epochs:    {config.num_train_epochs}")
    logger.info(f"  Eff. batch:{config.effective_batch_size()}")
    logger.info(f"  Output:    {config.output_dir}")

    if args.dry_run:
        logger.info("Dry run — config validated successfully. Exiting.")
        return

    from src.training.trainer import train, merge_and_export

    checkpoint_path = train(
        config,
        resume_from_checkpoint=args.resume,
        local_dataset_path=args.dataset,
    )
    logger.info(f"Training complete. Best checkpoint: {checkpoint_path}")

    if args.merge:
        merged_path = checkpoint_path.parent / "merged"
        merge_and_export(
            adapter_path=checkpoint_path,
            base_model_name=config.model_name_or_path,
            output_path=merged_path,
            push_to_hub=bool(args.push_to_hub),
            hub_repo=args.push_to_hub,
        )
        logger.info(f"Merged model saved to {merged_path}")


if __name__ == "__main__":
    main()
