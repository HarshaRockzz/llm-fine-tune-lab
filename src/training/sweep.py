"""W&B sweep configuration for hyperparameter search across 20+ experiments."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import wandb

from src.training.config import TrainingConfig, PRESETS


SWEEP_CONFIG_LORA = {
    "method": "bayes",
    "metric": {"name": "eval/loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 5e-4,
        },
        "lora_r": {"values": [8, 16, 32, 64]},
        "lora_alpha": {"values": [16, 32, 64]},
        "lora_dropout": {"values": [0.0, 0.05, 0.1]},
        "per_device_train_batch_size": {"values": [2, 4, 8]},
        "gradient_accumulation_steps": {"values": [2, 4, 8]},
        "warmup_ratio": {"distribution": "uniform", "min": 0.01, "max": 0.1},
        "weight_decay": {"values": [0.0, 0.01, 0.1]},
        "neftune_noise_alpha": {"values": [0.0, 5.0, 10.0]},
        "num_train_epochs": {"values": [1, 2, 3]},
        "lr_scheduler_type": {"values": ["cosine", "linear", "constant_with_warmup"]},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3,
        "eta": 3,
    },
}

SWEEP_CONFIG_QLORA = {
    **SWEEP_CONFIG_LORA,
    "parameters": {
        **SWEEP_CONFIG_LORA["parameters"],
        "bnb_4bit_quant_type": {"values": ["nf4", "fp4"]},
        "bnb_4bit_use_double_quant": {"values": [True, False]},
    },
}


def create_sweep(
    project: str = "llm-fine-tune-lab",
    model_preset: str = "llama3-qlora",
    use_qlora: bool = True,
    count: int = 20,
) -> str:
    """Create and launch a W&B sweep, returning the sweep ID."""
    sweep_config = SWEEP_CONFIG_QLORA if use_qlora else SWEEP_CONFIG_LORA
    sweep_config["name"] = f"sweep-{model_preset}-{count}runs"

    sweep_id = wandb.sweep(sweep_config, project=project)
    print(f"Created sweep: {sweep_id}")
    print(f"View at: https://wandb.ai/{project}/sweeps/{sweep_id}")
    return sweep_id


def sweep_train_fn() -> None:
    """Single sweep run — called by the W&B agent for each trial."""
    from src.training.trainer import train

    with wandb.init() as run:
        cfg_dict = dict(wandb.config)
        preset = os.environ.get("SWEEP_PRESET", "llama3-qlora")
        base_config = PRESETS[preset].model_copy(deep=True)

        # Patch config with sweep hyperparams
        base_config.learning_rate = cfg_dict.get("learning_rate", base_config.learning_rate)
        base_config.num_train_epochs = cfg_dict.get("num_train_epochs", base_config.num_train_epochs)
        base_config.per_device_train_batch_size = cfg_dict.get(
            "per_device_train_batch_size", base_config.per_device_train_batch_size
        )
        base_config.gradient_accumulation_steps = cfg_dict.get(
            "gradient_accumulation_steps", base_config.gradient_accumulation_steps
        )
        base_config.warmup_ratio = cfg_dict.get("warmup_ratio", base_config.warmup_ratio)
        base_config.weight_decay = cfg_dict.get("weight_decay", base_config.weight_decay)
        base_config.neftune_noise_alpha = cfg_dict.get(
            "neftune_noise_alpha", base_config.neftune_noise_alpha
        )
        base_config.lr_scheduler_type = cfg_dict.get(
            "lr_scheduler_type", base_config.lr_scheduler_type
        )

        # LoRA specific
        base_config.lora.r = cfg_dict.get("lora_r", base_config.lora.r)
        base_config.lora.lora_alpha = cfg_dict.get("lora_alpha", base_config.lora.lora_alpha)
        base_config.lora.lora_dropout = cfg_dict.get("lora_dropout", base_config.lora.lora_dropout)

        if base_config.use_qlora:
            base_config.qlora.r = base_config.lora.r
            base_config.qlora.lora_alpha = base_config.lora.lora_alpha
            base_config.qlora.lora_dropout = base_config.lora.lora_dropout
            base_config.qlora.bnb_4bit_quant_type = cfg_dict.get(
                "bnb_4bit_quant_type", base_config.qlora.bnb_4bit_quant_type
            )
            base_config.qlora.bnb_4bit_use_double_quant = cfg_dict.get(
                "bnb_4bit_use_double_quant", base_config.qlora.bnb_4bit_use_double_quant
            )

        base_config.run_name = run.name
        base_config.output_dir = Path(f"outputs/sweep/{run.id}")

        train(base_config)


def launch_sweep_agent(sweep_id: str, project: str, count: int = 5) -> None:
    """Launch a W&B agent to run `count` trials of the sweep."""
    wandb.agent(
        sweep_id,
        function=sweep_train_fn,
        project=project,
        count=count,
    )
