"""Training configuration using Pydantic v2 dataclasses."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class LoRAConfig(BaseModel):
    r: int = Field(16, description="LoRA rank")
    lora_alpha: int = Field(32, description="LoRA alpha scaling")
    target_modules: list[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_dropout: float = Field(0.05)
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"


class QLoRAConfig(LoRAConfig):
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True


class TrainingConfig(BaseModel):
    # Model
    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B"
    model_type: Literal["llama3", "mistral"] = "llama3"

    # Data
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    dataset_split: str = "train_sft"
    max_samples: int = 50_000
    max_seq_length: int = 2048
    instruction_template: str = "<|user|>"
    response_template: str = "<|assistant|>"

    # LoRA / QLoRA
    use_qlora: bool = True
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)

    # Training hyperparams
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    optim: str = "paged_adamw_32bit"
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    max_grad_norm: float = 0.3

    # Saving & logging
    output_dir: Path = Path("outputs/checkpoints")
    logging_steps: int = 10
    eval_steps: int = 250
    save_steps: int = 250
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # W&B
    report_to: str = "wandb"
    run_name: Optional[str] = None
    wandb_project: str = "llm-fine-tune-lab"

    # Hardware
    dataloader_num_workers: int = 4
    gradient_checkpointing: bool = True
    neftune_noise_alpha: float = 5.0

    @model_validator(mode="after")
    def validate_precision(self) -> "TrainingConfig":
        if self.bf16 and self.fp16:
            raise ValueError("Cannot use both bf16 and fp16")
        return self

    def effective_batch_size(self) -> int:
        return (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
        )


LLAMA3_8B_LORA = TrainingConfig(
    model_name_or_path="meta-llama/Meta-Llama-3-8B",
    model_type="llama3",
    use_qlora=False,
    lora=LoRAConfig(r=16, lora_alpha=32),
    learning_rate=2e-4,
    num_train_epochs=3,
    run_name="llama3-8b-lora",
)

LLAMA3_8B_QLORA = TrainingConfig(
    model_name_or_path="meta-llama/Meta-Llama-3-8B",
    model_type="llama3",
    use_qlora=True,
    qlora=QLoRAConfig(r=64, lora_alpha=16),
    learning_rate=2e-4,
    num_train_epochs=3,
    run_name="llama3-8b-qlora",
)

MISTRAL_7B_LORA = TrainingConfig(
    model_name_or_path="mistralai/Mistral-7B-v0.3",
    model_type="mistral",
    use_qlora=False,
    lora=LoRAConfig(r=16, lora_alpha=32),
    learning_rate=2e-4,
    num_train_epochs=3,
    instruction_template="[INST]",
    response_template="[/INST]",
    run_name="mistral-7b-lora",
)

MISTRAL_7B_QLORA = TrainingConfig(
    model_name_or_path="mistralai/Mistral-7B-v0.3",
    model_type="mistral",
    use_qlora=True,
    qlora=QLoRAConfig(r=64, lora_alpha=16),
    learning_rate=2e-4,
    num_train_epochs=3,
    instruction_template="[INST]",
    response_template="[/INST]",
    run_name="mistral-7b-qlora",
)

PRESETS: dict[str, TrainingConfig] = {
    "llama3-lora": LLAMA3_8B_LORA,
    "llama3-qlora": LLAMA3_8B_QLORA,
    "mistral-lora": MISTRAL_7B_LORA,
    "mistral-qlora": MISTRAL_7B_QLORA,
}
