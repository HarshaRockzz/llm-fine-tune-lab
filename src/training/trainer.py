"""Production LoRA and QLoRA fine-tuning trainer using PEFT + TRL."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import wandb
from datasets import DatasetDict
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from src.training.config import TrainingConfig
from src.training.dataset import load_and_prepare_dataset

logger = logging.getLogger(__name__)


def _build_bnb_config(config: TrainingConfig) -> Optional[BitsAndBytesConfig]:
    if not config.use_qlora:
        return None
    qcfg = config.qlora
    return BitsAndBytesConfig(
        load_in_4bit=qcfg.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, qcfg.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=qcfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=qcfg.bnb_4bit_use_double_quant,
    )


def _build_lora_config(config: TrainingConfig) -> LoraConfig:
    lcfg = config.qlora if config.use_qlora else config.lora
    return LoraConfig(
        r=lcfg.r,
        lora_alpha=lcfg.lora_alpha,
        target_modules=lcfg.target_modules,
        lora_dropout=lcfg.lora_dropout,
        bias=lcfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(config: TrainingConfig):
    """Load base model with optional 4-bit quantization + tokenizer."""
    logger.info(f"Loading model: {config.model_name_or_path} (QLoRA={config.use_qlora})")

    bnb_config = _build_bnb_config(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    if config.use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )

    lora_config = _build_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%) — "
        f"{'QLoRA 4-bit' if config.use_qlora else 'LoRA BF16'} active"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_training_arguments(config: TrainingConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        optim=config.optim,
        bf16=config.bf16,
        fp16=config.fp16,
        tf32=config.tf32,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to=config.report_to,
        run_name=config.run_name,
        dataloader_num_workers=config.dataloader_num_workers,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        neftune_noise_alpha=config.neftune_noise_alpha,
        group_by_length=True,
        ddp_find_unused_parameters=False,
    )


def train(
    config: TrainingConfig,
    resume_from_checkpoint: Optional[str] = None,
    local_dataset_path: Optional[Path] = None,
) -> Path:
    """Full training run. Returns path to best checkpoint."""

    if config.report_to == "wandb":
        wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config=config.model_dump(),
            tags=[config.model_type, "qlora" if config.use_qlora else "lora"],
        )

    model, tokenizer = load_model_and_tokenizer(config)
    dataset: DatasetDict = load_and_prepare_dataset(config, tokenizer, local_dataset_path)

    training_args = build_training_arguments(config)

    response_template_ids = tokenizer.encode(
        config.response_template, add_special_tokens=False
    )
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    best_ckpt = Path(training_args.output_dir) / "best_checkpoint"
    trainer.save_model(str(best_ckpt))
    tokenizer.save_pretrained(str(best_ckpt))

    if config.report_to == "wandb":
        wandb.finish()

    logger.info(f"Training complete. Best checkpoint saved to {best_ckpt}")
    return best_ckpt


def merge_and_export(
    adapter_path: Path,
    base_model_name: str,
    output_path: Path,
    push_to_hub: bool = False,
    hub_repo: Optional[str] = None,
) -> Path:
    """Merge LoRA adapter weights into the base model for deployment."""
    from peft import PeftModel

    logger.info(f"Merging adapter {adapter_path} into {base_model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged = peft_model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    if push_to_hub and hub_repo:
        merged.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)
        logger.info(f"Pushed merged model to {hub_repo}")

    logger.info(f"Merged model saved to {output_path}")
    return output_path
