"""Dataset loading, preprocessing, and tokenization for SFT fine-tuning."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from src.training.config import TrainingConfig

if TYPE_CHECKING:
    from datasets import DatasetDict, Dataset
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


CHAT_TEMPLATE_LLAMA3 = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<|user|>\n{{ message['content'] }}<|end|>\n"
    "{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}<|end|>\n"
    "{% elif message['role'] == 'system' %}<|system|>\n{{ message['content'] }}<|end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)

CHAT_TEMPLATE_MISTRAL = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]"
    "{% elif message['role'] == 'assistant' %}{{ message['content'] }}</s>"
    "{% endif %}{% endfor %}"
)


def _apply_chat_template(example: dict, tokenizer: Any, model_type: str) -> dict:
    """Convert messages list to a single formatted string."""
    messages = example.get("messages", [])
    if not messages:
        # Fallback for instruction/response format
        messages = [
            {"role": "user", "content": example.get("prompt", "")},
            {"role": "assistant", "content": example.get("completion", "")},
        ]

    if model_type == "llama3":
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    return {"text": text}


def _filter_long_sequences(
    example: dict, tokenizer: Any, max_length: int
) -> bool:
    tokens = tokenizer(example["text"], truncation=False)["input_ids"]
    return len(tokens) <= max_length


def load_and_prepare_dataset(
    config: TrainingConfig,
    tokenizer: "PreTrainedTokenizer",
    local_path: Optional[Path] = None,
) -> "DatasetDict":
    """Load dataset from HuggingFace hub or local path and apply preprocessing."""

    from datasets import load_dataset

    if local_path and local_path.exists():
        logger.info(f"Loading dataset from local path: {local_path}")
        raw = load_dataset("json", data_files=str(local_path))
        if "train" not in raw:
            raw = raw["train"].train_test_split(test_size=0.05, seed=42)
    else:
        logger.info(f"Loading {config.dataset_name} ({config.dataset_split}) from HuggingFace Hub")
        raw = load_dataset(config.dataset_name, split=config.dataset_split)
        raw = raw.train_test_split(test_size=0.05, seed=42)

    # Apply sampling
    if config.max_samples and len(raw["train"]) > config.max_samples:
        raw["train"] = raw["train"].shuffle(seed=42).select(range(config.max_samples))
        logger.info(f"Sampled {config.max_samples} training examples")

    # Apply chat template
    raw = raw.map(
        lambda ex: _apply_chat_template(ex, tokenizer, config.model_type),
        remove_columns=raw["train"].column_names,
        num_proc=4,
        desc="Applying chat template",
    )

    # Filter by length
    before_count = len(raw["train"])
    raw = raw.filter(
        lambda ex: _filter_long_sequences(ex, tokenizer, config.max_seq_length),
        num_proc=4,
        desc="Filtering long sequences",
    )
    after_count = len(raw["train"])
    logger.info(
        f"Filtered {before_count - after_count} sequences exceeding {config.max_seq_length} tokens "
        f"({(before_count - after_count) / before_count * 100:.1f}%)"
    )

    logger.info(
        f"Final dataset: {len(raw['train'])} train / {len(raw['test'])} eval examples"
    )
    return raw


def load_domain_dataset(domain: str, max_samples: int = 5000) -> "Dataset":
    """Load a specific domain dataset for domain-specific fine-tuning."""
    from datasets import load_dataset
    DOMAIN_MAP = {
        "medical": ("medalpaca/medical_meadow_medqa", "train"),
        "legal": ("nguyen-brat/legal-qa", "train"),
        "code": ("codeparrot/github-code-clean", "train"),
        "finance": ("FinGPT/fingpt-sentiment-train", "train"),
        "science": ("allenai/sciq", "train"),
    }
    if domain not in DOMAIN_MAP:
        raise ValueError(f"Unknown domain: {domain}. Choose from {list(DOMAIN_MAP.keys())}")

    name, split = DOMAIN_MAP[domain]
    ds = load_dataset(name, split=split)
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))
    return ds


def prepare_sample_data(output_path: Path, n_samples: int = 1000) -> Path:
    """Generate a small synthetic sample dataset for testing."""
    import json
    import random

    topics = ["Python", "machine learning", "data structures", "algorithms", "statistics"]
    examples = []
    for i in range(n_samples):
        topic = random.choice(topics)
        examples.append({
            "messages": [
                {"role": "user", "content": f"Explain {topic} concept #{i % 20} in detail."},
                {
                    "role": "assistant",
                    "content": (
                        f"Here is a detailed explanation of {topic} concept #{i % 20}. "
                        "This is a synthetic example for testing the training pipeline. "
                        "In production, replace this with real domain data."
                    ),
                },
            ]
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    logger.info(f"Wrote {n_samples} synthetic samples to {output_path}")
    return output_path
