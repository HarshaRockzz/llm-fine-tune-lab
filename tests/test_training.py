"""Unit tests for training config and dataset utilities (no GPU required)."""
import json
from pathlib import Path

import pytest

from src.training.config import (
    LoRAConfig,
    QLoRAConfig,
    TrainingConfig,
    PRESETS,
)
from src.training.dataset import prepare_sample_data


class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert "q_proj" in cfg.target_modules
        assert cfg.bias == "none"

    def test_custom_rank(self):
        cfg = LoRAConfig(r=64, lora_alpha=16)
        assert cfg.r == 64


class TestQLoRAConfig:
    def test_defaults(self):
        cfg = QLoRAConfig()
        assert cfg.load_in_4bit is True
        assert cfg.bnb_4bit_quant_type == "nf4"
        assert cfg.bnb_4bit_use_double_quant is True


class TestTrainingConfig:
    def test_bf16_fp16_mutual_exclusion(self):
        with pytest.raises(Exception):
            TrainingConfig(bf16=True, fp16=True)

    def test_effective_batch_size(self):
        cfg = TrainingConfig(
            model_name_or_path="test",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
        )
        assert cfg.effective_batch_size() == 32

    def test_presets_exist(self):
        for name in ["llama3-lora", "llama3-qlora", "mistral-lora", "mistral-qlora"]:
            assert name in PRESETS
            cfg = PRESETS[name]
            assert cfg.model_name_or_path
            assert cfg.learning_rate > 0

    def test_llama3_qlora_preset(self):
        cfg = PRESETS["llama3-qlora"]
        assert cfg.use_qlora is True
        assert "Meta-Llama-3" in cfg.model_name_or_path

    def test_mistral_preset(self):
        cfg = PRESETS["mistral-lora"]
        assert "mistralai" in cfg.model_name_or_path.lower() or "Mistral" in cfg.model_name_or_path


class TestSampleDataPrep:
    def test_creates_jsonl(self, tmp_path):
        out = tmp_path / "test.jsonl"
        result = prepare_sample_data(out, n_samples=50)
        assert result.exists()
        lines = result.read_text().strip().split("\n")
        assert len(lines) == 50

    def test_valid_json(self, tmp_path):
        out = tmp_path / "test.jsonl"
        prepare_sample_data(out, n_samples=10)
        for line in out.read_text().strip().split("\n"):
            data = json.loads(line)
            assert "messages" in data
            assert len(data["messages"]) == 2
            assert data["messages"][0]["role"] == "user"
            assert data["messages"][1]["role"] == "assistant"
