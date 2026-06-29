"""Tests for FastAPI schemas and adapter manager (no GPU required)."""
import json
import pytest
from pathlib import Path

from src.api.schemas import (
    GenerationRequest,
    ChatRequest,
    ChatMessage,
    EvalRequest,
)
from src.inference.adapter_manager import AdapterManager, AdapterInfo


class TestSchemas:
    def test_generation_request_defaults(self):
        req = GenerationRequest(prompt="hello")
        assert req.max_tokens == 512
        assert req.temperature == 0.7
        assert req.model == "base"
        assert req.stream is False

    def test_generation_request_validation(self):
        with pytest.raises(Exception):
            GenerationRequest(prompt="")  # min_length=1

        with pytest.raises(Exception):
            GenerationRequest(prompt="x", temperature=3.0)  # max=2.0

    def test_chat_request(self):
        req = ChatRequest(
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi!"),
            ]
        )
        assert len(req.messages) == 2
        assert req.messages[0].role == "user"

    def test_eval_request_defaults(self):
        req = EvalRequest()
        assert req.model == "base"
        assert "custom" in req.benchmarks
        assert req.max_samples == 100


class TestAdapterManager:
    def test_register_and_retrieve(self, tmp_path):
        manager = AdapterManager(registry_dir=tmp_path)
        info = manager.register(
            name="test-adapter",
            path="/fake/path",
            base_model="meta-llama/Meta-Llama-3-8B",
            description="Test adapter",
            tags=["test"],
            metrics={"mmlu": 0.71},
        )
        assert info.name == "test-adapter"
        assert info.active is True

        retrieved = manager.get("test-adapter")
        assert retrieved is not None
        assert retrieved.metrics["mmlu"] == 0.71

    def test_list_adapters(self, tmp_path):
        manager = AdapterManager(registry_dir=tmp_path)
        manager.register("a1", "/p1", "model-a")
        manager.register("a2", "/p2", "model-b")
        adapters = manager.list_adapters()
        assert len(adapters) == 2

    def test_deactivate(self, tmp_path):
        manager = AdapterManager(registry_dir=tmp_path)
        manager.register("ada", "/p", "model")
        manager.deactivate("ada")
        assert manager.list_adapters(active_only=True) == []
        assert len(manager.list_adapters(active_only=False)) == 1

    def test_to_vllm_dict(self, tmp_path):
        manager = AdapterManager(registry_dir=tmp_path)
        manager.register("lora1", "/adapters/lora1", "llama3")
        manager.register("lora2", "/adapters/lora2", "mistral")
        d = manager.to_vllm_dict()
        assert d["lora1"] == "/adapters/lora1"
        assert d["lora2"] == "/adapters/lora2"

    def test_registry_persistence(self, tmp_path):
        m1 = AdapterManager(registry_dir=tmp_path)
        m1.register("persistent", "/p", "model", metrics={"mmlu": 0.65})

        m2 = AdapterManager(registry_dir=tmp_path)
        assert m2.get("persistent") is not None
        assert m2.get("persistent").metrics["mmlu"] == 0.65

    def test_delete(self, tmp_path):
        manager = AdapterManager(registry_dir=tmp_path)
        manager.register("to-delete", "/p", "model")
        manager.delete("to-delete")
        assert manager.get("to-delete") is None

    def test_update_metrics(self, tmp_path):
        manager = AdapterManager(registry_dir=tmp_path)
        manager.register("ada", "/p", "model", metrics={"mmlu": 0.60})
        manager.update_metrics("ada", {"mmlu": 0.71, "truthfulqa": 0.57})
        assert manager.get("ada").metrics["mmlu"] == 0.71
        assert manager.get("ada").metrics["truthfulqa"] == 0.57
