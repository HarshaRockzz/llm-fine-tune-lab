"""LoRA adapter registry — load, validate, and hot-swap adapters at runtime."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ADAPTER_REGISTRY_FILE = "adapter_registry.json"


@dataclass
class AdapterInfo:
    name: str
    path: str
    base_model: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "base_model": self.base_model,
            "description": self.description,
            "tags": self.tags,
            "metrics": self.metrics,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AdapterInfo":
        return cls(**data)


class AdapterManager:
    """Manages LoRA adapters on disk with a JSON registry."""

    def __init__(self, registry_dir: Path = Path("outputs/adapters")):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = registry_dir / ADAPTER_REGISTRY_FILE
        self._adapters: dict[str, AdapterInfo] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        if self._registry_path.exists():
            with open(self._registry_path) as f:
                data = json.load(f)
            self._adapters = {k: AdapterInfo.from_dict(v) for k, v in data.items()}
            logger.info(f"Loaded {len(self._adapters)} adapter(s) from registry")

    def _save_registry(self) -> None:
        with open(self._registry_path, "w") as f:
            json.dump({k: v.to_dict() for k, v in self._adapters.items()}, f, indent=2)

    def register(
        self,
        name: str,
        path: str,
        base_model: str,
        description: str = "",
        tags: Optional[list[str]] = None,
        metrics: Optional[dict] = None,
        copy_to_registry: bool = False,
    ) -> AdapterInfo:
        """Register a new adapter. Optionally copy files to registry_dir."""
        dest_path = path

        if copy_to_registry:
            dest = self.registry_dir / name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(path, str(dest), dirs_exist_ok=True)
            dest_path = str(dest)
            logger.info(f"Copied adapter to {dest}")

        info = AdapterInfo(
            name=name,
            path=dest_path,
            base_model=base_model,
            description=description,
            tags=tags or [],
            metrics=metrics or {},
        )
        self._adapters[name] = info
        self._save_registry()
        logger.info(f"Registered adapter '{name}' ({base_model})")
        return info

    def get(self, name: str) -> Optional[AdapterInfo]:
        return self._adapters.get(name)

    def list_adapters(self, active_only: bool = True) -> list[AdapterInfo]:
        adapters = list(self._adapters.values())
        if active_only:
            adapters = [a for a in adapters if a.active]
        return adapters

    def deactivate(self, name: str) -> None:
        if name in self._adapters:
            self._adapters[name].active = False
            self._save_registry()

    def activate(self, name: str) -> None:
        if name in self._adapters:
            self._adapters[name].active = True
            self._save_registry()

    def update_metrics(self, name: str, metrics: dict) -> None:
        if name in self._adapters:
            self._adapters[name].metrics.update(metrics)
            self._save_registry()

    def delete(self, name: str, delete_files: bool = False) -> None:
        if name not in self._adapters:
            return
        info = self._adapters.pop(name)
        if delete_files and Path(info.path).exists():
            shutil.rmtree(info.path)
        self._save_registry()
        logger.info(f"Deleted adapter '{name}'")

    def to_vllm_dict(self) -> dict[str, str]:
        """Return name → path mapping for vLLM LoRA registration."""
        return {a.name: a.path for a in self.list_adapters()}
