"""Kernel inventory helpers for pto-kernels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import repo_root


INVENTORY_PATH = repo_root() / "bench" / "kernel_inventory.yaml"


@dataclass(frozen=True)
class KernelRecord:
    name: str
    family: str
    wave: str
    status: str
    ops_transformer_path: str


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_inventory() -> dict[str, Any]:
    return _read_yaml(INVENTORY_PATH)


def included_kernel_records() -> list[KernelRecord]:
    data = load_inventory()
    records: list[KernelRecord] = []
    for entry in data.get("included", []):
        records.append(
            KernelRecord(
                name=entry["name"],
                family=entry["family"],
                wave=entry["wave"],
                status=entry["status"],
                ops_transformer_path=entry["ops_transformer_path"],
            )
        )
    return records


def kernel_counts() -> dict[str, int]:
    data = load_inventory()
    return {
        "included": len(data.get("included", [])),
        "excluded_ai_cpu": len(data.get("excluded", {}).get("ai_cpu", [])),
        "excluded_a3_only": len(data.get("excluded", {}).get("a3_only", [])),
        "seed_kernels": len(data.get("seed_kernels", [])),
    }
