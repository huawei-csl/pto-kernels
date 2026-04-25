"""Spec loading for the PTO benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pto_kernels.config import repo_root


DEFAULT_FLAGS_PATH = repo_root() / "bench" / "canonical_compile_flags.yaml"


@dataclass(frozen=True)
class AdapterSpec:
    adapter: str
    kernel: str | None = None
    meta: str | None = None


@dataclass(frozen=True)
class BenchSettings:
    warmup: int
    repeat: int
    statistic: str
    parity_threshold: float


@dataclass(frozen=True)
class CorrectnessSettings:
    atol: float
    rtol: float
    shape_sets: list[str]


@dataclass(frozen=True)
class KernelBenchmarkSpec:
    name: str
    family: str
    wave: str
    inventory_ref: str
    status: str
    baseline: AdapterSpec
    pto: AdapterSpec
    bench: BenchSettings
    correctness: CorrectnessSettings
    device: dict[str, Any]
    notes: list[str]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_default_flags() -> dict[str, Any]:
    return _load_yaml(DEFAULT_FLAGS_PATH)


def load_spec(spec_path: str | Path) -> KernelBenchmarkSpec:
    spec_file = Path(spec_path)
    data = _load_yaml(spec_file)
    defaults = load_default_flags()
    bench_data = data.get("bench", {})
    correctness_data = data.get("correctness", {})
    device = data.get("device", defaults.get("device", {}))
    return KernelBenchmarkSpec(
        name=data["name"],
        family=data["family"],
        wave=data["wave"],
        inventory_ref=data["inventory_ref"],
        status=data.get("status", "planned"),
        baseline=AdapterSpec(**data["baseline"]),
        pto=AdapterSpec(**data["pto"]),
        bench=BenchSettings(
            warmup=int(bench_data.get("warmup", defaults["bench"]["warmup"])),
            repeat=int(bench_data.get("repeat", defaults["bench"]["repeat"])),
            statistic=str(bench_data.get("statistic", defaults["bench"]["statistic"])),
            parity_threshold=float(
                bench_data.get("parity_threshold", defaults["bench"]["parity_threshold"])
            ),
        ),
        correctness=CorrectnessSettings(
            atol=float(correctness_data.get("atol", defaults["correctness"]["atol"])),
            rtol=float(correctness_data.get("rtol", defaults["correctness"]["rtol"])),
            shape_sets=list(
                correctness_data.get("shape_sets", defaults["correctness"]["shape_sets"])
            ),
        ),
        device=device,
        notes=list(data.get("notes", [])),
    )
