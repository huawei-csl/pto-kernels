"""Shared helpers for benchmark adapters."""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blocked(reason: str) -> dict[str, Any]:
    return {"status": "blocked", "reason": reason}


def describe_baseline(repo_root: Path, family: str, name: str, ops_transformer_path: str) -> dict[str, Any]:
    return {
        "family": family,
        "name": name,
        "ops_transformer_path": ops_transformer_path,
        "baseline_root": str(repo_root.parent / "ops-transformer"),
    }


def describe_pto(repo_root: Path, kernel_path: str, meta_path: str) -> dict[str, Any]:
    meta_file = repo_root / meta_path
    kernel_file = repo_root / kernel_path
    result = {
        "kernel_path": str(kernel_file),
        "meta_path": str(meta_file),
    }
    if meta_file.exists():
        module = load_module(meta_file)
        result["meta"] = getattr(module, "META", {})
    return result


def compile_pto_kernel(repo_root: Path, kernel_path: str, output_dir: Path) -> dict[str, Any]:
    kernel_file = repo_root / kernel_path
    module = load_module(kernel_file)
    builder = getattr(module, "build_jit_wrapper", None)
    if not callable(builder):
        return blocked("kernel module does not expose build_jit_wrapper(output_dir)")
    try:
        wrapper = builder(output_dir=output_dir)
    except NotImplementedError as exc:
        return blocked(str(exc))
    build = getattr(wrapper, "_build", None)
    try:
        if callable(build):
            build()
    except Exception as exc:  # pragma: no cover - exercised on NPU hosts
        return blocked(f"PTO compile failed: {exc}")
    artifact_paths = getattr(wrapper, "_artifact_paths", lambda: ())()
    return {
        "status": "ready",
        "kernel_path": str(kernel_file),
        "output_dir": str(output_dir),
        "artifact_paths": [str(path) for path in artifact_paths],
        "library_path": getattr(wrapper, "library_path", None),
    }


@contextmanager
def temporary_env(updates: dict[str, int | str | None]):
    previous: dict[str, str | None] = {}
    try:
        for key, value in updates.items():
            previous[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
