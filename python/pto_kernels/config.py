"""Shared path and manifest helpers for the pto-kernels workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "external" / "manifest.lock"


def repo_root() -> Path:
    return REPO_ROOT


def manifest_path() -> Path:
    return MANIFEST_PATH


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a mapping: {MANIFEST_PATH}")
    return data


def resolve_workspace_repo(name: str) -> Path | None:
    manifest = load_manifest()
    repo_meta = manifest.get("repos", {}).get(name, {})
    path_value = repo_meta.get("workspace_path")
    if path_value:
        path = Path(path_value)
        if path.exists():
            return path
    default_path = REPO_ROOT / "external" / "src" / name
    if default_path.exists():
        return default_path
    sibling_path = REPO_ROOT.parent / name
    if sibling_path.exists():
        return sibling_path
    return None
