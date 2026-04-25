#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${REPO_ROOT}/external/manifest.lock"
TARGET_ROOT="${REPO_ROOT}/external/src"

mkdir -p "${TARGET_ROOT}"

python3 - <<'PY' "${MANIFEST}" "${TARGET_ROOT}"
import subprocess
import sys
from pathlib import Path

import yaml

manifest = Path(sys.argv[1])
target_root = Path(sys.argv[2])
data = yaml.safe_load(manifest.read_text())

for name, meta in data["repos"].items():
    repo_dir = target_root / name
    url = meta["url"]
    commit = meta["commit"]
    if not repo_dir.exists():
        subprocess.run(["git", "clone", url, str(repo_dir)], check=True)
    else:
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "--all", "--tags"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", commit], check=True)
    print(f"[bootstrap] {name}: {url} @ {commit}")
PY
