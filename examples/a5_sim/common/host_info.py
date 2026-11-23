"""Capture host CPU metadata for benchmark documentation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def capture_host_cpu() -> dict:
    lscpu = _run(["lscpu"])
    info: dict[str, str | int] = {"lscpu_excerpt": "\n".join(lscpu.splitlines()[:12]) if lscpu else ""}
    for line in lscpu.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
        info[key] = value.strip()
    nproc = _run(["nproc"])
    if nproc.isdigit():
        info["logical_cpus"] = int(nproc)
    return info


def readme_cpu_snippet(info: dict) -> str:
    model = info.get("model_name", "unknown")
    cpus = info.get("cpu_s", info.get("cpus", "?"))
    threads = info.get("thread_s_per_core", "?")
    logical = info.get("logical_cpus", "?")
    return (
        f"Host CPU: **{model}**, {cpus} cores × {threads} threads/core, "
        f"{logical} logical CPUs (from `lscpu`)."
    )


def write_host_info(path: Path) -> dict:
    info = capture_host_cpu()
    path.write_text(json.dumps(info, indent=2))
    return info
