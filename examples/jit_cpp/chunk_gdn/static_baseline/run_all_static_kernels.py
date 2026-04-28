"""Run all static PTO kernel tests in this directory (NPU required).

Each test runs in a **subprocess** so PyTorch/NPU RNG and device state match a fresh
``python run_*_static.py`` (in-process ``importlib`` runs were leaving non-deterministic
state that broke later tests, e.g. ``run_wy_fast_static``).
"""
from __future__ import annotations

import subprocess
import sys


def main():
    scripts = [
        "run_chunk_cumsum_static.py",
        "run_chunk_h_static.py",
        "run_chunk_o_static.py",
        "run_scaled_dot_kkt_static.py",
        "run_wy_fast_static.py",
    ]
    here = __file__.rsplit("/", 1)[0] or "."
    for name in scripts:
        print(f"--- {name} ---", flush=True)
        subprocess.run(
            [sys.executable, name],
            cwd=here,
            check=True,
        )
    print("All static kernel tests passed.")


if __name__ == "__main__":
    main()
