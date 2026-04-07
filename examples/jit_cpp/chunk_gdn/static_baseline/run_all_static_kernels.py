"""Run all static PTO kernel tests in this directory (NPU required)."""
from __future__ import annotations

import importlib


def main():
    modules = [
        "run_chunk_cumsum_static",
        "run_chunk_h_static",
        "run_chunk_o_static",
        "run_scaled_dot_kkt_static",
        "run_wy_fast_static",
    ]
    for name in modules:
        print(f"--- {name} ---")
        m = importlib.import_module(name)
        m.main()
    print("All static kernel tests passed.")


if __name__ == "__main__":
    main()
