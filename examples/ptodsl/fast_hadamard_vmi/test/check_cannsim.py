#!/usr/bin/env python3
"""cannsim entry: set argv and call the runner's pyACL correctness check so the
exact `run_hadamard_vmi.py --check` path (make_full(n).compile() -> per-N kernel
-> launch -> D2H -> compare vs golden) is exercised under cannsim.

Overridable via env: HAD_N, HAD_BATCH, HAD_GRID (defaults: N=256, batch=smallest
valid = grid*65536/N, grid=1)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.abspath(os.path.join(HERE, ".."))
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

N = os.environ.get("HAD_N", "256")
GRID = os.environ.get("HAD_GRID", "1")
argv = ["run_hadamard_vmi.py", "--check", "--n", N, "--grid", GRID]
if os.environ.get("HAD_BATCH"):
    argv += ["--batch", os.environ["HAD_BATCH"]]
sys.argv = argv

import run_hadamard_vmi  # noqa: E402
try:
    run_hadamard_vmi.main()
except SystemExit as e:
    os._exit(int(e.code or 0))
os._exit(0)
