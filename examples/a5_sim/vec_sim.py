#!/usr/bin/env python3
"""Run A5 pure-vector kernels (SiLU, SwiGLU) under Ascend950 simulators.

Usage::

    ./run_msprof.sh --kernel silu --mode correctness --num-elements 128
    ./run_cannsim.sh --kernel swiglu --mode correctness --batch 1 --input-n 256
    python3 vec_sim.py --kernel silu --mode sweep
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

_A5_DIR = Path(__file__).resolve().parent
if str(_A5_DIR) not in sys.path:
    sys.path.insert(0, str(_A5_DIR))

from common.build import build_kernel  # noqa: E402
from common.host_info import capture_host_cpu  # noqa: E402
from common.torch_runtime import (  # noqa: E402
    init_torch_npu,
    stream_ptr,
    sync,
    zeros_npu,
)

DEFAULT_BLOCK_DIM = 8
_DEFAULT_LADDER = _A5_DIR / "configs" / "scale_ladder.json"
_LIBS: dict[str, ctypes.CDLL] = {}


def _vp(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _load_lib(kernel: str) -> ctypes.CDLL:
    if kernel in _LIBS:
        return _LIBS[kernel]
    lib_path = build_kernel(kernel)
    lib = ctypes.CDLL(str(lib_path))
    if kernel == "silu":
        lib.call_kernel.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        lib.call_kernel.restype = None
    elif kernel == "swiglu":
        lib.call_swiglu_kernel.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.call_swiglu_kernel.restype = None
    else:
        raise ValueError(f"unknown kernel: {kernel}")
    _LIBS[kernel] = lib
    return lib


def _ladder_shape(entry: dict, kernel: str) -> tuple[str, int, dict]:
    n_seq = int(entry["n_seq"])
    l_seg = int(entry["l_seg"])
    t = n_seq * l_seg
    label = entry["label"]
    if kernel == "silu":
        return label, t, {"num_elements": t}
    return label, t, {"batch": n_seq, "input_n": 2 * l_seg}


def _make_silu_inputs(num_elements: int, seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    x = torch.randn(num_elements, dtype=torch.float16).to(device)
    y = zeros_npu((num_elements,), torch.float16)
    return {"x": x, "y": y, "T": num_elements}


def _make_swiglu_inputs(batch: int, input_n: int, seed: int, device: str) -> dict:
    if input_n <= 0 or (input_n & 1):
        raise ValueError("input_n must be a positive even integer")
    torch.manual_seed(seed)
    x = torch.randn(batch, input_n, dtype=torch.float16).to(device)
    y = zeros_npu((batch, input_n // 2), torch.float16)
    return {"x": x, "y": y, "T": batch * (input_n // 2), "batch": batch, "input_n": input_n}


def _make_inputs(kernel: str, shape: dict, seed: int, device: str) -> dict:
    if kernel == "silu":
        return _make_silu_inputs(shape["num_elements"], seed, device)
    return _make_swiglu_inputs(shape["batch"], shape["input_n"], seed, device)


def _launch_silu(data: dict, block_dim: int) -> None:
    lib = _load_lib("silu")
    lib.call_kernel(
        block_dim,
        stream_ptr(),
        _vp(data["y"]),
        _vp(data["x"]),
        data["T"],
    )


def _launch_swiglu(data: dict, block_dim: int) -> None:
    lib = _load_lib("swiglu")
    lib.call_swiglu_kernel(
        block_dim,
        stream_ptr(),
        _vp(data["x"]),
        _vp(data["y"]),
        data["batch"],
        data["input_n"],
    )


def _launch(kernel: str, data: dict, block_dim: int) -> None:
    if kernel == "silu":
        _launch_silu(data, block_dim)
    else:
        _launch_swiglu(data, block_dim)


def check_correctness(kernel: str, data: dict) -> tuple[bool, str]:
    _launch(kernel, data, data["block_dim"])
    sync()
    x_cpu = data["x"].cpu()
    out_cpu = data["y"].cpu()
    if kernel == "silu":
        ref = x_cpu * torch.sigmoid(x_cpu.float()).to(torch.float16)
        rtol, atol = 1e-1, 1e-5
    else:
        a, b = x_cpu.chunk(2, dim=-1)
        ref = a * torch.sigmoid(a.float()).to(torch.float16) * b
        rtol, atol = 1e-2, 1e-5
    try:
        torch.testing.assert_close(out_cpu, ref, rtol=rtol, atol=atol)
        return True, "PASS"
    except AssertionError as exc:
        diff = (out_cpu.float() - ref.float()).abs().max().item()
        return False, f"FAIL max_diff={diff:.4e}: {exc}"


def bench_once(kernel: str, data: dict) -> float:
    sync()
    t0 = time.perf_counter()
    _launch(kernel, data, data["block_dim"])
    sync()
    return time.perf_counter() - t0


def _load_ladder(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _run_case(
    kernel: str,
    label: str,
    t: int,
    shape: dict,
    *,
    mode: str,
    device: str,
    seed: int,
    repeat: int,
    check: bool,
    block_dim: int,
) -> dict:
    data = _make_inputs(kernel, shape, seed=seed, device=device)
    data["block_dim"] = block_dim
    row: dict = {
        "label": label,
        "kernel": kernel,
        "T": t,
        "block_dim": block_dim,
        "omp_threads": int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
        **shape,
    }

    if check or mode == "correctness":
        ok, msg = check_correctness(kernel, data)
        row["correctness_pass"] = ok
        row["correctness_msg"] = msg
        if not ok:
            print(f"  [{label}] CORRECTNESS FAIL: {msg}", file=sys.stderr)

    if mode in ("bench", "sweep", "correctness", "compare-tools"):
        times = [bench_once(kernel, data) for _ in range(max(1, repeat))]
        row["sim_wall_s"] = sum(times) / len(times)
        row["sim_wall_ms"] = row["sim_wall_s"] * 1000.0
        row["ms_per_element"] = (row["sim_wall_s"] / t) * 1000.0 if t else None
        if len(times) > 1:
            mean = row["sim_wall_s"]
            row["sim_wall_s_std"] = (
                sum((x - mean) ** 2 for x in times) / len(times)
            ) ** 0.5

    return row


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'label':<16} {'T':>8} {'sim_ms':>12} {'ms/elem':>10} {'ok':>5}")
    print("-" * 56)
    for r in sorted(rows, key=lambda x: x["T"]):
        sim = r.get("sim_wall_ms")
        mpe = r.get("ms_per_element")
        ok = r.get("correctness_pass")
        sim_s = f"{sim:.1f}" if sim is not None else "n/a"
        mpe_s = f"{mpe:.2f}" if mpe is not None else "n/a"
        ok_s = "yes" if ok else ("no" if ok is False else "-")
        print(f"{r.get('label', ''):<16} {r['T']:>8} {sim_s:>12} {mpe_s:>10} {ok_s:>5}")


def _spawn_tool(script: str, args: list[str]) -> dict:
    cmd = [str(_A5_DIR / script), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{script} failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"No JSON output from {script}:\n{proc.stdout}\n{proc.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description="A5 pure-vector simulator driver")
    parser.add_argument("--kernel", choices=("silu", "swiglu"), default="silu")
    parser.add_argument(
        "--mode",
        choices=("correctness", "bench", "sweep", "compare-tools"),
        default="correctness",
    )
    parser.add_argument("--num-elements", type=int, default=None, help="SiLU element count")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--input-n", type=int, default=None, help="SwiGLU input width (even)")
    parser.add_argument("--label", default="custom")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--block-dim", type=int, default=DEFAULT_BLOCK_DIM)
    parser.add_argument("--ladder", type=Path, default=_DEFAULT_LADDER)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    init_torch_npu(args.device)
    _load_lib(args.kernel)

    rows: list[dict] = []
    if args.mode == "sweep":
        for entry in _load_ladder(args.ladder):
            label, t, shape = _ladder_shape(entry, args.kernel)
            rows.append(
                _run_case(
                    args.kernel,
                    label,
                    t,
                    shape,
                    mode="sweep",
                    device=args.device,
                    seed=args.seed,
                    repeat=args.repeat,
                    check=not args.skip_correctness,
                    block_dim=args.block_dim,
                )
            )
        _print_table(rows)
    elif args.mode == "compare-tools":
        if args.kernel == "silu":
            if args.num_elements is None:
                args.num_elements = 128
            shape = {"num_elements": args.num_elements}
            t = args.num_elements
        else:
            if args.input_n is None:
                args.input_n = 256
            shape = {"batch": args.batch, "input_n": args.input_n}
            t = args.batch * (args.input_n // 2)
        base_args = [
            "--kernel",
            args.kernel,
            "--mode",
            "bench",
            "--label",
            args.label,
            "--repeat",
            str(args.repeat),
            "--skip-correctness",
            "--block-dim",
            str(args.block_dim),
        ]
        if args.kernel == "silu":
            base_args += ["--num-elements", str(shape["num_elements"])]
        else:
            base_args += [
                "--batch",
                str(shape["batch"]),
                "--input-n",
                str(shape["input_n"]),
            ]
        msprof = _spawn_tool("run_msprof.sh", base_args + ["--output-json", "/dev/stdout"])
        cannsim = _spawn_tool("run_cannsim.sh", base_args)
        ms_row = msprof.get("results", [msprof])[0]
        cn_row = cannsim.get("results", [cannsim])[0]
        rows = [
            {"tool": "msprof", **ms_row},
            {"tool": "cannsim", **cn_row},
        ]
        if ms_row.get("sim_wall_s") and cn_row.get("sim_wall_s"):
            rows.append(
                {
                    "tool": "ratio_msprof_over_cannsim",
                    "sim_wall_s": ms_row["sim_wall_s"] / cn_row["sim_wall_s"],
                }
            )
        print(json.dumps({"results": rows}, indent=2))
    else:
        if args.kernel == "silu":
            num_elements = args.num_elements if args.num_elements is not None else 128
            shape = {"num_elements": num_elements}
            t = num_elements
        else:
            input_n = args.input_n if args.input_n is not None else 256
            shape = {"batch": args.batch, "input_n": input_n}
            t = args.batch * (input_n // 2)
        rows.append(
            _run_case(
                args.kernel,
                args.label,
                t,
                shape,
                mode=args.mode,
                device=args.device,
                seed=args.seed,
                repeat=args.repeat,
                check=not args.skip_correctness
                and args.mode in ("correctness", "bench"),
                block_dim=args.block_dim,
            )
        )
        if args.mode == "correctness":
            ok = rows[0].get("correctness_pass", False)
            print(rows[0].get("correctness_msg", "FAIL"))
            if not ok:
                raise SystemExit(1)

    host_cpu = capture_host_cpu()
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "kernel": args.kernel,
        "soc_msprof": "Ascend950PR_9599",
        "soc_cannsim": "Ascend950",
        "arch": "dav-c310-vec",
        "host_cpu": host_cpu,
        "results": rows,
    }
    out = json.dumps(payload, indent=2)
    if args.output_json:
        if str(args.output_json) == "/dev/stdout":
            print(out)
        else:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(out)
            print(out)
    elif args.mode != "compare-tools":
        print(out)


if __name__ == "__main__":
    main()
