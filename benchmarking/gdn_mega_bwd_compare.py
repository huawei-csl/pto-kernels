# Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
"""Paired backward timings; baseline must use the same Plan/workspace ABI.

Run from the repository root with PYTHONPATH=python:.:tests. Pass the device
kernel.so from a saved build as the positional argument; native launcher and
forward caches are shared for FP16. FP32 measures full recomputation.
Emits JSONL including every synchronized sample; --quick is for iteration.
"""
import argparse
import hashlib
import ctypes, json, os, statistics, time
from pathlib import Path
import torch, torch_npu
from csrc.kernel.gdn_mega_bwd.backward import backward
from csrc.kernel.gdn_mega_bwd.binding import build
from test_gdn_mega import _prepare_gdn_mega, pto_gdn_mega

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("baseline", type=Path)
parser.add_argument(
    "--quick",
    action="store_true",
    help="Two shapes, one warmup and three measured pairs",
)
parser.add_argument(
    "--shape", nargs=3, type=int, action="append", metavar=("TOKENS", "HV", "HK")
)
parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
parser.add_argument("--warmup", type=int)
parser.add_argument("--repeats", type=int)
args = parser.parse_args()
warmup = args.warmup if args.warmup is not None else (1 if args.quick else 3)
repeats = args.repeats if args.repeats is not None else (3 if args.quick else 11)
if warmup < 0 or repeats < 1:
    parser.error("warmup must be nonnegative and repeats must be positive")
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = False
native = build()
current = native.stage_launcher
snapshot = {"device": str(args.baseline.resolve())}
old_library = ctypes.CDLL(snapshot["device"], mode=os.RTLD_LOCAL | os.RTLD_DEEPBIND)
old = ctypes.cast(old_library.launch_stages, ctypes.c_void_p)
libraries = {"snapshot": snapshot["device"], "current": native.device_library._name}
print(
    json.dumps(
        dict(
            libraries=libraries,
            sha256={
                key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for key, path in libraries.items()
            },
            layout_sha256=hashlib.sha256(
                Path("csrc/kernel/gdn_mega_bwd/layout.h").read_bytes()
            ).hexdigest(),
        )
    ),
    flush=True,
)


def measure(length, heads, key_heads):
    torch.manual_seed(42)
    q = (
        torch.randn(
            1, length, key_heads, 128, device="npu", dtype=getattr(torch, args.dtype)
        )
        * 0.1
    )
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(
        1, length, heads, 128, device="npu", dtype=getattr(torch, args.dtype)
    )
    g = -torch.rand(1, length, heads, device="npu") * 0.1
    beta = torch.rand_like(g).to(q.dtype)
    do = torch.randn_like(v)
    cache, gates = None, None
    if args.dtype == "float16":
        launch = _prepare_gdn_mega(
            q, k, v, g, beta, torch.tensor([0, length], device="npu", dtype=torch.int32)
        )
        cache = torch.empty(
            (length + 127) // 128, heads, 128, 128, device="npu", dtype=torch.bfloat16
        )
        pto_gdn_mega(**launch, inverse_cache=cache)
        gates = tuple(launch[x] for x in ("g_t", "g_sum", "beta_t"))
    torch.npu.synchronize()
    samples = {"snapshot": [], "current": []}
    for trial in range(warmup + repeats):
        modes = [("snapshot", old), ("current", current)]
        if trial % 2:
            modes.reverse()
        for label, pointer in modes:
            native.stage_launcher = pointer
            start = time.perf_counter()
            result = backward(
                q, k, v, g, beta, do, inverse_cache=cache, gate_cache=gates
            )
            torch.npu.synchronize()
            if trial >= warmup:
                samples[label].append((time.perf_counter() - start) * 1000)
            if trial == 0:
                assert all(torch.isfinite(x).all() for x in result)
                torch.npu.synchronize()
    assert all(torch.isfinite(x).all() for x in result)
    medians = {k: statistics.median(v) for k, v in samples.items()}
    print(
        json.dumps(
            dict(
                tokens=length,
                heads=heads,
                key_heads=key_heads,
                dtype=args.dtype,
                cache_dtype="bfloat16" if cache is not None else None,
                warmup=warmup,
                repeats=repeats,
                samples_ms=samples,
                medians_ms=medians,
                speedup=medians["snapshot"] / medians["current"],
            )
        ),
        flush=True,
    )


shapes = args.shape or (
    [(256, 8, 2), (513, 8, 2)]
    if args.dtype == "float32"
    else (
        [(2048, 16, 2), (8192, 64, 16)]
        if args.quick
        else [
            (2048, 16, 2),
            (8192, 16, 2),
            (8192, 64, 16),
            (8192, 48, 16),
            (8192, 32, 16),
            (8192, 32, 8),
            (32768, 64, 16),
        ]
    )
)
try:
    for shape in shapes:
        measure(*shape)
finally:
    native.stage_launcher = current
