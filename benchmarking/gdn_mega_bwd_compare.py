# Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
"""Paired cached backward timings; baseline must use the same Plan/workspace ABI.

Run from the repository root with PYTHONPATH=python:.:tests. Pass the device
kernel.so from a saved build as the positional argument; native launcher and
forward caches are shared. Emits JSONL including every synchronized sample.
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
args = parser.parse_args()
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
    q = torch.randn(1, length, key_heads, 128, device="npu", dtype=torch.float16) * 0.1
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(1, length, heads, 128, device="npu", dtype=torch.float16)
    g = -torch.rand(1, length, heads, device="npu") * 0.1
    beta = torch.rand_like(g).half()
    do = torch.randn_like(v)
    launch = _prepare_gdn_mega(
        q, k, v, g, beta, torch.tensor([0, length], device="npu", dtype=torch.int32)
    )
    cache = torch.empty(
        length // 128, heads, 128, 128, device="npu", dtype=torch.bfloat16
    )
    pto_gdn_mega(**launch, inverse_cache=cache)
    gates = tuple(launch[x] for x in ("g_t", "g_sum", "beta_t"))
    torch.npu.synchronize()
    samples = {"snapshot": [], "current": []}
    for trial in range(14):
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
            if trial >= 3:
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
                cache_dtype="bfloat16",
                samples_ms=samples,
                medians_ms=medians,
                speedup=medians["snapshot"] / medians["current"],
            )
        ),
        flush=True,
    )


try:
    for shape in (
        (2048, 16, 2),
        (8192, 16, 2),
        (8192, 64, 16),
        (8192, 48, 16),
        (8192, 32, 16),
        (8192, 32, 8),
        (32768, 64, 16),
    ):
        measure(*shape)
finally:
    native.stage_launcher = current
