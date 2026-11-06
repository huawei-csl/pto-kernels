"""Check native FP32/FP16/BF16 cache storage against fixed gradient bounds.

--roundtrip isolates rounding error using test-only Torch casts of FP32 caches;
the default exercises PTO writes and reads of actual 16-bit cache allocations.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from test_gdn_mega_bwd import test_backward, test_saved_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roundtrip", action="store_true")
    args = parser.parse_args()
    device = os.environ.get("NPU_DEVICE", "npu:0")
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    cases = next(
        mark.args[1]
        for mark in test_saved_forward.pytestmark
        if mark.name == "parametrize" and mark.args[0].startswith("lengths,")
    )
    results = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for lengths, heads, key_heads, mode in cases:
            row = dict(
                cache_dtype=str(dtype),
                lengths=lengths,
                heads=heads,
                key_heads=key_heads,
                mode=mode,
                gradients=[],
                passed=True,
            )
            try:
                test_backward(
                    lengths,
                    heads,
                    key_heads,
                    mode,
                    torch.float16,
                    device,
                    cache_mode="both",
                    cache_dtype=dtype,
                    error_report=row["gradients"],
                    roundtrip=args.roundtrip,
                )
            except AssertionError as error:
                row.update(passed=False, failure=str(error))
            results.append(row)
            print(
                f"{dtype} L={sum(lengths)} H={heads}/{key_heads} {mode}: "
                f"{'PASS' if row['passed'] else row['failure']}",
                flush=True,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
