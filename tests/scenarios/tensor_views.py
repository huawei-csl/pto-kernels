"""Check contiguous offset views with `python tests/scenarios/tensor_views.py`.

Requires the default GDN_H=16, GDN_C=128 build and an Ascend NPU.
"""

import torch
import torch_npu  # noqa: F401

from pto_kernels import pto_abs, pto_gdn_chunk_cumsum, pto_swiglu


def check(name, actual, expected, **tolerance):
    actual = actual.cpu()
    if not torch.allclose(actual, expected, **tolerance):
        error = (actual.float() - expected.float()).abs().max().item()
        raise RuntimeError(f"{name}: max absolute error {error}")
    print(f"passed {name}")


def main():
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False

    for dtype in (torch.float16, torch.float32):
        backing = torch.arange(512, dtype=dtype, device="npu") - 300
        view = backing[128:384]
        check(f"abs offset {dtype}", pto_abs(view), view.cpu().abs())
        check(f"abs standalone {dtype}", pto_abs(view.clone()), view.cpu().abs())

    backing = torch.linspace(-2, 2, 4 * 256, dtype=torch.float16).reshape(4, 256)
    view = backing.npu()[1:3]
    gate, up = view.cpu().float().chunk(2, dim=-1)
    expected = (gate * gate.sigmoid() * up).half()
    check("swiglu offset", pto_swiglu(view), expected, rtol=1e-2, atol=1e-5)

    # A sliced metadata tensor must preserve the same sequence boundaries.
    gates = torch.arange(384 * 16, dtype=torch.float32).reshape(384, 16) / 65536
    gates_npu = torch.cat((torch.full((128, 16), -1.0), gates)).npu()[128:]
    boundaries = torch.tensor([0, 0, 128, 384], dtype=torch.int32).npu()[1:]
    expected = torch.cat([chunk.cumsum(0) for chunk in gates.split(128)])
    result = pto_gdn_chunk_cumsum(
        gates_npu, batch_size=2, seq_len=0, cu_seqlens=boundaries
    )
    check("gdn offset inputs and metadata", result, expected, rtol=1e-4, atol=1e-5)
    check(
        "gdn output consumed by abs",
        pto_abs(result),
        expected.abs(),
        rtol=1e-4,
        atol=1e-5,
    )


if __name__ == "__main__":
    main()
