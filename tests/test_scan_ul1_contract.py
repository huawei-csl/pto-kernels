# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

"""Run scan input rejection and recovery through the installed NPU operator."""

import torch
import torch_npu
from pto_kernels import pto_scan_ul1


def test_scan_ul1_input_contract(npu_device):
    device = npu_device
    rejected = 0
    completed = 0
    for dtype in (torch.float16, torch.float32):
        storage = torch.full((516,), -7, dtype=dtype, device=device)
        storage[3:259] = 1
        storage[259:] = 5
        negative = torch._neg_view(storage[3:259])
        if not negative.is_neg():
            raise RuntimeError("failed to create a lazy negative view")
        invalid_inputs = [
            (torch.ones(size, dtype=dtype, device=device), "input size")
            for size in (0, 250, 255, 257, 1023, 2304, 4095, 6400, 9215, 12544, 16383)
        ]
        invalid_inputs.extend(
            [
                (storage[3:515:2], "contiguous"),
                (storage[:256].view(16, 16), "only 1D"),
                (storage[0], "only 1D"),
                (negative, "resolve_neg"),
            ]
        )
        allow_internal_format = torch.npu.config.allow_internal_format
        try:
            torch.npu.config.allow_internal_format = True
            matrix = torch.ones((64, 64), dtype=dtype, device=device)
            blocked = torch_npu.npu_format_cast(matrix, 29)
            blocked_view = blocked.view(-1)
            if (
                not blocked_view.is_contiguous()
                or torch_npu.get_npu_format(blocked_view)
                != torch_npu.get_npu_format(blocked)
                or torch_npu.get_npu_format(blocked_view)
                == torch_npu.get_npu_format(matrix)
            ):
                raise RuntimeError(
                    "failed to create an internal-format contiguous view"
                )
        finally:
            torch.npu.config.allow_internal_format = allow_internal_format
        invalid_inputs.append((blocked_view, "base NPU format"))

        for source, diagnostic in invalid_inputs:
            try:
                pto_scan_ul1(source)
            except RuntimeError as error:
                if "scan_ul1: " not in str(error) or diagnostic not in str(error):
                    raise
                rejected += 1
            else:
                raise RuntimeError(f"scan accepted unsupported input {source.shape}")

            source = storage[3:259]
            output = pto_scan_ul1(source)
            torch.npu.synchronize()
            expected = torch.arange(1, 257, dtype=torch.float32, device=device)
            if not torch.equal(output, expected):
                raise RuntimeError("scan recovery failed for a contiguous offset view")
            completed += 1

        source = torch_npu.npu_format_cast(blocked_view, 2)
        output = pto_scan_ul1(source)
        torch.npu.synchronize()
        expected = torch.arange(1, 4097, dtype=torch.float32, device=device)
        if not torch.equal(output, expected):
            raise RuntimeError("scan failed after converting an internal-format view")
        completed += 1

        source = negative.resolve_neg()
        output = pto_scan_ul1(source)
        torch.npu.synchronize()
        expected = -torch.arange(1, 257, dtype=torch.float32, device=device)
        if not torch.equal(output, expected):
            raise RuntimeError("scan failed after resolving a negative view")
        completed += 1

        for size in (256, 1024, 4096, 9216, 16384):
            source = torch.ones(size, dtype=dtype, device=device)
            output = pto_scan_ul1(source)
            torch.npu.synchronize()
            expected = torch.arange(1, size + 1, dtype=torch.float32, device=device)
            if not torch.equal(output, expected):
                raise RuntimeError(f"scan result differs at supported size {size}")
            completed += 1

    print(f"rejected={rejected}, completed={completed}")
