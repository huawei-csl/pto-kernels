# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

"""Exercise both operands through the installed matrix-multiplication operator."""

import torch
import torch_npu
from pto_kernels import pto_simple_matmul


def check_matmul(left, right, expected):
    output = pto_simple_matmul(left, right)
    if (
        output.shape != left.shape
        or output.dtype != torch.float32
        or output.device != left.device
    ):
        raise RuntimeError("matmul changed the output shape, dtype or device")
    if not torch.equal(output, expected.to(left.device)):
        raise RuntimeError("matmul differs from the independent product")


def test_simple_matmul_operand_contract(npu_device):
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    for dtype in dtypes:
        # Small integer entries keep products and sums exactly representable in fp32.
        for size in (0, 16, 32, 64, 96, 128):
            a_cpu = (
                (torch.arange(size * size).remainder(7) - 3)
                .to(dtype)
                .reshape(size, size)
            )
            b_cpu = (
                ((torch.arange(size * size) * 3).remainder(11) - 5)
                .to(dtype)
                .reshape(size, size)
            )
            expected = (a_cpu.double() @ b_cpu.double()).float()
            a, b = a_cpu.to(npu_device), b_cpu.to(npu_device)
            check_matmul(a, b, expected)
            check_matmul(a, a, (a_cpu.double() @ a_cpu.double()).float())
            if size == 0:
                check_matmul(torch._neg_view(a), b, expected)
                check_matmul(a, torch._neg_view(b), expected)

        a_cpu = (torch.arange(256).remainder(7) - 3).to(dtype).reshape(16, 16)
        b_cpu = ((torch.arange(256) * 3).remainder(11) - 5).to(dtype).reshape(16, 16)
        expected = (a_cpu.double() @ b_cpu.double()).float()
        left_storage = torch.full((258,), -31, dtype=dtype, device=npu_device)
        right_storage = torch.full((258,), 29, dtype=dtype, device=npu_device)
        a = left_storage[1:-1].view(16, 16)
        b = right_storage[1:-1].view(16, 16)
        a.copy_(a_cpu)
        b.copy_(b_cpu)
        check_matmul(a, b, expected)
        negative_a, negative_b = torch._neg_view(a), torch._neg_view(b)
        if not negative_a.is_neg() or not negative_b.is_neg():
            raise RuntimeError("failed to create lazy negative operand views")
        check_matmul(negative_a, negative_b, expected)

        invalid = [
            (a, b.to(other_dtype), "same dtype")
            for other_dtype in (*dtypes, torch.int32)
            if other_dtype != dtype
        ]
        invalid += [
            (a.cpu(), b, "must be on NPU"),
            (a, b.cpu(), "same NPU device"),
            (a.transpose(0, 1), b, "contiguous"),
            (a, b.transpose(0, 1), "contiguous"),
            (a.unsqueeze(0), b, "two-dimensional"),
            (a, b.unsqueeze(0), "two-dimensional"),
            (a[0], b, "two-dimensional"),
            (a, b[0, 0], "two-dimensional"),
            (a, b[:8], "same size"),
            (a[:8], b, "same size"),
            (a, torch.empty((16, 32), dtype=dtype, device=npu_device), "same size"),
            (a, torch.empty((32, 32), dtype=dtype, device=npu_device), "same size"),
            (
                torch.empty((0, 16), dtype=dtype, device=npu_device),
                torch.empty((0, 16), dtype=dtype, device=npu_device),
                "square matrices",
            ),
            (
                torch.empty((48, 48), dtype=dtype, device=npu_device),
                torch.empty((48, 48), dtype=dtype, device=npu_device),
                "matrix size",
            ),
            (a.to(torch.int32), b.to(torch.int32), "fp16, bf16, or float32"),
            (negative_a, b, "resolve_neg"),
            (a, negative_b, "resolve_neg"),
        ]
        for left, right, diagnostic in invalid:
            try:
                pto_simple_matmul(left, right)
            except RuntimeError as error:
                if "simple_matmul:" not in str(error) or diagnostic not in str(error):
                    raise
            else:
                raise RuntimeError(f"matmul accepted invalid operands for {diagnostic}")
            check_matmul(a, b, expected)
        check_matmul(negative_a.resolve_neg(), b, -expected)
        check_matmul(a, negative_b.resolve_neg(), -expected)

        matrix = (torch.arange(4096).remainder(7) - 3).to(dtype).reshape(64, 64)
        other = ((torch.arange(4096) * 3).remainder(11) - 5).to(dtype).reshape(64, 64)
        product = (matrix.double() @ other.double()).float()
        left, right = matrix.to(npu_device), other.to(npu_device)
        allow_internal_format = torch.npu.config.allow_internal_format
        try:
            torch.npu.config.allow_internal_format = True
            blocked_left = torch_npu.npu_format_cast(left, 29)
            blocked_right = torch_npu.npu_format_cast(right, 29)
            for blocked in (blocked_left, blocked_right):
                if (
                    not blocked.is_contiguous()
                    or torch_npu.get_npu_format(blocked) != 29
                ):
                    raise RuntimeError(
                        "failed to create contiguous internal-format operands"
                    )
        finally:
            torch.npu.config.allow_internal_format = allow_internal_format
        for operands in ((blocked_left, right), (left, blocked_right)):
            try:
                pto_simple_matmul(*operands)
            except RuntimeError as error:
                if "base NPU formats" not in str(error):
                    raise
            else:
                raise RuntimeError("matmul accepted an internal-format operand")
            check_matmul(left, right, product)
        check_matmul(
            torch_npu.npu_format_cast(blocked_left, 2),
            torch_npu.npu_format_cast(blocked_right, 2),
            product,
        )

        if torch.npu.device_count() > 1:
            other_device = (a.device.index + 1) % torch.npu.device_count()
            with torch.npu.device(other_device):
                check_matmul(a, b, expected)
                if torch.npu.current_device() != other_device:
                    raise RuntimeError("matmul did not restore the caller device")
                for operands in (
                    (a.to(f"npu:{other_device}"), b),
                    (a, b.to(f"npu:{other_device}")),
                ):
                    try:
                        pto_simple_matmul(*operands)
                    except RuntimeError as error:
                        if "same NPU device" not in str(error):
                            raise
                    else:
                        raise RuntimeError(
                            "matmul accepted operands on different devices"
                        )
                    check_matmul(a, b, expected)
