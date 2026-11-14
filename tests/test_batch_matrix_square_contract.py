# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

"""Exercise matrix-batch interpretation through the installed NPU operator."""

import math

import torch
import torch_npu
from pto_kernels import pto_batch_matrix_square


def check_matrix_squares(source, expected):
    output = pto_batch_matrix_square(source)
    if (
        output.shape != source.shape
        or output.dtype != torch.float32
        or output.device != source.device
    ):
        raise RuntimeError("matrix square changed the output shape, dtype or device")
    if not torch.equal(output, expected.to(source.device)):
        raise RuntimeError("matrix square differs from independent matmul")


def test_batch_matrix_square_contract(npu_device):
    # Small integer entries keep products and sums exactly representable in fp32.
    for dtype in (torch.float16, torch.float32):
        for matrix_size in (0, 16, 32, 64, 96, 128):
            rows = (
                torch.arange(matrix_size * matrix_size)
                .remainder(5)
                .reshape(matrix_size, matrix_size)
                - 2
            )
            for batch_shape in ((), (1,), (3,), (2, 3), (2, 1, 3), (0,), (2, 0)):
                count = math.prod(batch_shape)
                distinct = torch.arange(count).reshape(count, 1, 1)
                cpu = (
                    (rows.unsqueeze(0) + distinct)
                    .to(dtype)
                    .reshape(*batch_shape, matrix_size, matrix_size)
                )
                expected = (cpu.to(torch.float64) @ cpu.to(torch.float64)).to(
                    torch.float32
                )
                check_matrix_squares(cpu.to(npu_device), expected)

        empty = torch.empty((65536, 0, 0), dtype=dtype, device=npu_device)
        check_matrix_squares(empty, torch.empty_like(empty, dtype=torch.float32))

        cpu = (
            (torch.arange(6 * 16 * 16).remainder(7) - 3).to(dtype).reshape(2, 3, 16, 16)
        )
        expected = (cpu.to(torch.float64) @ cpu.to(torch.float64)).to(torch.float32)
        source = cpu.to(npu_device)
        storage = torch.full((source.numel() + 2,), -31, dtype=dtype, device=npu_device)
        offset = storage[1:-1].view_as(source)
        offset.copy_(source)
        check_matrix_squares(offset, expected)
        negative = torch._neg_view(offset)
        if not negative.is_neg():
            raise RuntimeError("failed to create a lazy negative view")
        check_matrix_squares(negative, expected)

        invalid = [
            (source.transpose(-1, -2), "contiguous"),
            (source.cpu(), "must be on NPU"),
            (source.to(torch.int32), "fp16 or float32"),
            (torch.zeros((), dtype=dtype, device=npu_device), "at least two"),
            (torch.zeros(16, dtype=dtype, device=npu_device), "at least two"),
            (
                torch.zeros((2, 16, 32), dtype=dtype, device=npu_device),
                "square matrices",
            ),
            (torch.zeros((2, 48, 48), dtype=dtype, device=npu_device), "matrix size"),
            (torch.empty((0, 48, 48), dtype=dtype, device=npu_device), "matrix size"),
            (
                torch.empty((2, 0, 16), dtype=dtype, device=npu_device),
                "square matrices",
            ),
            (
                torch.empty((0, 16, 0), dtype=dtype, device=npu_device),
                "square matrices",
            ),
            (
                torch.empty((0, 0), dtype=torch.int32, device=npu_device),
                "fp16 or float32",
            ),
            (torch.empty((65536, 16, 16), dtype=dtype, device=npu_device), "65535"),
        ]
        for value, diagnostic in invalid:
            try:
                pto_batch_matrix_square(value)
            except RuntimeError as error:
                if "batch_matrix_square:" not in str(error) or diagnostic not in str(
                    error
                ):
                    raise
            else:
                raise RuntimeError(
                    f"matrix square accepted invalid input for {diagnostic}"
                )
            check_matrix_squares(offset, expected)

        matrix = torch.arange(4096).remainder(5).to(dtype).reshape(64, 64)
        allow_internal_format = torch.npu.config.allow_internal_format
        try:
            torch.npu.config.allow_internal_format = True
            dense = matrix.to(npu_device)
            blocked = torch_npu.npu_format_cast(dense, 29)
            blocked_view = blocked.view(2, 2, 32, 32)
            if (
                not blocked_view.is_contiguous()
                or torch_npu.get_npu_format(blocked_view)
                != torch_npu.get_npu_format(blocked)
                or torch_npu.get_npu_format(blocked_view)
                == torch_npu.get_npu_format(dense)
            ):
                raise RuntimeError(
                    "failed to create an internal-format contiguous view"
                )
        finally:
            torch.npu.config.allow_internal_format = allow_internal_format
        try:
            pto_batch_matrix_square(blocked_view)
        except RuntimeError as error:
            if "base NPU format" not in str(error):
                raise
        else:
            raise RuntimeError("matrix square accepted internal-format storage")
        restored = torch_npu.npu_format_cast(blocked_view, 2)
        logical = matrix.view(2, 2, 32, 32).to(torch.float64)
        check_matrix_squares(restored, (logical @ logical).to(torch.float32))

        if torch.npu.device_count() > 1:
            other_device = (source.device.index + 1) % torch.npu.device_count()
            with torch.npu.device(other_device):
                check_matrix_squares(offset, expected)
                if torch.npu.current_device() != other_device:
                    raise RuntimeError(
                        "matrix square did not restore the caller device"
                    )
