# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

"""Exercise paired sparse entries, rejection and recovery through the NPU operator."""

import torch
import torch_npu
from pto_kernels import pto_csr_gather


def test_csr_gather_operand_contract(npu_device):
    for dtype in (torch.float16, torch.float32):
        values = torch.tensor([2, 3, 5, 7], dtype=dtype, device=npu_device)
        indices = torch.tensor([3, 0, 2, 1], dtype=torch.int32, device=npu_device)
        x = torch.tensor([11, 13, 17, 19], dtype=dtype, device=npu_device)
        original = [values, indices, x]
        expected = torch.tensor([38, 33, 85, 91], dtype=dtype, device=npu_device)
        offset_inputs = []
        for tensor, sentinel in zip(original, (-23, 0, -29)):
            storage = torch.full((10,), sentinel, dtype=tensor.dtype, device=npu_device)
            storage[3:7].copy_(tensor)
            offset_inputs.append(storage[3:7])

        invalid = [
            ([values[:3], indices, x], "same number of elements"),
            ([torch.cat((values, values)), indices, x], "same number of elements"),
            (
                [
                    values.to(
                        torch.float32 if dtype == torch.float16 else torch.float16
                    ),
                    indices,
                    x,
                ],
                "same dtype",
            ),
            ([values, indices.to(torch.int64), x], "int32"),
            ([values, indices.to(torch.float32), x], "int32"),
            ([values, indices, x[:0]], "nonempty"),
            (
                [values, indices, torch.ones(40961, dtype=dtype, device=npu_device)],
                "maximum supported size",
            ),
        ]
        for position, tensor in enumerate(original):
            args = original.copy()
            args[position] = tensor.cpu()
            invalid.append((args, "must be on NPU" if position == 2 else "same device"))
            storage = torch.zeros(8, dtype=tensor.dtype, device=npu_device)
            storage[::2].copy_(tensor)
            args = original.copy()
            args[position] = storage[::2]
            invalid.append((args, "contiguous"))
            args = original.copy()
            args[position] = torch._neg_view(-tensor)
            if not args[position].is_neg():
                raise RuntimeError("failed to create a lazy negative view")
            invalid.append((args, "resolve_neg"))
            resolved = args.copy()
            resolved[position] = args[position].resolve_neg()
            output = pto_csr_gather(*resolved)
            if not torch.equal(output, expected):
                raise RuntimeError(
                    "gather result differs after resolving a negative view"
                )

        for args, diagnostic in invalid:
            try:
                pto_csr_gather(*args)
            except RuntimeError as error:
                if "csr_gather:" not in str(error) or diagnostic not in str(error):
                    raise
            else:
                raise RuntimeError(f"gather accepted invalid operands for {diagnostic}")
            output = pto_csr_gather(*offset_inputs)
            torch.npu.synchronize()
            if not torch.equal(output, expected):
                raise RuntimeError("gather recovery failed for offset views")

        for dense in (x, x[:0]):
            output = pto_csr_gather(values[:0], indices[:0], dense)
            torch.npu.synchronize()
            if (
                output.numel() != 0
                or output.dtype != dtype
                or output.device != values.device
            ):
                raise RuntimeError("empty gather changed the output contract")

        output = pto_csr_gather(values.view(2, 2), indices, x.view(2, 2))
        if not torch.equal(output, expected.view(2, 2)):
            raise RuntimeError("gather did not preserve flattened operand order")

        for position, tensor in enumerate(original):
            allow_internal_format = torch.npu.config.allow_internal_format
            try:
                torch.npu.config.allow_internal_format = True
                matrix = torch.ones((64, 64), dtype=tensor.dtype, device=npu_device)
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
            args = [
                torch.ones(4096, dtype=dtype, device=npu_device),
                torch.ones(4096, dtype=torch.int32, device=npu_device),
                x,
            ]
            if position == 2:
                args = original.copy()
            args[position] = blocked_view
            try:
                pto_csr_gather(*args)
            except RuntimeError as error:
                if "base NPU format" not in str(error):
                    raise
            else:
                raise RuntimeError("gather accepted internal-format storage")
            args[position] = torch_npu.npu_format_cast(blocked_view, 2)
            output = pto_csr_gather(*args)
            reference = (
                values
                if position == 2
                else torch.full((4096,), 13, dtype=dtype, device=npu_device)
            )
            if not torch.equal(output, reference):
                raise RuntimeError(
                    "gather result differs after converting internal format"
                )

        if torch.npu.device_count() > 1:
            other_device = (values.device.index + 1) % torch.npu.device_count()
            with torch.npu.device(other_device):
                output = pto_csr_gather(*offset_inputs)
                if torch.npu.current_device() != other_device:
                    raise RuntimeError("gather did not restore the caller device")
            if not torch.equal(output, expected):
                raise RuntimeError("gather used the wrong device for its inputs")
