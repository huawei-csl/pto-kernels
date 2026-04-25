from mlir.dialects import arith

from ptodsl import pto, to_ir_module
from ptodsl import scalar as s

from pto_kernels.ops.tilekernels_common import int64_type, load_scalar, ptr, store_scalar


const = s.const


def _meta_data():
    i64 = int64_type()
    return {
        "ptr_i64": ptr(i64),
        "ptr_i32": ptr(pto.int32),
        "i32": pto.int32,
        "i64": i64,
    }


def build_group_count(num_topk: int, num_groups: int):
    """Build correctness-first scalar PTO port of TileKernels moe.group_count.

    This intentionally avoids atomics by running as a single-block expert scan.
    It is not the final high-throughput algorithm, but it is a real PTO kernel
    that matches the TileKernels reference semantics for validation.
    """
    if num_topk <= 0:
        raise ValueError("num_topk must be positive")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")

    def tilekernels_moe_group_count_kernel(
        group_idx_ptr: "ptr_i64",
        out_ptr: "ptr_i32",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_topk = const(num_topk)
        c_groups = const(num_groups)
        i32 = pto.int32
        i64 = int64_type()
        zero_i32 = s.index_cast(c0, i32)
        one_i32 = s.index_cast(c1, i32)
        num_tokens = s.index_cast(num_tokens_i32)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            with pto.if_context(bid == c0):
                for group in pto.range(c0, c_groups, c1):
                    store_scalar(out_ptr, group, zero_i32)
                    for token in pto.range(c0, num_tokens, c1):
                        base = token * c_topk
                        for topk_pos in range(num_topk):
                            elem = base + const(topk_pos)
                            expert64 = load_scalar(i64, group_idx_ptr, elem)
                            expert = s.index_cast(expert64)
                            with pto.if_context(expert == group):
                                current = load_scalar(i32, out_ptr, group)
                                updated = s.wrap_value(
                                    arith.AddIOp(
                                        s._unwrap(current), s._unwrap(one_i32)
                                    ).result
                                )
                                store_scalar(out_ptr, group, updated)

    tilekernels_moe_group_count_kernel.__name__ = (
        f"tilekernels_moe_group_count_k{num_topk}_g{num_groups}"
    )
    return to_ir_module(meta_data=_meta_data)(tilekernels_moe_group_count_kernel)
