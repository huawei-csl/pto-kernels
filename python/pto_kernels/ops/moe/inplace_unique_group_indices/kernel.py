from ptodsl import pto, to_ir_module
from ptodsl import scalar as s

from pto_kernels.ops.tilekernels_common import iand, int64_type, load_scalar, ptr, store_scalar


const = s.const


def _meta_data():
    i64 = int64_type()
    return {
        "ptr_i64": ptr(i64),
        "i32": pto.int32,
        "i64": i64,
    }


def build_inplace_unique_group_indices(num_topk: int, num_groups: int):
    """Build correctness-first per-token duplicate group removal.

    Later occurrences of a non-negative group id are replaced with -1. The
    TileKernels CUDA path uses bit masks; this PTO port keeps the same semantics
    with a small scalar scan over the fixed top-k row.
    """
    if num_topk <= 0:
        raise ValueError("num_topk must be positive")
    if num_groups <= 0 or num_groups > 128:
        raise ValueError("num_groups must be in [1, 128]")

    def tilekernels_moe_inplace_unique_group_indices_kernel(
        group_indices_ptr: "ptr_i64",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_neg1 = const(-1)
        c_topk = const(num_topk)
        i64 = int64_type()
        num_tokens = s.index_cast(num_tokens_i32)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(num_tokens, nblocks)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, num_tokens)

            for row in pto.range(row_start, row_end, c1):
                base = row * c_topk
                for topk_pos in range(num_topk):
                    pos = const(topk_pos)
                    elem = base + pos
                    value64 = load_scalar(i64, group_indices_ptr, elem)
                    value = s.index_cast(value64)
                    duplicate = iand(value < c0, value >= c0)

                    for prev_pos in range(topk_pos):
                        prev64 = load_scalar(
                            i64, group_indices_ptr, base + const(prev_pos)
                        )
                        prev = s.index_cast(prev64)
                        same_non_negative = iand(value >= c0, prev == value)
                        duplicate = s.select(same_non_negative, value >= c0, duplicate)

                    out_value = s.select(duplicate, c_neg1, value)
                    store_scalar(group_indices_ptr, elem, s.index_cast(out_value, i64))

    tilekernels_moe_inplace_unique_group_indices_kernel.__name__ = (
        f"tilekernels_moe_inplace_unique_group_indices_k{num_topk}_g{num_groups}"
    )
    return to_ir_module(meta_data=_meta_data)(
        tilekernels_moe_inplace_unique_group_indices_kernel
    )
