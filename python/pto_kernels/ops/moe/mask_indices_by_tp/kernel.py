from ptodsl import pto, to_ir_module
from ptodsl import scalar as s

from pto_kernels.ops.tilekernels_common import int64_type, load_scalar, ptr, store_scalar


const = s.const


def _meta_data():
    i64 = int64_type()
    return {
        "ptr_i64": ptr(i64),
        "i32": pto.int32,
        "i64": i64,
    }


def build_mask_indices_by_tp(
    num_topk: int,
    num_experts: int | None = None,
    num_ep_ranks: int | None = None,
    num_tp_ranks: int | None = None,
):
    """Build scalar PTO port of TileKernels moe.mask_indices_by_tp."""
    del num_experts, num_ep_ranks, num_tp_ranks
    if num_topk <= 0:
        raise ValueError("num_topk must be positive")

    def tilekernels_moe_mask_indices_by_tp_kernel(
        indices_ptr: "ptr_i64",
        masked_indices_ptr: "ptr_i64",
        per_gpu_i32: "i32",
        per_dp_i32: "i32",
        num_tp_ranks_i32: "i32",
        tp_rank_i32: "i32",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_topk = const(num_topk)
        c_neg1 = const(-1)
        i64 = int64_type()

        num_tokens = s.index_cast(num_tokens_i32)
        per_gpu = s.index_cast(per_gpu_i32)
        per_dp = s.index_cast(per_dp_i32)
        num_tp_ranks = s.index_cast(num_tp_ranks_i32)
        tp_rank = s.index_cast(tp_rank_i32)
        total = num_tokens * c_topk

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            elems_per_core = s.ceil_div(total, nblocks)
            elem_start = bid * elems_per_core
            elem_end = s.min_u(elem_start + elems_per_core, total)

            for elem in pto.range(elem_start, elem_end, c1):
                value64 = load_scalar(i64, indices_ptr, elem)
                value = s.index_cast(value64)
                expert_tp_rank = (value // per_gpu) % num_tp_ranks
                wrong_rank = expert_tp_rank != tp_rank

                local_value = value - tp_rank * per_gpu
                dp_rank = local_value // per_dp
                remapped = local_value - dp_rank * (per_dp - per_gpu)

                out_value = s.select(
                    value < c0,
                    c_neg1,
                    s.select(
                        wrong_rank,
                        c_neg1,
                        s.select(remapped < c0, c_neg1, remapped),
                    ),
                )
                store_scalar(masked_indices_ptr, elem, s.index_cast(out_value, i64))

    tilekernels_moe_mask_indices_by_tp_kernel.__name__ = (
        f"tilekernels_moe_mask_indices_by_tp_k{num_topk}"
    )
    return to_ir_module(meta_data=_meta_data)(tilekernels_moe_mask_indices_by_tp_kernel)
