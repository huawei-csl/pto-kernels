from mlir.dialects import arith

from ptodsl import pto, to_ir_module
from ptodsl import scalar as s

from pto_kernels.ops.tilekernels_common import int64_type, load_scalar, ptr, store_scalar


const = s.const


def _meta_data():
    i64 = int64_type()
    f32 = pto.float32
    return {
        "ptr_i64": ptr(i64),
        "ptr_f32": ptr(f32),
        "i32": pto.int32,
        "i64": i64,
        "f32": f32,
    }


def build_aux_fi(num_topk: int, num_experts: int):
    """Build correctness-first scalar PTO port of TileKernels moe.aux_fi."""
    if num_topk <= 0:
        raise ValueError("num_topk must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    def tilekernels_moe_aux_fi_kernel(
        topk_idx_ptr: "ptr_i64",
        out_ptr: "ptr_f32",
        num_aux_topk_i32: "i32",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_topk = const(num_topk)
        c_experts = const(num_experts)
        i32 = pto.int32
        i64 = int64_type()
        f32 = pto.float32
        zero_f32 = s.wrap_value(arith.ConstantOp(f32, 0.0).result)
        one_f32 = s.wrap_value(arith.ConstantOp(f32, 1.0).result)
        num_experts_f32 = s.wrap_value(arith.ConstantOp(f32, float(num_experts)).result)
        num_tokens = s.index_cast(num_tokens_i32)

        denom_i32 = s.wrap_value(
            arith.MulIOp(s._unwrap(num_tokens_i32), s._unwrap(num_aux_topk_i32)).result
        )
        denom_f32 = s.wrap_value(arith.SIToFPOp(f32, s._unwrap(denom_i32)).result)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            with pto.if_context(bid == c0):
                for expert in pto.range(c0, c_experts, c1):
                    store_scalar(out_ptr, expert, zero_f32)
                    for token in pto.range(c0, num_tokens, c1):
                        base = token * c_topk
                        for topk_pos in range(num_topk):
                            elem = base + const(topk_pos)
                            expert64 = load_scalar(i64, topk_idx_ptr, elem)
                            selected_expert = s.index_cast(expert64)
                            with pto.if_context(selected_expert == expert):
                                current = load_scalar(f32, out_ptr, expert)
                                updated = s.wrap_value(
                                    arith.AddFOp(
                                        s._unwrap(current), s._unwrap(one_f32)
                                    ).result
                                )
                                store_scalar(out_ptr, expert, updated)

                    count_f32 = load_scalar(f32, out_ptr, expert)
                    numerator = s.wrap_value(
                        arith.MulFOp(
                            s._unwrap(count_f32), s._unwrap(num_experts_f32)
                        ).result
                    )
                    scaled = s.wrap_value(
                        arith.DivFOp(s._unwrap(numerator), s._unwrap(denom_f32)).result
                    )
                    store_scalar(out_ptr, expert, scaled)

    tilekernels_moe_aux_fi_kernel.__name__ = (
        f"tilekernels_moe_aux_fi_k{num_topk}_e{num_experts}"
    )
    return to_ir_module(meta_data=_meta_data)(tilekernels_moe_aux_fi_kernel)
