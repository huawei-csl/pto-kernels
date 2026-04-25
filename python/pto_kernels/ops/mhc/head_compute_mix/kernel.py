from mlir.dialects import arith, pto as _pto_dialect

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s


const = s.const


def _f32(value: float):
    return arith.ConstantOp(pto.float32, value).result


def _adds(src, value: float, dst) -> None:
    _pto_dialect.TAddSOp(src, _f32(value), dst)


def _muls(src, value: float, dst) -> None:
    _pto_dialect.TMulSOp(src, _f32(value), dst)


def _meta_data(mhc_mult: int):
    f32 = pto.float32
    tile_cfg = pto.TileBufConfig()
    row_shape = [1, mhc_mult]
    scalar_shape = [1, 1]
    return {
        "ptr_f32": pto.PtrType(f32),
        "i32": pto.int32,
        "tensor2_f32": pto.TensorType(rank=2, dtype=f32),
        "sub_row": pto.SubTensorType(shape=row_shape, dtype=f32),
        "sub_scalar": pto.SubTensorType(shape=scalar_shape, dtype=f32),
        "tile_row": pto.TileBufType(
            shape=row_shape,
            valid_shape=row_shape,
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_scalar": pto.TileBufType(
            shape=scalar_shape,
            valid_shape=scalar_shape,
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
    }


def _row_tensor(ptr_value, num_tokens, mhc_mult: int):
    c1 = const(1)
    c_mhc = const(mhc_mult)
    return pto.as_tensor(
        tensor2_f32,
        ptr=ptr_value,
        shape=[num_tokens, c_mhc],
        strides=[c_mhc, c1],
    )


def _base_tensor(ptr_value, mhc_mult: int):
    c1 = const(1)
    c_mhc = const(mhc_mult)
    return pto.as_tensor(
        tensor2_f32,
        ptr=ptr_value,
        shape=[c1, c_mhc],
        strides=[c_mhc, c1],
    )


def _scalar_tensor(ptr_value):
    c1 = const(1)
    return pto.as_tensor(
        tensor2_f32,
        ptr=ptr_value,
        shape=[c1, c1],
        strides=[c1, c1],
    )


def _slice_row(tensor, row_idx, mhc_mult: int):
    c0 = const(0)
    c1 = const(1)
    c_mhc = const(mhc_mult)
    return pto.slice_view(
        sub_row,
        source=tensor,
        offsets=[row_idx, c0],
        sizes=[c1, c_mhc],
    )


def _slice_scalar(tensor):
    c0 = const(0)
    c1 = const(1)
    return pto.slice_view(
        sub_scalar,
        source=tensor,
        offsets=[c0, c0],
        sizes=[c1, c1],
    )


def _sigmoid_from_logits(logits, neg, exp_tile, denom, out) -> None:
    _muls(logits, -1.0, neg)
    tile.exp(neg, exp_tile)
    _adds(exp_tile, 1.0, denom)
    tile.reciprocal(denom, out)


def _load_base_and_scale(base_ptr, scale_ptr, mhc_mult: int, base, scale) -> None:
    base_t = _base_tensor(base_ptr, mhc_mult)
    scale_t = _scalar_tensor(scale_ptr)
    pto.load(_slice_row(base_t, const(0), mhc_mult), base)
    pto.load(_slice_scalar(scale_t), scale)


def build_head_compute_mix_fwd(mhc_mult: int = 4, eps: float = 1e-6):
    """Build MHC head-compute forward.

    Computes ``sigmoid(input_mix * mhc_scale[0] + mhc_base) + eps`` for each
    token row. Sigmoid is emitted through PTO tile exp/reciprocal operations so
    local ``ptoas`` does not need scalar ``math.exp`` lowering.
    """
    if mhc_mult <= 0:
        raise ValueError("mhc_mult must be positive")

    meta_data = lambda: _meta_data(mhc_mult)

    def tilekernels_mhc_head_compute_mix_fwd_kernel(
        input_mix_ptr: "ptr_f32",
        mhc_scale_ptr: "ptr_f32",
        mhc_base_ptr: "ptr_f32",
        output_mix_ptr: "ptr_f32",
        num_tokens_i32: "i32",
    ) -> None:
        c1 = const(1)
        num_tokens = s.index_cast(num_tokens_i32)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(num_tokens, nblocks)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, num_tokens)

            input_t = _row_tensor(input_mix_ptr, num_tokens, mhc_mult)
            output_t = _row_tensor(output_mix_ptr, num_tokens, mhc_mult)

            row = pto.alloc_tile(tile_row)
            base = pto.alloc_tile(tile_row)
            scale = pto.alloc_tile(tile_scalar)
            scale_row = pto.alloc_tile(tile_row)
            logits = pto.alloc_tile(tile_row)
            neg = pto.alloc_tile(tile_row)
            exp_tile = pto.alloc_tile(tile_row)
            denom = pto.alloc_tile(tile_row)
            sig = pto.alloc_tile(tile_row)
            out = pto.alloc_tile(tile_row)

            _load_base_and_scale(mhc_base_ptr, mhc_scale_ptr, mhc_mult, base, scale)
            tile.row_expand(scale, scale_row)

            for row_idx in pto.range(row_start, row_end, c1):
                pto.load(_slice_row(input_t, row_idx, mhc_mult), row)
                tile.mul(row, scale_row, logits)
                tile.add(logits, base, logits)
                _sigmoid_from_logits(logits, neg, exp_tile, denom, sig)
                _adds(sig, eps, out)
                pto.store(out, _slice_row(output_t, row_idx, mhc_mult))

    tilekernels_mhc_head_compute_mix_fwd_kernel.__name__ = (
        f"tilekernels_mhc_head_compute_mix_fwd_m{mhc_mult}"
    )
    return to_ir_module(meta_data=meta_data)(
        tilekernels_mhc_head_compute_mix_fwd_kernel
    )


def build_head_compute_mix_bwd(mhc_mult: int = 4):
    """Build correctness-first MHC head-compute backward.

    The generated kernel is configured with one block in the registry, so the
    partial scale/base gradients match the one-row partial validation ABI.
    """
    if mhc_mult <= 0:
        raise ValueError("mhc_mult must be positive")

    meta_data = lambda: _meta_data(mhc_mult)

    def tilekernels_mhc_head_compute_mix_bwd_kernel(
        output_mix_grad_ptr: "ptr_f32",
        input_mix_ptr: "ptr_f32",
        mhc_scale_ptr: "ptr_f32",
        mhc_base_ptr: "ptr_f32",
        input_mix_grad_ptr: "ptr_f32",
        mhc_scale_grad_partial_ptr: "ptr_f32",
        mhc_base_grad_partial_ptr: "ptr_f32",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        num_tokens = s.index_cast(num_tokens_i32)

        with pto.vector_section():
            input_t = _row_tensor(input_mix_ptr, num_tokens, mhc_mult)
            output_grad_t = _row_tensor(output_mix_grad_ptr, num_tokens, mhc_mult)
            input_grad_t = _row_tensor(input_mix_grad_ptr, num_tokens, mhc_mult)
            scale_grad_t = _scalar_tensor(mhc_scale_grad_partial_ptr)
            base_grad_t = _base_tensor(mhc_base_grad_partial_ptr, mhc_mult)

            row = pto.alloc_tile(tile_row)
            out_grad = pto.alloc_tile(tile_row)
            base = pto.alloc_tile(tile_row)
            scale = pto.alloc_tile(tile_scalar)
            scale_row = pto.alloc_tile(tile_row)
            logits = pto.alloc_tile(tile_row)
            neg = pto.alloc_tile(tile_row)
            exp_tile = pto.alloc_tile(tile_row)
            denom = pto.alloc_tile(tile_row)
            sig = pto.alloc_tile(tile_row)
            one_minus_sig = pto.alloc_tile(tile_row)
            grad = pto.alloc_tile(tile_row)
            input_grad = pto.alloc_tile(tile_row)
            scale_terms = pto.alloc_tile(tile_row)
            scale_term = pto.alloc_tile(tile_scalar)
            scale_acc = pto.alloc_tile(tile_scalar)
            base_acc = pto.alloc_tile(tile_row)
            tmp = pto.alloc_tile(tile_row)

            _load_base_and_scale(mhc_base_ptr, mhc_scale_ptr, mhc_mult, base, scale)
            tile.row_expand(scale, scale_row)
            _muls(scale, 0.0, scale_acc)
            _muls(base, 0.0, base_acc)

            for row_idx in pto.range(c0, num_tokens, c1):
                pto.load(_slice_row(input_t, row_idx, mhc_mult), row)
                pto.load(_slice_row(output_grad_t, row_idx, mhc_mult), out_grad)

                tile.mul(row, scale_row, logits)
                tile.add(logits, base, logits)
                _sigmoid_from_logits(logits, neg, exp_tile, denom, sig)

                _muls(sig, -1.0, one_minus_sig)
                _adds(one_minus_sig, 1.0, one_minus_sig)
                tile.mul(sig, one_minus_sig, grad)
                tile.mul(grad, out_grad, grad)

                tile.mul(grad, scale_row, input_grad)
                pto.store(input_grad, _slice_row(input_grad_t, row_idx, mhc_mult))

                tile.add(base_acc, grad, base_acc)
                tile.mul(grad, row, scale_terms)
                tile.row_sum(scale_terms, tmp, scale_term)
                tile.add(scale_acc, scale_term, scale_acc)

            pto.store(scale_acc, _slice_scalar(scale_grad_t))
            pto.store(base_acc, _slice_row(base_grad_t, c0, mhc_mult))

    tilekernels_mhc_head_compute_mix_bwd_kernel.__name__ = (
        f"tilekernels_mhc_head_compute_mix_bwd_m{mhc_mult}"
    )
    return to_ir_module(meta_data=meta_data)(
        tilekernels_mhc_head_compute_mix_bwd_kernel
    )
