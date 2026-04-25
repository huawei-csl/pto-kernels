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
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    return {
        "ptr_f32": pto.PtrType(f32),
        "i32": pto.int32,
        "tensor2_f32": pto.TensorType(rank=2, dtype=f32),
        "sub_row4": pto.SubTensorType(shape=[1, mhc_mult], dtype=f32),
        "sub_row16": pto.SubTensorType(shape=[1, mhc_mult2], dtype=f32),
        "sub_row24": pto.SubTensorType(shape=[1, mhc_mult3], dtype=f32),
        "sub_scalar": pto.SubTensorType(shape=[1, 1], dtype=f32),
        "tile_row4": pto.TileBufType(
            shape=[1, mhc_mult],
            valid_shape=[1, mhc_mult],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_row16": pto.TileBufType(
            shape=[1, mhc_mult2],
            valid_shape=[1, mhc_mult2],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_row24": pto.TileBufType(
            shape=[1, mhc_mult3],
            valid_shape=[1, mhc_mult3],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_scalar": pto.TileBufType(
            shape=[1, 1],
            valid_shape=[1, 1],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
    }


def _tensor(ptr_value, rows, cols: int):
    c1 = const(1)
    c_cols = const(cols)
    return pto.as_tensor(
        tensor2_f32,
        ptr=ptr_value,
        shape=[rows, c_cols],
        strides=[c_cols, c1],
    )


def _slice_row4(tensor, row_idx, col_offset):
    c1 = const(1)
    return pto.slice_view(
        sub_row4,
        source=tensor,
        offsets=[row_idx, col_offset],
        sizes=[c1, const(4)],
    )


def _slice_row16(tensor, row_idx, col_offset):
    c1 = const(1)
    return pto.slice_view(
        sub_row16,
        source=tensor,
        offsets=[row_idx, col_offset],
        sizes=[c1, const(16)],
    )


def _slice_row24(tensor, row_idx):
    c0 = const(0)
    c1 = const(1)
    return pto.slice_view(
        sub_row24,
        source=tensor,
        offsets=[row_idx, c0],
        sizes=[c1, const(24)],
    )


def _slice_scalar(tensor, row_idx, col_offset):
    c1 = const(1)
    return pto.slice_view(
        sub_scalar,
        source=tensor,
        offsets=[row_idx, col_offset],
        sizes=[c1, c1],
    )


def _sigmoid_from_logits(logits, neg, exp_tile, denom, out) -> None:
    _muls(logits, -1.0, neg)
    tile.exp(neg, exp_tile)
    _adds(exp_tile, 1.0, denom)
    tile.reciprocal(denom, out)


def _load_constants(mhc_scale_ptr, mhc_base_ptr, mhc_mult3: int, scale0, scale1, scale2, base):
    c0 = const(0)
    scale_t = _tensor(mhc_scale_ptr, const(1), 3)
    base_t = _tensor(mhc_base_ptr, const(1), mhc_mult3)
    pto.load(_slice_scalar(scale_t, c0, c0), scale0)
    pto.load(_slice_scalar(scale_t, c0, const(1)), scale1)
    pto.load(_slice_scalar(scale_t, c0, const(2)), scale2)
    pto.load(_slice_row24(base_t, c0), base)


def build_pre_split_mixes_fwd(
    mhc_mult: int = 4,
    mhc_post_mult_value: float = 2.0,
    mhc_pre_eps: float = 1e-2,
):
    """Build MHC pre-split forward for f32 mix rows."""
    if mhc_mult != 4:
        raise ValueError("pre_split_mixes currently supports mhc_mult=4")

    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    meta_data = lambda: _meta_data(mhc_mult)

    def tilekernels_mhc_pre_split_mixes_fwd_kernel(
        input_mixes_ptr: "ptr_f32",
        mhc_scale_ptr: "ptr_f32",
        mhc_base_ptr: "ptr_f32",
        pre_layer_mix_ptr: "ptr_f32",
        post_layer_mix_ptr: "ptr_f32",
        comb_res_mix_ptr: "ptr_f32",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        num_tokens = s.index_cast(num_tokens_i32)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(num_tokens, nblocks)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, num_tokens)

            input_t = _tensor(input_mixes_ptr, num_tokens, mhc_mult3)
            pre_t = _tensor(pre_layer_mix_ptr, num_tokens, mhc_mult)
            post_t = _tensor(post_layer_mix_ptr, num_tokens, mhc_mult)
            comb_t = _tensor(comb_res_mix_ptr, num_tokens, mhc_mult2)

            input_row = pto.alloc_tile(tile_row24)
            base = pto.alloc_tile(tile_row24)
            scale0 = pto.alloc_tile(tile_scalar)
            scale1 = pto.alloc_tile(tile_scalar)
            scale2 = pto.alloc_tile(tile_scalar)
            scale0_row = pto.alloc_tile(tile_row4)
            scale1_row = pto.alloc_tile(tile_row4)
            scale2_row = pto.alloc_tile(tile_row16)
            logits4 = pto.alloc_tile(tile_row4)
            neg4 = pto.alloc_tile(tile_row4)
            exp4 = pto.alloc_tile(tile_row4)
            denom4 = pto.alloc_tile(tile_row4)
            sig4 = pto.alloc_tile(tile_row4)
            out4 = pto.alloc_tile(tile_row4)
            out16 = pto.alloc_tile(tile_row16)

            _load_constants(
                mhc_scale_ptr, mhc_base_ptr, mhc_mult3, scale0, scale1, scale2, base
            )
            tile.row_expand(scale0, scale0_row)
            tile.row_expand(scale1, scale1_row)
            tile.row_expand(scale2, scale2_row)

            base_pre = tile.subset(base, [c0, c0], [1, mhc_mult])
            base_post = tile.subset(base, [c0, const(mhc_mult)], [1, mhc_mult])
            base_comb = tile.subset(base, [c0, const(mhc_mult * 2)], [1, mhc_mult2])

            for row_idx in pto.range(row_start, row_end, c1):
                pto.load(_slice_row24(input_t, row_idx), input_row)

                input_pre = tile.subset(input_row, [c0, c0], [1, mhc_mult])
                tile.mul(input_pre, scale0_row, logits4)
                tile.add(logits4, base_pre, logits4)
                _sigmoid_from_logits(logits4, neg4, exp4, denom4, sig4)
                _adds(sig4, mhc_pre_eps, out4)
                pto.store(out4, _slice_row4(pre_t, row_idx, c0))

                input_post = tile.subset(input_row, [c0, const(mhc_mult)], [1, mhc_mult])
                tile.mul(input_post, scale1_row, logits4)
                tile.add(logits4, base_post, logits4)
                _sigmoid_from_logits(logits4, neg4, exp4, denom4, sig4)
                _muls(sig4, mhc_post_mult_value, out4)
                pto.store(out4, _slice_row4(post_t, row_idx, c0))

                input_comb = tile.subset(input_row, [c0, const(mhc_mult * 2)], [1, mhc_mult2])
                tile.mul(input_comb, scale2_row, out16)
                tile.add(out16, base_comb, out16)
                pto.store(out16, _slice_row16(comb_t, row_idx, c0))

    tilekernels_mhc_pre_split_mixes_fwd_kernel.__name__ = (
        f"tilekernels_mhc_pre_split_mixes_fwd_m{mhc_mult}"
    )
    return to_ir_module(meta_data=meta_data)(
        tilekernels_mhc_pre_split_mixes_fwd_kernel
    )


def build_pre_split_mixes_bwd(
    mhc_mult: int = 4,
    mhc_post_mult_value: float = 2.0,
):
    """Build correctness-first MHC pre-split backward for f32 mix rows."""
    if mhc_mult != 4:
        raise ValueError("pre_split_mixes currently supports mhc_mult=4")

    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    meta_data = lambda: _meta_data(mhc_mult)

    def tilekernels_mhc_pre_split_mixes_bwd_kernel(
        pre_layer_mix_grad_ptr: "ptr_f32",
        post_layer_mix_grad_ptr: "ptr_f32",
        comb_res_mix_grad_ptr: "ptr_f32",
        input_mixes_ptr: "ptr_f32",
        post_layer_mix_ptr: "ptr_f32",
        mhc_scale_ptr: "ptr_f32",
        mhc_base_ptr: "ptr_f32",
        input_mixes_grad_ptr: "ptr_f32",
        mhc_scale_grad_partial_ptr: "ptr_f32",
        mhc_base_grad_partial_ptr: "ptr_f32",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        num_tokens = s.index_cast(num_tokens_i32)

        with pto.vector_section():
            pre_grad_t = _tensor(pre_layer_mix_grad_ptr, num_tokens, mhc_mult)
            post_grad_t = _tensor(post_layer_mix_grad_ptr, num_tokens, mhc_mult)
            comb_grad_t = _tensor(comb_res_mix_grad_ptr, num_tokens, mhc_mult2)
            input_t = _tensor(input_mixes_ptr, num_tokens, mhc_mult3)
            post_t = _tensor(post_layer_mix_ptr, num_tokens, mhc_mult)
            input_grad_t = _tensor(input_mixes_grad_ptr, num_tokens, mhc_mult3)
            scale_grad_t = _tensor(mhc_scale_grad_partial_ptr, const(1), 3)
            base_grad_t = _tensor(mhc_base_grad_partial_ptr, const(1), mhc_mult3)

            input_row = pto.alloc_tile(tile_row24)
            base = pto.alloc_tile(tile_row24)
            scale0 = pto.alloc_tile(tile_scalar)
            scale1 = pto.alloc_tile(tile_scalar)
            scale2 = pto.alloc_tile(tile_scalar)
            scale0_row = pto.alloc_tile(tile_row4)
            scale1_row = pto.alloc_tile(tile_row4)
            scale2_row = pto.alloc_tile(tile_row16)
            pre_grad = pto.alloc_tile(tile_row4)
            post_grad = pto.alloc_tile(tile_row4)
            comb_grad = pto.alloc_tile(tile_row16)
            post_mix = pto.alloc_tile(tile_row4)
            logits4 = pto.alloc_tile(tile_row4)
            neg4 = pto.alloc_tile(tile_row4)
            exp4 = pto.alloc_tile(tile_row4)
            denom4 = pto.alloc_tile(tile_row4)
            sig4 = pto.alloc_tile(tile_row4)
            one_minus = pto.alloc_tile(tile_row4)
            unscaled4 = pto.alloc_tile(tile_row4)
            scaled4 = pto.alloc_tile(tile_row4)
            scaled16 = pto.alloc_tile(tile_row16)
            terms4 = pto.alloc_tile(tile_row4)
            terms16 = pto.alloc_tile(tile_row16)
            tmp4 = pto.alloc_tile(tile_row4)
            tmp16 = pto.alloc_tile(tile_row16)
            scale_term = pto.alloc_tile(tile_scalar)
            scale0_acc = pto.alloc_tile(tile_scalar)
            scale1_acc = pto.alloc_tile(tile_scalar)
            scale2_acc = pto.alloc_tile(tile_scalar)
            pre_acc = pto.alloc_tile(tile_row4)
            post_acc = pto.alloc_tile(tile_row4)
            comb_acc = pto.alloc_tile(tile_row16)

            _load_constants(
                mhc_scale_ptr, mhc_base_ptr, mhc_mult3, scale0, scale1, scale2, base
            )
            tile.row_expand(scale0, scale0_row)
            tile.row_expand(scale1, scale1_row)
            tile.row_expand(scale2, scale2_row)
            _muls(scale0, 0.0, scale0_acc)
            _muls(scale1, 0.0, scale1_acc)
            _muls(scale2, 0.0, scale2_acc)
            _muls(tile.subset(base, [c0, c0], [1, mhc_mult]), 0.0, pre_acc)
            _muls(tile.subset(base, [c0, const(mhc_mult)], [1, mhc_mult]), 0.0, post_acc)
            _muls(tile.subset(base, [c0, const(mhc_mult * 2)], [1, mhc_mult2]), 0.0, comb_acc)

            base_pre = tile.subset(base, [c0, c0], [1, mhc_mult])
            base_post = tile.subset(base, [c0, const(mhc_mult)], [1, mhc_mult])

            for row_idx in pto.range(c0, num_tokens, c1):
                pto.load(_slice_row24(input_t, row_idx), input_row)
                pto.load(_slice_row4(pre_grad_t, row_idx, c0), pre_grad)
                pto.load(_slice_row4(post_grad_t, row_idx, c0), post_grad)
                pto.load(_slice_row16(comb_grad_t, row_idx, c0), comb_grad)
                pto.load(_slice_row4(post_t, row_idx, c0), post_mix)

                input_pre = tile.subset(input_row, [c0, c0], [1, mhc_mult])
                tile.mul(input_pre, scale0_row, logits4)
                tile.add(logits4, base_pre, logits4)
                _sigmoid_from_logits(logits4, neg4, exp4, denom4, sig4)
                _muls(sig4, -1.0, one_minus)
                _adds(one_minus, 1.0, one_minus)
                tile.mul(sig4, one_minus, unscaled4)
                tile.mul(unscaled4, pre_grad, unscaled4)
                tile.mul(unscaled4, scale0_row, scaled4)
                pto.store(scaled4, _slice_row4(input_grad_t, row_idx, c0))
                tile.add(pre_acc, unscaled4, pre_acc)
                tile.mul(unscaled4, input_pre, terms4)
                tile.row_sum(terms4, tmp4, scale_term)
                tile.add(scale0_acc, scale_term, scale0_acc)

                input_post = tile.subset(input_row, [c0, const(mhc_mult)], [1, mhc_mult])
                _muls(post_mix, 1.0 / mhc_post_mult_value, one_minus)
                _muls(one_minus, -1.0, one_minus)
                _adds(one_minus, 1.0, one_minus)
                tile.mul(post_grad, post_mix, unscaled4)
                tile.mul(unscaled4, one_minus, unscaled4)
                tile.mul(unscaled4, scale1_row, scaled4)
                pto.store(scaled4, _slice_row4(input_grad_t, row_idx, const(mhc_mult)))
                tile.add(post_acc, unscaled4, post_acc)
                tile.mul(unscaled4, input_post, terms4)
                tile.row_sum(terms4, tmp4, scale_term)
                tile.add(scale1_acc, scale_term, scale1_acc)

                input_comb = tile.subset(input_row, [c0, const(mhc_mult * 2)], [1, mhc_mult2])
                tile.mul(comb_grad, scale2_row, scaled16)
                pto.store(scaled16, _slice_row16(input_grad_t, row_idx, const(mhc_mult * 2)))
                tile.add(comb_acc, comb_grad, comb_acc)
                tile.mul(comb_grad, input_comb, terms16)
                tile.row_sum(terms16, tmp16, scale_term)
                tile.add(scale2_acc, scale_term, scale2_acc)

            pto.store(scale0_acc, _slice_scalar(scale_grad_t, c0, c0))
            pto.store(scale1_acc, _slice_scalar(scale_grad_t, c0, const(1)))
            pto.store(scale2_acc, _slice_scalar(scale_grad_t, c0, const(2)))
            pto.store(pre_acc, _slice_row4(base_grad_t, c0, c0))
            pto.store(post_acc, _slice_row4(base_grad_t, c0, const(mhc_mult)))
            pto.store(comb_acc, _slice_row16(base_grad_t, c0, const(mhc_mult * 2)))

    tilekernels_mhc_pre_split_mixes_bwd_kernel.__name__ = (
        f"tilekernels_mhc_pre_split_mixes_bwd_m{mhc_mult}"
    )
    return to_ir_module(meta_data=meta_data)(
        tilekernels_mhc_pre_split_mixes_bwd_kernel
    )
