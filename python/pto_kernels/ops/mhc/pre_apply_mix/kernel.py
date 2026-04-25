from mlir.dialects import pto as _pto_dialect

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

from pto_kernels.ops.tilekernels_common import ptr, tile_type


const = s.const
_TILE = 1024


def _meta_data():
    bf16 = pto.bfloat16
    f32 = pto.float32
    return {
        "ptr_bf16": ptr(bf16),
        "ptr_f32": ptr(f32),
        "i32": pto.int32,
        "tensor2_bf16": pto.TensorType(rank=2, dtype=bf16),
        "tensor2_f32": pto.TensorType(rank=2, dtype=f32),
        "sub_bf16": pto.SubTensorType(shape=[1, _TILE], dtype=bf16),
        "sub_scalar": pto.SubTensorType(shape=[1, 1], dtype=f32),
        "tile_bf16": tile_type(bf16, [1, _TILE], valid_shape=[1, -1]),
        "tile_f32": tile_type(f32, [1, _TILE], valid_shape=[1, -1]),
        "tile_scalar": tile_type(f32, [1, 1]),
    }


def _tensor_bf16(ptr_value, rows, cols):
    c1 = const(1)
    return pto.as_tensor(
        tensor2_bf16,
        ptr=ptr_value,
        shape=[rows, cols],
        strides=[cols, c1],
    )


def _tensor_f32(ptr_value, rows, cols):
    c1 = const(1)
    return pto.as_tensor(
        tensor2_f32,
        ptr=ptr_value,
        shape=[rows, cols],
        strides=[cols, c1],
    )


def _slice_bf16(tensor, row, col, cols_this):
    c1 = const(1)
    return pto.slice_view(
        sub_bf16,
        source=tensor,
        offsets=[row, col],
        sizes=[c1, cols_this],
    )


def _slice_scalar(tensor, row, col):
    c1 = const(1)
    return pto.slice_view(
        sub_scalar,
        source=tensor,
        offsets=[row, col],
        sizes=[c1, c1],
    )


def build_pre_apply_mix_fwd(mhc_mult: int = 4):
    """Build MHC pre-apply mix forward.

    Computes ``out[token, hidden] = sum_m x[token, m, hidden] * mix[token, m]``
    with bf16 input/output and f32 accumulation.
    """
    if mhc_mult <= 0:
        raise ValueError("mhc_mult must be positive")

    meta_data = _meta_data

    def tilekernels_mhc_pre_apply_mix_fwd_kernel(
        x_ptr: "ptr_bf16",
        mix_ptr: "ptr_f32",
        out_ptr: "ptr_bf16",
        tokens_i32: "i32",
        hidden_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(_TILE)
        c_mhc = const(mhc_mult)
        tokens = s.index_cast(tokens_i32)
        hidden = s.index_cast(hidden_i32)

        x = _tensor_bf16(x_ptr, tokens * c_mhc, hidden)
        mix = _tensor_f32(mix_ptr, tokens, c_mhc)
        out = _tensor_bf16(out_ptr, tokens, hidden)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(tokens, nblocks)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, tokens)
            tiles_per_row = s.ceil_div(hidden, c_tile)

            for row in pto.range(row_start, row_end, c1):
                for tile_idx in pto.range(c0, tiles_per_row, c1):
                    col = tile_idx * c_tile
                    cols_this = s.min_u(c_tile, hidden - col)
                    x_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                    x_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    mix_scalar = pto.alloc_tile(tile_scalar)
                    mix_row = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    term = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    acc = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    out_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)

                    for mhc_idx in range(mhc_mult):
                        c_m = const(mhc_idx)
                        x_row = row * c_mhc + c_m
                        pto.load(_slice_bf16(x, x_row, col, cols_this), x_bf16)
                        _pto_dialect.TCvtOp(x_bf16, x_f32)
                        pto.load(_slice_scalar(mix, row, c_m), mix_scalar)
                        tile.row_expand(mix_scalar, mix_row)
                        tile.mul(x_f32, mix_row, term)
                        if mhc_idx == 0:
                            tile.mov(term, acc)
                        else:
                            tile.add(acc, term, acc)

                    _pto_dialect.TCvtOp(acc, out_bf16)
                    pto.store(out_bf16, _slice_bf16(out, row, col, cols_this))

    tilekernels_mhc_pre_apply_mix_fwd_kernel.__name__ = (
        f"tilekernels_mhc_pre_apply_mix_fwd_m{mhc_mult}"
    )
    return to_ir_module(meta_data=meta_data)(tilekernels_mhc_pre_apply_mix_fwd_kernel)


def build_pre_apply_mix_bwd(mhc_mult: int = 4):
    """Build MHC pre-apply mix backward.

    Computes ``x_grad += mix * out_grad`` and per-token ``mix_grad`` as the
    hidden-axis dot product between ``out_grad`` and ``x``.
    """
    if mhc_mult <= 0:
        raise ValueError("mhc_mult must be positive")

    meta_data = _meta_data

    def tilekernels_mhc_pre_apply_mix_bwd_kernel(
        out_grad_ptr: "ptr_bf16",
        x_ptr: "ptr_bf16",
        mix_ptr: "ptr_f32",
        x_grad_ptr: "ptr_bf16",
        mix_grad_ptr: "ptr_f32",
        tokens_i32: "i32",
        hidden_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(_TILE)
        c_mhc = const(mhc_mult)
        tokens = s.index_cast(tokens_i32)
        hidden = s.index_cast(hidden_i32)

        out_grad = _tensor_bf16(out_grad_ptr, tokens, hidden)
        x = _tensor_bf16(x_ptr, tokens * c_mhc, hidden)
        mix = _tensor_f32(mix_ptr, tokens, c_mhc)
        x_grad = _tensor_bf16(x_grad_ptr, tokens * c_mhc, hidden)
        mix_grad = _tensor_f32(mix_grad_ptr, tokens, c_mhc)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(tokens, nblocks)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, tokens)
            tiles_per_row = s.ceil_div(hidden, c_tile)

            for row in pto.range(row_start, row_end, c1):
                for mhc_idx in range(mhc_mult):
                    c_m = const(mhc_idx)
                    x_row = row * c_mhc + c_m
                    mix_scalar = pto.alloc_tile(tile_scalar)
                    partial = pto.alloc_tile(tile_scalar)
                    acc = pto.alloc_tile(tile_scalar)

                    pto.load(_slice_scalar(mix, row, c_m), mix_scalar)
                    tile.mov(mix_scalar, acc)
                    tile.sub(acc, mix_scalar, acc)

                    for tile_idx in pto.range(c0, tiles_per_row, c1):
                        col = tile_idx * c_tile
                        cols_this = s.min_u(c_tile, hidden - col)
                        og_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                        og_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        x_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                        x_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        xg_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                        xg_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        mix_row = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        term = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        tmp = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        out_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)

                        pto.load(_slice_bf16(out_grad, row, col, cols_this), og_bf16)
                        pto.load(_slice_bf16(x, x_row, col, cols_this), x_bf16)
                        pto.load(_slice_bf16(x_grad, x_row, col, cols_this), xg_bf16)
                        _pto_dialect.TCvtOp(og_bf16, og_f32)
                        _pto_dialect.TCvtOp(x_bf16, x_f32)
                        _pto_dialect.TCvtOp(xg_bf16, xg_f32)

                        tile.row_expand(mix_scalar, mix_row)
                        tile.mul(og_f32, mix_row, term)
                        tile.add(xg_f32, term, xg_f32)
                        _pto_dialect.TCvtOp(xg_f32, out_bf16)
                        pto.store(out_bf16, _slice_bf16(x_grad, x_row, col, cols_this))

                        tile.mul(og_f32, x_f32, term)
                        tile.row_sum(term, tmp, partial)
                        tile.add(acc, partial, acc)

                    pto.store(acc, _slice_scalar(mix_grad, row, c_m))

    tilekernels_mhc_pre_apply_mix_bwd_kernel.__name__ = (
        f"tilekernels_mhc_pre_apply_mix_bwd_m{mhc_mult}"
    )
    return to_ir_module(meta_data=meta_data)(tilekernels_mhc_pre_apply_mix_bwd_kernel)
