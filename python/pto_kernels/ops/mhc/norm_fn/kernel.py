from mlir.dialects import arith, pto as _pto_dialect

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

from pto_kernels.ops.tilekernels_common import ptr, tile_type


const = s.const
_TILE = 1024


def _f32(value: float):
    return arith.ConstantOp(pto.float32, float(value)).result


def _adds(src, value: float, dst) -> None:
    _pto_dialect.TAddSOp(src, _f32(value), dst)


def _muls(src, value: float, dst) -> None:
    _pto_dialect.TMulSOp(src, _f32(value), dst)


def _cvt(src, dst) -> None:
    _pto_dialect.TCvtOp(src, dst)


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
        "sub_f32": pto.SubTensorType(shape=[1, _TILE], dtype=f32),
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


def _slice_f32(tensor, row, col, cols_this):
    c1 = const(1)
    return pto.slice_view(
        sub_f32,
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


def build_pre_norm_fn_fwd(
    hidden_size: int,
    mhc_mult: int = 4,
    eps: float = 1e-6,
):
    """Build baseline MHC pre-norm forward without optional norm-weight merge."""
    if mhc_mult != 4:
        raise ValueError("pre_norm_fn forward currently supports mhc_mult=4")
    total_hidden = mhc_mult * hidden_size
    if total_hidden <= 0 or total_hidden % _TILE != 0:
        raise ValueError(f"mhc_mult * hidden_size must be a positive multiple of {_TILE}")

    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    num_tiles = total_hidden // _TILE
    inv_total_hidden = 1.0 / float(total_hidden)
    meta_data = _meta_data

    def tilekernels_mhc_pre_norm_fn_fwd_kernel(
        residual_ptr: "ptr_bf16",
        mhc_fn_ptr: "ptr_f32",
        output_ptr: "ptr_f32",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_total_hidden = const(total_hidden)
        c_out_cols = const(mhc_mult3)
        num_tokens = s.index_cast(num_tokens_i32)

        residual = _tensor_bf16(residual_ptr, num_tokens, c_total_hidden)
        mhc_fn = _tensor_f32(mhc_fn_ptr, c_out_cols, c_total_hidden)
        output = _tensor_f32(output_ptr, num_tokens, c_out_cols)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(num_tokens, nblocks)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, num_tokens)

            for row in pto.range(row_start, row_end, c1):
                sqsum = pto.alloc_tile(tile_scalar)
                for tile_idx in range(num_tiles):
                    col = const(tile_idx * _TILE)
                    x_bf16 = pto.alloc_tile(tile_bf16, valid_col=const(_TILE))
                    x_f32 = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                    tmp = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                    tmp2 = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                    partial = pto.alloc_tile(tile_scalar)
                    pto.load(_slice_bf16(residual, row, col, const(_TILE)), x_bf16)
                    _cvt(x_bf16, x_f32)
                    tile.mul(x_f32, x_f32, tmp)
                    tile.row_sum(tmp, tmp2, partial)
                    if tile_idx == 0:
                        tile.mov(partial, sqsum)
                    else:
                        tile.add(sqsum, partial, sqsum)

                _muls(sqsum, inv_total_hidden, sqsum)
                _adds(sqsum, eps, sqsum)
                tile.rsqrt(sqsum, sqsum)

                for out_col in range(mhc_mult3):
                    c_out = const(out_col)
                    dot = pto.alloc_tile(tile_scalar)
                    for tile_idx in range(num_tiles):
                        col = const(tile_idx * _TILE)
                        x_bf16 = pto.alloc_tile(tile_bf16, valid_col=const(_TILE))
                        x_f32 = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                        fn = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                        tmp = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                        tmp2 = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                        partial = pto.alloc_tile(tile_scalar)
                        pto.load(_slice_bf16(residual, row, col, const(_TILE)), x_bf16)
                        pto.load(_slice_f32(mhc_fn, c_out, col, const(_TILE)), fn)
                        _cvt(x_bf16, x_f32)
                        tile.mul(x_f32, fn, tmp)
                        tile.row_sum(tmp, tmp2, partial)
                        if tile_idx == 0:
                            tile.mov(partial, dot)
                        else:
                            tile.add(dot, partial, dot)

                    tile.mul(dot, sqsum, dot)
                    pto.store(dot, _slice_scalar(output, row, c_out))

    tilekernels_mhc_pre_norm_fn_fwd_kernel.__name__ = (
        f"tilekernels_mhc_pre_norm_fn_fwd_m{mhc_mult}_h{hidden_size}"
    )
    return to_ir_module(meta_data=meta_data)(tilekernels_mhc_pre_norm_fn_fwd_kernel)


def build_fn_normw_merge_fwd(hidden_size: int, mhc_mult: int = 4):
    """Build optional MHC FN/norm-weight merge forward."""
    if mhc_mult != 4:
        raise ValueError("fn_normw merge currently supports mhc_mult=4")
    total_hidden = mhc_mult * hidden_size
    if total_hidden <= 0 or total_hidden % _TILE != 0:
        raise ValueError(f"mhc_mult * hidden_size must be a positive multiple of {_TILE}")

    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    num_tiles = total_hidden // _TILE
    meta_data = _meta_data

    def tilekernels_mhc_fn_normw_merge_fwd_kernel(
        fn_ptr: "ptr_f32",
        normw_ptr: "ptr_f32",
        out_fn_ptr: "ptr_f32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_total_hidden = const(total_hidden)
        c_out_cols = const(mhc_mult3)
        fn = _tensor_f32(fn_ptr, c_out_cols, c_total_hidden)
        normw = _tensor_f32(normw_ptr, c1, c_total_hidden)
        out_fn = _tensor_f32(out_fn_ptr, c_out_cols, c_total_hidden)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(c_out_cols, nblocks)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, c_out_cols)

            for row in pto.range(row_start, row_end, c1):
                for tile_idx in range(num_tiles):
                    col = const(tile_idx * _TILE)
                    fn_tile = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                    normw_tile = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                    out_tile = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                    pto.load(_slice_f32(fn, row, col, const(_TILE)), fn_tile)
                    pto.load(_slice_f32(normw, c0, col, const(_TILE)), normw_tile)
                    tile.mul(fn_tile, normw_tile, out_tile)
                    pto.store(out_tile, _slice_f32(out_fn, row, col, const(_TILE)))

    tilekernels_mhc_fn_normw_merge_fwd_kernel.__name__ = (
        f"tilekernels_mhc_fn_normw_merge_fwd_m{mhc_mult}_h{hidden_size}"
    )
    return to_ir_module(meta_data=meta_data)(tilekernels_mhc_fn_normw_merge_fwd_kernel)


def build_fn_normw_merge_bwd(hidden_size: int, mhc_mult: int = 4):
    """Build optional MHC FN/norm-weight merge backward."""
    if mhc_mult != 4:
        raise ValueError("fn_normw merge currently supports mhc_mult=4")
    total_hidden = mhc_mult * hidden_size
    if total_hidden <= 0 or total_hidden % _TILE != 0:
        raise ValueError(f"mhc_mult * hidden_size must be a positive multiple of {_TILE}")

    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    num_tiles = total_hidden // _TILE
    meta_data = _meta_data

    def tilekernels_mhc_fn_normw_merge_bwd_kernel(
        fn_ptr: "ptr_f32",
        normw_ptr: "ptr_f32",
        out_fn_grad_ptr: "ptr_f32",
        fn_grad_ptr: "ptr_f32",
        normw_grad_ptr: "ptr_f32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_total_hidden = const(total_hidden)
        c_out_cols = const(mhc_mult3)
        fn = _tensor_f32(fn_ptr, c_out_cols, c_total_hidden)
        normw = _tensor_f32(normw_ptr, c1, c_total_hidden)
        out_fn_grad = _tensor_f32(out_fn_grad_ptr, c_out_cols, c_total_hidden)
        fn_grad = _tensor_f32(fn_grad_ptr, c_out_cols, c_total_hidden)
        normw_grad = _tensor_f32(normw_grad_ptr, c1, c_total_hidden)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            tiles_per_core = s.ceil_div(const(num_tiles), nblocks)
            tile_start = bid * tiles_per_core
            tile_end = s.min_u(tile_start + tiles_per_core, const(num_tiles))

            for dyn_tile in pto.range(tile_start, tile_end, c1):
                col = dyn_tile * const(_TILE)
                normw_tile = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                normw_grad_acc = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                pto.load(_slice_f32(normw, c0, col, const(_TILE)), normw_tile)
                pto.load(_slice_f32(normw_grad, c0, col, const(_TILE)), normw_grad_acc)

                for row_idx in range(mhc_mult3):
                    row = const(row_idx)
                    fn_tile = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                    out_grad_tile = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                    fn_grad_tile = pto.alloc_tile(tile_f32, valid_col=const(_TILE))
                    term = pto.alloc_tile(tile_f32, valid_col=const(_TILE))

                    pto.load(_slice_f32(fn, row, col, const(_TILE)), fn_tile)
                    pto.load(_slice_f32(out_fn_grad, row, col, const(_TILE)), out_grad_tile)
                    pto.load(_slice_f32(fn_grad, row, col, const(_TILE)), fn_grad_tile)

                    tile.mul(out_grad_tile, normw_tile, term)
                    tile.add(fn_grad_tile, term, fn_grad_tile)
                    pto.store(fn_grad_tile, _slice_f32(fn_grad, row, col, const(_TILE)))

                    tile.mul(out_grad_tile, fn_tile, term)
                    tile.add(normw_grad_acc, term, normw_grad_acc)

                pto.store(normw_grad_acc, _slice_f32(normw_grad, c0, col, const(_TILE)))

    tilekernels_mhc_fn_normw_merge_bwd_kernel.__name__ = (
        f"tilekernels_mhc_fn_normw_merge_bwd_m{mhc_mult}_h{hidden_size}"
    )
    return to_ir_module(meta_data=meta_data)(tilekernels_mhc_fn_normw_merge_bwd_kernel)
