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


def build_post_fwd(mhc_mult: int = 4):
    """Build MHC post forward.

    Computes ``out[token, out_mhc, h] = post_mix[token, out_mhc] * x[token, h]
    + sum_in comb[token, in_mhc, out_mhc] * residual[token, in_mhc, h]``.
    """
    if mhc_mult <= 0:
        raise ValueError("mhc_mult must be positive")

    meta_data = _meta_data

    def tilekernels_mhc_post_fwd_kernel(
        comb_res_mix_ptr: "ptr_f32",
        residual_ptr: "ptr_bf16",
        post_layer_mix_ptr: "ptr_f32",
        x_ptr: "ptr_bf16",
        out_ptr: "ptr_bf16",
        tokens_i32: "i32",
        hidden_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(_TILE)
        c_mhc = const(mhc_mult)
        c_mhc2 = const(mhc_mult * mhc_mult)
        tokens = s.index_cast(tokens_i32)
        hidden = s.index_cast(hidden_i32)

        comb = _tensor_f32(comb_res_mix_ptr, tokens, c_mhc2)
        residual = _tensor_bf16(residual_ptr, tokens * c_mhc, hidden)
        post = _tensor_f32(post_layer_mix_ptr, tokens, c_mhc)
        x = _tensor_bf16(x_ptr, tokens, hidden)
        out = _tensor_bf16(out_ptr, tokens * c_mhc, hidden)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(tokens, nblocks)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, tokens)
            tiles_per_row = s.ceil_div(hidden, c_tile)

            for row in pto.range(row_start, row_end, c1):
                for out_mhc in range(mhc_mult):
                    c_out = const(out_mhc)
                    out_row = row * c_mhc + c_out
                    post_scalar = pto.alloc_tile(tile_scalar)
                    pto.load(_slice_scalar(post, row, c_out), post_scalar)

                    for tile_idx in pto.range(c0, tiles_per_row, c1):
                        col = tile_idx * c_tile
                        cols_this = s.min_u(c_tile, hidden - col)
                        x_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                        x_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        residual_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                        residual_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        scale = pto.alloc_tile(tile_scalar)
                        scale_row = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        acc = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        term = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        out_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)

                        pto.load(_slice_bf16(x, row, col, cols_this), x_bf16)
                        _pto_dialect.TCvtOp(x_bf16, x_f32)
                        tile.row_expand(post_scalar, scale_row)
                        tile.mul(x_f32, scale_row, acc)

                        for in_mhc in range(mhc_mult):
                            c_in = const(in_mhc)
                            residual_row = row * c_mhc + c_in
                            comb_col = const(in_mhc * mhc_mult + out_mhc)
                            pto.load(
                                _slice_bf16(residual, residual_row, col, cols_this),
                                residual_bf16,
                            )
                            _pto_dialect.TCvtOp(residual_bf16, residual_f32)
                            pto.load(_slice_scalar(comb, row, comb_col), scale)
                            tile.row_expand(scale, scale_row)
                            tile.mul(residual_f32, scale_row, term)
                            tile.add(acc, term, acc)

                        _pto_dialect.TCvtOp(acc, out_bf16)
                        pto.store(out_bf16, _slice_bf16(out, out_row, col, cols_this))

    tilekernels_mhc_post_fwd_kernel.__name__ = (
        f"tilekernels_mhc_post_fwd_m{mhc_mult}"
    )
    return to_ir_module(meta_data=meta_data)(tilekernels_mhc_post_fwd_kernel)


def build_post_bwd(mhc_mult: int = 4):
    """Build correctness-first MHC post backward for ``mhc_mult=4``."""
    if mhc_mult != 4:
        raise ValueError("post backward currently supports mhc_mult=4")

    meta_data = _meta_data

    def tilekernels_mhc_post_bwd_kernel(
        out_grad_ptr: "ptr_bf16",
        comb_res_mix_ptr: "ptr_f32",
        residual_ptr: "ptr_bf16",
        post_layer_mix_ptr: "ptr_f32",
        x_ptr: "ptr_bf16",
        comb_res_mix_grad_ptr: "ptr_f32",
        residual_grad_ptr: "ptr_bf16",
        post_layer_mix_grad_ptr: "ptr_f32",
        x_grad_ptr: "ptr_bf16",
        tokens_i32: "i32",
        hidden_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(_TILE)
        c_mhc = const(4)
        c_mhc2 = const(16)
        tokens = s.index_cast(tokens_i32)
        hidden = s.index_cast(hidden_i32)

        out_grad = _tensor_bf16(out_grad_ptr, tokens * c_mhc, hidden)
        comb = _tensor_f32(comb_res_mix_ptr, tokens, c_mhc2)
        residual = _tensor_bf16(residual_ptr, tokens * c_mhc, hidden)
        post = _tensor_f32(post_layer_mix_ptr, tokens, c_mhc)
        x = _tensor_bf16(x_ptr, tokens, hidden)
        comb_grad = _tensor_f32(comb_res_mix_grad_ptr, tokens, c_mhc2)
        residual_grad = _tensor_bf16(residual_grad_ptr, tokens * c_mhc, hidden)
        post_grad = _tensor_f32(post_layer_mix_grad_ptr, tokens, c_mhc)
        x_grad = _tensor_bf16(x_grad_ptr, tokens, hidden)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(tokens, nblocks)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, tokens)
            tiles_per_row = s.ceil_div(hidden, c_tile)

            for row in pto.range(row_start, row_end, c1):
                for out_mhc in range(4):
                    c_out = const(out_mhc)
                    og_row = row * c_mhc + c_out
                    post_acc = pto.alloc_tile(tile_scalar)
                    pto.load(_slice_scalar(post, row, c_out), post_acc)
                    tile.sub(post_acc, post_acc, post_acc)

                    for tile_idx in pto.range(c0, tiles_per_row, c1):
                        col = tile_idx * c_tile
                        cols_this = s.min_u(c_tile, hidden - col)
                        og_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                        og_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        x_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                        x_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        term = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        tmp = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        partial = pto.alloc_tile(tile_scalar)

                        pto.load(_slice_bf16(out_grad, og_row, col, cols_this), og_bf16)
                        pto.load(_slice_bf16(x, row, col, cols_this), x_bf16)
                        _pto_dialect.TCvtOp(og_bf16, og_f32)
                        _pto_dialect.TCvtOp(x_bf16, x_f32)
                        tile.mul(og_f32, x_f32, term)
                        tile.row_sum(term, tmp, partial)
                        tile.add(post_acc, partial, post_acc)

                    pto.store(post_acc, _slice_scalar(post_grad, row, c_out))

                for in_mhc in range(4):
                    c_in = const(in_mhc)
                    res_row = row * c_mhc + c_in
                    for out_mhc in range(4):
                        c_out = const(out_mhc)
                        og_row = row * c_mhc + c_out
                        comb_col = const(in_mhc * 4 + out_mhc)
                        comb_acc = pto.alloc_tile(tile_scalar)
                        pto.load(_slice_scalar(comb, row, comb_col), comb_acc)
                        tile.sub(comb_acc, comb_acc, comb_acc)

                        for tile_idx in pto.range(c0, tiles_per_row, c1):
                            col = tile_idx * c_tile
                            cols_this = s.min_u(c_tile, hidden - col)
                            og_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                            og_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                            res_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                            res_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                            term = pto.alloc_tile(tile_f32, valid_col=cols_this)
                            tmp = pto.alloc_tile(tile_f32, valid_col=cols_this)
                            partial = pto.alloc_tile(tile_scalar)

                            pto.load(
                                _slice_bf16(out_grad, og_row, col, cols_this),
                                og_bf16,
                            )
                            pto.load(
                                _slice_bf16(residual, res_row, col, cols_this),
                                res_bf16,
                            )
                            _pto_dialect.TCvtOp(og_bf16, og_f32)
                            _pto_dialect.TCvtOp(res_bf16, res_f32)
                            tile.mul(og_f32, res_f32, term)
                            tile.row_sum(term, tmp, partial)
                            tile.add(comb_acc, partial, comb_acc)

                        pto.store(comb_acc, _slice_scalar(comb_grad, row, comb_col))

                for in_mhc in range(4):
                    c_in = const(in_mhc)
                    res_row = row * c_mhc + c_in
                    for tile_idx in pto.range(c0, tiles_per_row, c1):
                        col = tile_idx * c_tile
                        cols_this = s.min_u(c_tile, hidden - col)
                        acc = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        og_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                        og_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        scale = pto.alloc_tile(tile_scalar)
                        scale_row = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        term = pto.alloc_tile(tile_f32, valid_col=cols_this)
                        out_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)

                        for out_mhc in range(4):
                            c_out = const(out_mhc)
                            og_row = row * c_mhc + c_out
                            comb_col = const(in_mhc * 4 + out_mhc)
                            pto.load(
                                _slice_bf16(out_grad, og_row, col, cols_this),
                                og_bf16,
                            )
                            _pto_dialect.TCvtOp(og_bf16, og_f32)
                            pto.load(_slice_scalar(comb, row, comb_col), scale)
                            tile.row_expand(scale, scale_row)
                            tile.mul(og_f32, scale_row, term)
                            if out_mhc == 0:
                                tile.mov(term, acc)
                            else:
                                tile.add(acc, term, acc)

                        _pto_dialect.TCvtOp(acc, out_bf16)
                        pto.store(
                            out_bf16,
                            _slice_bf16(residual_grad, res_row, col, cols_this),
                        )

                for tile_idx in pto.range(c0, tiles_per_row, c1):
                    col = tile_idx * c_tile
                    cols_this = s.min_u(c_tile, hidden - col)
                    acc = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    og_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                    og_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    scale = pto.alloc_tile(tile_scalar)
                    scale_row = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    term = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    out_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)

                    for out_mhc in range(4):
                        c_out = const(out_mhc)
                        og_row = row * c_mhc + c_out
                        pto.load(_slice_bf16(out_grad, og_row, col, cols_this), og_bf16)
                        _pto_dialect.TCvtOp(og_bf16, og_f32)
                        pto.load(_slice_scalar(post, row, c_out), scale)
                        tile.row_expand(scale, scale_row)
                        tile.mul(og_f32, scale_row, term)
                        if out_mhc == 0:
                            tile.mov(term, acc)
                        else:
                            tile.add(acc, term, acc)

                    _pto_dialect.TCvtOp(acc, out_bf16)
                    pto.store(out_bf16, _slice_bf16(x_grad, row, col, cols_this))

    tilekernels_mhc_post_bwd_kernel.__name__ = "tilekernels_mhc_post_bwd_m4"
    return to_ir_module(meta_data=meta_data)(tilekernels_mhc_post_bwd_kernel)
