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
        "i32": pto.int32,
        "tensor2_bf16": pto.TensorType(rank=2, dtype=bf16),
        "sub_bf16": pto.SubTensorType(shape=[1, _TILE], dtype=bf16),
        "tile_bf16": tile_type(bf16, [1, _TILE], valid_shape=[1, -1]),
        "tile_f32": tile_type(f32, [1, _TILE], valid_shape=[1, -1]),
    }


def build_expand_to_mhc_fwd(mhc_mult: int):
    """Build bf16 MHC expand forward: out[token, mhc, h] = x[token, h]."""
    if mhc_mult <= 0:
        raise ValueError("mhc_mult must be positive")

    meta_data = _meta_data

    def tilekernels_mhc_expand_to_mhc_fwd_kernel(
        x_ptr: "ptr_bf16",
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

        x = pto.as_tensor(
            tensor2_bf16,
            ptr=x_ptr,
            shape=[tokens, hidden],
            strides=[hidden, c1],
        )
        out = pto.as_tensor(
            tensor2_bf16,
            ptr=out_ptr,
            shape=[tokens * c_mhc, hidden],
            strides=[hidden, c1],
        )

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
                    buf = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                    x_view = pto.slice_view(
                        sub_bf16,
                        source=x,
                        offsets=[row, col],
                        sizes=[c1, cols_this],
                    )
                    pto.load(x_view, buf)
                    for mhc_idx in range(mhc_mult):
                        c_m = const(mhc_idx)
                        out_row = row * c_mhc + c_m
                        out_view = pto.slice_view(
                            sub_bf16,
                            source=out,
                            offsets=[out_row, col],
                            sizes=[c1, cols_this],
                        )
                        pto.store(buf, out_view)

    tilekernels_mhc_expand_to_mhc_fwd_kernel.__name__ = (
        f"tilekernels_mhc_expand_to_mhc_fwd_m{mhc_mult}"
    )
    return to_ir_module(meta_data=meta_data)(tilekernels_mhc_expand_to_mhc_fwd_kernel)


def build_expand_to_mhc_bwd(mhc_mult: int):
    """Build bf16 MHC expand backward: x_grad[token,h] = sum_m out_grad[token,m,h]."""
    if mhc_mult <= 0:
        raise ValueError("mhc_mult must be positive")

    meta_data = _meta_data

    def tilekernels_mhc_expand_to_mhc_bwd_kernel(
        out_grad_ptr: "ptr_bf16",
        x_grad_ptr: "ptr_bf16",
        tokens_i32: "i32",
        hidden_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(_TILE)
        c_mhc = const(mhc_mult)
        tokens = s.index_cast(tokens_i32)
        hidden = s.index_cast(hidden_i32)

        out_grad = pto.as_tensor(
            tensor2_bf16,
            ptr=out_grad_ptr,
            shape=[tokens * c_mhc, hidden],
            strides=[hidden, c1],
        )
        x_grad = pto.as_tensor(
            tensor2_bf16,
            ptr=x_grad_ptr,
            shape=[tokens, hidden],
            strides=[hidden, c1],
        )

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
                    grad_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                    grad_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    acc_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    out_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                    for mhc_idx in range(mhc_mult):
                        c_m = const(mhc_idx)
                        grad_row = row * c_mhc + c_m
                        grad_view = pto.slice_view(
                            sub_bf16,
                            source=out_grad,
                            offsets=[grad_row, col],
                            sizes=[c1, cols_this],
                        )
                        pto.load(grad_view, grad_bf16)
                        _pto_dialect.TCvtOp(grad_bf16, grad_f32)
                        if mhc_idx == 0:
                            tile.mov(grad_f32, acc_f32)
                        else:
                            tile.add(acc_f32, grad_f32, acc_f32)

                    _pto_dialect.TCvtOp(acc_f32, out_bf16)
                    out_view = pto.slice_view(
                        sub_bf16,
                        source=x_grad,
                        offsets=[row, col],
                        sizes=[c1, cols_this],
                    )
                    pto.store(out_bf16, out_view)

    tilekernels_mhc_expand_to_mhc_bwd_kernel.__name__ = (
        f"tilekernels_mhc_expand_to_mhc_bwd_m{mhc_mult}"
    )
    return to_ir_module(meta_data=meta_data)(tilekernels_mhc_expand_to_mhc_bwd_kernel)
