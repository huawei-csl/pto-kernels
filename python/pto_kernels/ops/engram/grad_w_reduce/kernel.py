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
        "sub_f32": pto.SubTensorType(shape=[1, _TILE], dtype=f32),
        "tile_bf16": tile_type(bf16, [1, _TILE], valid_shape=[1, -1]),
        "tile_f32": tile_type(f32, [1, _TILE], valid_shape=[1, -1]),
    }


def build_grad_w_reduce(num_persistent_blocks: int, hc_mult: int = 4):
    """Build Engram grad_w_reduce.

    Reduces ``grad_w_partial`` over persistent blocks, then accumulates into
    ``grad_weight_hidden`` and ``grad_weight_embed`` after multiplying by the
    opposite bf16 weight converted to f32.
    """
    if num_persistent_blocks <= 0:
        raise ValueError("num_persistent_blocks must be positive")
    if hc_mult <= 0:
        raise ValueError("hc_mult must be positive")

    def tilekernels_engram_grad_w_reduce_kernel(
        grad_w_partial_ptr: "ptr_f32",
        weight_hidden_ptr: "ptr_bf16",
        weight_embed_ptr: "ptr_bf16",
        grad_weight_hidden_ptr: "ptr_f32",
        grad_weight_embed_ptr: "ptr_f32",
        hidden_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(_TILE)
        c_hc = const(hc_mult)
        hidden = s.index_cast(hidden_i32)

        grad_w_partial = pto.as_tensor(
            tensor2_f32,
            ptr=grad_w_partial_ptr,
            shape=[const(num_persistent_blocks) * c_hc, hidden],
            strides=[hidden, c1],
        )
        weight_hidden = pto.as_tensor(
            tensor2_bf16,
            ptr=weight_hidden_ptr,
            shape=[c_hc, hidden],
            strides=[hidden, c1],
        )
        weight_embed = pto.as_tensor(
            tensor2_bf16,
            ptr=weight_embed_ptr,
            shape=[c_hc, hidden],
            strides=[hidden, c1],
        )
        grad_weight_hidden = pto.as_tensor(
            tensor2_f32,
            ptr=grad_weight_hidden_ptr,
            shape=[c_hc, hidden],
            strides=[hidden, c1],
        )
        grad_weight_embed = pto.as_tensor(
            tensor2_f32,
            ptr=grad_weight_embed_ptr,
            shape=[c_hc, hidden],
            strides=[hidden, c1],
        )

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(c_hc, nblocks)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, c_hc)
            tiles_per_row = s.ceil_div(hidden, c_tile)

            for hc_idx in pto.range(row_start, row_end, c1):
                for tile_idx in pto.range(c0, tiles_per_row, c1):
                    col = tile_idx * c_tile
                    cols_this = s.min_u(c_tile, hidden - col)
                    partial = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    acc = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    wh_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                    we_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                    wh_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    we_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    grad_wh = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    grad_we = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    tmp = pto.alloc_tile(tile_f32, valid_col=cols_this)

                    for block_idx in range(num_persistent_blocks):
                        partial_row = const(block_idx) * c_hc + hc_idx
                        pto.load(
                            pto.slice_view(
                                sub_f32,
                                source=grad_w_partial,
                                offsets=[partial_row, col],
                                sizes=[c1, cols_this],
                            ),
                            partial,
                        )
                        if block_idx == 0:
                            tile.mov(partial, acc)
                        else:
                            tile.add(acc, partial, acc)

                    pto.load(
                        pto.slice_view(
                            sub_bf16,
                            source=weight_hidden,
                            offsets=[hc_idx, col],
                            sizes=[c1, cols_this],
                        ),
                        wh_bf16,
                    )
                    pto.load(
                        pto.slice_view(
                            sub_bf16,
                            source=weight_embed,
                            offsets=[hc_idx, col],
                            sizes=[c1, cols_this],
                        ),
                        we_bf16,
                    )
                    pto.load(
                        pto.slice_view(
                            sub_f32,
                            source=grad_weight_hidden,
                            offsets=[hc_idx, col],
                            sizes=[c1, cols_this],
                        ),
                        grad_wh,
                    )
                    pto.load(
                        pto.slice_view(
                            sub_f32,
                            source=grad_weight_embed,
                            offsets=[hc_idx, col],
                            sizes=[c1, cols_this],
                        ),
                        grad_we,
                    )

                    _pto_dialect.TCvtOp(wh_bf16, wh_f32)
                    _pto_dialect.TCvtOp(we_bf16, we_f32)
                    tile.mul(acc, we_f32, tmp)
                    tile.add(grad_wh, tmp, grad_wh)
                    tile.mul(acc, wh_f32, tmp)
                    tile.add(grad_we, tmp, grad_we)

                    pto.store(
                        grad_wh,
                        pto.slice_view(
                            sub_f32,
                            source=grad_weight_hidden,
                            offsets=[hc_idx, col],
                            sizes=[c1, cols_this],
                        ),
                    )
                    pto.store(
                        grad_we,
                        pto.slice_view(
                            sub_f32,
                            source=grad_weight_embed,
                            offsets=[hc_idx, col],
                            sizes=[c1, cols_this],
                        ),
                    )

    tilekernels_engram_grad_w_reduce_kernel.__name__ = (
        f"tilekernels_engram_grad_w_reduce_p{num_persistent_blocks}_h{hc_mult}"
    )
    return to_ir_module(meta_data=_meta_data)(tilekernels_engram_grad_w_reduce_kernel)
