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


def build_fused_weight():
    """Build bf16 x bf16 -> f32 elementwise Engram fused-weight kernel."""

    def tilekernels_engram_fused_weight_kernel(
        weight_hidden_ptr: "ptr_bf16",
        weight_embed_ptr: "ptr_bf16",
        weight_fused_ptr: "ptr_f32",
        hc_i32: "i32",
        hidden_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(_TILE)
        hc = s.index_cast(hc_i32)
        hidden = s.index_cast(hidden_i32)

        weight_hidden = pto.as_tensor(
            tensor2_bf16,
            ptr=weight_hidden_ptr,
            shape=[hc, hidden],
            strides=[hidden, c1],
        )
        weight_embed = pto.as_tensor(
            tensor2_bf16,
            ptr=weight_embed_ptr,
            shape=[hc, hidden],
            strides=[hidden, c1],
        )
        weight_fused = pto.as_tensor(
            tensor2_f32,
            ptr=weight_fused_ptr,
            shape=[hc, hidden],
            strides=[hidden, c1],
        )

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(hc, nblocks)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, hc)
            tiles_per_row = s.ceil_div(hidden, c_tile)

            for row in pto.range(row_start, row_end, c1):
                for tile_idx in pto.range(c0, tiles_per_row, c1):
                    col = tile_idx * c_tile
                    cols_this = s.min_u(c_tile, hidden - col)
                    wh_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                    we_bf16 = pto.alloc_tile(tile_bf16, valid_col=cols_this)
                    wh_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    we_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    out_f32 = pto.alloc_tile(tile_f32, valid_col=cols_this)
                    wh_view = pto.slice_view(
                        sub_bf16,
                        source=weight_hidden,
                        offsets=[row, col],
                        sizes=[c1, cols_this],
                    )
                    we_view = pto.slice_view(
                        sub_bf16,
                        source=weight_embed,
                        offsets=[row, col],
                        sizes=[c1, cols_this],
                    )
                    out_view = pto.slice_view(
                        sub_f32,
                        source=weight_fused,
                        offsets=[row, col],
                        sizes=[c1, cols_this],
                    )

                    pto.load(wh_view, wh_bf16)
                    pto.load(we_view, we_bf16)
                    _pto_dialect.TCvtOp(wh_bf16, wh_f32)
                    _pto_dialect.TCvtOp(we_bf16, we_f32)
                    tile.mul(wh_f32, we_f32, out_f32)
                    pto.store(out_f32, out_view)

    return to_ir_module(meta_data=_meta_data)(tilekernels_engram_fused_weight_kernel)
