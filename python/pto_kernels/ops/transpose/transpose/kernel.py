from mlir.dialects import pto as _pto_dialect

from ptodsl import pto, to_ir_module
from ptodsl import scalar as s

from pto_kernels.ops.tilekernels_common import dtype_from_name, ptr, tile_type


const = s.const
_TILE = 32


def _meta_data(dtype_name: str):
    dtype = dtype_from_name(dtype_name)
    shape = [_TILE, _TILE]
    return {
        "ptr_t": ptr(dtype),
        "i32": pto.int32,
        "tensor2_t": pto.TensorType(rank=2, dtype=dtype),
        "sub_tile": pto.SubTensorType(shape=shape, dtype=dtype),
        "tile_t": tile_type(dtype, shape, valid_shape=[-1, -1]),
    }


def build_transpose(dtype: str = "bf16"):
    """Build a dynamic 2-D transpose kernel.

    Runtime args:
      - x_ptr: input [rows, cols] with runtime row stride
      - out_ptr: output [cols, rows], contiguous
      - rows_i32, cols_i32, stride_i32
    """
    meta_data = lambda: _meta_data(dtype)

    def tilekernels_transpose_kernel(
        x_ptr: "ptr_t",
        out_ptr: "ptr_t",
        rows_i32: "i32",
        cols_i32: "i32",
        stride_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(_TILE)
        rows = s.index_cast(rows_i32)
        cols = s.index_cast(cols_i32)
        stride = s.index_cast(stride_i32)

        x = pto.as_tensor(
            tensor2_t,
            ptr=x_ptr,
            shape=[rows, cols],
            strides=[stride, c1],
        )
        out = pto.as_tensor(
            tensor2_t,
            ptr=out_ptr,
            shape=[cols, rows],
            strides=[rows, c1],
        )

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            row_tiles = s.ceil_div(rows, c_tile)
            col_tiles = s.ceil_div(cols, c_tile)
            total_tiles = row_tiles * col_tiles
            tiles_per_core = s.ceil_div(total_tiles, nblocks)
            tile_start = bid * tiles_per_core
            tile_end = s.min_u(tile_start + tiles_per_core, total_tiles)

            for tile_idx in pto.range(tile_start, tile_end, c1):
                row_tile_idx = tile_idx // col_tiles
                col_tile_idx = tile_idx % col_tiles
                row = row_tile_idx * c_tile
                col = col_tile_idx * c_tile
                rows_this = s.min_u(c_tile, rows - row)
                cols_this = s.min_u(c_tile, cols - col)
                src_tile = pto.alloc_tile(
                    tile_t, valid_row=rows_this, valid_col=cols_this
                )
                tmp_tile = pto.alloc_tile(
                    tile_t, valid_row=rows_this, valid_col=cols_this
                )
                dst_tile = pto.alloc_tile(
                    tile_t, valid_row=cols_this, valid_col=rows_this
                )

                src = pto.slice_view(
                    sub_tile,
                    source=x,
                    offsets=[row, col],
                    sizes=[rows_this, cols_this],
                )
                dst = pto.slice_view(
                    sub_tile,
                    source=out,
                    offsets=[col, row],
                    sizes=[cols_this, rows_this],
                )

                pto.load(src, src_tile)
                _pto_dialect.TTransOp(src_tile, tmp_tile, dst_tile)
                pto.store(dst_tile, dst)

    tilekernels_transpose_kernel.__name__ = f"tilekernels_transpose_{dtype}"
    return to_ir_module(meta_data=meta_data)(tilekernels_transpose_kernel)


def build_batched_transpose(dtype: str = "bf16"):
    """Build a dynamic batched transpose kernel for [B, rows, cols]."""
    meta_data = lambda: _meta_data(dtype)

    def tilekernels_batched_transpose_kernel(
        x_ptr: "ptr_t",
        out_ptr: "ptr_t",
        batches_i32: "i32",
        rows_i32: "i32",
        cols_i32: "i32",
        stride_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(_TILE)
        batches = s.index_cast(batches_i32)
        rows = s.index_cast(rows_i32)
        cols = s.index_cast(cols_i32)
        stride = s.index_cast(stride_i32)

        x = pto.as_tensor(
            tensor2_t,
            ptr=x_ptr,
            shape=[batches * rows, cols],
            strides=[stride, c1],
        )
        out = pto.as_tensor(
            tensor2_t,
            ptr=out_ptr,
            shape=[batches * cols, rows],
            strides=[rows, c1],
        )

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            row_tiles = s.ceil_div(rows, c_tile)
            col_tiles = s.ceil_div(cols, c_tile)
            tiles_per_batch = row_tiles * col_tiles
            total_tiles = batches * tiles_per_batch
            tiles_per_core = s.ceil_div(total_tiles, nblocks)
            tile_start = bid * tiles_per_core
            tile_end = s.min_u(tile_start + tiles_per_core, total_tiles)

            for tile_idx in pto.range(tile_start, tile_end, c1):
                batch_idx = tile_idx // tiles_per_batch
                tile_in_batch = tile_idx % tiles_per_batch
                row_tile_idx = tile_in_batch // col_tiles
                col_tile_idx = tile_in_batch % col_tiles
                row = row_tile_idx * c_tile
                col = col_tile_idx * c_tile
                rows_this = s.min_u(c_tile, rows - row)
                cols_this = s.min_u(c_tile, cols - col)
                src_tile = pto.alloc_tile(
                    tile_t, valid_row=rows_this, valid_col=cols_this
                )
                tmp_tile = pto.alloc_tile(
                    tile_t, valid_row=rows_this, valid_col=cols_this
                )
                dst_tile = pto.alloc_tile(
                    tile_t, valid_row=cols_this, valid_col=rows_this
                )

                src_row = batch_idx * rows + row
                dst_row = batch_idx * cols + col
                src = pto.slice_view(
                    sub_tile,
                    source=x,
                    offsets=[src_row, col],
                    sizes=[rows_this, cols_this],
                )
                dst = pto.slice_view(
                    sub_tile,
                    source=out,
                    offsets=[dst_row, row],
                    sizes=[cols_this, rows_this],
                )

                pto.load(src, src_tile)
                _pto_dialect.TTransOp(src_tile, tmp_tile, dst_tile)
                pto.store(dst_tile, dst)

    tilekernels_batched_transpose_kernel.__name__ = (
        f"tilekernels_batched_transpose_{dtype}"
    )
    return to_ir_module(meta_data=meta_data)(tilekernels_batched_transpose_kernel)
