from mlir.dialects import arith, pto as _pto_dialect

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s


const = s.const


def _meta_data(num_topk: int):
    f32 = pto.float32
    i32 = pto.int32
    tile_cfg = pto.TileBufConfig()
    row_shape = [1, num_topk]

    return {
        "ptr_f32": pto.PtrType(f32),
        "i32": i32,
        "tensor2_f32": pto.TensorType(rank=2, dtype=f32),
        "sub_row": pto.SubTensorType(shape=row_shape, dtype=f32),
        "sub_scalar": pto.SubTensorType(shape=[1, 1], dtype=f32),
        "tile_row": pto.TileBufType(
            shape=row_shape,
            valid_shape=row_shape,
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


def build_normalize_weight(num_topk: int):
    """Build PTO-DSL IR for TileKernels ``moe.normalize_weight``.

    Runtime pointers:
      - ``topk_weights_ptr``: [num_tokens, num_topk] float32
      - ``denominator_ptr``: [num_tokens, 1] float32
      - ``normalized_weights_ptr``: [num_tokens, num_topk] float32
      - ``num_tokens_i32``: runtime token count

    The denominator uses a 2-D [N, 1] tensor view so PTO tile stores can write it
    through the same partitioned-tensor mechanism as the normalized row.
    """
    if num_topk <= 0:
        raise ValueError("num_topk must be positive")

    meta_data = lambda: _meta_data(num_topk)

    def normalize_weight_kernel(
        topk_weights_ptr: "ptr_f32",
        denominator_ptr: "ptr_f32",
        normalized_weights_ptr: "ptr_f32",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_topk = const(num_topk)
        num_tokens = s.index_cast(num_tokens_i32)

        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        vid = s.index_cast(cid * sub_bnum + sub_bid)
        n_blocks = s.index_cast(pto.get_block_num() * sub_bnum)
        rows_per_core = s.ceil_div(num_tokens, n_blocks)
        row_start = vid * rows_per_core
        row_end_raw = row_start + rows_per_core
        row_end = s.min_u(row_end_raw, num_tokens)

        weights = pto.as_tensor(
            tensor2_f32,
            ptr=topk_weights_ptr,
            shape=[num_tokens, c_topk],
            strides=[c_topk, c1],
        )
        denominators = pto.as_tensor(
            tensor2_f32,
            ptr=denominator_ptr,
            shape=[num_tokens, c1],
            strides=[c1, c1],
        )
        normalized = pto.as_tensor(
            tensor2_f32,
            ptr=normalized_weights_ptr,
            shape=[num_tokens, c_topk],
            strides=[c_topk, c1],
        )

        with pto.vector_section():
            row = pto.alloc_tile(tile_row)
            tmp = pto.alloc_tile(tile_row)
            denom = pto.alloc_tile(tile_scalar)
            denom_broadcast = pto.alloc_tile(tile_row)
            out = pto.alloc_tile(tile_row)

            for row_idx in pto.range(row_start, row_end, c1):
                weight_row = pto.slice_view(
                    sub_row,
                    source=weights,
                    offsets=[row_idx, c0],
                    sizes=[c1, c_topk],
                )
                denom_row = pto.slice_view(
                    sub_scalar,
                    source=denominators,
                    offsets=[row_idx, c0],
                    sizes=[c1, c1],
                )
                out_row = pto.slice_view(
                    sub_row,
                    source=normalized,
                    offsets=[row_idx, c0],
                    sizes=[c1, c_topk],
                )

                pto.load(weight_row, row)
                tile.row_sum(row, tmp, denom)
                eps = arith.ConstantOp(pto.float32, 1e-20).result
                _pto_dialect.TAddSOp(denom, eps, denom)
                tile.row_expand(denom, denom_broadcast)
                tile.row_expand_div(row, denom_broadcast, out)
                pto.store(denom, denom_row)
                pto.store(out, out_row)

    normalize_weight_kernel.__name__ = f"tilekernels_moe_normalize_weight_k{num_topk}"
    return to_ir_module(meta_data=meta_data)(normalize_weight_kernel)
