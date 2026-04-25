from mlir.dialects import pto as _pto_dialect

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s


const = s.const
_DST_STRIDE = 2
_SORT_BLOCK_LEN = 32


def _sort_width(num_experts: int) -> int:
    """Return a PTO top-k sort width supported by sort32/mrgsort."""
    width = _SORT_BLOCK_LEN
    while width < num_experts:
        width *= 4
    return width


def _assert_clean_merge(sort_cols: int) -> None:
    hw_block_len = _SORT_BLOCK_LEN * _DST_STRIDE
    block = hw_block_len
    while block * 4 <= sort_cols:
        block *= 4
    if block != sort_cols:
        raise ValueError(
            f"sort_cols={sort_cols} is not a clean power-of-4 merge width"
        )


def _tfillpad_expand(src, dst) -> None:
    _pto_dialect.TFillPadExpandOp(s._unwrap(src), s._unwrap(dst))


def _tinsert(src, row, col, dst) -> None:
    _pto_dialect.TInsertOp(
        src=s._unwrap(src),
        indexRow=s._unwrap(row),
        indexCol=s._unwrap(col),
        dst=s._unwrap(dst),
    )


def _meta_data(num_experts: int, sort_ncols: int, num_topk: int):
    f32 = pto.float32
    u32 = pto.uint32
    tile_cfg = pto.TileBufConfig()
    padded_cfg = pto.TileBufConfig(pad="Min")
    sort_cols = sort_ncols * _DST_STRIDE
    return {
        "ptr_f32": pto.PtrType(f32),
        "ptr_u32": pto.PtrType(u32),
        "i32": pto.int32,
        "tensor_scores": pto.TensorType(rank=2, dtype=f32),
        "tensor_inidx": pto.TensorType(rank=2, dtype=u32),
        "tensor_indices": pto.TensorType(rank=2, dtype=u32),
        "sub_scores": pto.SubTensorType(shape=[1, num_experts], dtype=f32),
        "sub_inidx": pto.SubTensorType(shape=[1, sort_ncols], dtype=u32),
        "sub_indices": pto.SubTensorType(shape=[1, num_topk], dtype=u32),
        "tile_src": pto.TileBufType(
            shape=[1, num_experts],
            valid_shape=[1, num_experts],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_src_padded": pto.TileBufType(
            shape=[1, sort_ncols],
            valid_shape=[1, sort_ncols],
            dtype=f32,
            memory_space="VEC",
            config=padded_cfg,
        ),
        "tile_inidx": pto.TileBufType(
            shape=[1, sort_ncols],
            valid_shape=[1, sort_ncols],
            dtype=u32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_sort_f32": pto.TileBufType(
            shape=[1, sort_cols],
            valid_shape=[1, sort_cols],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_sort_u32": pto.TileBufType(
            shape=[1, sort_cols],
            valid_shape=[1, sort_cols],
            dtype=u32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_gather_win_u32": pto.TileBufType(
            shape=[1, sort_cols],
            valid_shape=[1, num_topk * _DST_STRIDE],
            dtype=u32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_topk_u32": pto.TileBufType(
            shape=[1, num_topk],
            valid_shape=[1, num_topk],
            dtype=u32,
            memory_space="VEC",
            config=tile_cfg,
        ),
    }


def _group_meta_data(
    num_experts: int,
    num_groups: int,
    experts_per_group: int,
    group_sort_ncols: int,
    group_score_sort_ncols: int,
    num_group_sum_topk: int,
    num_topk_groups: int,
):
    f32 = pto.float32
    u32 = pto.uint32
    tile_cfg = pto.TileBufConfig()
    padded_cfg = pto.TileBufConfig(pad="Min")
    group_sort_cols = group_sort_ncols * _DST_STRIDE
    group_score_sort_cols = group_score_sort_ncols * _DST_STRIDE
    max_sort_ncols = max(group_sort_ncols, group_score_sort_ncols)
    return {
        "ptr_f32": pto.PtrType(f32),
        "ptr_u32": pto.PtrType(u32),
        "i32": pto.int32,
        "tensor_scores": pto.TensorType(rank=2, dtype=f32),
        "tensor_inidx": pto.TensorType(rank=2, dtype=u32),
        "tensor_group_idx": pto.TensorType(rank=2, dtype=u32),
        "sub_scores": pto.SubTensorType(shape=[1, num_experts], dtype=f32),
        "sub_inidx_max": pto.SubTensorType(shape=[1, max_sort_ncols], dtype=u32),
        "sub_group_idx": pto.SubTensorType(shape=[1, num_topk_groups], dtype=u32),
        "tile_scores": pto.TileBufType(
            shape=[1, num_experts],
            valid_shape=[1, num_experts],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_group_src": pto.TileBufType(
            shape=[1, experts_per_group],
            valid_shape=[1, experts_per_group],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_group_src_padded": pto.TileBufType(
            shape=[1, group_sort_ncols],
            valid_shape=[1, group_sort_ncols],
            dtype=f32,
            memory_space="VEC",
            config=padded_cfg,
        ),
        "tile_inidx_max": pto.TileBufType(
            shape=[1, max_sort_ncols],
            valid_shape=[1, max_sort_ncols],
            dtype=u32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_expert_inidx": pto.TileBufType(
            shape=[1, group_sort_ncols],
            valid_shape=[1, group_sort_ncols],
            dtype=u32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_group_inidx": pto.TileBufType(
            shape=[1, group_score_sort_ncols],
            valid_shape=[1, group_score_sort_ncols],
            dtype=u32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_group_sort_f32": pto.TileBufType(
            shape=[1, group_sort_cols],
            valid_shape=[1, group_sort_cols],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_group_gather_win_f32": pto.TileBufType(
            shape=[1, group_sort_cols],
            valid_shape=[1, num_group_sum_topk * _DST_STRIDE],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_group_top_scores": pto.TileBufType(
            shape=[1, num_group_sum_topk],
            valid_shape=[1, num_group_sum_topk],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_scalar_f32": pto.TileBufType(
            shape=[1, 1],
            valid_shape=[1, 1],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_group_scores": pto.TileBufType(
            shape=[1, num_groups],
            valid_shape=[1, num_groups],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_group_scores_padded": pto.TileBufType(
            shape=[1, group_score_sort_ncols],
            valid_shape=[1, group_score_sort_ncols],
            dtype=f32,
            memory_space="VEC",
            config=padded_cfg,
        ),
        "tile_group_score_sort_f32": pto.TileBufType(
            shape=[1, group_score_sort_cols],
            valid_shape=[1, group_score_sort_cols],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_group_score_gather_win_u32": pto.TileBufType(
            shape=[1, group_score_sort_cols],
            valid_shape=[1, num_topk_groups * _DST_STRIDE],
            dtype=u32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_top_group_idx": pto.TileBufType(
            shape=[1, num_topk_groups],
            valid_shape=[1, num_topk_groups],
            dtype=u32,
            memory_space="VEC",
            config=tile_cfg,
        ),
    }


def build_topk_gate(num_experts: int, num_topk: int):
    """Build TopK expert selection with PTO sort32/mrgsort/gather primitives.

    This follows the PTO-ISA manual TopK kernel structure: the source row is
    padded to a supported sort width with PadValue::Min, TSORT32 attaches the
    caller-provided uint32 column index vector, TMRGSORT merges the sorted
    blocks, and TGATHER extracts the index lanes.
    """
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if num_topk <= 0 or num_topk > num_experts:
        raise ValueError("num_topk must be in [1, num_experts]")

    sort_ncols = _sort_width(num_experts)
    sort_cols = sort_ncols * _DST_STRIDE
    _assert_clean_merge(sort_cols)

    def tilekernels_moe_topk_gate_kernel(
        scores_ptr: "ptr_f32",
        inidx_ptr: "ptr_u32",
        topk_idx_ptr: "ptr_u32",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_experts = const(num_experts)
        c_sort_ncols = const(sort_ncols)
        c_topk = const(num_topk)
        c_bdim = s.index_cast(pto.get_block_num())
        num_tokens = s.index_cast(num_tokens_i32)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            rows_per_core = s.ceil_div(num_tokens, c_bdim)
            row_start = bid * rows_per_core
            row_end_raw = row_start + rows_per_core
            need_clamp = row_end_raw > num_tokens
            rows_this_core = s.select(need_clamp, num_tokens - row_start, rows_per_core)

            tv_scores = pto.as_tensor(
                tensor_scores,
                ptr=scores_ptr,
                shape=[num_tokens, c_experts],
                strides=[c_experts, c1],
            )
            tv_inidx = pto.as_tensor(
                tensor_inidx,
                ptr=inidx_ptr,
                shape=[c1, c_sort_ncols],
                strides=[c_sort_ncols, c1],
            )
            tv_indices = pto.as_tensor(
                tensor_indices,
                ptr=topk_idx_ptr,
                shape=[num_tokens, c_topk],
                strides=[c_topk, c1],
            )

            tb_src = pto.alloc_tile(tile_src)
            tb_src_padded = pto.alloc_tile(tile_src_padded)
            tb_inidx = pto.alloc_tile(tile_inidx)
            tb_sort = pto.alloc_tile(tile_sort_f32)
            tb_sort_tmp = pto.alloc_tile(tile_sort_f32)
            tb_gather_win_u = pto.alloc_tile(tile_gather_win_u32)
            tb_indices = pto.alloc_tile(tile_topk_u32)

            sv_inidx = pto.slice_view(
                sub_inidx,
                source=tv_inidx,
                offsets=[c0, c0],
                sizes=[c1, c_sort_ncols],
            )
            pto.load(sv_inidx, tb_inidx)

            with pto.if_context(row_start < num_tokens):
                with pto.if_context(rows_this_core > c0):
                    for i in pto.range(c0, rows_this_core, c1):
                        row = row_start + i
                        sv_scores = pto.slice_view(
                            sub_scores,
                            source=tv_scores,
                            offsets=[row, c0],
                            sizes=[c1, c_experts],
                        )
                        pto.load(sv_scores, tb_src)
                        if sort_ncols == num_experts:
                            tile.mov(tb_src, tb_src_padded)
                        else:
                            _tfillpad_expand(tb_src, tb_src_padded)

                        tile.sort32(tb_src_padded, tb_sort, tb_inidx)

                        cur_block = _SORT_BLOCK_LEN * _DST_STRIDE
                        while cur_block * 4 <= sort_cols:
                            tile.mrgsort(tb_sort, tb_sort_tmp, const(cur_block))
                            tile.mov(tb_sort_tmp, tb_sort)
                            cur_block *= 4

                        tile.mov(tb_sort, tb_gather_win_u)
                        tile.gather(tb_gather_win_u, tb_indices, mask_pattern="P1010")

                        sv_indices = pto.slice_view(
                            sub_indices,
                            source=tv_indices,
                            offsets=[row, c0],
                            sizes=[c1, c_topk],
                        )
                        pto.store(tb_indices, sv_indices)

    tilekernels_moe_topk_gate_kernel.__name__ = (
        f"tilekernels_moe_topk_gate_sort_e{num_experts}_p{sort_ncols}_k{num_topk}"
    )
    meta_data = lambda: _meta_data(num_experts, sort_ncols, num_topk)
    return to_ir_module(meta_data=meta_data)(
        tilekernels_moe_topk_gate_kernel
    )


def build_topk_sum_and_topk_group_idx(
    num_experts: int,
    num_groups: int,
    num_group_sum_topk: int,
    num_topk_groups: int,
):
    """Build grouped TopK using PTO sort32/mrgsort/gather primitives.

    For each token row, each group is independently sorted to get its top
    ``num_group_sum_topk`` scores. Those per-group sums are assembled into a
    score row and sorted again to return the top group indices.
    """
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if num_groups <= 0 or num_experts % num_groups != 0:
        raise ValueError("num_groups must divide num_experts")
    if num_group_sum_topk not in {1, 2}:
        raise ValueError("num_group_sum_topk must be 1 or 2")
    if num_topk_groups <= 0 or num_topk_groups > num_groups:
        raise ValueError("num_topk_groups must be in [1, num_groups]")

    experts_per_group = num_experts // num_groups
    if num_group_sum_topk > experts_per_group:
        raise ValueError("num_group_sum_topk cannot exceed experts_per_group")

    group_sort_ncols = _sort_width(experts_per_group)
    group_score_sort_ncols = _sort_width(num_groups)
    group_sort_cols = group_sort_ncols * _DST_STRIDE
    group_score_sort_cols = group_score_sort_ncols * _DST_STRIDE
    max_sort_ncols = max(group_sort_ncols, group_score_sort_ncols)
    _assert_clean_merge(group_sort_cols)
    _assert_clean_merge(group_score_sort_cols)

    def tilekernels_moe_topk_sum_and_topk_group_idx_kernel(
        scores_ptr: "ptr_f32",
        inidx_ptr: "ptr_u32",
        group_idx_ptr: "ptr_u32",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_experts = const(num_experts)
        c_groups = const(num_groups)
        c_topk_groups = const(num_topk_groups)
        c_max_sort_ncols = const(max_sort_ncols)
        c_bdim = s.index_cast(pto.get_block_num())
        num_tokens = s.index_cast(num_tokens_i32)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            rows_per_core = s.ceil_div(num_tokens, c_bdim)
            row_start = bid * rows_per_core
            row_end_raw = row_start + rows_per_core
            need_clamp = row_end_raw > num_tokens
            rows_this_core = s.select(need_clamp, num_tokens - row_start, rows_per_core)

            tv_scores = pto.as_tensor(
                tensor_scores,
                ptr=scores_ptr,
                shape=[num_tokens, c_experts],
                strides=[c_experts, c1],
            )
            tv_inidx = pto.as_tensor(
                tensor_inidx,
                ptr=inidx_ptr,
                shape=[c1, c_max_sort_ncols],
                strides=[c_max_sort_ncols, c1],
            )
            tv_group_idx = pto.as_tensor(
                tensor_group_idx,
                ptr=group_idx_ptr,
                shape=[num_tokens, c_topk_groups],
                strides=[c_topk_groups, c1],
            )

            tb_scores = pto.alloc_tile(tile_scores)
            tb_group_src = pto.alloc_tile(tile_group_src)
            tb_group_src_padded = pto.alloc_tile(tile_group_src_padded)
            tb_inidx_max = pto.alloc_tile(tile_inidx_max)
            tb_expert_inidx = pto.alloc_tile(tile_expert_inidx)
            tb_group_inidx = pto.alloc_tile(tile_group_inidx)
            tb_group_sort = pto.alloc_tile(tile_group_sort_f32)
            tb_group_sort_tmp = pto.alloc_tile(tile_group_sort_f32)
            tb_group_gather_win = pto.alloc_tile(tile_group_gather_win_f32)
            tb_group_top_scores = pto.alloc_tile(tile_group_top_scores)
            tb_group_top_scores_tmp = pto.alloc_tile(tile_group_top_scores)
            tb_group_sum = pto.alloc_tile(tile_scalar_f32)
            tb_group_scores = pto.alloc_tile(tile_group_scores)
            tb_group_scores_padded = pto.alloc_tile(tile_group_scores_padded)
            tb_group_score_sort = pto.alloc_tile(tile_group_score_sort_f32)
            tb_group_score_sort_tmp = pto.alloc_tile(tile_group_score_sort_f32)
            tb_group_score_gather_win_u = pto.alloc_tile(tile_group_score_gather_win_u32)
            tb_top_group_idx = pto.alloc_tile(tile_top_group_idx)

            sv_inidx = pto.slice_view(
                sub_inidx_max,
                source=tv_inidx,
                offsets=[c0, c0],
                sizes=[c1, c_max_sort_ncols],
            )
            pto.load(sv_inidx, tb_inidx_max)
            tile.mov(tile.subset(tb_inidx_max, [c0, c0], [1, group_sort_ncols]), tb_expert_inidx)
            tile.mov(tile.subset(tb_inidx_max, [c0, c0], [1, group_score_sort_ncols]), tb_group_inidx)

            with pto.if_context(row_start < num_tokens):
                with pto.if_context(rows_this_core > c0):
                    for i in pto.range(c0, rows_this_core, c1):
                        row = row_start + i
                        sv_scores = pto.slice_view(
                            sub_scores,
                            source=tv_scores,
                            offsets=[row, c0],
                            sizes=[c1, c_experts],
                        )
                        pto.load(sv_scores, tb_scores)

                        for group in range(num_groups):
                            group_offset = const(group * experts_per_group)
                            tile.mov(
                                tile.subset(
                                    tb_scores,
                                    [c0, group_offset],
                                    [1, experts_per_group],
                                ),
                                tb_group_src,
                            )
                            if group_sort_ncols == experts_per_group:
                                tile.mov(tb_group_src, tb_group_src_padded)
                            else:
                                _tfillpad_expand(tb_group_src, tb_group_src_padded)

                            tile.sort32(tb_group_src_padded, tb_group_sort, tb_expert_inidx)

                            cur_block = _SORT_BLOCK_LEN * _DST_STRIDE
                            while cur_block * 4 <= group_sort_cols:
                                tile.mrgsort(tb_group_sort, tb_group_sort_tmp, const(cur_block))
                                tile.mov(tb_group_sort_tmp, tb_group_sort)
                                cur_block *= 4

                            tile.mov(tb_group_sort, tb_group_gather_win)
                            tile.gather(
                                tb_group_gather_win,
                                tb_group_top_scores,
                                mask_pattern="P0101",
                            )
                            tile.row_sum(
                                tb_group_top_scores,
                                tb_group_top_scores_tmp,
                                tb_group_sum,
                            )
                            _tinsert(tb_group_sum, c0, const(group), tb_group_scores)

                        if group_score_sort_ncols == num_groups:
                            tile.mov(tb_group_scores, tb_group_scores_padded)
                        else:
                            _tfillpad_expand(tb_group_scores, tb_group_scores_padded)

                        tile.sort32(
                            tb_group_scores_padded,
                            tb_group_score_sort,
                            tb_group_inidx,
                        )

                        cur_block = _SORT_BLOCK_LEN * _DST_STRIDE
                        while cur_block * 4 <= group_score_sort_cols:
                            tile.mrgsort(
                                tb_group_score_sort,
                                tb_group_score_sort_tmp,
                                const(cur_block),
                            )
                            tile.mov(tb_group_score_sort_tmp, tb_group_score_sort)
                            cur_block *= 4

                        tile.mov(tb_group_score_sort, tb_group_score_gather_win_u)
                        tile.gather(
                            tb_group_score_gather_win_u,
                            tb_top_group_idx,
                            mask_pattern="P1010",
                        )

                        sv_group_idx = pto.slice_view(
                            sub_group_idx,
                            source=tv_group_idx,
                            offsets=[row, c0],
                            sizes=[c1, c_topk_groups],
                        )
                        pto.store(tb_top_group_idx, sv_group_idx)

    tilekernels_moe_topk_sum_and_topk_group_idx_kernel.__name__ = (
        "tilekernels_moe_topk_sum_group_idx_"
        f"e{num_experts}_g{num_groups}_s{num_group_sum_topk}_k{num_topk_groups}"
    )
    meta_data = lambda: _group_meta_data(
        num_experts,
        num_groups,
        experts_per_group,
        group_sort_ncols,
        group_score_sort_ncols,
        num_group_sum_topk,
        num_topk_groups,
    )
    return to_ir_module(meta_data=meta_data)(
        tilekernels_moe_topk_sum_and_topk_group_idx_kernel
    )
