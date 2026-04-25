from mlir.dialects import arith, pto as _pto_dialect

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s


const = s.const


def _meta_data(mhc: int):
    f32 = pto.float32
    tile_cfg = pto.TileBufConfig()
    return {
        "ptr_f32": pto.PtrType(f32),
        "i32": pto.int32,
        "tensor2_f32": pto.TensorType(rank=2, dtype=f32),
        "sub_matrix": pto.SubTensorType(shape=[mhc, mhc], dtype=f32),
        "tile_matrix": pto.TileBufType(
            shape=[mhc, mhc],
            valid_shape=[mhc, mhc],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_row": pto.TileBufType(
            shape=[mhc, 1],
            valid_shape=[mhc, 1],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_col": pto.TileBufType(
            shape=[1, mhc],
            valid_shape=[1, mhc],
            dtype=f32,
            memory_space="VEC",
            config=tile_cfg,
        ),
    }


def _f32(value: float):
    return arith.ConstantOp(pto.float32, value).result


def _adds_inplace(tile_buf, value: float) -> None:
    _pto_dialect.TAddSOp(tile_buf, _f32(value), tile_buf)


def _make_matrix_tensor(ptr_value, num_tokens, mhc: int):
    c1 = const(1)
    c_mhc = const(mhc)
    return pto.as_tensor(
        tensor2_f32,
        ptr=ptr_value,
        shape=[num_tokens * c_mhc, c_mhc],
        strides=[c_mhc, c1],
    )


def _slice_token(tensor, token, mhc: int):
    c0 = const(0)
    c_mhc = const(mhc)
    return pto.slice_view(
        sub_matrix,
        source=tensor,
        offsets=[token * c_mhc, c0],
        sizes=[c_mhc, c_mhc],
    )


def _row_normalize(src, tmp, row_sum, row_broadcast, out, eps: float) -> None:
    tile.row_sum(src, tmp, row_sum)
    _adds_inplace(row_sum, eps)
    tile.row_expand(row_sum, row_broadcast)
    tile.div(src, row_broadcast, out)


def _col_normalize(src, tmp, col_sum, col_broadcast, out, eps: float) -> None:
    tile.col_sum(src, tmp, col_sum)
    _adds_inplace(col_sum, eps)
    tile.col_expand(col_sum, col_broadcast)
    tile.div(src, col_broadcast, out)


def _softmax_rows(src, tmp, row_stat, row_broadcast, centered, exp_tile, out) -> None:
    tile.row_max(src, tmp, row_stat)
    tile.row_expand_sub(src, row_stat, centered)
    tile.exp(centered, exp_tile)
    tile.row_sum(exp_tile, tmp, row_stat)
    tile.row_expand(row_stat, row_broadcast)
    tile.div(exp_tile, row_broadcast, out)


def _backward_col_norm(
    grad,
    x_inter,
    norm_sum,
    tmp_matrix,
    tmp_matrix2,
    tmp_col,
    tmp_col2,
    col_broadcast,
    grad_next,
    eps: float,
) -> None:
    tile.mul(grad, x_inter, tmp_matrix)
    tile.col_sum(tmp_matrix, tmp_matrix2, tmp_col)
    tile.mov(norm_sum, tmp_col2)
    _adds_inplace(tmp_col2, eps)
    tile.div(tmp_col, tmp_col2, tmp_col)
    tile.col_expand(tmp_col, col_broadcast)
    tile.sub(grad, col_broadcast, tmp_matrix)
    tile.col_expand(tmp_col2, col_broadcast)
    tile.div(tmp_matrix, col_broadcast, grad_next)
    tile.mov(grad_next, grad)


def _backward_row_norm(
    grad,
    x_inter,
    norm_sum,
    tmp_matrix,
    tmp_matrix2,
    tmp_row,
    tmp_row2,
    row_broadcast,
    grad_next,
    eps: float,
) -> None:
    tile.mul(grad, x_inter, tmp_matrix)
    tile.row_sum(tmp_matrix, tmp_matrix2, tmp_row)
    tile.mov(norm_sum, tmp_row2)
    _adds_inplace(tmp_row2, eps)
    tile.div(tmp_row, tmp_row2, tmp_row)
    tile.row_expand(tmp_row, row_broadcast)
    tile.sub(grad, row_broadcast, tmp_matrix)
    tile.row_expand(tmp_row2, row_broadcast)
    tile.div(tmp_matrix, row_broadcast, grad_next)
    tile.mov(grad_next, grad)


def _forward_tiles(src, *, mhc: int, repeat: int, eps: float, keep_stages: bool):
    tmp = pto.alloc_tile(tile_matrix)
    tmp2 = pto.alloc_tile(tile_matrix)
    row_stat = pto.alloc_tile(tile_row)
    col_stat = pto.alloc_tile(tile_col)
    row_broadcast = pto.alloc_tile(tile_matrix)
    col_broadcast = pto.alloc_tile(tile_matrix)
    centered = pto.alloc_tile(tile_matrix)
    exp_tile = pto.alloc_tile(tile_matrix)
    current = pto.alloc_tile(tile_matrix)
    next_tile = pto.alloc_tile(tile_matrix)

    stages = []
    sums = []
    _softmax_rows(src, tmp, row_stat, row_broadcast, centered, exp_tile, current)
    if keep_stages:
        softmax = pto.alloc_tile(tile_matrix)
        tile.mov(current, softmax)
        stages.append(softmax)
        sums.append(pto.alloc_tile(tile_row))

    _adds_inplace(current, eps)
    if keep_stages:
        stage = pto.alloc_tile(tile_matrix)
        tile.mov(current, stage)
        stages.append(stage)
    _col_normalize(current, tmp, col_stat, col_broadcast, next_tile, eps)
    if keep_stages:
        col_sum = pto.alloc_tile(tile_col)
        tile.mov(col_stat, col_sum)
        sums.append(col_sum)
    tile.mov(next_tile, current)

    for _ in range(repeat - 1):
        if keep_stages:
            stage = pto.alloc_tile(tile_matrix)
            tile.mov(current, stage)
            stages.append(stage)
        _row_normalize(current, tmp, row_stat, row_broadcast, next_tile, eps)
        if keep_stages:
            row_sum = pto.alloc_tile(tile_row)
            tile.mov(row_stat, row_sum)
            sums.append(row_sum)
        tile.mov(next_tile, current)

        if keep_stages:
            stage = pto.alloc_tile(tile_matrix)
            tile.mov(current, stage)
            stages.append(stage)
        _col_normalize(current, tmp, col_stat, col_broadcast, next_tile, eps)
        if keep_stages:
            col_sum = pto.alloc_tile(tile_col)
            tile.mov(col_stat, col_sum)
            sums.append(col_sum)
        tile.mov(next_tile, current)

    return current, stages, sums


def build_sinkhorn_normalize_fwd(mhc: int = 4, repeat: int = 10, eps: float = 1e-6):
    """Build tile f32 Sinkhorn forward for [num_tokens, mhc, mhc]."""
    if mhc <= 0:
        raise ValueError("mhc must be positive")
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    meta_data = lambda: _meta_data(mhc)

    def tilekernels_mhc_sinkhorn_normalize_fwd_kernel(
        x_ptr: "ptr_f32",
        out_ptr: "ptr_f32",
        num_tokens_i32: "i32",
    ) -> None:
        c1 = const(1)
        num_tokens = s.index_cast(num_tokens_i32)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(num_tokens, nblocks)
            token_start = bid * rows_per_core
            token_end = s.min_u(token_start + rows_per_core, num_tokens)
            x_tensor = _make_matrix_tensor(x_ptr, num_tokens, mhc)
            out_tensor = _make_matrix_tensor(out_ptr, num_tokens, mhc)
            src = pto.alloc_tile(tile_matrix)

            for token in pto.range(token_start, token_end, c1):
                pto.load(_slice_token(x_tensor, token, mhc), src)
                out, _, _ = _forward_tiles(
                    src, mhc=mhc, repeat=repeat, eps=eps, keep_stages=False
                )
                pto.store(out, _slice_token(out_tensor, token, mhc))

    tilekernels_mhc_sinkhorn_normalize_fwd_kernel.__name__ = (
        f"tilekernels_mhc_sinkhorn_fwd_m{mhc}_r{repeat}"
    )
    return to_ir_module(meta_data=meta_data)(
        tilekernels_mhc_sinkhorn_normalize_fwd_kernel
    )


def build_sinkhorn_normalize_bwd(mhc: int = 4, repeat: int = 10, eps: float = 1e-6):
    """Build tile f32 Sinkhorn backward for [num_tokens, mhc, mhc]."""
    if mhc <= 0:
        raise ValueError("mhc must be positive")
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    meta_data = lambda: _meta_data(mhc)

    def tilekernels_mhc_sinkhorn_normalize_bwd_kernel(
        grad_output_ptr: "ptr_f32",
        x_ptr: "ptr_f32",
        grad_input_ptr: "ptr_f32",
        num_tokens_i32: "i32",
    ) -> None:
        c1 = const(1)
        num_tokens = s.index_cast(num_tokens_i32)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(num_tokens, nblocks)
            token_start = bid * rows_per_core
            token_end = s.min_u(token_start + rows_per_core, num_tokens)
            grad_tensor = _make_matrix_tensor(grad_output_ptr, num_tokens, mhc)
            x_tensor = _make_matrix_tensor(x_ptr, num_tokens, mhc)
            out_tensor = _make_matrix_tensor(grad_input_ptr, num_tokens, mhc)

            x = pto.alloc_tile(tile_matrix)
            grad = pto.alloc_tile(tile_matrix)
            tmp_matrix = pto.alloc_tile(tile_matrix)
            tmp_matrix2 = pto.alloc_tile(tile_matrix)
            tmp_row = pto.alloc_tile(tile_row)
            tmp_row2 = pto.alloc_tile(tile_row)
            tmp_col = pto.alloc_tile(tile_col)
            tmp_col2 = pto.alloc_tile(tile_col)
            broadcast = pto.alloc_tile(tile_matrix)
            grad_next = pto.alloc_tile(tile_matrix)

            for token in pto.range(token_start, token_end, c1):
                pto.load(_slice_token(x_tensor, token, mhc), x)
                pto.load(_slice_token(grad_tensor, token, mhc), grad)
                _, stages, sums = _forward_tiles(
                    x, mhc=mhc, repeat=repeat, eps=eps, keep_stages=True
                )

                for stage_idx in range(2 * repeat - 1, 0, -1):
                    if stage_idx % 2 == 1:
                        _backward_col_norm(
                            grad,
                            stages[stage_idx],
                            sums[stage_idx],
                            tmp_matrix,
                            tmp_matrix2,
                            tmp_col,
                            tmp_col2,
                            broadcast,
                            grad_next,
                            eps,
                        )
                    else:
                        _backward_row_norm(
                            grad,
                            stages[stage_idx],
                            sums[stage_idx],
                            tmp_matrix,
                            tmp_matrix2,
                            tmp_row,
                            tmp_row2,
                            broadcast,
                            grad_next,
                            eps,
                        )

                tile.mul(grad, stages[0], tmp_matrix)
                tile.row_sum(tmp_matrix, tmp_matrix2, tmp_row)
                tile.row_expand(tmp_row, broadcast)
                tile.sub(grad, broadcast, tmp_matrix)
                tile.mul(tmp_matrix, stages[0], grad)
                pto.store(grad, _slice_token(out_tensor, token, mhc))

    tilekernels_mhc_sinkhorn_normalize_bwd_kernel.__name__ = (
        f"tilekernels_mhc_sinkhorn_bwd_m{mhc}_r{repeat}"
    )
    return to_ir_module(meta_data=meta_data)(
        tilekernels_mhc_sinkhorn_normalize_bwd_kernel
    )
