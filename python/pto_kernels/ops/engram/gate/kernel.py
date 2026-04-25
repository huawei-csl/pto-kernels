from mlir.dialects import arith, pto as _pto_dialect
from mlir.ir import Context

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


def _maxs(src, value: float, dst) -> None:
    _pto_dialect.TMaxSOp(src, _f32(value), dst)


def _cmps(src, value: float, dst, mode) -> None:
    _pto_dialect.TCmpSOp(
        src,
        _f32(value),
        dst,
        cmpMode=_pto_dialect.CmpModeAttr.get(Context.current, mode),
    )


def _select(mask, lhs, rhs, tmp, dst) -> None:
    _pto_dialect.TSelOp(mask, lhs, rhs, tmp, dst)


def _cvt(src, dst) -> None:
    _pto_dialect.TCvtOp(src, dst)


def _sigmoid_from_logits(logits, neg, exp_tile, denom, out) -> None:
    _muls(logits, -1.0, neg)
    tile.exp(neg, exp_tile)
    _adds(exp_tile, 1.0, denom)
    tile.reciprocal(denom, out)


def _meta_data(hidden_size: int):
    bf16 = pto.bfloat16
    f32 = pto.float32
    return {
        "ptr_bf16": ptr(bf16),
        "ptr_f32": ptr(f32),
        "i32": pto.int32,
        "tensor2_bf16": pto.TensorType(rank=2, dtype=bf16),
        "tensor2_f32": pto.TensorType(rank=2, dtype=f32),
        "sub_row_bf16": pto.SubTensorType(shape=[1, _TILE], dtype=bf16),
        "sub_row_f32": pto.SubTensorType(shape=[1, _TILE], dtype=f32),
        "sub_scalar": pto.SubTensorType(shape=[1, 1], dtype=f32),
        "tile_bf16": tile_type(bf16, [1, _TILE], valid_shape=[1, -1]),
        "tile_f32": tile_type(f32, [1, _TILE], valid_shape=[1, -1]),
        "tile_scalar": tile_type(f32, [1, 1]),
    }


def _row_view(tensor, row, col, cols):
    c1 = const(1)
    return pto.slice_view(
        sub_row_bf16,
        source=tensor,
        offsets=[row, col],
        sizes=[c1, cols],
    )


def _row_view_f32(tensor, row, col, cols):
    c1 = const(1)
    return pto.slice_view(
        sub_row_f32,
        source=tensor,
        offsets=[row, col],
        sizes=[c1, cols],
    )


def _scalar_view(tensor, token, head):
    c1 = const(1)
    return pto.slice_view(
        sub_scalar,
        source=tensor,
        offsets=[token, head],
        sizes=[c1, c1],
    )


def build_gate_fwd(
    hidden_size: int,
    hc_mult: int = 4,
    eps: float = 1e-20,
    clamp_value: float = 1e-6,
):
    """Build Engram gate forward with saved backward intermediates.

    The TileKernels CUDA path specializes on hidden size. This PTO-DSL port
    does the same so the RMSNorm scale and signed-sqrt gate constants stay in
    tile ops instead of scalar math.
    """
    if hidden_size <= 0 or hidden_size % _TILE != 0:
        raise ValueError(f"hidden_size must be a positive multiple of {_TILE}")
    if hc_mult <= 0:
        raise ValueError("hc_mult must be positive")

    num_tiles = hidden_size // _TILE
    inv_hidden = 1.0 / float(hidden_size)
    gate_scalar = hidden_size**-0.5

    def tilekernels_engram_gate_fwd_kernel(
        hidden_states_ptr: "ptr_bf16",
        k_ptr: "ptr_bf16",
        v_ptr: "ptr_bf16",
        weight_fused_ptr: "ptr_f32",
        output_ptr: "ptr_bf16",
        dot_out_ptr: "ptr_f32",
        gate_score_ptr: "ptr_f32",
        rstd_x_ptr: "ptr_f32",
        rstd_k_ptr: "ptr_f32",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(_TILE)
        c_hidden = const(hidden_size)
        c_hc = const(hc_mult)
        num_tokens = s.index_cast(num_tokens_i32)

        hidden_states = pto.as_tensor(
            tensor2_bf16,
            ptr=hidden_states_ptr,
            shape=[num_tokens * c_hc, c_hidden],
            strides=[c_hidden, c1],
        )
        key = pto.as_tensor(
            tensor2_bf16,
            ptr=k_ptr,
            shape=[num_tokens * c_hc, c_hidden],
            strides=[c_hidden, c1],
        )
        value = pto.as_tensor(
            tensor2_bf16,
            ptr=v_ptr,
            shape=[num_tokens, c_hidden],
            strides=[c_hidden, c1],
        )
        weight_fused = pto.as_tensor(
            tensor2_f32,
            ptr=weight_fused_ptr,
            shape=[c_hc, c_hidden],
            strides=[c_hidden, c1],
        )
        output = pto.as_tensor(
            tensor2_bf16,
            ptr=output_ptr,
            shape=[num_tokens * c_hc, c_hidden],
            strides=[c_hidden, c1],
        )
        dot_out = pto.as_tensor(
            tensor2_f32,
            ptr=dot_out_ptr,
            shape=[num_tokens, c_hc],
            strides=[c_hc, c1],
        )
        gate_score = pto.as_tensor(
            tensor2_f32,
            ptr=gate_score_ptr,
            shape=[num_tokens, c_hc],
            strides=[c_hc, c1],
        )
        rstd_x = pto.as_tensor(
            tensor2_f32,
            ptr=rstd_x_ptr,
            shape=[num_tokens, c_hc],
            strides=[c_hc, c1],
        )
        rstd_k = pto.as_tensor(
            tensor2_f32,
            ptr=rstd_k_ptr,
            shape=[num_tokens, c_hc],
            strides=[c_hc, c1],
        )

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            nblocks = s.index_cast(pto.get_block_num())
            rows_per_core = s.ceil_div(num_tokens, nblocks)
            token_start = bid * rows_per_core
            token_end = s.min_u(token_start + rows_per_core, num_tokens)

            x_bf16 = pto.alloc_tile(tile_bf16, valid_col=c_tile)
            k_bf16 = pto.alloc_tile(tile_bf16, valid_col=c_tile)
            v_bf16 = pto.alloc_tile(tile_bf16, valid_col=c_tile)
            x = pto.alloc_tile(tile_f32, valid_col=c_tile)
            k_f32 = pto.alloc_tile(tile_f32, valid_col=c_tile)
            v_f32 = pto.alloc_tile(tile_f32, valid_col=c_tile)
            weight = pto.alloc_tile(tile_f32, valid_col=c_tile)
            tmp = pto.alloc_tile(tile_f32, valid_col=c_tile)
            tmp2 = pto.alloc_tile(tile_f32, valid_col=c_tile)
            out_f32 = pto.alloc_tile(tile_f32, valid_col=c_tile)
            out_bf16 = pto.alloc_tile(tile_bf16, valid_col=c_tile)

            chunk_sum = pto.alloc_tile(tile_scalar)
            x_sum = pto.alloc_tile(tile_scalar)
            k_sum = pto.alloc_tile(tile_scalar)
            dot_sum = pto.alloc_tile(tile_scalar)
            norm_dot = pto.alloc_tile(tile_scalar)
            abs_dot = pto.alloc_tile(tile_scalar)
            sqrt_dot = pto.alloc_tile(tile_scalar)
            neg_sqrt_dot = pto.alloc_tile(tile_scalar)
            signed_sqrt = pto.alloc_tile(tile_scalar)
            zero_scalar = pto.alloc_tile(tile_scalar)
            gate = pto.alloc_tile(tile_scalar)
            gate_row = pto.alloc_tile(tile_f32, valid_col=c_tile)
            neg = pto.alloc_tile(tile_scalar)
            exp_tile = pto.alloc_tile(tile_scalar)
            denom = pto.alloc_tile(tile_scalar)
            mask = pto.alloc_tile(tile_scalar)
            zero_mask = pto.alloc_tile(tile_scalar)
            sel_tmp = pto.alloc_tile(tile_scalar)

            for token in pto.range(token_start, token_end, c1):
                for head_idx in range(hc_mult):
                    head = const(head_idx)
                    row = token * c_hc + head

                    for tile_idx in range(num_tiles):
                        col = const(tile_idx * _TILE)
                        pto.load(_row_view(hidden_states, row, col, c_tile), x_bf16)
                        pto.load(_row_view(key, row, col, c_tile), k_bf16)
                        pto.load(_row_view_f32(weight_fused, head, col, c_tile), weight)
                        _cvt(x_bf16, x)
                        _cvt(k_bf16, k_f32)

                        tile.mul(x, x, tmp)
                        tile.row_sum(tmp, tmp2, chunk_sum)
                        if tile_idx == 0:
                            tile.mov(chunk_sum, x_sum)
                        else:
                            tile.add(x_sum, chunk_sum, x_sum)

                        tile.mul(k_f32, k_f32, tmp)
                        tile.row_sum(tmp, tmp2, chunk_sum)
                        if tile_idx == 0:
                            tile.mov(chunk_sum, k_sum)
                        else:
                            tile.add(k_sum, chunk_sum, k_sum)

                        tile.mul(x, weight, tmp)
                        tile.mul(tmp, k_f32, tmp)
                        tile.row_sum(tmp, tmp2, chunk_sum)
                        if tile_idx == 0:
                            tile.mov(chunk_sum, dot_sum)
                        else:
                            tile.add(dot_sum, chunk_sum, dot_sum)

                    pto.store(dot_sum, _scalar_view(dot_out, token, head))

                    _muls(x_sum, inv_hidden, x_sum)
                    _adds(x_sum, eps, x_sum)
                    tile.rsqrt(x_sum, x_sum)
                    pto.store(x_sum, _scalar_view(rstd_x, token, head))

                    _muls(k_sum, inv_hidden, k_sum)
                    _adds(k_sum, eps, k_sum)
                    tile.rsqrt(k_sum, k_sum)
                    pto.store(k_sum, _scalar_view(rstd_k, token, head))

                    tile.mul(dot_sum, x_sum, norm_dot)
                    tile.mul(norm_dot, k_sum, norm_dot)
                    _muls(norm_dot, gate_scalar, norm_dot)

                    tile.abs(norm_dot, abs_dot)
                    _maxs(abs_dot, clamp_value, abs_dot)
                    tile.sqrt(abs_dot, sqrt_dot)
                    _muls(sqrt_dot, -1.0, neg_sqrt_dot)
                    _cmps(norm_dot, 0.0, mask, _pto_dialect.CmpMode.LT)
                    _select(mask, neg_sqrt_dot, sqrt_dot, sel_tmp, signed_sqrt)
                    _muls(sqrt_dot, 0.0, zero_scalar)
                    _cmps(norm_dot, 0.0, zero_mask, _pto_dialect.CmpMode.EQ)
                    _select(zero_mask, zero_scalar, signed_sqrt, sel_tmp, signed_sqrt)
                    _sigmoid_from_logits(signed_sqrt, neg, exp_tile, denom, gate)
                    pto.store(gate, _scalar_view(gate_score, token, head))
                    tile.row_expand(gate, gate_row)

                    for tile_idx in range(num_tiles):
                        col = const(tile_idx * _TILE)
                        pto.load(_row_view(hidden_states, row, col, c_tile), x_bf16)
                        pto.load(_row_view(value, token, col, c_tile), v_bf16)
                        _cvt(x_bf16, x)
                        _cvt(v_bf16, v_f32)
                        tile.mul(gate_row, v_f32, tmp)
                        tile.add(x, tmp, out_f32)
                        _cvt(out_f32, out_bf16)
                        pto.store(out_bf16, _row_view(output, row, col, c_tile))

    tilekernels_engram_gate_fwd_kernel.__name__ = (
        f"tilekernels_engram_gate_fwd_h{hidden_size}_m{hc_mult}"
    )
    return to_ir_module(meta_data=lambda: _meta_data(hidden_size))(
        tilekernels_engram_gate_fwd_kernel
    )


def build_gate_bwd(
    hidden_size: int,
    num_persistent_blocks: int = 4,
    hc_mult: int = 4,
    clamp_value: float = 1e-6,
):
    """Build Engram gate backward with per-block partial weight gradients."""
    if hidden_size <= 0 or hidden_size % _TILE != 0:
        raise ValueError(f"hidden_size must be a positive multiple of {_TILE}")
    if num_persistent_blocks <= 0:
        raise ValueError("num_persistent_blocks must be positive")
    if hc_mult <= 0:
        raise ValueError("hc_mult must be positive")

    num_tiles = hidden_size // _TILE
    inv_hidden = 1.0 / float(hidden_size)
    gate_scalar = hidden_size**-0.5

    def tilekernels_engram_gate_bwd_kernel(
        grad_out_ptr: "ptr_bf16",
        hidden_states_ptr: "ptr_bf16",
        k_ptr: "ptr_bf16",
        v_ptr: "ptr_bf16",
        weight_fused_ptr: "ptr_f32",
        dot_in_ptr: "ptr_f32",
        gate_in_ptr: "ptr_f32",
        rstd_x_in_ptr: "ptr_f32",
        rstd_k_in_ptr: "ptr_f32",
        grad_x_ptr: "ptr_bf16",
        grad_k_ptr: "ptr_bf16",
        grad_v_ptr: "ptr_bf16",
        grad_w_partial_ptr: "ptr_f32",
        num_tokens_i32: "i32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(_TILE)
        c_hidden = const(hidden_size)
        c_hc = const(hc_mult)
        c_blocks = const(num_persistent_blocks)
        num_tokens = s.index_cast(num_tokens_i32)

        grad_out = pto.as_tensor(
            tensor2_bf16,
            ptr=grad_out_ptr,
            shape=[num_tokens * c_hc, c_hidden],
            strides=[c_hidden, c1],
        )
        hidden_states = pto.as_tensor(
            tensor2_bf16,
            ptr=hidden_states_ptr,
            shape=[num_tokens * c_hc, c_hidden],
            strides=[c_hidden, c1],
        )
        key = pto.as_tensor(
            tensor2_bf16,
            ptr=k_ptr,
            shape=[num_tokens * c_hc, c_hidden],
            strides=[c_hidden, c1],
        )
        value = pto.as_tensor(
            tensor2_bf16,
            ptr=v_ptr,
            shape=[num_tokens, c_hidden],
            strides=[c_hidden, c1],
        )
        weight_fused = pto.as_tensor(
            tensor2_f32,
            ptr=weight_fused_ptr,
            shape=[c_hc, c_hidden],
            strides=[c_hidden, c1],
        )
        dot_in = pto.as_tensor(
            tensor2_f32,
            ptr=dot_in_ptr,
            shape=[num_tokens, c_hc],
            strides=[c_hc, c1],
        )
        gate_in = pto.as_tensor(
            tensor2_f32,
            ptr=gate_in_ptr,
            shape=[num_tokens, c_hc],
            strides=[c_hc, c1],
        )
        rstd_x_in = pto.as_tensor(
            tensor2_f32,
            ptr=rstd_x_in_ptr,
            shape=[num_tokens, c_hc],
            strides=[c_hc, c1],
        )
        rstd_k_in = pto.as_tensor(
            tensor2_f32,
            ptr=rstd_k_in_ptr,
            shape=[num_tokens, c_hc],
            strides=[c_hc, c1],
        )
        grad_x = pto.as_tensor(
            tensor2_bf16,
            ptr=grad_x_ptr,
            shape=[num_tokens * c_hc, c_hidden],
            strides=[c_hidden, c1],
        )
        grad_k = pto.as_tensor(
            tensor2_bf16,
            ptr=grad_k_ptr,
            shape=[num_tokens * c_hc, c_hidden],
            strides=[c_hidden, c1],
        )
        grad_v = pto.as_tensor(
            tensor2_bf16,
            ptr=grad_v_ptr,
            shape=[num_tokens, c_hidden],
            strides=[c_hidden, c1],
        )
        grad_w_partial = pto.as_tensor(
            tensor2_f32,
            ptr=grad_w_partial_ptr,
            shape=[c_blocks * c_hc, c_hidden],
            strides=[c_hidden, c1],
        )

        with pto.vector_section():
            pid = s.index_cast(pto.get_block_idx())
            rows_per_block = s.ceil_div(num_tokens, c_blocks)
            token_start = pid * rows_per_block
            token_end = s.min_u(token_start + rows_per_block, num_tokens)

            go_bf16 = pto.alloc_tile(tile_bf16, valid_col=c_tile)
            x_bf16 = pto.alloc_tile(tile_bf16, valid_col=c_tile)
            k_bf16 = pto.alloc_tile(tile_bf16, valid_col=c_tile)
            v_bf16 = pto.alloc_tile(tile_bf16, valid_col=c_tile)
            grad_bf16 = pto.alloc_tile(tile_bf16, valid_col=c_tile)

            go = pto.alloc_tile(tile_f32, valid_col=c_tile)
            x = pto.alloc_tile(tile_f32, valid_col=c_tile)
            k_f32 = pto.alloc_tile(tile_f32, valid_col=c_tile)
            v_f32 = pto.alloc_tile(tile_f32, valid_col=c_tile)
            weight = pto.alloc_tile(tile_f32, valid_col=c_tile)
            tmp = pto.alloc_tile(tile_f32, valid_col=c_tile)
            tmp2 = pto.alloc_tile(tile_f32, valid_col=c_tile)
            grad_v_acc = pto.alloc_tile(tile_f32, valid_col=c_tile)
            grad_x_f32 = pto.alloc_tile(tile_f32, valid_col=c_tile)
            grad_k_f32 = pto.alloc_tile(tile_f32, valid_col=c_tile)
            grad_w_acc = pto.alloc_tile(tile_f32, valid_col=c_tile)

            chunk_sum = pto.alloc_tile(tile_scalar)
            dldg = pto.alloc_tile(tile_scalar)
            dldg_r = pto.alloc_tile(tile_scalar)
            dot = pto.alloc_tile(tile_scalar)
            gate = pto.alloc_tile(tile_scalar)
            rstd_x = pto.alloc_tile(tile_scalar)
            rstd_k = pto.alloc_tile(tile_scalar)
            coeff = pto.alloc_tile(tile_scalar)
            abs_dot = pto.alloc_tile(tile_scalar)
            norm_abs = pto.alloc_tile(tile_scalar)
            sqrt_arg = pto.alloc_tile(tile_scalar)
            gate_deriv = pto.alloc_tile(tile_scalar)
            one_minus_gate = pto.alloc_tile(tile_scalar)
            dot_x = pto.alloc_tile(tile_scalar)
            dot_k = pto.alloc_tile(tile_scalar)
            zero_scalar = pto.alloc_tile(tile_scalar)
            mask = pto.alloc_tile(tile_scalar)
            sel_tmp = pto.alloc_tile(tile_scalar)
            dldg_row = pto.alloc_tile(tile_f32, valid_col=c_tile)
            gate_row = pto.alloc_tile(tile_f32, valid_col=c_tile)
            dot_x_row = pto.alloc_tile(tile_f32, valid_col=c_tile)
            dot_k_row = pto.alloc_tile(tile_f32, valid_col=c_tile)

            for token in pto.range(token_start, token_end, c1):
                for tile_idx in range(num_tiles):
                    col = const(tile_idx * _TILE)
                    for head_idx in range(hc_mult):
                        head = const(head_idx)
                        row = token * c_hc + head
                        pto.load(_scalar_view(gate_in, token, head), gate)
                        tile.row_expand(gate, gate_row)
                        pto.load(_row_view(grad_out, row, col, c_tile), go_bf16)
                        _cvt(go_bf16, go)
                        tile.mul(go, gate_row, tmp)
                        if head_idx == 0:
                            tile.mov(tmp, grad_v_acc)
                        else:
                            tile.add(grad_v_acc, tmp, grad_v_acc)
                    _cvt(grad_v_acc, grad_bf16)
                    pto.store(grad_bf16, _row_view(grad_v, token, col, c_tile))

            for head_idx in range(hc_mult):
                head = const(head_idx)
                for tile_idx in range(num_tiles):
                    col = const(tile_idx * _TILE)
                    pto.load(_row_view_f32(weight_fused, head, col, c_tile), weight)
                    _muls(weight, 0.0, grad_w_acc)

                    for token in pto.range(token_start, token_end, c1):
                        row = token * c_hc + head

                        for reduce_idx in range(num_tiles):
                            reduce_col = const(reduce_idx * _TILE)
                            pto.load(
                                _row_view(grad_out, row, reduce_col, c_tile),
                                go_bf16,
                            )
                            pto.load(
                                _row_view(value, token, reduce_col, c_tile),
                                v_bf16,
                            )
                            _cvt(go_bf16, go)
                            _cvt(v_bf16, v_f32)
                            tile.mul(go, v_f32, tmp)
                            tile.row_sum(tmp, tmp2, chunk_sum)
                            if reduce_idx == 0:
                                tile.mov(chunk_sum, dldg)
                            else:
                                tile.add(dldg, chunk_sum, dldg)

                        pto.load(_scalar_view(dot_in, token, head), dot)
                        pto.load(_scalar_view(gate_in, token, head), gate)
                        pto.load(_scalar_view(rstd_x_in, token, head), rstd_x)
                        pto.load(_scalar_view(rstd_k_in, token, head), rstd_k)

                        tile.mul(rstd_x, rstd_k, coeff)
                        _muls(coeff, gate_scalar, coeff)
                        tile.abs(dot, abs_dot)
                        tile.mul(abs_dot, coeff, norm_abs)
                        _cmps(norm_abs, clamp_value, mask, _pto_dialect.CmpMode.LT)
                        _maxs(abs_dot, 1e-20, abs_dot)
                        tile.div(coeff, abs_dot, sqrt_arg)
                        tile.sqrt(sqrt_arg, sqrt_arg)
                        _muls(gate, -1.0, one_minus_gate)
                        _adds(one_minus_gate, 1.0, one_minus_gate)
                        tile.mul(gate, one_minus_gate, gate_deriv)
                        _muls(gate_deriv, 0.5, gate_deriv)
                        tile.mul(dldg, gate_deriv, dldg_r)
                        tile.mul(dldg_r, sqrt_arg, dldg_r)
                        _muls(dldg, 0.0, zero_scalar)
                        _select(mask, zero_scalar, dldg_r, sel_tmp, dldg_r)
                        tile.row_expand(dldg_r, dldg_row)

                        tile.mul(rstd_x, rstd_x, dot_x)
                        tile.mul(dot_x, dot, dot_x)
                        _muls(dot_x, inv_hidden, dot_x)
                        tile.row_expand(dot_x, dot_x_row)

                        tile.mul(rstd_k, rstd_k, dot_k)
                        tile.mul(dot_k, dot, dot_k)
                        _muls(dot_k, inv_hidden, dot_k)
                        tile.row_expand(dot_k, dot_k_row)

                        pto.load(_row_view(grad_out, row, col, c_tile), go_bf16)
                        pto.load(_row_view(hidden_states, row, col, c_tile), x_bf16)
                        pto.load(_row_view(key, row, col, c_tile), k_bf16)
                        _cvt(go_bf16, go)
                        _cvt(x_bf16, x)
                        _cvt(k_bf16, k_f32)

                        tile.mul(k_f32, weight, tmp)
                        tile.mul(x, dot_x_row, tmp2)
                        tile.sub(tmp, tmp2, tmp)
                        tile.mul(dldg_row, tmp, tmp)
                        tile.add(go, tmp, grad_x_f32)
                        _cvt(grad_x_f32, grad_bf16)
                        pto.store(grad_bf16, _row_view(grad_x, row, col, c_tile))

                        tile.mul(x, weight, tmp)
                        tile.mul(k_f32, dot_k_row, tmp2)
                        tile.sub(tmp, tmp2, tmp)
                        tile.mul(dldg_row, tmp, grad_k_f32)
                        _cvt(grad_k_f32, grad_bf16)
                        pto.store(grad_bf16, _row_view(grad_k, row, col, c_tile))

                        tile.mul(x, k_f32, tmp)
                        tile.mul(dldg_row, tmp, tmp)
                        tile.add(grad_w_acc, tmp, grad_w_acc)

                    partial_row = pid * c_hc + head
                    pto.store(
                        grad_w_acc,
                        _row_view_f32(grad_w_partial, partial_row, col, c_tile),
                    )

    tilekernels_engram_gate_bwd_kernel.__name__ = (
        f"tilekernels_engram_gate_bwd_h{hidden_size}_p{num_persistent_blocks}_m{hc_mult}"
    )
    return to_ir_module(meta_data=lambda: _meta_data(hidden_size))(
        tilekernels_engram_gate_bwd_kernel
    )
