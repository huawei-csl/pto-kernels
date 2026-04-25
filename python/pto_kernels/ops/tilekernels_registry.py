"""TileKernels PTO-DSL cases mapped into pto-kernels op packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class KernelArg:
    name: str
    ctype: str
    pointer: bool = True


@dataclass(frozen=True)
class KernelCase:
    name: str
    family: str
    builder: str
    status: str
    description: str
    configs: tuple[dict, ...]
    args: tuple[KernelArg, ...]
    block_dim: str = "20"


_MOE_BASE_CONFIGS = tuple(
    {
        "num_topk": num_topk,
        "num_experts": num_experts // num_ep_ranks,
        "num_ep_ranks": num_ep_ranks,
    }
    for num_topk in (2, 6, 8, 9)
    for num_experts in (72, 256)
    for num_ep_ranks in (8, 64)
    if num_experts % num_ep_ranks == 0
)

_MOE_EDGE_CONFIGS = (
    {"num_topk": 1, "num_experts": 1, "num_ep_ranks": 1},
    {"num_topk": 7, "num_experts": 4, "num_ep_ranks": 64},
)

_MOE_COUNT_CONFIGS = tuple(
    sorted(
        {
        (cfg["num_topk"], cfg["num_experts"])
        for cfg in _MOE_BASE_CONFIGS + _MOE_EDGE_CONFIGS
        }
    )
)

_TOPK_GATE_CONFIGS = (
    {"num_experts": 72, "num_topk": 6},
    {"num_experts": 32, "num_topk": 6},
    {"num_experts": 64, "num_topk": 6},
    {"num_experts": 96, "num_topk": 6},
    {"num_experts": 16, "num_topk": 6},
    {"num_experts": 36, "num_topk": 6},
    {"num_experts": 108, "num_topk": 6},
    {"num_experts": 128, "num_topk": 6},
    {"num_experts": 144, "num_topk": 6},
    {"num_experts": 256, "num_topk": 8},
)

_TOPK_GROUP_CONFIGS = tuple(
    {
        "num_experts": num_experts,
        "num_groups": num_groups,
        "num_group_sum_topk": num_group_sum_topk,
        "num_topk_groups": num_topk_groups,
    }
    for num_experts in (72, 256)
    for num_groups in (4, 8, 12, 16)
    if num_experts % num_groups == 0
    for num_group_sum_topk in (1, 2)
    for num_topk_groups in (2, 4)
)

_INPLACE_UNIQUE_GROUP_CONFIGS = (
    {"num_topk": 1, "num_groups": 8},
    {"num_topk": 2, "num_groups": 8},
    {"num_topk": 6, "num_groups": 8},
    {"num_topk": 7, "num_groups": 16},
    {"num_topk": 8, "num_groups": 16},
    {"num_topk": 9, "num_groups": 72},
)

_CASES: tuple[KernelCase, ...] = (
    KernelCase(
        name="moe.normalize_weight",
        family="moe",
        builder="pto_kernels.ops.moe.normalize_weight.kernel:build_normalize_weight",
        status="implemented",
        description="Row-wise top-k weight normalization using PTO row reductions.",
        configs=(
            {"num_topk": 1},
            {"num_topk": 2},
            {"num_topk": 6},
            {"num_topk": 7},
            {"num_topk": 8},
            {"num_topk": 9},
        ),
        args=(
            KernelArg("topk_weights", "float"),
            KernelArg("denominator", "float"),
            KernelArg("normalized_weights", "float"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="moe.mask_indices_by_tp",
        family="moe",
        builder="pto_kernels.ops.moe.mask_indices_by_tp.kernel:build_mask_indices_by_tp",
        status="implemented",
        description="Scalar PTO remap/mask of expert indices by TP rank.",
        configs=tuple(
            {
                **cfg,
                "num_tp_ranks": num_tp_ranks,
            }
            for cfg in _MOE_BASE_CONFIGS + _MOE_EDGE_CONFIGS
            for num_tp_ranks in (2, 4, 8)
        ),
        args=(
            KernelArg("indices", "int64_t"),
            KernelArg("masked_indices", "int64_t"),
            KernelArg("per_gpu", "int32_t", pointer=False),
            KernelArg("per_dp", "int32_t", pointer=False),
            KernelArg("num_tp_ranks", "int32_t", pointer=False),
            KernelArg("tp_rank", "int32_t", pointer=False),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="moe.group_count",
        family="moe",
        builder="pto_kernels.ops.moe.group_count.kernel:build_group_count",
        status="implemented",
        description="Correctness-first scalar PTO expert-count scan.",
        configs=tuple(
            {"num_topk": num_topk, "num_groups": num_experts}
            for num_topk, num_experts in _MOE_COUNT_CONFIGS
        ),
        args=(
            KernelArg("group_idx", "int64_t"),
            KernelArg("out", "int32_t"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
        block_dim="1",
    ),
    KernelCase(
        name="moe.aux_fi",
        family="moe",
        builder="pto_kernels.ops.moe.aux_fi.kernel:build_aux_fi",
        status="implemented",
        description="Correctness-first scalar PTO auxiliary frequency indicator.",
        configs=tuple(
            {"num_topk": num_topk, "num_experts": num_experts}
            for num_topk, num_experts in _MOE_COUNT_CONFIGS
        ),
        args=(
            KernelArg("topk_idx", "int64_t"),
            KernelArg("out", "float"),
            KernelArg("num_aux_topk", "int32_t", pointer=False),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
        block_dim="1",
    ),
    KernelCase(
        name="moe.topk_gate",
        family="moe",
        builder="pto_kernels.ops.moe.topk.kernel:build_topk_gate",
        status="implemented",
        description="PTO sort32/mrgsort/gather TopK expert selection with PadValue::Min tail fill.",
        configs=_TOPK_GATE_CONFIGS,
        args=(
            KernelArg("scores", "float"),
            KernelArg("inidx", "uint32_t"),
            KernelArg("topk_idx", "uint32_t"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="moe.topk_sum_and_topk_group_idx",
        family="moe",
        builder="pto_kernels.ops.moe.topk.kernel:build_topk_sum_and_topk_group_idx",
        status="implemented",
        description="Grouped PTO sort TopK over per-group top score sums.",
        configs=_TOPK_GROUP_CONFIGS,
        args=(
            KernelArg("scores", "float"),
            KernelArg("inidx", "uint32_t"),
            KernelArg("group_idx", "uint32_t"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="moe.inplace_unique_group_indices",
        family="moe",
        builder="pto_kernels.ops.moe.inplace_unique_group_indices.kernel:build_inplace_unique_group_indices",
        status="implemented",
        description="Correctness-first per-token in-place duplicate group removal.",
        configs=_INPLACE_UNIQUE_GROUP_CONFIGS,
        args=(
            KernelArg("group_indices", "int64_t"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="transpose.transpose",
        family="transpose",
        builder="pto_kernels.ops.transpose.transpose.kernel:build_transpose",
        status="implemented",
        description="Dynamic 2-D transpose using PTO TTrans tiles.",
        configs=({"dtype": "bf16"}, {"dtype": "f32"}),
        args=(
            KernelArg("x", "__bf16"),
            KernelArg("out", "__bf16"),
            KernelArg("rows", "int32_t", pointer=False),
            KernelArg("cols", "int32_t", pointer=False),
            KernelArg("stride", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="transpose.batched_transpose",
        family="transpose",
        builder="pto_kernels.ops.transpose.transpose.kernel:build_batched_transpose",
        status="implemented",
        description="Dynamic batched 3-D transpose using PTO TTrans tiles.",
        configs=({"dtype": "bf16"}, {"dtype": "f32"}),
        args=(
            KernelArg("x", "__bf16"),
            KernelArg("out", "__bf16"),
            KernelArg("batches", "int32_t", pointer=False),
            KernelArg("rows", "int32_t", pointer=False),
            KernelArg("cols", "int32_t", pointer=False),
            KernelArg("stride", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="engram.fused_weight",
        family="engram",
        builder="pto_kernels.ops.engram.fused_weight.kernel:build_fused_weight",
        status="implemented",
        description="Engram bf16 weight_hidden * weight_embed -> fp32.",
        configs=({},),
        args=(
            KernelArg("weight_hidden", "__bf16"),
            KernelArg("weight_embed", "__bf16"),
            KernelArg("weight_fused", "float"),
            KernelArg("hc", "int32_t", pointer=False),
            KernelArg("hidden", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="engram.engram_hash",
        family="engram",
        builder="pto_kernels.ops.engram.engram_hash.kernel:build_engram_hash",
        status="implemented",
        description="Scalar/token-parallel Engram n-gram hash index generation.",
        configs=({"max_ngram_size": 3, "num_ngram_layers": 2, "num_embed_table_per_ngram": 8},),
        args=(
            KernelArg("ngram_token_ids", "int32_t"),
            KernelArg("multipliers", "int64_t"),
            KernelArg("vocab_sizes", "int32_t"),
            KernelArg("offsets", "int32_t"),
            KernelArg("output", "int32_t"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="engram.grad_w_reduce",
        family="engram",
        builder="pto_kernels.ops.engram.grad_w_reduce.kernel:build_grad_w_reduce",
        status="implemented",
        description="Engram f32 partial weight-gradient reduction and accumulation.",
        configs=(
            {"num_persistent_blocks": 4, "hc_mult": 4},
            {"num_persistent_blocks": 8, "hc_mult": 4},
        ),
        args=(
            KernelArg("grad_w_partial", "float"),
            KernelArg("weight_hidden", "__bf16"),
            KernelArg("weight_embed", "__bf16"),
            KernelArg("grad_weight_hidden", "float"),
            KernelArg("grad_weight_embed", "float"),
            KernelArg("hidden", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="engram.engram_gate_fwd",
        family="engram",
        builder="pto_kernels.ops.engram.gate.kernel:build_gate_fwd",
        status="implemented",
        description="Engram gate forward with saved dot/gate/rstd intermediates.",
        configs=(
            {"hidden_size": 2048, "hc_mult": 4},
            {"hidden_size": 4096, "hc_mult": 4},
            {"hidden_size": 7168, "hc_mult": 4},
        ),
        args=(
            KernelArg("hidden_states", "__bf16"),
            KernelArg("k", "__bf16"),
            KernelArg("v", "__bf16"),
            KernelArg("weight_fused", "float"),
            KernelArg("output", "__bf16"),
            KernelArg("dot_out", "float"),
            KernelArg("gate_score", "float"),
            KernelArg("rstd_x", "float"),
            KernelArg("rstd_k", "float"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="engram.engram_gate_bwd",
        family="engram",
        builder="pto_kernels.ops.engram.gate.kernel:build_gate_bwd",
        status="implemented",
        description="Engram gate backward with grad_x/grad_k/grad_v and grad_w_partial.",
        configs=(
            {"hidden_size": 2048, "num_persistent_blocks": 4, "hc_mult": 4},
            {"hidden_size": 4096, "num_persistent_blocks": 4, "hc_mult": 4},
            {"hidden_size": 7168, "num_persistent_blocks": 4, "hc_mult": 4},
        ),
        args=(
            KernelArg("grad_out", "__bf16"),
            KernelArg("hidden_states", "__bf16"),
            KernelArg("k", "__bf16"),
            KernelArg("v", "__bf16"),
            KernelArg("weight_fused", "float"),
            KernelArg("dot_in", "float"),
            KernelArg("gate_in", "float"),
            KernelArg("rstd_x_in", "float"),
            KernelArg("rstd_k_in", "float"),
            KernelArg("grad_x", "__bf16"),
            KernelArg("grad_k", "__bf16"),
            KernelArg("grad_v", "__bf16"),
            KernelArg("grad_w_partial", "float"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
        block_dim="4",
    ),
    KernelCase(
        name="mhc.expand_to_mhc_fwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.expand_to_mhc.kernel:build_expand_to_mhc_fwd",
        status="implemented",
        description="MHC bf16 expand forward.",
        configs=({"mhc_mult": 2}, {"mhc_mult": 4}, {"mhc_mult": 8}),
        args=(
            KernelArg("x", "__bf16"),
            KernelArg("out", "__bf16"),
            KernelArg("tokens", "int32_t", pointer=False),
            KernelArg("hidden", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="mhc.expand_to_mhc_bwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.expand_to_mhc.kernel:build_expand_to_mhc_bwd",
        status="implemented",
        description="MHC bf16 expand backward reduction over mhc axis.",
        configs=({"mhc_mult": 2}, {"mhc_mult": 4}, {"mhc_mult": 8}),
        args=(
            KernelArg("out_grad", "__bf16"),
            KernelArg("x_grad", "__bf16"),
            KernelArg("tokens", "int32_t", pointer=False),
            KernelArg("hidden", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="mhc.pre_apply_mix_fwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.pre_apply_mix.kernel:build_pre_apply_mix_fwd",
        status="implemented",
        description="MHC bf16 pre-apply mix forward reduction.",
        configs=({"mhc_mult": 4},),
        args=(
            KernelArg("x", "__bf16"),
            KernelArg("mix", "float"),
            KernelArg("out", "__bf16"),
            KernelArg("tokens", "int32_t", pointer=False),
            KernelArg("hidden", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="mhc.pre_apply_mix_bwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.pre_apply_mix.kernel:build_pre_apply_mix_bwd",
        status="implemented",
        description="MHC bf16 pre-apply mix backward with per-token mix gradients.",
        configs=({"mhc_mult": 4},),
        args=(
            KernelArg("out_grad", "__bf16"),
            KernelArg("x", "__bf16"),
            KernelArg("mix", "float"),
            KernelArg("x_grad", "__bf16"),
            KernelArg("mix_grad", "float"),
            KernelArg("tokens", "int32_t", pointer=False),
            KernelArg("hidden", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="mhc.post_fwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.post.kernel:build_post_fwd",
        status="implemented",
        description="MHC bf16 post forward residual combination.",
        configs=({"mhc_mult": 4},),
        args=(
            KernelArg("comb_res_mix", "float"),
            KernelArg("residual", "__bf16"),
            KernelArg("post_layer_mix", "float"),
            KernelArg("x", "__bf16"),
            KernelArg("out", "__bf16"),
            KernelArg("tokens", "int32_t", pointer=False),
            KernelArg("hidden", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="mhc.post_bwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.post.kernel:build_post_bwd",
        status="implemented",
        description="MHC bf16 post backward for residual, x, and mix gradients.",
        configs=({"mhc_mult": 4},),
        args=(
            KernelArg("out_grad", "__bf16"),
            KernelArg("comb_res_mix", "float"),
            KernelArg("residual", "__bf16"),
            KernelArg("post_layer_mix", "float"),
            KernelArg("x", "__bf16"),
            KernelArg("comb_res_mix_grad", "float"),
            KernelArg("residual_grad", "__bf16"),
            KernelArg("post_layer_mix_grad", "float"),
            KernelArg("x_grad", "__bf16"),
            KernelArg("tokens", "int32_t", pointer=False),
            KernelArg("hidden", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="mhc.pre_norm_fn_fwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.norm_fn.kernel:build_pre_norm_fn_fwd",
        status="implemented",
        description="MHC f32 RMS-normalized residual/FN projection forward.",
        configs=(
            {"hidden_size": 1280, "mhc_mult": 4, "eps": 1e-6},
            {"hidden_size": 2560, "mhc_mult": 4, "eps": 1e-6},
            {"hidden_size": 7168, "mhc_mult": 4, "eps": 1e-6},
        ),
        args=(
            KernelArg("residual", "__bf16"),
            KernelArg("mhc_fn", "float"),
            KernelArg("output", "float"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="mhc.fn_normw_merge_fwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.norm_fn.kernel:build_fn_normw_merge_fwd",
        status="implemented",
        description="MHC f32 optional FN/norm-weight merge forward.",
        configs=(
            {"hidden_size": 1280, "mhc_mult": 4},
            {"hidden_size": 2560, "mhc_mult": 4},
            {"hidden_size": 7168, "mhc_mult": 4},
        ),
        args=(
            KernelArg("fn", "float"),
            KernelArg("normw", "float"),
            KernelArg("out_fn", "float"),
        ),
    ),
    KernelCase(
        name="mhc.fn_normw_merge_bwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.norm_fn.kernel:build_fn_normw_merge_bwd",
        status="implemented",
        description="MHC f32 optional FN/norm-weight merge backward.",
        configs=(
            {"hidden_size": 1280, "mhc_mult": 4},
            {"hidden_size": 2560, "mhc_mult": 4},
            {"hidden_size": 7168, "mhc_mult": 4},
        ),
        args=(
            KernelArg("fn", "float"),
            KernelArg("normw", "float"),
            KernelArg("out_fn_grad", "float"),
            KernelArg("fn_grad", "float"),
            KernelArg("normw_grad", "float"),
        ),
    ),
    KernelCase(
        name="mhc.sinkhorn_normalize_fwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.sinkhorn_normalize.kernel:build_sinkhorn_normalize_fwd",
        status="implemented",
        description="Tile f32 Sinkhorn forward for 4x4 MHC mix matrices.",
        configs=({"mhc": 4, "repeat": 10, "eps": 1e-6},),
        args=(
            KernelArg("x", "float"),
            KernelArg("out", "float"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="mhc.head_compute_mix_fwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.head_compute_mix.kernel:build_head_compute_mix_fwd",
        status="implemented",
        description="MHC f32 sigmoid head mix forward.",
        configs=({"mhc_mult": 4, "eps": 1e-6},),
        args=(
            KernelArg("input_mix", "float"),
            KernelArg("mhc_scale", "float"),
            KernelArg("mhc_base", "float"),
            KernelArg("output_mix", "float"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="mhc.head_compute_mix_bwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.head_compute_mix.kernel:build_head_compute_mix_bwd",
        status="implemented",
        description="MHC f32 sigmoid head mix backward with one partial row.",
        configs=({"mhc_mult": 4},),
        args=(
            KernelArg("output_mix_grad", "float"),
            KernelArg("input_mix", "float"),
            KernelArg("mhc_scale", "float"),
            KernelArg("mhc_base", "float"),
            KernelArg("input_mix_grad", "float"),
            KernelArg("mhc_scale_grad_partial", "float"),
            KernelArg("mhc_base_grad_partial", "float"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
        block_dim="1",
    ),
    KernelCase(
        name="mhc.pre_split_mixes_fwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.pre_split_mixes.kernel:build_pre_split_mixes_fwd",
        status="implemented",
        description="MHC f32 pre-split mix forward for pre/post/comb tensors.",
        configs=({"mhc_mult": 4, "mhc_post_mult_value": 2.0, "mhc_pre_eps": 1e-2},),
        args=(
            KernelArg("input_mixes", "float"),
            KernelArg("mhc_scale", "float"),
            KernelArg("mhc_base", "float"),
            KernelArg("pre_layer_mix", "float"),
            KernelArg("post_layer_mix", "float"),
            KernelArg("comb_res_mix", "float"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
    ),
    KernelCase(
        name="mhc.pre_split_mixes_bwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.pre_split_mixes.kernel:build_pre_split_mixes_bwd",
        status="implemented",
        description="MHC f32 pre-split mix backward with one partial gradient row.",
        configs=({"mhc_mult": 4, "mhc_post_mult_value": 2.0},),
        args=(
            KernelArg("pre_layer_mix_grad", "float"),
            KernelArg("post_layer_mix_grad", "float"),
            KernelArg("comb_res_mix_grad", "float"),
            KernelArg("input_mixes", "float"),
            KernelArg("post_layer_mix", "float"),
            KernelArg("mhc_scale", "float"),
            KernelArg("mhc_base", "float"),
            KernelArg("input_mixes_grad", "float"),
            KernelArg("mhc_scale_grad_partial", "float"),
            KernelArg("mhc_base_grad_partial", "float"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
        block_dim="1",
    ),
    KernelCase(
        name="mhc.sinkhorn_normalize_bwd",
        family="mhc",
        builder="pto_kernels.ops.mhc.sinkhorn_normalize.kernel:build_sinkhorn_normalize_bwd",
        status="implemented",
        description="Tile f32 Sinkhorn backward for 4x4 MHC mix matrices.",
        configs=({"mhc": 4, "repeat": 10, "eps": 1e-6},),
        args=(
            KernelArg("grad_output", "float"),
            KernelArg("x", "float"),
            KernelArg("grad_input", "float"),
            KernelArg("num_tokens", "int32_t", pointer=False),
        ),
    ),
)


def iter_cases() -> Iterable[KernelCase]:
    return iter(_CASES)
