# pto-kernels

A collection of high-performance custom kernels for **Ascend NPUs**, built on top of
[pto-isa](https://github.com/PTO-ISA/pto-isa) — the Parallel Tile Operation virtual
instruction set architecture designed by Ascend CANN.

PTO focuses on tile-level operations, enabling efficient, composable kernel development
targeting Huawei's Ascend AI processors.


---

## Prerequisites

- A configured **torch-npu** environment
- Ascend toolkit installed at `/usr/local/Ascend/ascend-toolkit`

Run the one-time setup before building:

```bash
make setup_once
```

## Install repository using pip

The repository is "pip installable", i.e.,

```bash
export CMAKE_GENERATOR="Unix Makefiles" && pip install -v git+https://github.com/huawei-csl/pto-kernels.git
```

---

## Build

```bash
bash scripts/source_env.sh
pip3 install -r requirements.txt
make build_wheel
```

This produces an installable Python wheel:

```text
pto_kernels-X.Y.Z-*.whl
```

---

## Installation

```bash
pip install --force-reinstall pto_kernels-*.whl
```

---

## Testing

```bash
make test
```

## Contributor Workflow

Use the lightweight local path first:

```bash
make setup
make test
make check
```

Use the CANN/NPU path only on a configured Ascend host:

```bash
make bootstrap
make check-env
make test-npu
```

Useful entrypoints:

- `make help`: list common commands.
- `make build-wheel`: build the Python wheel.
- `docs/cann_recipes_infer_notes.md`: notes from the pinned CANN recipes
  reference.
- `skills/pto-kernel-writer/SKILL.md`: repo-native kernel writer workflow.
- `templates/kernel_writer/`: starter files for new kernel entries.

---

## Kernel Inventory

| Name | Category | Description | Status |
| --- | --- | --- | --- |
| `moe/normalize_weight` | MoE | Normalizes top-k routing weights and writes per-token denominators. | `local_ptoas` |
| `moe/mask_indices_by_tp` | MoE | Masks and remaps expert indices for tensor-parallel ranks. | `local_ptoas` |
| `moe/group_count` | MoE | Counts routed tokens per MoE group or expert bucket. | `local_ptoas` |
| `moe/aux_fi` | MoE | Computes auxiliary load-balancing frequency information for MoE routing. | `local_ptoas` |
| `moe/topk_gate` | MoE | Selects top-k experts from routing scores. | `local_ptoas` |
| `moe/topk_sum_and_topk_group_idx` | MoE | Computes grouped top-k routing scores and selected group indices. | `local_ptoas` |
| `moe/inplace_unique_group_indices` | MoE | Deduplicates per-token group indices in place. | `local_ptoas` |
| `transpose/transpose` | Transpose | Transposes a 2D tensor. | `local_ptoas` |
| `transpose/batched_transpose` | Transpose | Transposes each matrix in a batched tensor. | `local_ptoas` |
| `engram/fused_weight` | Engram | Fuses Engram hidden and embedding weights into an f32 weight buffer. | `local_ptoas` |
| `engram/engram_hash` | Engram | Computes Engram hash ids from n-gram token inputs. | `local_ptoas` |
| `engram/grad_w_reduce` | Engram | Reduces partial Engram weight gradients. | `local_ptoas` |
| `engram/engram_gate_fwd` | Engram | Runs the Engram gate forward path and saves intermediates. | `local_ptoas` |
| `engram/engram_gate_bwd` | Engram | Runs the Engram gate backward path for activations and partial weights. | `local_ptoas` |
| `mhc/expand_to_mhc_fwd` | MHC | Expands activations across the MHC dimension. | `local_ptoas` |
| `mhc/expand_to_mhc_bwd` | MHC | Reduces gradients from the expanded MHC dimension. | `local_ptoas` |
| `mhc/pre_apply_mix_fwd` | MHC | Applies pre-layer MHC mixing in the forward path. | `local_ptoas` |
| `mhc/pre_apply_mix_bwd` | MHC | Computes gradients for pre-layer MHC mixing. | `local_ptoas` |
| `mhc/post_fwd` | MHC | Combines residual, post-layer mix, and MHC output tensors. | `local_ptoas` |
| `mhc/post_bwd` | MHC | Computes gradients for the MHC post-combine path. | `local_ptoas` |
| `mhc/pre_norm_fn_fwd` | MHC | Computes pre-normalized residual and feed-forward MHC features. | `local_ptoas` |
| `mhc/fn_normw_merge_fwd` | MHC | Applies norm-weight merge to MHC feed-forward features. | `local_ptoas` |
| `mhc/fn_normw_merge_bwd` | MHC | Computes gradients for MHC norm-weight merge. | `local_ptoas` |
| `mhc/sinkhorn_normalize_fwd` | MHC | Runs Sinkhorn normalization over MHC routing scores. | `local_ptoas` |
| `mhc/head_compute_mix_fwd` | MHC | Computes per-head MHC mix values. | `local_ptoas` |
| `mhc/head_compute_mix_bwd` | MHC | Computes gradients for per-head MHC mix values. | `local_ptoas` |
| `mhc/pre_split_mixes_fwd` | MHC | Splits input MHC mix values into pre, post, and residual mixes. | `local_ptoas` |
| `mhc/pre_split_mixes_bwd` | MHC | Computes gradients for split MHC mix values. | `local_ptoas` |
| `mhc/sinkhorn_normalize_bwd` | MHC | Computes gradients for Sinkhorn normalization. | `local_ptoas` |
| `posembedding/apply_rotary_pos_emb` | Position embedding | Applies rotary position embedding to query/key tensors. | `planned` |
| `posembedding/dequant_rope_quant_kvcache` | Position embedding | Dequantizes inputs, applies RoPE, and quantizes KV cache output. | `planned` |
| `posembedding/interleave_rope` | Position embedding | Applies interleaved rotary position embedding layout. | `planned` |
| `posembedding/qkv_rms_norm_rope_cache` | Position embedding | Runs QKV RMSNorm, RoPE, and cache update. | `planned` |
| `posembedding/rope_quant_kvcache` | Position embedding | Applies RoPE and quantizes KV cache tensors. | `planned` |
| `posembedding/rope_with_sin_cos_cache` | Position embedding | Applies RoPE using cached sine and cosine tensors. | `planned` |
| `posembedding/rotary_position_embedding` | Position embedding | Computes rotary position embedding output. | `planned` |
| `posembedding/rotary_position_embedding_grad` | Position embedding | Computes gradients for rotary position embedding. | `planned` |
| `gmm/grouped_matmul` | Grouped matmul | Runs grouped matrix multiplication. | `planned` |
| `gmm/grouped_matmul_add` | Grouped matmul | Runs grouped matrix multiplication with an add epilogue. | `planned` |
| `gmm/grouped_matmul_finalize_routing` | Grouped matmul | Finalizes MoE routing data for grouped matmul. | `planned` |
| `gmm/grouped_matmul_swiglu_quant` | Grouped matmul | Runs grouped matmul with SwiGLU and quantization. | `planned` |
| `gmm/grouped_matmul_swiglu_quant_v2` | Grouped matmul | Runs the v2 grouped matmul SwiGLU quantization path. | `planned` |
| `gmm/quant_grouped_matmul_inplace_add` | Grouped matmul | Runs quantized grouped matmul with in-place add. | `planned` |
| `ffn/ffn` | FFN | Runs a feed-forward network block. | `planned` |
| `ffn/swin_attention_ffn` | FFN | Runs the Swin attention feed-forward block. | `planned` |
| `ffn/swin_transformer_ln_qkv` | FFN | Runs Swin transformer layernorm and QKV projection. | `planned` |
| `ffn/swin_transformer_ln_qkv_quant` | FFN | Runs quantized Swin transformer layernorm and QKV projection. | `planned` |
| `moe/moe_compute_expert_tokens` | MoE | Counts tokens assigned to each MoE expert. | `planned` |
| `moe/moe_finalize_routing` | MoE | Finalizes routed MoE token placement. | `planned` |
| `moe/moe_finalize_routing_v2` | MoE | Finalizes routed MoE token placement with the v2 path. | `planned` |
| `moe/moe_finalize_routing_v2_grad` | MoE | Computes gradients for the v2 MoE finalize-routing path. | `planned` |
| `moe/moe_gating_top_k` | MoE | Selects top-k experts from MoE gating scores. | `planned` |
| `moe/moe_gating_top_k_softmax` | MoE | Applies softmax and top-k selection to MoE gating scores. | `planned` |
| `moe/moe_gating_top_k_softmax_v2` | MoE | Applies the v2 softmax top-k MoE gating path. | `planned` |
| `moe/moe_init_routing` | MoE | Initializes MoE token routing metadata. | `planned` |
| `moe/moe_init_routing_quant` | MoE | Initializes quantized MoE token routing metadata. | `planned` |
| `moe/moe_init_routing_quant_v2` | MoE | Initializes quantized MoE token routing metadata with the v2 path. | `planned` |
| `moe/moe_init_routing_v2` | MoE | Initializes MoE token routing metadata with the v2 path. | `planned` |
| `moe/moe_init_routing_v2_grad` | MoE | Computes gradients for the v2 MoE init-routing path. | `planned` |
| `moe/moe_init_routing_v3` | MoE | Initializes MoE token routing metadata with the v3 path. | `planned` |
| `moe/moe_re_routing` | MoE | Recomputes MoE routing assignments. | `planned` |
| `moe/moe_token_permute` | MoE | Permutes tokens into expert order. | `planned` |
| `moe/moe_token_permute_grad` | MoE | Computes gradients for token permutation. | `planned` |
| `moe/moe_token_permute_with_ep` | MoE | Permutes tokens into expert-parallel expert order. | `planned` |
| `moe/moe_token_permute_with_ep_grad` | MoE | Computes gradients for expert-parallel token permutation. | `planned` |
| `moe/moe_token_permute_with_routing_map` | MoE | Permutes tokens using a routing map. | `planned` |
| `moe/moe_token_permute_with_routing_map_grad` | MoE | Computes gradients for routing-map token permutation. | `planned` |
| `moe/moe_token_unpermute` | MoE | Restores tokens from expert order. | `planned` |
| `moe/moe_token_unpermute_grad` | MoE | Computes gradients for token unpermutation. | `planned` |
| `moe/moe_token_unpermute_with_ep` | MoE | Restores tokens from expert-parallel expert order. | `planned` |
| `moe/moe_token_unpermute_with_ep_grad` | MoE | Computes gradients for expert-parallel token unpermutation. | `planned` |
| `moe/moe_token_unpermute_with_routing_map` | MoE | Restores tokens using a routing map. | `planned` |
| `moe/moe_token_unpermute_with_routing_map_grad` | MoE | Computes gradients for routing-map token unpermutation. | `planned` |
| `attention/attention_update` | Attention | Updates attention state tensors. | `planned` |
| `attention/flash_attention_score` | Attention | Computes flash attention scores and output. | `planned` |
| `attention/flash_attention_score_grad` | Attention | Computes flash attention gradients. | `planned` |
| `attention/fused_infer_attention_score` | Attention | Computes fused inference attention scores. | `planned` |
| `attention/incre_flash_attention` | Attention | Runs incremental flash attention. | `planned` |
| `attention/prompt_flash_attention` | Attention | Runs prompt flash attention. | `planned` |
| `attention/recurrent_gated_delta_rule` | Attention | Runs recurrent gated delta-rule attention. | `planned` |
| `attention/ring_attention_update` | Attention | Updates ring attention state. | `planned` |
| `attention/scatter_pa_cache` | Attention | Scatters paged-attention cache entries. | `planned` |
| `attention/gather_pa_kv_cache` | Attention | Gathers paged-attention KV cache entries. | `planned` |
| `attention/kv_quant_sparse_flash_attention` | Attention | Runs sparse flash attention with quantized KV inputs. | `planned` |
| `attention/lightning_indexer` | Attention | Builds indices for lightning attention. | `planned` |
| `attention/mla_preprocess` | Attention | Preprocesses tensors for multi-head latent attention. | `planned` |
| `attention/mla_prolog` | Attention | Runs the MLA prolog path. | `planned` |
| `attention/mla_prolog_v2` | Attention | Runs the v2 MLA prolog path. | `planned` |
| `attention/mla_prolog_v3` | Attention | Runs the v3 MLA prolog path. | `planned` |
| `attention/nsa_compress` | Attention | Compresses tensors for native sparse attention. | `planned` |
| `attention/nsa_compress_attention` | Attention | Runs compressed native sparse attention. | `planned` |
| `attention/nsa_compress_attention_infer` | Attention | Runs inference compressed native sparse attention. | `planned` |
| `attention/nsa_compress_grad` | Attention | Computes gradients for native sparse attention compression. | `planned` |
| `attention/nsa_compress_with_cache` | Attention | Compresses native sparse attention tensors with cache output. | `planned` |
| `attention/nsa_selected_attention` | Attention | Runs selected native sparse attention. | `planned` |
| `attention/nsa_selected_attention_grad` | Attention | Computes gradients for selected native sparse attention. | `planned` |
| `attention/nsa_selected_attention_infer` | Attention | Runs inference selected native sparse attention. | `planned` |
| `attention/quant_lightning_indexer` | Attention | Builds quantized lightning attention indices. | `planned` |
| `attention/sparse_flash_attention` | Attention | Runs sparse flash attention. | `planned` |
| `mc2/all_gather_matmul` | MC2 | Runs all-gather followed by matrix multiplication. | `planned` |
| `mc2/grouped_mat_mul_all_reduce` | MC2 | Runs grouped matmul with all-reduce. | `planned` |
| `mc2/inplace_matmul_all_reduce_add_rms_norm` | MC2 | Runs matmul, all-reduce, add, and RMSNorm in place. | `planned` |
| `mc2/matmul_all_reduce` | MC2 | Runs matrix multiplication with all-reduce. | `planned` |
| `mc2/matmul_all_reduce_add_rms_norm` | MC2 | Runs matmul, all-reduce, add, and RMSNorm. | `planned` |
| `mc2/matmul_reduce_scatter` | MC2 | Runs matrix multiplication with reduce-scatter. | `planned` |
| `mc2/moe_distribute_combine` | MC2 | Combines distributed MoE token outputs. | `planned` |
| `mc2/moe_distribute_combine_v2` | MC2 | Combines distributed MoE token outputs with the v2 path. | `planned` |
| `mc2/moe_distribute_dispatch` | MC2 | Dispatches MoE tokens across distributed ranks. | `planned` |
| `mc2/moe_distribute_dispatch_v2` | MC2 | Dispatches MoE tokens across distributed ranks with the v2 path. | `planned` |
| `mc2/moe_distribute_dispatch_v3` | MC2 | Dispatches MoE tokens across distributed ranks with the v3 path. | `planned` |
| `attention/attention_worker_scheduler` | Attention | Schedules attention worker tasks on AI CPU. | `excluded_ai_cpu` |
| `ffn/ffn_worker_scheduler` | FFN | Schedules FFN worker tasks on AI CPU. | `excluded_ai_cpu` |
| `mc2/allto_all_all_gather_batch_mat_mul` | MC2 | Runs all-to-all, all-gather, and batched matmul. | `excluded_a3_only` |
| `mc2/allto_all_matmul` | MC2 | Runs all-to-all followed by matmul. | `excluded_a3_only` |
| `mc2/allto_allv_grouped_mat_mul` | MC2 | Runs all-to-all-v followed by grouped matmul. | `excluded_a3_only` |
| `mc2/ffn_to_attention` | MC2 | Transfers distributed state from FFN to attention. | `excluded_a3_only` |
| `mc2/attention_to_ffn` | MC2 | Transfers distributed state from attention to FFN. | `excluded_a3_only` |
| `mc2/batch_mat_mul_reduce_scatter_allto_all` | MC2 | Runs batched matmul, reduce-scatter, and all-to-all. | `excluded_a3_only` |
| `mc2/distribute_barrier` | MC2 | Synchronizes distributed MC2 work. | `excluded_a3_only` |
| `mc2/grouped_mat_mul_allto_allv` | MC2 | Runs grouped matmul followed by all-to-all-v. | `excluded_a3_only` |
| `mc2/matmul_allto_all` | MC2 | Runs matmul followed by all-to-all. | `excluded_a3_only` |
| `mc2/moe_distribute_combine_add_rms_norm` | MC2 | Combines distributed MoE outputs with add and RMSNorm. | `excluded_a3_only` |
| `mc2/moe_update_expert` | MC2 | Updates distributed MoE expert state. | `excluded_a3_only` |

## Repository Structure

```text
pto-kernels/
├── csrc/                  # C++ kernel source files
├── python/pto_kernels/    # Python bindings and utilities
├── bench/                 # Benchmark specs, adapters, and inventory
├── docs/                  # Repository notes and references
├── examples/jit_cpp/      # JIT compilation examples
├── skills/                # Agent workflow skills
├── templates/             # Kernel writer templates
├── tests/                 # Test suite
├── scripts/               # Helper scripts
├── doxygen/               # API documentation config
└── CMakeLists.txt         # CMake build configuration
```

## Tutorial

If you are new to this repository, start with the Chinese tutorial under
[`tutorial/`](./tutorial/README.md). It explains the full `PTO-DSL -> PTOAS -> PTO-ISA -> Bisheng -> .so` workflow and walks through real kernels such as `grouped_matmul`, `flash_attention_score`, and `moe_token_permute`.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## License

BSD-3-Clause-Clear — see [LICENSE](LICENSE) for details.
