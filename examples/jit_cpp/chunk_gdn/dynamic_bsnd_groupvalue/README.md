# Dynamic BSND group-value heads (`H ≠ Hg`)

PTO kernels for GQA-style layouts where **value/query heads `H`** exceed **shared key heads `Hg`** (same mapping as FLA/Triton: `head_g = head // (H // Hg)`).

| Kernel | C++ | Role |
|--------|-----|------|
| `scaled_dot_kkt` | `scaled_dot_kkt_kernel.cpp` | Intra-chunk gated `KKᵀ` (`K` stride `Hg`; `β`,`g`,`A` per value head `H`) |
| `chunk_h` | `chunk_h_kernel.cpp` | Recurrent hidden-state update (`K`/`W`/`U` strides split) |
| `wy_fast` | `wy_fast_kernel.cpp` | WY recompute `W`,`U` from `A`,`β`,`g` (`K` vs `V` strides split) |
| `chunk_o` | `chunk_o_kernel.cpp` | Chunk output `O = (QK_gated @ V) + exp(g)·(Q @ S)` |

Same batch / packed-varlen semantics as ``dynamic_bsnd/``; see parent ``dynamic_bsnd/README.md``.

## Build / load

Uses ``bisheng`` via ``pto_dynamic_common.compile_pto_kernel``. Macros:

- ``GDN_H`` — value head count ``H``
- ``GDN_HG`` — key head count ``Hg`` (default ``GDN_H`` if omitted)
- ``GDN_D``, ``GDN_C`` — hidden size and chunk size

Cached shared objects:

- ``compiled_lib/scaled_dot_kkt_bsnd_groupvalue_H{H}_Hg{Hg}_D{D}_C{C}.so``
- ``compiled_lib/chunk_h_bsnd_groupvalue_H{H}_Hg{Hg}_D{D}_C{C}.so``
- ``compiled_lib/wy_fast_bsnd_groupvalue_H{H}_Hg{Hg}_D{D}_C{C}.so``
- ``compiled_lib/chunk_o_bsnd_groupvalue_H{H}_Hg{Hg}_D{D}_C{C}.so``

## Verification (NPU)

From ``chunk_gdn/dynamic_bsnd_groupvalue``:

```bash
export ASCEND_TOOLKIT_HOME=/path/to/Ascend/cann   # or ASCEND_HOME_PATH
export PTO_LIB_PATH=/path/to/pto-isa/include/..   # header tree parent
export GDN_NPU_DEVICE=npu:7                       # prefer a free NPU id

python3 verify_dynamic_bsnd_groupvalue.py --device npu:7 --H-list 16,32,48,64
python3 verify_dynamic_bsnd_groupvalue.py --device npu:7 --quick

python3 verify_chunk_o_groupvalue.py --device npu:7 --H-list 16,32,48,64
python3 verify_chunk_o_groupvalue.py --device npu:7 --quick

python3 verify_wy_fast_groupvalue.py --device npu:7 --H-list 16,32,48,64
python3 verify_wy_fast_groupvalue.py --device npu:7 --quick

python3 verify_scaled_dot_kkt_groupvalue.py --device npu:7 --H-list 16,32,48,64
python3 verify_scaled_dot_kkt_groupvalue.py --device npu:7 --quick
```

Expectations:

- ``verify_scaled_dot_kkt_groupvalue.py``: ``k`` ``[B,T,Hg,D]``, ``β``/``g``/``A`` over ``H``; CPU ref uses ``head_g = head // (H // Hg)`` (matches FLA/Triton).
- ``verify_dynamic_bsnd_groupvalue.py``: **same case list** as ``dynamic_bsnd/verify_dynamic_bsnd.py`` lines 222–280; checks ``h_states`` and ``v_new``.
- ``verify_chunk_o_groupvalue.py``: runs ``chunk_h`` then ``chunk_o``; compares ``chunk_o`` to a CPU fp32 reference (PTO ``exp(min(Δg,0))`` gating).
- ``verify_wy_fast_groupvalue.py``: **``wy_fast`` only** with synthetic ``A`` tiles; compares ``w`` and ``u`` to a CPU fp32 reference (FLA-style ``hg`` for ``K``).

## Benchmark

Same default workload as ``dynamic_bsnd/bench_dynamic_bsnd.py``: ``N_seq=16``, ``L_seg=16384``, ``T=262144``, ``D=128``, ``C=128``.

Read **`dynamic_bsnd/README.md` → [PTO vs Triton chunk tile](../dynamic_bsnd/README.md#pto-vs-triton-chunk-tile)** before comparing numbers: **PTO uses chunk size 128 (`GDN_C`)**; **`bench_scaled_dot_kkt_groupvalue.py`** times Triton **`chunk_scaled_dot_kkt_fwd`** at **`BT=64`** by default (env **`GDN_TRITON_KKT_CHUNK`**, avoids Ascend MLIR compile failures seen at **`BT=128`**). After that run it **optionally** tries **`BT=128`** when **`GDN_TRITON_KKT_TRY128`** is non-zero and reports timings **only if compile + execution succeed**. Ratio columns use **`ms_triton / ms_pto`** (**values > 1 ⇒ PTO faster**).

```bash
export ASCEND_TOOLKIT_HOME=...
export GDN_NPU_DEVICE=npu:7
GDN_BENCH_H=32 GDN_BENCH_HG=16 python3 bench_scaled_dot_kkt_groupvalue.py
GDN_BENCH_H=32 GDN_BENCH_HG=16 python3 bench_dynamic_bsnd_groupvalue.py
GDN_BENCH_H=32 GDN_BENCH_HG=16 python3 bench_chunk_o_groupvalue.py
GDN_BENCH_H=32 GDN_BENCH_HG=16 python3 bench_wy_fast_groupvalue.py
```

For **`scaled_dot_kkt`** only: optional **`GDN_TRITON_KKT_CHUNK=64`** (default primary Triton tile), **`GDN_TRITON_KKT_TRY128=1`** (attempt optional **`BT=128`** timing).

### Measured latency (910B2, ``npu:7``, ``cube_core_num=24``)

Recorded **2026-04-28** from this directory with ``ASCEND_TOOLKIT_HOME`` set and ``GDN_NPU_DEVICE=npu:7``. Shape: ``N_seq=16``, ``L_seg=16384``, ``T=262144``, ``D=128``, ``Hg=16``. **PTO** chunk kernels use **`C=128`**; **Triton** ``chunk_fwd_o`` column uses **`BT=64`** by default (see env ``GDN_TRITON_CHUNK_O_CHUNK`` in ``bench_chunk_o_groupvalue.py``). Failures at ``BT=128`` on Ascend: omitted here with reason in parent README.

**``scaled_dot_kkt``**: PTO kernel compiled at **`C=128`**. Triton uses **`chunk_scaled_dot_kkt_fwd`** at **`BT=64`** (baseline for Ascend); **`BT=128`** is timed **only when compile + launch succeed**. Ratio **`Triton_ms / PTO_ms`** (**``> 1`` ⇒ PTO faster**).

| ``H`` | PTO ``C=128`` (ms) | Triton ``BT=64`` (ms) | ``T64/PTO`` | Triton ``BT=128`` (ms) | ``T128/PTO`` |
| --: | --: | --: | --: | --: | --: |
| 16 | 4.31 | 4.08 | 0.95 | — | — |
| 32 | 7.40 | 7.50 | 1.01 | — | — |
| 48 | 11.87 | 11.02 | 0.93 | — | — |
| 64 | 17.32 | 14.54 | 0.84 | — | — |

Optional **`BT=128`** did not compile on this host (**``MLIRCompilationError``**); rerun after **`bench_scaled_dot_kkt_groupvalue.py`** when Triton **`BT=128`** succeeds (e.g. on CUDA or newer stacks).

**Other kernels** (unchanged methodology):

| ``H`` | PTO chunk_h (ms) | Triton chunk_h (ms) | PTO chunk_o ``C=128`` (ms) | Triton chunk_o ``BT=64`` (ms) |
| --: | --: | --: | --: | --: |
| 16 | 9.08 | 15.61 | 9.59 | 16.13 |
| 32 | 17.83 | 30.54 | 19.49 | 31.50 |
| 48 | 25.09 | 45.47 | 30.25 | 46.63 |
| 64 | 38.04 | 60.62 | 38.97 | — |

``—``: Triton ``chunk_fwd_o`` failed at ``H=64`` (AICore error 507015) on the measurement host; PTO paths succeeded.

**``wy_fast``** (same shape; PTO vs Triton ``recompute_w_u_fwd``, both at ``C=128``):

| ``H`` | PTO wy_fast (ms) | Triton wy_fast (ms) |
| --: | --: | --: |
| 16 | 6.02 | 11.92 |
| 32 | 12.28 | 23.37 |
| 48 | 16.69 | 34.83 |
| 64 | 22.48 | 46.30 |

## Implementation notes

- Vec-stage GM loads for ``K`` (and ``chunk_o`` ``Q``) use ``(token·Hg + head_g)·D`` row indexing with stride ``Hg·D`` (see ``scaled_dot_kkt_kernel.cpp`` / ``chunk_h_kernel.cpp`` / ``chunk_o_kernel.cpp`` / ``wy_fast_kernel.cpp`` Cube loads).
- UB packing in ``chunk_h`` uses a fixed leading slack matching the legacy ``GDN_H=16`` kernel so large compile-time ``H`` does not exceed the vector UB budget (~192 KiB on 910B2).
