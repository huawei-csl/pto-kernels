# Dynamic BSND group-value heads (`H ≠ Hg`)

PTO kernels for GQA-style layouts where **value/query heads `H`** exceed **shared key heads `Hg`** (same mapping as FLA/Triton: `head_g = head // (H // Hg)`).

| Kernel | C++ | Role |
|--------|-----|------|
| `chunk_h` | `chunk_h_kernel.cpp` | Recurrent hidden-state update (`K`/`W`/`U` strides split) |
| `chunk_o` | `chunk_o_kernel.cpp` | Chunk output `O = (QK_gated @ V) + exp(g)·(Q @ S)` |

Same batch / packed-varlen semantics as ``dynamic_bsnd/``; see parent ``dynamic_bsnd/README.md``.

## Build / load

Uses ``bisheng`` via ``pto_dynamic_common.compile_pto_kernel``. Macros:

- ``GDN_H`` — value head count ``H``
- ``GDN_HG`` — key head count ``Hg`` (default ``GDN_H`` if omitted)
- ``GDN_D``, ``GDN_C`` — hidden size and chunk size

Cached shared objects:

- ``compiled_lib/chunk_h_bsnd_groupvalue_H{H}_Hg{Hg}_D{D}_C{C}.so``
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
```

Expectations:

- ``verify_dynamic_bsnd_groupvalue.py``: **same case list** as ``dynamic_bsnd/verify_dynamic_bsnd.py`` lines 222–280; checks ``h_states`` and ``v_new``.
- ``verify_chunk_o_groupvalue.py``: runs ``chunk_h`` then ``chunk_o``; compares ``chunk_o`` to a CPU fp32 reference (PTO ``exp(min(Δg,0))`` gating).

## Benchmark

Same default workload as ``dynamic_bsnd/bench_dynamic_bsnd.py``: ``N_seq=16``, ``L_seg=16384``, ``T=262144``, ``D=128``, ``C=128``.

Read **`dynamic_bsnd/README.md` → [PTO vs Triton chunk tile](../dynamic_bsnd/README.md#pto-vs-triton-chunk-tile)** before comparing numbers: **PTO uses chunk size 128**; **Triton baseline defaults to chunk size 64 (`BT`)**. Different chunk sizes are still reported together as comparable configurations; optional **128** on Triton only when it compiles and runs—otherwise omit and note the failure.

```bash
export ASCEND_TOOLKIT_HOME=...
export GDN_NPU_DEVICE=npu:7
GDN_BENCH_H=32 GDN_BENCH_HG=16 python3 bench_dynamic_bsnd_groupvalue.py
GDN_BENCH_H=32 GDN_BENCH_HG=16 python3 bench_chunk_o_groupvalue.py
```

### Measured latency (910B2, ``npu:7``, ``cube_core_num=24``)

Shape: ``N_seq=16``, ``L_seg=16384``, ``T=262144``, ``D=128``, ``Hg=16``. **PTO** chunk kernels use **`C=128`**; **Triton** ``chunk_fwd_o`` column uses **`BT=64`** by default (see env ``GDN_TRITON_CHUNK_O_CHUNK`` in ``bench_chunk_o_groupvalue.py``). Failures at ``BT=128`` on Ascend: omitted here with reason in parent README.

| ``H`` | PTO chunk_h (ms) | Triton chunk_h (ms) | PTO chunk_o ``C=128`` (ms) | Triton chunk_o ``BT=64`` (ms) |
| --: | --: | --: | --: | --: |
| 16 | 9.47 | 15.55 | 10.59 | 16.10 |
| 32 | 17.81 | 30.57 | 19.59 | 31.60 |
| 48 | 26.41 | 45.50 | 30.87 | 46.63 |
| 64 | 35.37 | 60.62 | 39.25 | — |

``—``: Triton ``chunk_fwd_o`` failed at ``H=64`` (AICore error 507015) on the measurement host; PTO paths succeeded.

## Implementation notes

- Vec-stage GM loads for ``K`` (and ``chunk_o`` ``Q``) use ``(token·Hg + head_g)·D`` row indexing with stride ``Hg·D`` (see ``chunk_h_kernel.cpp`` / ``chunk_o_kernel.cpp``).
- UB packing in ``chunk_h`` uses a fixed leading slack matching the legacy ``GDN_H=16`` kernel so large compile-time ``H`` does not exceed the vector UB budget (~192 KiB on 910B2).
