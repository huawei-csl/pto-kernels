# Porting kernels from `H == Hg` to GQA-style `H != Hg`

This documents what changed when extending **dynamic BSND** PTO kernels so **value/query heads `H`** can exceed **shared key heads `Hg`** (same grouping rule as FLA/Triton: `head_g = head // (H // Hg)`).

## Tensor roles

| Role | BSND slice | Row stride along sequence |
|------|------------|---------------------------|
| Keys `K`, queries `Q` | `[total_tokens, Hg, D]` | `Hg * D` elements |
| Values `V`, gates `G`, wy outputs `W`,`U`, chunk_o output `O`, chunk_h state over value heads | `[total_tokens, H, D]` or `[H, T]` for `G` | `H * D` or `H` |
| Hidden state `S` snapshots | `[chunks, H, D, D]` | Indexed per **value** head |

Triton references: `chunk_delta_h.py` / `chunk_o.py` (`stride_k = Hg * K`, `stride_v = H * V`, shared key row for grouped heads).

## C++ indexing pattern

1. **Compile-time**: add `NumKeyHeads` (`Hg`), `GROUP = NumHeads / NumKeyHeads`, `static_assert(NumHeads % NumKeyHeads == 0)`.
2. **Per value head index `head`** (what you already iterate): **`head_g = head / GROUP`** (integer divide).
3. **GM byte/element offset** for a token `t` and head dimension:
   - **Q/K**: `(t * Hg + head_g) * D` with stride **`Hg * D`** (`BSND_QK_STRIDE`).
   - **V / outputs tied to value heads**: `(t * H + head) * D` with stride **`H * D`** (`BSND_V_STRIDE`).
4. **Gates `G`** stay **`[H, total_tokens]`** per **value** head — unchanged.

## `chunk_h`-specific notes

- Cube loads **only `W`,`V`** from value stride; Vec loads **`K`** from key stride — split offsets accordingly.
- **Vector UB**: the legacy leading scratch `C * NumHeads * sizeof(float)` before `zero_ub` scaled with **`H`** and pushed UB past ~192 KiB on **910B2** when compiling `GDN_H ∈ {32,48,64}`. Fix: **fixed slack** matching the historical **`GDN_H=16`** hole (`ChunkSize * 16 * sizeof(float)`), not proportional to template `NumHeads`.

## `chunk_o`-specific notes

- **GEMM 1 & 2** use **`Q`,`K`** from the shared key head → **`qk_off`** + **`BSND_QK_STRIDE`** on `GlobalTensor` strides.
- **GEMM 3** uses **`V`** → **`v_off`** + **`BSND_V_STRIDE`**.
- **`S`** (chunk_h states) stays **`(chunk_idx * H + head) * D²`** — state is per **value** head.
- **Vec writes `O`** with value-head stride (`NumHeads * HiddenSize` in the original equals **`BSND_V_STRIDE`**).

## Python / verification

- Avoid **`torch.randn` gates** alone for recurrence-heavy ops — match **`verify_dynamic_bsnd`**: **`logsigmoid`** then **chunk-local `cumsum`** per sequence.
- **Normalize `Q`,`K`** like upstream (`F.normalize(..., dim=-1, p=2)`) so numerical checks align with the full pipeline tests.
- Import **`pto_dynamic_common`** only from **this directory** when loading ctypes libs (`sys.modules['pto_dynamic_common'] = …`) so **`key_heads`** reaches **`compile_pto_kernel`** (otherwise an older module shadowing breaks `-DGDN_HG=`).
- Scripts: **`verify_dynamic_bsnd_groupvalue.py`** (chunk_h), **`verify_chunk_o_groupvalue.py`** (chunk_h → chunk_o chain), **`bench_dynamic_bsnd_groupvalue.py`** (chunk_h), **`bench_chunk_o_groupvalue.py`** (chunk_o).

## Benchmarking

- Compare **PTO vs Triton** with **matching tensor layouts** (`k`/`q` `[B,T,Hg,D]`, `v`/`o` `[B,T,H,D]`).
- Original **`dynamic_bsnd`** bench remains valid when **`H == Hg`**; group-value timings live beside it or in a dedicated **`bench_*_groupvalue.py`** / **`bench_chunk_o_groupvalue.py`**.
