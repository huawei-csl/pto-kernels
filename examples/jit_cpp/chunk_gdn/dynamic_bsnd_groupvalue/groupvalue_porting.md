# Porting kernels from `H == Hg` to GQA-style `H != Hg`

This documents what changed when extending **dynamic BSND** PTO kernels so **value/query heads `H`** can exceed **shared key heads `Hg`** (same grouping rule as FLA/Triton: `head_g = head // (H // Hg)`).

## Tensor roles

| Role | BSND slice | Row stride along sequence |
|------|------------|---------------------------|
| Keys `K`, queries `Q` | `[total_tokens, Hg, D]` | `Hg * D` elements |
| Values `V`, gates `G`, wy outputs `W`,`U`, chunk_o output `O`, chunk_h state over value heads | `[total_tokens, H, D]` or `[H, T]` for `G` | `H * D` or `H` |
| Hidden state `S` snapshots | `[chunks, H, D, D]` | Indexed per **value** head |
| Attention blocks `A` (from scaled-dot / KKT stage) | `[batch, seq, H, C]` | Stride `H * C` along seq (per **value** head) |

Triton references: `chunk_delta_h.py` / `chunk_o.py` / `wy_fast.py` (`stride_k = Hg * K`, `stride_v = H * V`, shared key row for grouped heads).

## C++ indexing pattern

1. **Compile-time**: add `NumKeyHeads` (`Hg`), `GROUP = NumHeads / NumKeyHeads`, `static_assert(NumHeads % NumKeyHeads == 0)`.
2. **Per value head index `head`** (what you already iterate): **`head_g = head / GROUP`** (integer divide).
3. **GM byte/element offset** for a token `t` and head dimension:
   - **Q/K**: `(t * Hg + head_g) * D` with stride **`Hg * D`** (`BSND_QK_STRIDE`).
   - **V / outputs tied to value heads**: `(t * H + head) * D` with stride **`H * D`** (`BSND_V_STRIDE`).
4. **Gates `G`** stay **`[H, total_tokens]`** per **value** head — unchanged.

Launcher macros: **`GDN_H`** = value heads, **`GDN_HG`** = key heads (default **`GDN_H`**). Wrapper invokes **`kernel<GDN_H, GDN_HG, GDN_D, GDN_C>`**.

## `chunk_h`-specific notes

- Cube loads **only `W`,`V`** from value stride; Vec loads **`K`** from key stride — split offsets accordingly.
- **Vector UB**: the legacy leading scratch `C * NumHeads * sizeof(float)` before `zero_ub` scaled with **`H`** and pushed UB past ~192 KiB on **910B2** when compiling `GDN_H ∈ {32,48,64}`. Fix: **fixed slack** matching the historical **`GDN_H=16`** hole (`ChunkSize * 16 * sizeof(float)`), not proportional to template `NumHeads`.

## `chunk_o`-specific notes

Porting mirrored **`chunk_h`**: introduce **`qk_off`** / **`v_off`**, **`head_g`**, and explicit **`BSND_QK_STRIDE`** vs **`BSND_V_STRIDE`** anywhere **`GlobalTensor`** touches **`Q`,`K`** vs **`V`** (dense **and** **`cu_seqlens`** Cube paths).

- **GEMM 1 & 2** (`Q @ Kᵀ`, `Q @ S`): load **`Q`** and **`K`** via **`qk_off`** + **`BSND_QK_STRIDE`**.
- **GEMM 3** (`QK_gated @ V`): load **`V`** via **`v_off`** + **`BSND_V_STRIDE`**.
- **`S`** chunk states: **`(chunk_global_idx * H + head_idx) * D²`** — still **value** heads (**`NumHeads`** in template = **`H`**).
- **Vec stores `O`**: row offset **`(chunk_token_start * H + head_idx) * D`** + half-chunk **`vid`** skew; **`Stride`** uses **`BSND_V_STRIDE`** (same numeric size as **`H * HiddenSize`**).

There is **no** unified **`qkv_offset`** once **`H ≠ Hg`**: **`K`** cannot share the same leading dimension stride as **`V`**.

## `wy_fast`-specific notes

Math unchanged: **`U = (A ⊙ β₂d) @ V`**, **`W = (A ⊙ (eᵍβ)₂d) @ K`** with **`β`,`g`,`A`** per **value** head.

- **Cube GM loads**: **`K`** uses **`k_off`** + **`BSND_QK_STRIDE`**; **`V`**, and **`W`/`U` stores**, use **`v_off`** + **`BSND_V_STRIDE`** (same **`v_off`** pattern as **`chunk_h`** outputs).
- **Vec** loads **`β`**, **`g`**, stores **`A`** unchanged vs **`H == Hg`** — **[batch, seq, H, …]** / **`[H,T]`** transposed for **value** heads **`H`** (template **`NumHeads`**).

## `scaled_dot_kkt`-specific notes

Same split as **`chunk_o`** / **`wy_fast`** on the Cube **`K`** path only:

- **Cube `TLOAD` / `GlobalTensor` for `K`**: token offset **`(bos + chunk_start) * Hg + head_g`** with **`head_g = head_idx / GROUP`**; stride **`BSND_QK_STRIDE = Hg * D`** (not **`H * D`**).
- **Vec `β` / `g` loads**, **`A` GM store**, and **`pid → head_idx`** over **`H`** value heads — unchanged from the **`H == Hg`** kernel (**`Stride … NumHeads * ChunkSize`** along sequence for **`A`**).

Reference: FLA **`chunk_scaled_dot_kkt`** / Triton indexing **`k + (bos * Hg + i_h // GROUP) * K`**.

## Python / verification

- Avoid **`torch.randn` gates** alone for recurrence-heavy ops — match **`verify_dynamic_bsnd`**: **`logsigmoid`** then **chunk-local `cumsum`** per sequence where applicable.
- **Normalize `Q`,`K`** like upstream (`F.normalize(..., dim=-1, p=2)`) when comparing to pipeline-style tests.
- Import **`pto_dynamic_common`** only from **this directory** when loading ctypes libs (`sys.modules['pto_dynamic_common'] = …`) so **`key_heads`** reaches **`compile_pto_kernel`** (otherwise an older module shadowing breaks `-DGDN_HG=`).

Scripts (single entry points):

| Script | Role |
|--------|------|
| **`verify_dynamic_bsnd_groupvalue.py`** | **`--stage`** among **`kkt`**, **`chunk_h`**, **`wy_fast`**, **`chunk_o`** (same packed-varlen case list as **`dynamic_bsnd/verify_dynamic_bsnd.py`**) |
| **`bench_dynamic_bsnd_groupvalue.py`** | Times each stage vs FLA Triton; **`--stage`** filter; **`GDN_TRITON_KKT_CHUNK`** / **`GDN_TRITON_CHUNK_O_CHUNK`** |

## Benchmarking

- Compare **PTO vs Triton** with **matching tensor layouts**. **`bench_dynamic_bsnd_groupvalue.py`** benchmarks **`scaled_dot_kkt`** with Triton **`BT=64`** by default and optionally **`BT=128`** when it compiles; ratios **`ms_triton/ms_pto`** (**``>1`` ⇒ PTO faster**).
- **`dynamic_bsnd/bench_dynamic_bsnd.py`** remains the **`H == Hg`** pipeline bench; group-value numbers are in **`README.md`** here.
