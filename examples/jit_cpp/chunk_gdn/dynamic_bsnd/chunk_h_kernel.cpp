// ============================================================================
// chunk_h_kernel.cpp — Recurrent hidden state update for GatedDeltaNet
//
// Mathematical recurrence per chunk c:
//   S_{c+1} = exp(g_last) * S_c  +  K^T @ V
//
// where g_last = exp(g[valid-1]) is the chunk's final gate value, S is the
// D×D hidden state, K ∈ ℝ^{C×D}, V ∈ ℝ^{C×D}, and g ∈ ℝ^C is the per-token
// gate.
//
// ── Cube phase (two GEMMs per chunk, sequentially): ──────────────────────
//   1. WS = W @ S       project current state through W (wy_fast output)
//      W ∈ ℝ^{C×D}, S ∈ ℝ^{D×D}  →  WS ∈ ℝ^{C×D}
//   2. KV = K^T @ V     outer product of keys and values (transpose_A!)
//      K stored as D×C, V ∈ ℝ^{C×D}  →  KV ∈ ℝ^{D×D}
//
// ── Vec phase (two sub-blocks handle upper/lower C/2 rows): ─────────────
//   For each chunk:
//     1. Load K, G (pre-transposed), U (from wy_fast)
//     2. Compute coeff[i] = exp(g[i] - g[valid-1])  — time-decay scaling
//        Uses TROWEXPAND to broadcast coefficients across D columns
//     3. Scale K: K_scaled[i,:] = K[i,:] * coeff[i]
//     4. Load WS from Cube workspace, compute V_new = U - WS (residual)
//     5. Store V_new and K_scaled to workspace for Cube's next iteration
//     6. Update state: S = exp(g_last) * S + KV (from Cube workspace)
//     7. Store final state FS after last chunk
//
// Cross-core sync: Cube→Vec flags for WS/KV ready, Vec→Cube flags for
// K/S ready.
//
// Inputs:
//   K  [total_tokens, H, D]  half   — keys (BSND layout)
//   W  [total_tokens, H, D]  half   — wy_fast output (BSND layout)
//   U  [total_tokens, H, D]  half   — values pre-residual (BSND layout)
//   G  [H, total_tokens]     float  — pre-transposed cumulative gates
//   S  [total_chunks, H, D, D] half — per-chunk state snapshots (output)
//   V  [total_tokens, H, D]  half   — residual-corrected values (output)
//   FS [batch, H, D, D]      half   — final state per sequence (output)
//   workspace [per-core scratch]     — Cube↔Vec communication buffer
//
// NPU memory hierarchy:
//   GM → L1 (Cube-accessible) → L0A/L0B/L0C (Cube GEMM registers)
//   GM → UB (Vec-accessible, on-chip SRAM)
//   Cross-core sync via FFTS (Fast Fine-grained Task Synchronization)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This is the most complex kernel in the GDN suite. It implements the
// recurrent state update, requiring sequential chunk processing (chunks
// within a sequence CANNOT be parallelized — each depends on the previous).
//
// Key PTO APIs (numpy/torch equivalents):
//   TLOAD(dst, gm)          — dst = gm_data        (DMA: GM→L1 or GM→UB)
//   TSTORE(gm, src)         — gm_data = src        (DMA: UB/L0C→GM)
//   TASSIGN(tile, addr)     — tile = memory[addr]   (bind tile to buffer address)
//   TCVT(dst, src, mode)    — dst = src.float()/.half()
//   TMOV(dst, src)          — dst = src.clone()
//   TADD(d, a, b)           — d = a + b
//   TSUB(d, a, b)           — d = a - b
//   TMUL(d, a, b)           — d = a * b
//   TMULS(d, s, scalar)     — d = s * scalar       (scalar multiply)
//   TADDS(d, s, scalar)     — d = s + scalar       (scalar add)
//   TEXP(d, s)              — d = torch.exp(s)
//   TEXPANDS(tile, scalar)  — tile[:] = scalar     (fill with constant)
//   TROWEXPAND(2d, col)     — 2d[i,j] = col[i]    (broadcast col across row dim)
//   TFILLPAD(dst, src)      — zero-fill L1 tile padding (for tail chunks)
//   TEXTRACT(l0, l1, r, c)  — L1 sub-tile → L0A/L0B
//   TRESHAPE(zn, nz)        — reinterpret layout NZ↔ZN (logical transpose, free)
//   TMATMUL(C, A, B)        — C = A @ B (Cube GEMM, fp16 inputs → fp32 accum)
//   set_flag/wait_flag      — pipe sync within same core
//   ffts_cross_core_sync    — cross-core signal Cube↔Vec
//   wait_flag_dev(flag)     — wait for cross-core signal
//   GetValue(idx)           — read a single scalar from a UB tile (slow, use sparingly)
//
// ── Workspace memory layout (shared between Cube and Vec via GM) ──────
// Each AI core has its own workspace region to avoid contention:
//   WS_WS [C×D]:  Cube writes WS = W @ S here → Vec reads it
//   WS_K  [D×C]:  Vec writes K_scaled here → Cube reads it for KV = K^T @ V
//   WS_S  [D×D]:  Vec writes current state S here → Cube reads it for GEMM 1
//   WS_KV [D×D]:  Cube writes KV = K^T @ V here → Vec reads it to update S
//
// Data flow per chunk (think of it as a ping-pong between Cube and Vec):
//   Vec: write S₀ to WS_S → signal Cube (flag 3)
//   Cube: read S from WS_S, load W → compute WS = W@S → write WS_WS → signal Vec (flag 0)
//   Vec: read WS, compute V_new = U - WS, compute K_scaled → write WS_K → signal Cube (flag 1)
//   Cube: read K from WS_K, load V → compute KV = K^T@V → write WS_KV → signal Vec (flag 2)
//   Vec: read KV, update S = exp(g_last)*S + KV → write S to WS_S → signal Cube (flag 3)
//   ... repeat for next chunk ...
// ============================================================================

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

#ifndef GDN_H
#define GDN_H 16
#endif

#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

// ── PTO type aliases (device-only, guarded by __CCE_AICORE__) ───────────────
// The bisheng compiler makes 3 passes: Vec core, Cube core (both define
// __CCE_AICORE__), and Host (does NOT define it).  All PTO tile types
// must be hidden from the host pass.
//
// Quick tile taxonomy for beginners:
//   UbND  — Vec engine tile, row-major (ND). For element-wise math in UB SRAM.
//   UbDN  — Vec engine tile, col-major (DN). Needed for TROWEXPAND broadcasts.
//   L1Mat — Cube engine tile in L1 cache, NZ fractal format (standard input layout).
//   L1MatZN — Cube engine tile, ZN fractal format (used when you need transpose_A).
//   TileAcc — Cube accumulator in L0C (fp32). TMATMUL writes results here.
//   TileLeft/TileRight — GEMM operands in L0A/L0B respectively.
//
// The template parameters are:
//   <Dtype, StaticRows, StaticCols, DynValidRows, DynValidCols, PadValue>
//   Static shape = tile allocation size. Dynamic valid = how much data is real.
//   Padding fills unused slots with zeros (important for tail chunks < C tokens).
#ifdef __CCE_AICORE__

// UB tile, row-major (ND) layout — used by Vec engine for element-wise ops.
// T=dtype, R×C=static shape, RV×CV=dynamic valid region, P=pad fill for TLOAD.
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;

// UB tile, col-major (DN) layout — needed for TROWEXPAND (broadcasts a
// column vector across rows).
template <typename T, int R, int C, int RV = R, int CV = C>
using UbDN = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::ColMajor,
                       RV, CV, pto::SLayout::NoneBox, 512>;

// L1 matrix tile, col-major base / row-major sub-layout (NZ fractal format).
// Used as Cube GEMM operand source in L1 cache.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::ColMajor,
                        RV, CV, pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// L1 matrix tile, row-major base / col-major sub-layout (ZN fractal format).
// Needed when transposing A before GEMM (TRESHAPE from NZ → ZN).
template <typename T, int R, int C, int RV = R, int CV = C>
using L1MatZN = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::RowMajor,
                          RV, CV, pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

#endif  // __CCE_AICORE__

// ── Kernel function signature ────────────────────────────────────────────
// Template params: NumHeads (H), HiddenSize (D), ChunkSize (C) are compile-time.
// __gm__ pointers point to Global Memory (device DRAM). Each AI core gets
// a unique cid (core ID) and picks its share of work from the total_work pool.
template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void chunk_h_kernel(
    __gm__ half *K_handle, __gm__ half *W_handle, __gm__ half *U_handle,
    __gm__ float *G_handle,
    __gm__ half *S_handle, __gm__ half *V_handle, __gm__ half *FS_handle,
    __gm__ half *workspace_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens,
    uint64_t ffts_addr)
{
  // cid = which AI core am I? block_num = total AI cores launched.
  // Each core processes a subset of (sequence, head) pairs.
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  // FFTS base address enables cross-core synchronization (Cube↔Vec signaling).
  set_ffts_base_addr(ffts_addr);

  constexpr int32_t D = HiddenSize;
  constexpr int32_t C = ChunkSize;
  constexpr int32_t H = NumHeads;
  constexpr int32_t HalfC = C / 2;       // Each Vec sub-block handles C/2 rows
  constexpr int32_t BSND_QKV_STRIDE = H * D;  // Stride between consecutive tokens in BSND layout
  constexpr int32_t DD = D * D;           // Size of the D×D state matrix

  // ── Workspace layout (per AI-core, in half-element units) ─────────────
  // Cube and Vec share workspace via GM for cross-core data exchange.
  // Think of this as a shared mailbox: one engine writes, signals, and the
  // other reads. Each AI core gets its own region (ws_base offset) so cores
  // don't step on each other.
  constexpr int32_t WS_WS = 0;         // WS = W @ S result (C×D) — Cube writes, Vec reads
  constexpr int32_t WS_K  = DD;        // scaled keys from Vec (D×C) — Vec writes, Cube reads
  constexpr int32_t WS_S  = DD * 2;    // current state S (D×D) — Vec writes, Cube reads
  constexpr int32_t WS_KV = DD * 3;    // KV = K^T @ V result (D×D) — Cube writes, Vec reads
  constexpr int32_t WS_PER_CORE = DD * 4;  // Total workspace per core = 4 × D² half elements

  // ── L1 tile assignments (Cube GEMM operands) ─────────────────────────
  // L1 cache is the Cube engine's working memory. We manually partition it
  // into tiles at specific byte offsets using TASSIGN (like malloc, but static).
  //
  // L1 cache layout (Cube engine's working memory):
  //   Address 0:                s_l1 [D×D]  — current state S
  //   Address D*D*2:            w_l1 [C×D]  — W matrix (or K_scaled later)
  //   Address (DD+C*D)*2:       k_l1 [D×C]  — K_scaled (from Vec workspace)
  //   Address (DD+C*D+D*C)*2:   v_l1 [C×D]  — V (value vectors from GM)
  // Cube reads S and W for GEMM 1 (WS = W@S), then K and V for GEMM 2 (KV = K^T@V)
  //
  // Accumulators live in L0C (on-chip registers, fp32):
  //   ws_l0 [C×D]  — result of GEMM 1 (W@S)
  //   kv_l0 [D×D]  — result of GEMM 2 (K^T@V)
  L1Mat<half, D, D, D, D> s_l1;
  TASSIGN(s_l1, 0);
  L1Mat<half, C, D, C, D> w_l1;
  TASSIGN(w_l1, D * D * sizeof(half));
  TileAcc<float, C, D, C, D> ws_l0;
  TASSIGN(ws_l0, 0);
  L1Mat<half, D, C, D, C> k_l1;
  TASSIGN(k_l1, (DD + C * D) * sizeof(half));
  L1Mat<half, C, D, C, D> v_l1;
  TASSIGN(v_l1, (DD + C * D + D * C) * sizeof(half));
  TileAcc<float, D, D, D, D> kv_l0;
  TASSIGN(kv_l0, C * D * sizeof(float));

  // ── UB memory layout (Vec sub-block local SRAM) ──────────────────────
  // UB (Unified Buffer) is the Vec engine's on-chip SRAM (~256 KB).
  // We manually partition it into tiles at specific byte offsets.
  // Think of it as: UB[offset .. offset+size] = one named tensor.
  //
  // Layout map (offsets in bytes):
  //   G_BLOCK_UB:  g_sum values for all heads (pre-fetched for block of chunks)
  //   ZERO_UB:     a tile filled with 0.0 (used for negation via TSUB(0, x))
  //   S_UB:        current state [C/2, D] float (Vec's copy of state)
  //   K_UB_HALF:   keys in half precision [C/2, D]
  //   G_UB:        gate values for current chunk [1, C] float
  //   U_UB_HALF:   wy_fast output in half [C/2, D]
  //   K_UB:        keys in float [C/2, D] (after TCVT from half)
  //   G_V_UB:      gate values for this sub-block [1, 64] float
  //   COEFF_UB:    exp(g - g_last) coefficients [1, 64] float
  //   U_UB:        wy_fast output in float [C/2, D]
  //   WS_UB:       W@S result loaded from workspace [C/2, D] float
  //   KV_UB:       aliases U_UB_HALF (reuses memory — KV is loaded after U is consumed)
  //   S_UB_HALF:   state in half precision (for DMA store to workspace)
  constexpr int32_t G_BLOCK_UB = 0;
  constexpr int32_t G_BLOCK_SIZE = C * H * sizeof(float);
  constexpr int32_t EXPAND_UB = 0;
  constexpr int32_t EXPAND_ROWS = 16;
  constexpr int32_t ZERO_UB = G_BLOCK_SIZE;
  constexpr int32_t S_UB = ZERO_UB + 64 * sizeof(float);
  constexpr int32_t K_UB_HALF = S_UB + HalfC * D * sizeof(float);
  constexpr int32_t G_UB = K_UB_HALF + HalfC * D * sizeof(half);
  constexpr int32_t U_UB_HALF = G_UB + C * sizeof(float);
  constexpr int32_t K_UB = U_UB_HALF + HalfC * D * sizeof(half);
  constexpr int32_t G_V_UB = K_UB + HalfC * D * sizeof(float);
  constexpr int32_t COEFF_UB = G_V_UB + 64 * sizeof(float);
  constexpr int32_t U_UB = COEFF_UB + 64 * sizeof(float);
  constexpr int32_t WS_UB = U_UB + HalfC * D * sizeof(float);
  constexpr int32_t KV_UB = U_UB_HALF;
  constexpr int32_t S_UB_HALF = WS_UB + HalfC * D * sizeof(float);

  // ── UB tile declarations ─────────────────────────────────────────────
  // Each tile is a "view" into UB memory at a fixed offset. TASSIGN binds
  // the tile variable to its memory address — no data is moved, it's like
  // creating a numpy view: zero_ub = ub_memory[ZERO_UB:ZERO_UB+size]
  UbND<float, 1, 64, 1, 64> zero_ub;
  TASSIGN(zero_ub, ZERO_UB);
  UbND<float, HalfC, D, HalfC, D> s_ub;
  TASSIGN(s_ub, S_UB);
  UbND<half, HalfC, D, HalfC, D> k_ub_half;
  TASSIGN(k_ub_half, K_UB_HALF);
  UbND<float, 1, C, 1, C> g_ub;
  TASSIGN(g_ub, G_UB);
  UbND<half, HalfC, D, HalfC, D> s_ub_half;
  TASSIGN(s_ub_half, S_UB_HALF);
  UbND<half, HalfC, D, HalfC, D> u_ub_half;
  TASSIGN(u_ub_half, U_UB_HALF);
  UbND<float, HalfC, D, HalfC, D> k_ub;
  TASSIGN(k_ub, K_UB);
  UbND<float, 1, 64, 1, 64> g_v_ub;
  TASSIGN(g_v_ub, G_V_UB);
  UbND<float, 1, 64, 1, 64> coeff_ub;
  TASSIGN(coeff_ub, COEFF_UB);
  UbND<float, HalfC, D, HalfC, D> u_ub;
  TASSIGN(u_ub, U_UB);
  UbND<float, HalfC, D, HalfC, D> ws_ub;
  TASSIGN(ws_ub, WS_UB);
  UbND<float, HalfC, D, HalfC, D> kv_ub;
  TASSIGN(kv_ub, KV_UB);

  // vid = Vec sub-block ID (0 or 1). The Vec engine has 2 sub-blocks that
  // run in parallel. vid=0 handles rows [0..C/2), vid=1 handles [C/2..C).
  // This doubles Vec throughput by splitting row-wise work.
  auto vid = get_subblockid();

  // Total work items = num_sequences × num_heads. Each AI core picks every
  // block_num-th item (strided distribution across cores).
  int64_t num_seqs = batch_size;
  int64_t total_work = num_seqs * H;

  // ========================================================================
  // CUBE PHASE — two GEMMs per chunk: WS = W @ S, then KV = K^T @ V
  //
  // The Cube engine is the NPU's matrix-multiply unit (like a GPU's tensor
  // cores). It can only do GEMM — no element-wise ops. All element-wise
  // math happens on the Vec engine. Cube and Vec run on SEPARATE hardware
  // cores and communicate through GM workspace + FFTS signals.
  //
  // For each chunk, Cube performs two matrix multiplications:
  //   GEMM 1: WS = W @ S   → projects state through W matrix
  //   GEMM 2: KV = K^T @ V → computes key-value outer product
  // Between GEMMs, it waits for Vec to prepare K_scaled.
  // ========================================================================
#if defined(__DAV_C220_CUBE__)
  // Outer work loop: each iteration processes one (sequence, head) pair.
  // Cores stripe through work items: core 0 gets items 0, N, 2N, ...
  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;  // This core's work item index
    if (pid >= total_work) break;

    // Decode which head and sequence this work item corresponds to.
    int64_t head = pid % H;
    int64_t seq_idx = pid / H;

    // ── Compute sequence boundaries (variable-length support) ──────────
    // cu_seqlens (cumulative sequence lengths) enables packed/ragged batches:
    //   bos = beginning-of-sequence token index in the packed tensor
    //   slen = this sequence's actual length
    //   chunk_offset = how many chunks precede this sequence in S_handle
    int64_t bos, slen;
    int64_t chunk_offset = 0;
    if (cu_seqlens != nullptr) {
      // Variable-length mode: sequences are packed end-to-end
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
      slen = eos - bos;
      // Count total chunks from all preceding sequences
      for (int64_t si = 0; si < seq_idx; ++si) {
        int64_t sb = static_cast<int64_t>(cu_seqlens[si]);
        int64_t se = static_cast<int64_t>(cu_seqlens[si + 1]);
        chunk_offset += (se - sb + C - 1) / C;
      }
    } else {
      // Fixed-length mode: all sequences have the same length
      bos = seq_idx * seq_len;
      slen = seq_len;
      chunk_offset = seq_idx * ((seq_len + C - 1) / C);
    }
    // ceil(slen / C) = number of chunks in this sequence
    int64_t num_chunks = (slen + C - 1) / C;
    // Each core's workspace starts at a different GM offset
    int64_t ws_base = static_cast<int64_t>(cid) * WS_PER_CORE;

    // ── Sequential chunk loop (CANNOT be parallelized — recurrence!) ───
    for (int32_t ci = 0; ci < num_chunks; ++ci) {
      // Wait for Vec to finish writing S to workspace (flag 3)
      // This is the Cube's "start of chunk" sync point — it cannot proceed
      // until Vec has provided the current state S.
      wait_flag_dev(3);

      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      // valid = min(C, remaining tokens). The last chunk may be shorter.
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) valid = C;

      // ── Load S (D×D state) from workspace → L1 ──────────────────────
      {
        L1Mat<half, D, D, DYNAMIC, DYNAMIC> _l1(D, D);
        TASSIGN(_l1, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = D; _gs.shape[4] = D;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, D, 1>>
            _gm(workspace_handle + ws_base + WS_S, _gs);
        TLOAD(_l1, _gm);
      }

      // ── Load W (C×D) from GM → L1, BSND stride ─────────────────────
      // W_handle points to the wy_fast output in BSND layout. The stride
      // between consecutive tokens is H*D (skipping over all heads).
      // If this is a tail chunk (valid < C), we TFILLPAD to zero-fill the
      // padding rows so the GEMM doesn't produce garbage in unused rows.
      {
        int64_t w_offset = ((chunk_start) * H + head) * D;
        L1Mat<half, C, D, DYNAMIC, DYNAMIC> _l1(static_cast<int32_t>(valid), D);
        TASSIGN(_l1, D * D * static_cast<int32_t>(sizeof(half)));
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = static_cast<int32_t>(valid); _gs.shape[4] = D;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_QKV_STRIDE, 1>>
            _gm(W_handle + w_offset, _gs);
        TLOAD(_l1, _gm);
        if (static_cast<int32_t>(valid) != C)
          TFILLPAD(_l1, _l1);
      }

      // ── GEMM 1: WS = W @ S  (no transpose) ─────────────────────────
      // W ∈ L1 (C×D), S ∈ L1 (D×D) → WS ∈ L0C (C×D float accumulator)
      // numpy equivalent: WS = W @ S  →  [C×D] @ [D×D] = [C×D]
      //
      // Pipeline sync dance explained:
      //   set_flag(A, B, id) = "pipe A signals pipe B on event id"
      //   wait_flag(A, B, id) = "pipe B waits for pipe A's signal on event id"
      //   TEXTRACT loads tiles from L1 → L0A/L0B (MTE1 pipe)
      //   TMATMUL runs on the M pipe (matrix multiply hardware)
      //   The flags ensure data is in L0 before GEMM starts, and GEMM is
      //   done before we try to store the result.
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      {
        TileLeft<half, C, D, C, D> _l0a;
        TileRight<half, D, D, D, D> _l0b;
        TASSIGN(_l0a, 0x0); TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we); wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we); wait_flag(PIPE_M, PIPE_MTE1, _we);
        TEXTRACT(_l0a, w_l1, 0, 0);
        TEXTRACT(_l0b, s_l1, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we); wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(ws_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we); wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we); wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // ── Store WS (C×D) from L0C → workspace GM (with half conversion) ─
      // The accumulator is fp32, but workspace stores fp16 (half). TSTORE
      // automatically converts fp32 L0C → fp16 GM (hardware-accelerated).
      // After storing, we signal Vec that WS is ready to read.
      {
        TileAcc<float, C, D, DYNAMIC, DYNAMIC> _l0(C, D);
        TASSIGN(_l0, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = C; _gs.shape[4] = D;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, D, 1>>
            _gm(workspace_handle + ws_base + WS_WS, _gs);
        TSTORE(_gm, _l0);
      }
      // Signal Vec: WS is ready (Cube→Vec flag 0)
      // ffts_cross_core_sync encodes: direction | (core_mask << 4) | (flag_id << 8)
      //   1 = signal (not wait), 2 = target core mask, 0 = flag ID
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));

      // Wait for Vec to finish writing K_scaled to workspace (flag 1)
      wait_flag_dev(1);

      // ── Load K_scaled (D×C) from workspace → L1 ────────────────────
      {
        L1Mat<half, D, C, DYNAMIC, DYNAMIC> _l1(D, C);
        TASSIGN(_l1, (DD + C * D) * static_cast<int32_t>(sizeof(half)));
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = D; _gs.shape[4] = C;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, C, 1>>
            _gm(workspace_handle + ws_base + WS_K, _gs);
        TLOAD(_l1, _gm);
      }

      // ── Load V (C×D) from GM → L1, BSND stride ─────────────────────
      {
        int64_t v_offset = ((chunk_start) * H + head) * D;
        L1Mat<half, C, D, DYNAMIC, DYNAMIC> _l1(static_cast<int32_t>(valid), D);
        TASSIGN(_l1, (DD + C * D + D * C) * static_cast<int32_t>(sizeof(half)));
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = static_cast<int32_t>(valid); _gs.shape[4] = D;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_QKV_STRIDE, 1>>
            _gm(V_handle + v_offset, _gs);
        TLOAD(_l1, _gm);
        if (static_cast<int32_t>(valid) != C)
          TFILLPAD(_l1, _l1);
      }

      // ── GEMM 2: KV = K^T @ V  (transpose_A) ───────────────────────
      // K ∈ L1 (D×C NZ) → reshape to ZN for transpose, V ∈ L1 (C×D)
      // Result: KV ∈ L0C (D×D float accumulator)
      //
      // numpy: KV = K_scaled.T @ V  →  [D×C] @ [C×D] = [D×D]
      // To transpose K_scaled for the Cube, we TRESHAPE the L1 tile from
      // NZ→ZN format. TRESHAPE is a zero-cost operation — it just
      // reinterprets the fractal memory layout, effectively transposing
      // the matrix without moving any data. This is possible because the
      // NZ fractal format stores data in 16×16 sub-blocks, and swapping
      // the interpretation of "row-major sub / col-major base" to
      // "col-major sub / row-major base" is equivalent to transposing.
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      {
        TileLeft<half, D, C, D, C> _l0a;
        TileRight<half, C, D, C, D> _l0b;
        TASSIGN(_l0a, 0x0); TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we); wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we); wait_flag(PIPE_M, PIPE_MTE1, _we);
        // TRESHAPE NZ→ZN implements the transpose of K before extraction
        L1MatZN<half, D, C> _azn; TRESHAPE(_azn, k_l1); TEXTRACT(_l0a, _azn, 0, 0);
        TEXTRACT(_l0b, v_l1, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we); wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(kv_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we); wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we); wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // ── Store KV (D×D) from L0C → workspace GM ─────────────────────
      {
        TileAcc<float, D, D, DYNAMIC, DYNAMIC> _l0(D, D);
        TASSIGN(_l0, C * D * static_cast<int32_t>(sizeof(float)));
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = D; _gs.shape[4] = D;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, D, 1>>
            _gm(workspace_handle + ws_base + WS_KV, _gs);
        TSTORE(_gm, _l0);
      }
      // Signal Vec: KV is ready (Cube→Vec flag 2)
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
    }
  }
#endif

  // ========================================================================
  // VEC PHASE — gate scaling, state update, cross-core data exchange
  // Two Vec sub-blocks (vid=0,1) each handle C/2 rows independently.
  //
  // The Vec engine handles all element-wise operations: exp, add, sub, mul,
  // type conversion, etc. It cannot do matrix multiply (that's Cube's job).
  // The two sub-blocks (vid=0 and vid=1) split the C rows in half so they
  // can process in parallel, doubling throughput.
  //
  // The Vec phase orchestrates the entire chunk pipeline:
  //   1. Initialize state S = 0
  //   2. For each chunk:
  //      a. Load K, G, U from GM
  //      b. Compute decay coefficients and scale K
  //      c. Wait for Cube's WS, compute V_new = U - WS
  //      d. Send K_scaled + V_new to Cube for GEMM 2
  //      e. Wait for Cube's KV, update S = exp(g_last)*S + KV
  //      f. Send updated S back to Cube for next iteration
  //   3. Store final state FS
  // ========================================================================
#if defined(__DAV_C220_VEC__)
  // set_mask_norm + set_vector_mask(-1,-1): enable all Vec lanes (no masking).
  // The Vec engine processes 256 bits per cycle; masking selects which lanes
  // are active. -1 = all bits set = all lanes active.
  set_mask_norm();
  set_vector_mask(-1, -1);

  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) break;

    // Same (head, sequence) decoding as Cube phase — both engines must
    // process the same work item so their workspace reads/writes match.
    int64_t head = pid % H;
    int64_t seq_idx = pid / H;

    // Compute sequence boundaries (same logic as Cube — see comments above)
    int64_t bos, slen;
    int64_t chunk_offset = 0;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
      slen = eos - bos;
      for (int64_t si = 0; si < seq_idx; ++si) {
        int64_t sb = static_cast<int64_t>(cu_seqlens[si]);
        int64_t se = static_cast<int64_t>(cu_seqlens[si + 1]);
        chunk_offset += (se - sb + C - 1) / C;
      }
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
      chunk_offset = seq_idx * ((seq_len + C - 1) / C);
    }
    int64_t num_chunks = (slen + C - 1) / C;
    int64_t ws_base = static_cast<int64_t>(cid) * WS_PER_CORE;

    // ── Initialize state S = 0 for the first chunk ────────────────────────
    // For the first chunk of each sequence, S starts at zero.
    // TEXPANDS(s_ub, 0.0f) fills the state tile with zeros:
    //   numpy equivalent: S = np.zeros((D, D), dtype=np.float32)
    //
    // We also fill zero_ub with 0.0 — this constant tile is used later to
    // negate values via TSUB(zero, x) = -x (since there's no TNEG instruction).
    //
    // The set_flag/wait_flag pairs around TEXPANDS synchronize the Vec pipe (V)
    // with the scalar pipe (S) — TEXPANDS uses the scalar unit to broadcast.
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(zero_ub, 0.0f);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(s_ub, 0.0f);

    // Convert zero state to half and store to workspace for Cube.
    // numpy equivalent: workspace['S'] = S.astype(np.float16)
    // The Cube can only read fp16 from workspace (it feeds into GEMM which
    // requires fp16 inputs), so we must convert before storing.
    TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    {
      Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
      _gs.shape[3] = HalfC; _gs.shape[4] = D;
      GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, D, 1>>
          _gm(workspace_handle + ws_base * sizeof(half) + WS_S * sizeof(half)
               + vid * HalfC * D * sizeof(half), _gs);
      UbND<half, HalfC, D, DYNAMIC, DYNAMIC> _st(HalfC, D);
      TASSIGN(_st, S_UB_HALF);
      TSTORE(_gm, _st);
    }
    // Signal Cube: initial S is ready (Vec→Cube flag 3)
    // This kicks off the first iteration — Cube is waiting on flag 3 to read S.
    ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));

    // ── Prefetch K and G for the first chunk ────────────────────────────
    // We start loading K and G from GM → UB BEFORE entering the chunk loop.
    // This "primes the pump" so data is ready when the loop body needs it.
    // Subsequent prefetches happen inside the loop (overlapped with Cube work).
    int64_t chunk_start_0 = bos;
    // vid * HalfC * BSND_QKV_STRIDE: skip to this sub-block's rows.
    // vid=0 reads rows [0..C/2), vid=1 reads rows [C/2..C).
    int64_t k_offset_0 = (chunk_start_0 * H + head) * D + vid * HalfC * BSND_QKV_STRIDE;
    {
      Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
      _gs.shape[3] = HalfC; _gs.shape[4] = D;
      GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_QKV_STRIDE, 1>>
          _gm(K_handle + k_offset_0, _gs);
      UbND<half, HalfC, D, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfC, D);
      TASSIGN(_ld, K_UB_HALF);
      TLOAD(_ld, _gm);
    }

    // G is pre-transposed to [H, total_tokens] float — contiguous per head.
    // This layout means all gate values for one head are adjacent in memory,
    // enabling efficient DMA. The transpose was done on the host/prior kernel.
    {
      Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
      _gs.shape[3] = 1; _gs.shape[4] = C;
      GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>>
          _gm(G_handle + head * total_tokens + chunk_start_0, _gs);
      UbND<float, 1, C, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, C);
      TASSIGN(_ld, G_UB);
      TLOAD(_ld, _gm);
    }

    // Wait for the prefetch DMA to finish before Vec starts using the data.
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // ── Main chunk loop ─────────────────────────────────────────────────
    // Each iteration processes one chunk of C tokens. Chunks MUST be
    // processed sequentially because S_{c+1} depends on S_c.
    for (int32_t ci = 0; ci < static_cast<int32_t>(num_chunks); ++ci) {
      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      // valid = actual number of tokens in this chunk (last chunk may be < C)
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) valid = C;

      // Load U (wy_fast output) for this chunk — this is the "uncorrected"
      // value that will become V_new = U - W@S after the residual subtraction.
      {
        int64_t u_offset = (chunk_start * H + head) * D + vid * HalfC * BSND_QKV_STRIDE;
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HalfC; _gs.shape[4] = D;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_QKV_STRIDE, 1>>
            _gm(U_handle + u_offset, _gs);
        UbND<half, HalfC, D, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfC, D);
        TASSIGN(_ld, U_UB_HALF);
        TLOAD(_ld, _gm);
      }

      // K half→float for scaling (Vec math operates on fp32 for precision)
      TCVT(k_ub, k_ub_half, pto::RoundMode::CAST_NONE);

      // Extract this sub-block's gate slice (vid selects upper/lower half).
      // g_ub holds all C gate values; vid=0 reads g[0..63], vid=1 reads g[64..127].
      UbND<float, 1, 64, 1, 64> g_ub_temp;
      TASSIGN(g_ub_temp, G_UB + vid * 64 * sizeof(float));
      TMOV(g_v_ub, g_ub_temp);

      // ── Time-decay coefficient: coeff[i] = exp(g_last - g[i]) ────────
      // This scales each token's key by how "old" it is relative to the
      // chunk end. Tokens near the end get coeff ≈ 1 (recent), tokens at
      // the start get coeff > 1 (but after K scaling and the state update
      // recurrence, the net effect is proper exponential gating).
      //
      // numpy equivalent:
      //   g_last = g[valid - 1]                    # last gate value in chunk
      //   coeff = np.exp(g_last - g[my_rows])      # decay from token to end
      //
      // Step by step:
      //   1. TADDS(coeff, g_v, -g_last)  → coeff = g[i] - g_last  (≤ 0, since g is cumsum)
      //   2. TSUB(coeff, zero, coeff)    → coeff = -(g[i] - g_last) = g_last - g[i]  (≥ 0)
      //   3. TEXP(coeff, coeff)          → coeff = exp(g_last - g[i])
      //
      // Result: K_scaled[i] = K[i] * exp(g_last - g[i])
      // This ensures recent tokens (near chunk end) have larger keys.
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // GetValue reads a scalar from a UB tile — slow (stalls pipeline),
      // but we only need one value per chunk so it's acceptable.
      float g_last = g_ub.GetValue(static_cast<int32_t>(valid) - 1);
      TADDS(coeff_ub, g_v_ub, -g_last);
      pipe_barrier(PIPE_V);
      TSUB(coeff_ub, zero_ub, coeff_ub);
      pipe_barrier(PIPE_V);
      TEXP(coeff_ub, coeff_ub);

      // exp(g) for the full chunk — we need g_ub = exp(cumulative_gate) later
      // for the state decay: S *= exp(g_last). The TEXP here converts all C
      // gate values in-place, so g_ub[valid-1] will be exp(g_last) afterwards.
      TEXP(g_ub, g_ub);

      // Wait for the U load DMA to finish, then convert U from half to float.
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(u_ub, u_ub_half, pto::RoundMode::CAST_NONE);

      // ── Scale K rows by decay coefficients ────────────────────────────
      // We need: K_scaled[i, d] = K[i, d] * coeff[i] for all d.
      // This is a "row broadcast multiply" — each row of K gets multiplied
      // by a scalar from coeff.
      //
      // TROWEXPAND(expanded, coeff_col): broadcasts coeff_col into a 2D tile:
      //   expanded[i, j] = coeff_col[i] for all j
      //   (Like numpy: np.tile(coeff[:, None], (1, D)))
      // Then TMUL(k_blk, k_blk, expanded) = element-wise multiply.
      //
      // We process in blocks of EXPAND_ROWS=16 because TROWEXPAND has a max
      // tile size it can handle efficiently on the Vec hardware.
      for (int32_t blk = 0; blk < HalfC / EXPAND_ROWS; ++blk) {
        UbDN<float, EXPAND_ROWS, 1,
             EXPAND_ROWS, 1> coeff_blk;
        TASSIGN(coeff_blk, COEFF_UB + blk * EXPAND_ROWS *
                                          static_cast<int32_t>(sizeof(float)));
        UbND<float, EXPAND_ROWS, D,
             EXPAND_ROWS, D> expanded;
        TASSIGN(expanded, EXPAND_UB);
        TROWEXPAND(expanded, coeff_blk);
        pipe_barrier(PIPE_V);

        UbND<float, EXPAND_ROWS, D,
             EXPAND_ROWS, D> k_blk;
        TASSIGN(k_blk, K_UB + blk * EXPAND_ROWS * D *
                                   static_cast<int32_t>(sizeof(float)));
        TMUL(k_blk, k_blk, expanded);
        pipe_barrier(PIPE_V);
      }

      // ── Wait for Cube's WS result, compute V_new = U - WS ──────────
      // flag 0: Cube signals WS is ready in workspace.
      // V_new = U - WS (residual correction):
      //   numpy: V_new = U - (W @ S)
      //   U comes from wy_fast kernel, WS = W @ S comes from Cube via workspace.
      //   The subtraction "corrects" U by removing the state-projected component.
      //   This is the "delta" in GatedDeltaNet — we update S with only the
      //   residual information not already captured by the current state.
      wait_flag_dev(0);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HalfC; _gs.shape[4] = D;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, D, 1>>
            _gm(workspace_handle + ws_base * sizeof(half) + WS_WS * sizeof(half)
                 + vid * HalfC * D * sizeof(half), _gs);
        UbND<half, HalfC, D, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfC, D);
        TASSIGN(_ld, U_UB_HALF);
        TLOAD(_ld, _gm);
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // WS was loaded as half → convert to float for subtraction
      TCVT(ws_ub, u_ub_half, pto::RoundMode::CAST_NONE);
      // V_new = U - WS (the core "delta rule" residual correction)
      TSUB(u_ub, u_ub, ws_ub);
      // Convert results back to half for DMA store to GM
      TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);
      TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

      // ── Store V_new to output V (BSND layout) ──────────────────────
      // This is a final output — V_new goes to the V output tensor in GM,
      // which downstream kernels will read.
      {
        int64_t v_offset = (chunk_start * H + head) * D + vid * HalfC * BSND_QKV_STRIDE;
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HalfC; _gs.shape[4] = D;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_QKV_STRIDE, 1>>
            _gm(V_handle + v_offset, _gs);
        UbND<half, HalfC, D, DYNAMIC, DYNAMIC> _st(HalfC, D);
        TASSIGN(_st, U_UB_HALF);
        TSTORE(_gm, _st);
      }

      // ── Store K_scaled to workspace for Cube's GEMM 2 ─────────────
      // Cube will read K_scaled from WS_K to compute KV = K_scaled^T @ V_new.
      // Note: K_scaled is stored as [HalfC, D] per sub-block; the two sub-blocks
      // write to different halves of the D×C workspace region.
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HalfC; _gs.shape[4] = D;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, D, 1>>
            _gm(workspace_handle + ws_base * sizeof(half) + WS_K * sizeof(half)
                 + vid * HalfC * D * sizeof(half), _gs);
        UbND<half, HalfC, D, DYNAMIC, DYNAMIC> _st(HalfC, D);
        TASSIGN(_st, K_UB_HALF);
        TSTORE(_gm, _st);
      }

      // Signal Cube: K_scaled is ready (Vec→Cube flag 1)
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));

      // ── State decay: S = exp(g_last) * S ────────────────────────────
      // This is the first half of the state update recurrence:
      //   S_{c+1} = exp(g_last) * S_c + KV
      // We compute exp(g_last)*S now, and add KV after Cube finishes GEMM 2.
      //
      // exp_g_last = exp(g[valid-1]) was pre-computed by TEXP(g_ub, g_ub) above.
      // TMULS multiplies every element of s_ub by this scalar.
      // numpy: S = exp(g[valid-1]) * S
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      float exp_g_last = g_ub.GetValue(static_cast<int32_t>(valid) - 1);
      TMULS(s_ub, s_ub, exp_g_last);

      // ── Prefetch next chunk's K and G while waiting for Cube's KV ────
      // While waiting for Cube to finish GEMM 2 (KV = K^T @ V), we use MTE2
      // (the DMA-in pipe) to start loading the NEXT chunk's K and G from GM → UB.
      // This hides DMA latency behind Cube computation time — a key optimization
      // that keeps the Vec engine busy instead of idling.
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        int64_t next_start = bos + static_cast<int64_t>(ci + 1) * C;
        int64_t next_valid = slen - static_cast<int64_t>(ci + 1) * C;
        if (next_valid > C) next_valid = C;

        int64_t nk_off = (next_start * H + head) * D + vid * HalfC * BSND_QKV_STRIDE;
        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = HalfC; _gs.shape[4] = D;
          GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_QKV_STRIDE, 1>>
              _gm(K_handle + nk_off, _gs);
          UbND<half, HalfC, D, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfC, D);
          TASSIGN(_ld, K_UB_HALF);
          TLOAD(_ld, _gm);
        }

        // G is pre-transposed to [H, total_tokens] float.
        // If this is the last chunk and it's shorter than C, we load only
        // next_valid elements and zero-pad the rest with TFILLPAD_INPLACE
        // so the unused gate values don't corrupt the computation.
        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = 1; _gs.shape[4] = static_cast<int32_t>(next_valid);
          GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>>
              _gm(G_handle + head * total_tokens + next_start, _gs);
          UbND<float, 1, C, DYNAMIC, DYNAMIC, PadValue::Zero>
              _ld(1, static_cast<int32_t>(next_valid));
          TASSIGN(_ld, G_UB);
          TLOAD(_ld, _gm);
          if (static_cast<int32_t>(next_valid) != C) {
            UbND<float, 1, C, 1, C, PadValue::Zero> _pd;
            TASSIGN(_pd, G_UB);
            TFILLPAD_INPLACE(_pd, _ld);
          }
        }
      }

      // ── Wait for Cube's KV result, accumulate into S ────────────────
      // flag 2: Cube signals KV is ready in workspace.
      // This completes the state update: S_{c+1} = exp(g_last)*S_c + KV
      // We already computed exp(g_last)*S above; now we add KV.
      wait_flag_dev(2);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HalfC; _gs.shape[4] = D;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, D, 1>>
            _gm(workspace_handle + ws_base * sizeof(half) + WS_KV * sizeof(half)
                 + vid * HalfC * D * sizeof(half), _gs);
        UbND<half, HalfC, D, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfC, D);
        TASSIGN(_ld, S_UB_HALF);
        TLOAD(_ld, _gm);
      }

      // ── State update: S_{c+1} = exp(g_last) * S_c + KV ──────────────
      // numpy: S = exp(g[valid-1]) * S + K_scaled.T @ V_new
      // exp(g_last) decays the old state, then we add the new key-value outer
      // product. This is the core recurrence of GatedDeltaNet's linear attention.
      //
      // s_ub already holds exp(g_last)*S from the decay step above.
      // kv_ub holds the KV result from Cube (loaded from workspace, converted to float).
      // TADD performs the final accumulation.
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // Convert KV from half (workspace format) to float (computation format)
      TCVT(kv_ub, s_ub_half, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_ALL);
      // S = exp(g_last)*S + KV  (the GatedDeltaNet recurrence!)
      TADD(s_ub, s_ub, kv_ub);
      // Convert updated state back to half for storage
      TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);

      // ── Store updated S to workspace and snapshot output ────────────
      // Two stores happen here:
      //   1. S → workspace WS_S: so Cube can read it for the NEXT chunk's GEMM 1
      //   2. S → S_handle output: a snapshot of S after each chunk (for backward pass)
      // We only do this if there's a next chunk; the final state goes to FS.
      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = HalfC; _gs.shape[4] = D;
          GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, D, 1>>
              _gm(workspace_handle + ws_base * sizeof(half) + WS_S * sizeof(half)
                   + vid * HalfC * D * sizeof(half), _gs);
          UbND<half, HalfC, D, DYNAMIC, DYNAMIC> _st(HalfC, D);
          TASSIGN(_st, S_UB_HALF);
          TSTORE(_gm, _st);
        }

        {
          int64_t s_out_offset = ((chunk_offset + ci + 1) * H + head) * DD;
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = HalfC; _gs.shape[4] = D;
          GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, D, 1>>
              _gm(S_handle + s_out_offset + vid * HalfC * D, _gs);
          UbND<half, HalfC, D, DYNAMIC, DYNAMIC> _st(HalfC, D);
          TASSIGN(_st, S_UB_HALF);
          TSTORE(_gm, _st);
        }
        // Signal Cube: updated S is ready (Vec→Cube flag 3)
        // This unblocks Cube's wait_flag_dev(3) at the top of the next chunk iteration.
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
      }

      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      }
    }

    // ── Store final state FS for this sequence ──────────────────────────
    // After all chunks are processed, the final state S is the "memory" that
    // carries over to the next forward pass (or is used by the backward pass).
    // FS[seq_idx, head, :, :] = S_final  (shape [batch, H, D, D] in half)
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    {
      int64_t fs_offset = (seq_idx * H + head) * DD;
      Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
      _gs.shape[3] = HalfC; _gs.shape[4] = D;
      GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, D, 1>>
          _gm(FS_handle + fs_offset + vid * HalfC * D, _gs);
      UbND<half, HalfC, D, DYNAMIC, DYNAMIC> _st(HalfC, D);
      TASSIGN(_st, S_UB_HALF);
      TSTORE(_gm, _st);
    }
  }
#endif
}

// ── Device entry point ────────────────────────────────────────────────
// extern "C" __global__ AICORE: this is the NPU kernel entry point.
// Each AI core runs one instance of this function in parallel.
// Pointers are uint8_t* (type-erased) — standard NPU calling convention.
// The actual types are reinterpret_cast'd inside to half*/float*/int32_t*.
extern "C" __global__ AICORE void launch_chunk_h(
    __gm__ uint8_t *K, __gm__ uint8_t *W, __gm__ uint8_t *U,
    __gm__ uint8_t *G,
    __gm__ uint8_t *S, __gm__ uint8_t *V, __gm__ uint8_t *FS,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens,
    uint64_t ffts_addr)
{
  chunk_h_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K),
      reinterpret_cast<__gm__ half *>(W),
      reinterpret_cast<__gm__ half *>(U),
      reinterpret_cast<__gm__ float *>(G),
      reinterpret_cast<__gm__ half *>(S),
      reinterpret_cast<__gm__ half *>(V),
      reinterpret_cast<__gm__ half *>(FS),
      reinterpret_cast<__gm__ half *>(workspace),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, ffts_addr);
}

// ── Host launcher (called from Python via ctypes) ─────────────────────
// block_dim = number of AI cores to launch.
// rtGetC2cCtrlAddr obtains the FFTS (cross-core sync) hardware address.
// <<<block_dim, nullptr, stream>>> is the NPU kernel launch syntax
// (analogous to CUDA's <<<grid, block, stream>>>).
extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *K, uint8_t *W, uint8_t *U, uint8_t *G,
    uint8_t *S, uint8_t *V, uint8_t *FS,
    uint8_t *workspace,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_h<<<block_dim, nullptr, stream>>>(
      K, W, U, G, S, V, FS, workspace, cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
