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
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  set_ffts_base_addr(ffts_addr);

  constexpr int32_t D = HiddenSize;
  constexpr int32_t C = ChunkSize;
  constexpr int32_t H = NumHeads;
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t BSND_QKV_STRIDE = H * D;
  constexpr int32_t DD = D * D;

  // ── Workspace layout (per AI-core, in half-element units) ─────────────
  // Cube and Vec share workspace via GM for cross-core data exchange.
  constexpr int32_t WS_WS = 0;         // WS = W @ S result (C×D)
  constexpr int32_t WS_K  = DD;        // scaled keys from Vec (D×C)
  constexpr int32_t WS_S  = DD * 2;    // current state S (D×D)
  constexpr int32_t WS_KV = DD * 3;    // KV = K^T @ V result (D×D)
  constexpr int32_t WS_PER_CORE = DD * 4;

  // ── L1 tile assignments (Cube GEMM operands) ─────────────────────────
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

  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;
  int64_t total_work = num_seqs * H;

  // ========================================================================
  // CUBE PHASE — two GEMMs per chunk: WS = W @ S, then KV = K^T @ V
  // ========================================================================
#if defined(__DAV_C220_CUBE__)
  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) break;

    int64_t head = pid % H;
    int64_t seq_idx = pid / H;

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

    for (int32_t ci = 0; ci < num_chunks; ++ci) {
      // Wait for Vec to finish writing S to workspace (flag 3)
      wait_flag_dev(3);

      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
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
  // ========================================================================
#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) break;

    int64_t head = pid % H;
    int64_t seq_idx = pid / H;

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

    // ── Initialize S = 0 for the first chunk ────────────────────────────
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(zero_ub, 0.0f);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(s_ub, 0.0f);

    // Convert zero state to half and store to workspace for Cube
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
    ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));

    // ── Prefetch K and G for the first chunk ────────────────────────────
    int64_t chunk_start_0 = bos;
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

    // G is pre-transposed to [H, total_tokens] float — contiguous per head
    {
      Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
      _gs.shape[3] = 1; _gs.shape[4] = C;
      GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>>
          _gm(G_handle + head * total_tokens + chunk_start_0, _gs);
      UbND<float, 1, C, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, C);
      TASSIGN(_ld, G_UB);
      TLOAD(_ld, _gm);
    }

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // ── Main chunk loop ─────────────────────────────────────────────────
    for (int32_t ci = 0; ci < static_cast<int32_t>(num_chunks); ++ci) {
      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) valid = C;

      // Load U (wy_fast output) for this chunk
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

      // K half→float for scaling
      TCVT(k_ub, k_ub_half, pto::RoundMode::CAST_NONE);

      // Extract this sub-block's gate slice (vid selects upper/lower half)
      UbND<float, 1, 64, 1, 64> g_ub_temp;
      TASSIGN(g_ub_temp, G_UB + vid * 64 * sizeof(float));
      TMOV(g_v_ub, g_ub_temp);

      // ── Compute coeff[i] = exp(g[i] - g[valid-1]) ──────────────────
      // This gives the time-decay factor relative to the chunk's last token.
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      float g_last = g_ub.GetValue(static_cast<int32_t>(valid) - 1);
      TADDS(coeff_ub, g_v_ub, -g_last);
      pipe_barrier(PIPE_V);
      TSUB(coeff_ub, zero_ub, coeff_ub);
      pipe_barrier(PIPE_V);
      TEXP(coeff_ub, coeff_ub);

      // exp(g) for the full chunk (used later for state decay)
      TEXP(g_ub, g_ub);

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(u_ub, u_ub_half, pto::RoundMode::CAST_NONE);

      // ── Scale K rows by coeff via TROWEXPAND ────────────────────────
      // K_scaled[i,:] = K[i,:] * exp(g[i] - g_last)
      // Process in blocks of EXPAND_ROWS for TROWEXPAND tile size.
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
      // flag 0: Cube signals WS is ready in workspace
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
      TCVT(ws_ub, u_ub_half, pto::RoundMode::CAST_NONE);
      // V_new = U - WS (residual correction)
      TSUB(u_ub, u_ub, ws_ub);
      TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);
      TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

      // ── Store V_new to output V (BSND layout) ──────────────────────
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

      // ── Store K_scaled to workspace for Cube's next GEMM 2 ─────────
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
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      float exp_g_last = g_ub.GetValue(static_cast<int32_t>(valid) - 1);
      TMULS(s_ub, s_ub, exp_g_last);

      // ── Prefetch next chunk's K and G while waiting for KV ──────────
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

        // G is pre-transposed to [H, total_tokens] float
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
      // flag 2: Cube signals KV is ready in workspace
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

      // S_{c+1} = exp(g_last) * S_c + KV
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(kv_ub, s_ub_half, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_ALL);
      TADD(s_ub, s_ub, kv_ub);
      TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);

      // ── Store updated S to workspace and snapshot output ────────────
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
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
      }

      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      }
    }

    // ── Store final state FS for this sequence ──────────────────────────
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
