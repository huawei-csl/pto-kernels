/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

// ============================================================================
// kernel_chunk_o.cpp — Output computation for GatedDeltaNet (chunk-wise)
//
// Mathematical operation (per chunk of C tokens, per head h):
//
//   O = (QK_gated @ V) + exp(g) * (Q @ S)
//     = intra_chunk_attention + inter_chunk_state_contribution
//
// where:
//   Q, K, V ∈ ℝ^{C×D}    — query/key/value projections for this chunk
//   S ∈ ℝ^{D×D}           — accumulated hidden state entering this chunk
//   G ∈ ℝ^{C}             — cumulative gate values (pre-transposed [H,T])
//   Msk ∈ ℝ^{C×C}         — lower-triangular causal mask
//
// Cube phase (3 GEMMs per chunk):
//   1. QK   = Q @ K^T         — intra-chunk attention scores
//   2. QS   = Q @ S           — query applied to accumulated state
//   3. QKV  = QK_gated @ V    — gated attention applied to values
//
// Vec phase (two sub-blocks process upper/lower C/2 rows):
//   a. Load G → compute gating coefficients:
//        coeff[i,j] = exp(min(g[i] - g[j], 0)) * mask[i,j]
//   b. Apply gating to QK: QK_gated = QK * coeff
//   c. Scale QS by exp(g): QS_gated = QS * exp(g_row)
//   d. Combine: O = QS_gated + QKV
//   e. Store O to GM in BSND layout
//
// Cross-core sync protocol (Cube ↔ Vec via FFTS):
//   flag 0: Cube→Vec  — QK and QS results ready in workspace
//   flag 1: Vec→Cube  — QK_gated written back, Cube can proceed to GEMM 3
//   flag 2: Cube→Vec  — QKV result ready in workspace
//   flag 3: Vec→Cube  — Vec done with this chunk, Cube can reuse workspace
//
// NPU memory hierarchy used:
//   GM → L1 (Cube-accessible) → L0A/L0B (matrix engines) → L0C (accumulator)
//   GM → UB (Vec-accessible, on-chip SRAM)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This kernel combines matrix multiplication (Cube) with element-wise gating
// (Vec) in a tightly coordinated 3-GEMM + gating pipeline per chunk.
//
// Execution timeline for one chunk:
//   Cube: GEMM1(Q@K^T) → GEMM2(Q@S) → store QK,QS → signal Vec ──────┐
//   Vec:  (meanwhile) load G, compute gating coefficients                │
//   Vec:  ←── wait for Cube signal ──── apply gating to QK → QK_gated  │
//   Vec:  store QK_gated → signal Cube ────────────────────────────────┐│
//   Cube: ←── wait for Vec signal ──── GEMM3(QK_gated@V) → store QKV ─┘│
//   Vec:  ←── wait for Cube signal ──── scale QS, combine O=QKV+QS_g   │
//   Vec:  store O → signal Cube "done" ─────────────────────────────────┘
//
// numpy pseudocode for the entire chunk computation:
//   QK = Q @ K.T                                          # GEMM 1
//   QS = Q @ S                                            # GEMM 2
//   coeff = exp(min(g_row - g_col, 0)) * mask             # gating (dynamic
//   PTO) QK_gated = QK * coeff                                 # apply gating
//   QKV = QK_gated @ V                                    # GEMM 3
//   O = QKV + QS * np.exp(g_row).reshape(-1, 1)           # final output
// ============================================================================

#include "kernel_utils.h"

using namespace pto;

// ── Compile-time configuration (overridable at build time via -D flags) ──
#ifndef GDN_H
#define GDN_H 16
#endif

#ifndef GDN_HG
#define GDN_HG GDN_H
#endif

#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

// ── PTO type aliases (device-only, guarded for host pass safety) ────────────
#ifdef __CCE_AICORE__

// UbND = Unified Buffer tile, row-major (ND) layout, for Vec SIMD ops.
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor, RV,
                       CV, pto::SLayout::NoneBox, 512, P>;

// UbDN = UB tile in column-major (DN) layout for TROWEXPAND.
template <typename T, int R, int C, int RV = R, int CV = C>
using UbDN = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::ColMajor, RV,
                       CV, pto::SLayout::NoneBox, 512>;

// L1Mat = L1 cache tile in NZ fractal format — standard Cube GEMM input.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::ColMajor, RV,
                        CV, pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// L1MatZN = ZN fractal format — used for transposed GEMM operands.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1MatZN =
    pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::RowMajor, RV, CV,
              pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

#endif  // __CCE_AICORE__

template <int32_t NumHeads, int32_t NumKeyHeads, int32_t HiddenSize,
          int32_t ChunkSize>
AICORE void chunk_o_kernel(__gm__ half* Q_handle, __gm__ half* K_handle,
                           __gm__ half* V_handle, __gm__ half* S_handle,
                           __gm__ float* G_handle, __gm__ float* Msk_handle,
                           __gm__ half* workspace_qk_handle,
                           __gm__ half* workspace_qs_qkv_handle,
                           __gm__ half* workspace_qk_gated_handle,
                           __gm__ half* O_handle, __gm__ int32_t* cu_seqlens,
                           int64_t batch_size, int64_t seq_len,
                           int64_t total_tokens) {
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr uint32_t KTail = (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);
  constexpr uint32_t CTail = (ChunkSize % 128 == 0) ? 128 : (ChunkSize % 128);

  constexpr int32_t H = NumHeads;
  constexpr int32_t Hg = NumKeyHeads;
  static_assert(Hg > 0 && H % Hg == 0,
                "NumHeads must be divisible by NumKeyHeads");
  constexpr int32_t GROUP = H / Hg;
  constexpr int32_t BSND_V_STRIDE = H * HiddenSize;
  constexpr int32_t BSND_QK_STRIDE = Hg * HiddenSize;

  constexpr int32_t WsQKSize = ChunkSize * ChunkSize;
  constexpr int32_t WsQSSize = ChunkSize * HiddenSize;
  constexpr int32_t WsGatedSize = ChunkSize * ChunkSize;

  // ── UB memory map (byte addresses within Unified Buffer) ─────────────
  constexpr int32_t GUbAddr = 0;
  constexpr int32_t MskUbAddr = 512;
  constexpr int32_t QKUbAddr = 33280;
  constexpr int32_t GvUbAddr = 66048;
  constexpr int32_t CoeffUbAddr = 66304;
  constexpr int32_t QKHalfUbAddr = 99072;
  constexpr int32_t QSHalfUbAddr = 115456;
  constexpr int32_t QSUbAddr = 131840;
  constexpr int32_t OHalfUbAddr = 164608;
  constexpr int32_t OUbAddr = QKUbAddr;

  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;

  // ── L1 tiles for Cube GEMM operands ──────────────────────────────────
  L1Mat<half, ChunkSize, HiddenSize> q_l1;
  TASSIGN(q_l1, 0);
  L1Mat<half, ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 32768);
  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> qk_l0;
  TASSIGN(qk_l0, 0);
  L1Mat<half, HiddenSize, HiddenSize> s_l1;
  TASSIGN(s_l1, 65536);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> qs_l0;
  TASSIGN(qs_l0, 65536);
  L1Mat<half, ChunkSize, ChunkSize> qk_gated_l1;
  TASSIGN(qk_gated_l1, 98304);
  L1Mat<half, ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 131072);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> qkv_l0;
  TASSIGN(qkv_l0, 0);

  // ── UB tiles for Vec element-wise operations ─────────────────────────
  UbND<float, 1, ChunkSize> g_ub;
  TASSIGN(g_ub, GUbAddr);
  UbND<float, HalfChunk, ChunkSize> msk_ub;
  TASSIGN(msk_ub, MskUbAddr);
  UbND<float, HalfChunk, ChunkSize> qk_ub;
  TASSIGN(qk_ub, QKUbAddr);
  UbND<float, 1, HalfChunk> g_v_ub;
  TASSIGN(g_v_ub, GvUbAddr);
  UbND<float, HalfChunk, ChunkSize> coeff_ub;
  TASSIGN(coeff_ub, CoeffUbAddr);
  UbND<half, HalfChunk, ChunkSize, HalfChunk, ChunkSize, PadValue::Zero>
      qk_ub_half;
  TASSIGN(qk_ub_half, QKHalfUbAddr);
  UbND<half, HalfChunk, HiddenSize, HalfChunk, HiddenSize, PadValue::Zero>
      qs_ub_half;
  TASSIGN(qs_ub_half, QSHalfUbAddr);
  UbND<float, HalfChunk, HiddenSize> qs_ub;
  TASSIGN(qs_ub, QSUbAddr);
  UbND<half, HalfChunk, HiddenSize, HalfChunk, HiddenSize, PadValue::Zero>
      o_ub_half;
  TASSIGN(o_ub_half, OHalfUbAddr);
  UbND<float, HalfChunk, HiddenSize> o_ub;
  TASSIGN(o_ub, OUbAddr);

  int64_t total_work = 0;
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    total_work = num_seqs * chunks_per_seq * NumHeads;
  }

// =====================================================================
// CUBE CORE — Three GEMMs per chunk: QK, QS, QKV
// =====================================================================
#if defined(__DAV_C220_CUBE__)
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    int64_t global_chunk_base = 0;
    bool first_cube_iter = true;

    for (int64_t work_idx = static_cast<int64_t>(cid); work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      if (!first_cube_iter) wait_flag_dev(3);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      int32_t head_idx = static_cast<int32_t>(work_idx % NumHeads);
      int32_t head_g = head_idx / GROUP;
      int64_t chunk_head_idx = work_idx / NumHeads;
      int64_t seq_idx = chunk_head_idx / chunks_per_seq;
      int64_t ci = chunk_head_idx % chunks_per_seq;

      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows =
          static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);
      int64_t chunk_token_start = bos + chunk_start;
      int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
      int32_t local_rows = valid_rows - row_offset;
      if (local_rows < 0) local_rows = 0;
      if (local_rows > HalfChunk) local_rows = HalfChunk;

      int64_t qk_off = (chunk_token_start * static_cast<int64_t>(Hg) +
                        static_cast<int64_t>(head_g)) *
                       static_cast<int64_t>(HiddenSize);
      int64_t v_off = (chunk_token_start * static_cast<int64_t>(H) +
                       static_cast<int64_t>(head_idx)) *
                      static_cast<int64_t>(HiddenSize);

      int64_t chunk_global_idx = seq_idx * chunks_per_seq + ci;
      int64_t s_offset = (chunk_global_idx * NumHeads + head_idx) *
                         static_cast<int64_t>(HiddenSize) *
                         static_cast<int64_t>(HiddenSize);

      // ── Load Q [valid_rows × D] from GM → L1 ────────────────────────
      {
        L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows,
                                                                 HiddenSize);
        TASSIGN(_l1, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_QK_STRIDE, 1>>
            _gm(Q_handle + qk_off, _gs);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }
      // ── Load K [valid_rows × D] from GM → L1 ────────────────────────
      {
        L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows,
                                                                 HiddenSize);
        TASSIGN(_l1, 32768);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_QK_STRIDE, 1>>
            _gm(K_handle + qk_off, _gs);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }

      // ── GEMM 1: QK = Q @ K^T ─────────────────────────────────────────
      {
        TileLeft<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
        TileRight<half, HiddenSize, ChunkSize, HiddenSize, ChunkSize> _l0b;
        TASSIGN(_l0a, 0x0);
        TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we);
        wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we);
        wait_flag(PIPE_M, PIPE_MTE1, _we);
        TEXTRACT(_l0a, q_l1, 0, 0);
        L1MatZN<half, HiddenSize, ChunkSize> _bzn;
        TRESHAPE(_bzn, k_l1);
        TEXTRACT(_l0b, _bzn, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we);
        wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(qk_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we);
        wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we);
        wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // ── Load S [D × D] from GM → L1 ─────────────────────────────────
      {
        L1Mat<half, HiddenSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(HiddenSize,
                                                                  HiddenSize);
        TASSIGN(_l1, 65536);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HiddenSize;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
            S_handle + s_offset, _gs);
        TLOAD(_l1, _gm);
      }

      // ── GEMM 2: QS = Q @ S ───────────────────────────────────────────
      {
        TileLeft<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
        TileRight<half, HiddenSize, HiddenSize, HiddenSize, HiddenSize> _l0b;
        TASSIGN(_l0a, 0x0);
        TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we);
        wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we);
        wait_flag(PIPE_M, PIPE_MTE1, _we);
        TEXTRACT(_l0a, q_l1, 0, 0);
        TEXTRACT(_l0b, s_l1, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we);
        wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(qs_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we);
        wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we);
        wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // ── Store QK [C × C] from L0C → GM workspace ────────────────────
      {
        TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l0(ChunkSize,
                                                                   ChunkSize);
        TASSIGN(_l0, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize;
        _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_qk_handle + static_cast<int64_t>(cid) * WsQKSize, _gs);
        TSTORE(_gm, _l0);
      }

      // ── Store QS [C × D] from L0C → GM workspace ────────────────────
      {
        TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(ChunkSize,
                                                                    HiddenSize);
        TASSIGN(_l0, 65536);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
            workspace_qs_qkv_handle + static_cast<int64_t>(cid) * WsQSSize,
            _gs);
        TSTORE(_gm, _l0);
      }

      // Signal Vec: QK and QS are ready (flag 0, Cube→Vec)
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));

      // Wait for Vec to write QK_gated back (flag 1, Vec→Cube)
      wait_flag_dev(1);

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      // ── Load QK_gated [C × C] from GM workspace → L1 ────────────────
      {
        L1Mat<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(ChunkSize,
                                                                ChunkSize);
        TASSIGN(_l1, 98304);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize;
        _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_qk_gated_handle + static_cast<int64_t>(cid) * WsGatedSize,
            _gs);
        TLOAD(_l1, _gm);
      }
      // ── Load V [valid_rows × D] from GM → L1 ────────────────────────
      {
        L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows,
                                                                 HiddenSize);
        TASSIGN(_l1, 131072);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_V_STRIDE, 1>>
            _gm(V_handle + v_off, _gs);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }

      // ── GEMM 3: QKV = QK_gated @ V ───────────────────────────────────
      {
        TileLeft<half, ChunkSize, ChunkSize, ChunkSize, ChunkSize> _l0a;
        TileRight<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0b;
        TASSIGN(_l0a, 0x0);
        TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we);
        wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we);
        wait_flag(PIPE_M, PIPE_MTE1, _we);
        TEXTRACT(_l0a, qk_gated_l1, 0, 0);
        TEXTRACT(_l0b, v_l1, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we);
        wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(qkv_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we);
        wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we);
        wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // ── Store QKV [C × D] from L0C → GM workspace ───────────────────
      // workspace_qs_qkv_handle is reused for QKV (safe: Vec has finished
      // reading QS before Cube writes QKV here, enforced by flag 1).
      {
        TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(ChunkSize,
                                                                    HiddenSize);
        TASSIGN(_l0, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
            workspace_qs_qkv_handle + static_cast<int64_t>(cid) * WsQSSize,
            _gs);
        TSTORE(_gm, _l0);
      }

      // Signal Vec: QKV is ready (flag 2, Cube→Vec)
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
      first_cube_iter = false;
    }
  } else {
    // ── Variable-length sequence path (cu_seqlens != nullptr) ──────────
    int64_t gi = 0;
    int64_t chunk_global_idx = 0;
    bool first_cube_iter_v = true;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            if (!first_cube_iter_v) wait_flag_dev(3);
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;
            int32_t head_g = head_idx / GROUP;

            int64_t qk_off = (chunk_token_start * static_cast<int64_t>(Hg) +
                              static_cast<int64_t>(head_g)) *
                             static_cast<int64_t>(HiddenSize);
            int64_t v_off = (chunk_token_start * static_cast<int64_t>(H) +
                             static_cast<int64_t>(head_idx)) *
                            static_cast<int64_t>(HiddenSize);
            int64_t s_offset = (chunk_global_idx * NumHeads + head_idx) *
                               static_cast<int64_t>(HiddenSize) *
                               static_cast<int64_t>(HiddenSize);

            // Load Q
            {
              L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(
                  valid_rows, HiddenSize);
              TASSIGN(_l1, 0);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows;
              _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs),
                           Stride<1, 1, 1, BSND_QK_STRIDE, 1>>
                  _gm(Q_handle + qk_off, _gs);
              TLOAD(_l1, _gm);
              if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
            }
            // Load K
            {
              L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(
                  valid_rows, HiddenSize);
              TASSIGN(_l1, 32768);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows;
              _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs),
                           Stride<1, 1, 1, BSND_QK_STRIDE, 1>>
                  _gm(K_handle + qk_off, _gs);
              TLOAD(_l1, _gm);
              if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
            }

            // GEMM 1: QK = Q @ K^T
            {
              TileLeft<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
              TileRight<half, HiddenSize, ChunkSize, HiddenSize, ChunkSize>
                  _l0b;
              TASSIGN(_l0a, 0x0);
              TASSIGN(_l0b, 0x0);
              auto _we = EVENT_ID1;
              set_flag(PIPE_MTE2, PIPE_MTE1, _we);
              wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
              set_flag(PIPE_M, PIPE_MTE1, _we);
              wait_flag(PIPE_M, PIPE_MTE1, _we);
              TEXTRACT(_l0a, q_l1, 0, 0);
              L1MatZN<half, HiddenSize, ChunkSize> _bzn;
              TRESHAPE(_bzn, k_l1);
              TEXTRACT(_l0b, _bzn, 0, 0);
              set_flag(PIPE_MTE1, PIPE_M, _we);
              wait_flag(PIPE_MTE1, PIPE_M, _we);
              TMATMUL(qk_l0, _l0a, _l0b);
              set_flag(PIPE_MTE1, PIPE_MTE2, _we);
              wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
              set_flag(PIPE_M, PIPE_FIX, _we);
              wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            // Load S
            {
              L1Mat<half, HiddenSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(
                  HiddenSize, HiddenSize);
              TASSIGN(_l1, 65536);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = HiddenSize;
              _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>>
                  _gm(S_handle + s_offset, _gs);
              TLOAD(_l1, _gm);
            }

            // GEMM 2: QS = Q @ S
            {
              TileLeft<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
              TileRight<half, HiddenSize, HiddenSize, HiddenSize, HiddenSize>
                  _l0b;
              TASSIGN(_l0a, 0x0);
              TASSIGN(_l0b, 0x0);
              auto _we = EVENT_ID1;
              set_flag(PIPE_MTE2, PIPE_MTE1, _we);
              wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
              set_flag(PIPE_M, PIPE_MTE1, _we);
              wait_flag(PIPE_M, PIPE_MTE1, _we);
              TEXTRACT(_l0a, q_l1, 0, 0);
              TEXTRACT(_l0b, s_l1, 0, 0);
              set_flag(PIPE_MTE1, PIPE_M, _we);
              wait_flag(PIPE_MTE1, PIPE_M, _we);
              TMATMUL(qs_l0, _l0a, _l0b);
              set_flag(PIPE_MTE1, PIPE_MTE2, _we);
              wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
              set_flag(PIPE_M, PIPE_FIX, _we);
              wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            // Store QK → workspace
            {
              TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l0(
                  ChunkSize, ChunkSize);
              TASSIGN(_l0, 0);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize;
              _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>>
                  _gm(workspace_qk_handle +
                          static_cast<int64_t>(cid) * WsQKSize,
                      _gs);
              TSTORE(_gm, _l0);
            }

            // Store QS → workspace
            {
              TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(
                  ChunkSize, HiddenSize);
              TASSIGN(_l0, 65536);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize;
              _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>>
                  _gm(workspace_qs_qkv_handle +
                          static_cast<int64_t>(cid) * WsQSSize,
                      _gs);
              TSTORE(_gm, _l0);
            }

            // Cube→Vec: QK & QS ready (flag 0)
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));

            // Wait Vec→Cube: QK_gated ready (flag 1)
            wait_flag_dev(1);

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

            // Load QK_gated
            {
              L1Mat<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(
                  ChunkSize, ChunkSize);
              TASSIGN(_l1, 98304);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize;
              _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>>
                  _gm(workspace_qk_gated_handle +
                          static_cast<int64_t>(cid) * WsGatedSize,
                      _gs);
              TLOAD(_l1, _gm);
            }
            // Load V
            {
              L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(
                  valid_rows, HiddenSize);
              TASSIGN(_l1, 131072);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows;
              _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs),
                           Stride<1, 1, 1, BSND_V_STRIDE, 1>>
                  _gm(V_handle + v_off, _gs);
              TLOAD(_l1, _gm);
              if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
            }

            // GEMM 3: QKV = QK_gated @ V
            {
              TileLeft<half, ChunkSize, ChunkSize, ChunkSize, ChunkSize> _l0a;
              TileRight<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize>
                  _l0b;
              TASSIGN(_l0a, 0x0);
              TASSIGN(_l0b, 0x0);
              auto _we = EVENT_ID1;
              set_flag(PIPE_MTE2, PIPE_MTE1, _we);
              wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
              set_flag(PIPE_M, PIPE_MTE1, _we);
              wait_flag(PIPE_M, PIPE_MTE1, _we);
              TEXTRACT(_l0a, qk_gated_l1, 0, 0);
              TEXTRACT(_l0b, v_l1, 0, 0);
              set_flag(PIPE_MTE1, PIPE_M, _we);
              wait_flag(PIPE_MTE1, PIPE_M, _we);
              TMATMUL(qkv_l0, _l0a, _l0b);
              set_flag(PIPE_MTE1, PIPE_MTE2, _we);
              wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
              set_flag(PIPE_M, PIPE_FIX, _we);
              wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            {
              TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(
                  ChunkSize, HiddenSize);
              TASSIGN(_l0, 0);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize;
              _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>>
                  _gm(workspace_qs_qkv_handle +
                          static_cast<int64_t>(cid) * WsQSSize,
                      _gs);
              TSTORE(_gm, _l0);
            }

            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
            first_cube_iter_v = false;
          }
          gi++;
        }
        chunk_global_idx++;
      }
    }
  }
#endif

// =====================================================================
// VEC CORE — Gating, element-wise ops, output assembly
// =====================================================================
#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  // ── Load causal mask once (reused across all chunks) ─────────────────
  {
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = HalfChunk;
    _gs.shape[4] = ChunkSize;
    GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
        Msk_handle + static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
    UbND<float, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
        HalfChunk, ChunkSize);
    TASSIGN(_ld, MskUbAddr);
    TLOAD(_ld, _gm);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;

    for (int64_t work_idx = static_cast<int64_t>(cid); work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      int32_t head_idx = static_cast<int32_t>(work_idx % NumHeads);
      int64_t chunk_head_idx = work_idx / NumHeads;
      int64_t seq_idx = chunk_head_idx / chunks_per_seq;
      int64_t ci = chunk_head_idx % chunks_per_seq;

      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows =
          static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);
      int64_t chunk_token_start = bos + chunk_start;
      int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
      int32_t local_rows = valid_rows - row_offset;
      if (local_rows < 0) local_rows = 0;
      if (local_rows > HalfChunk) local_rows = HalfChunk;

      if (local_rows > 0) {
        // ── Load G [1 × valid_rows] ────────────────────────────────────
        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = 1;
          _gs.shape[4] = valid_rows;
          GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
              G_handle + static_cast<int64_t>(head_idx) * total_tokens +
                  chunk_token_start,
              _gs);
          UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
              1, valid_rows);
          TASSIGN(_ld, GUbAddr);
          TLOAD(_ld, _gm);
          if (valid_rows != ChunkSize) {
            UbND<float, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
            TASSIGN(_pd, GUbAddr);
            TFILLPAD_INPLACE(_pd, _ld);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // ── Compute gating coefficients ────────────────────────────────
        // coeff[i,j] = exp(min(g[i] - g[j], 0)) * mask[i,j]
        UbND<float, 1, HalfChunk> g_ub_temp_0;
        TASSIGN(g_ub_temp_0, GUbAddr + static_cast<int32_t>(vid) * HalfChunk *
                                           static_cast<int32_t>(sizeof(float)));
        TMOV(g_v_ub, g_ub_temp_0);

        UbND<float, HalfChunk, ChunkSize> g_r_2d;
        TASSIGN(g_r_2d, QSUbAddr);
        UbDN<float, HalfChunk, 1> g_v_col;
        TASSIGN(g_v_col, GvUbAddr);
        TROWEXPAND(g_r_2d, g_v_col);
        TCOLEXPAND(coeff_ub, g_ub);
        TSUB(coeff_ub, g_r_2d, coeff_ub);
        pipe_barrier(PIPE_V);
        TMINS(coeff_ub, coeff_ub, 0.0f);
        pipe_barrier(PIPE_V);
        TEXP(coeff_ub, coeff_ub);
        pipe_barrier(PIPE_V);
        TMUL(coeff_ub, coeff_ub, msk_ub);
        pipe_barrier(PIPE_V);
        TEXP(g_v_ub, g_v_ub);
      }

      // ── Wait for Cube→Vec flag 0: QK & QS ready ─────────────────────
      wait_flag_dev(0);
      if (local_rows == 0) {
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
        wait_flag_dev(2);
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
        continue;
      }

      // ── Load QK [C/2 × C] from workspace → UB ───────────────────────
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows;
        _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_qk_handle + static_cast<int64_t>(cid) * WsQKSize +
                static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
            _gs);
        UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
            local_rows, ChunkSize);
        TASSIGN(_ld, QKHalfUbAddr);
        TLOAD(_ld, _gm);
        if (local_rows != HalfChunk) {
          TFILLPAD_INPLACE(qk_ub_half, _ld);
        }
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

      // ── Load QS [C/2 × D] from workspace → UB ───────────────────────
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
            workspace_qs_qkv_handle + static_cast<int64_t>(cid) * WsQSSize +
                static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
            _gs);
        UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
            local_rows, HiddenSize);
        TASSIGN(_ld, QSHalfUbAddr);
        TLOAD(_ld, _gm);
        if (local_rows != HalfChunk) {
          TFILLPAD_INPLACE(qs_ub_half, _ld);
        }
      }

      // ── Apply gating: QK_gated = QK * coeff ─────────────────────────
      TMUL(qk_ub, qk_ub, coeff_ub);
      TCVT(qk_ub_half, qk_ub, pto::RoundMode::CAST_NONE);

      // ── Store QK_gated [C/2 × C] → workspace for Cube's GEMM 3 ─────
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows;
        _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_qk_gated_handle +
                static_cast<int64_t>(cid) * WsGatedSize +
                static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
            _gs);
        UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(local_rows,
                                                               ChunkSize);
        TASSIGN(_st, QKHalfUbAddr);
        TSTORE(_gm, _st);
      }
      // Vec→Cube: QK_gated ready (flag 1)
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));

      // ── Scale QS by exp(g): QS_gated = QS * exp(g_row) ──────────────
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(qs_ub, qs_ub_half, pto::RoundMode::CAST_NONE);
      UbND<float, HalfChunk, HiddenSize> g_exp_2d;
      TASSIGN(g_exp_2d, CoeffUbAddr);
      UbDN<float, HalfChunk, 1> g_v_col2;
      TASSIGN(g_v_col2, GvUbAddr);
      TROWEXPAND(g_exp_2d, g_v_col2);
      pipe_barrier(PIPE_V);
      TMUL(qs_ub, qs_ub, g_exp_2d);

      // ── Wait for Cube→Vec flag 2: QKV ready ─────────────────────────
      wait_flag_dev(2);

      // ── Load QKV [C/2 × D] from workspace → UB ──────────────────────
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
            workspace_qs_qkv_handle + static_cast<int64_t>(cid) * WsQSSize +
                static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
            _gs);
        UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
            local_rows, HiddenSize);
        TASSIGN(_ld, OHalfUbAddr);
        TLOAD(_ld, _gm);
        if (local_rows != HalfChunk) {
          TFILLPAD_INPLACE(o_ub_half, _ld);
        }
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      // ── Combine: O = QS_gated + QKV ─────────────────────────────────
      TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);
      TADD(o_ub, qs_ub, o_ub);
      TCVT(o_ub_half, o_ub, pto::RoundMode::CAST_NONE);

      // ── Store O [C/2 × D] → GM in BSND layout ───────────────────────
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

      int64_t o_offset = (chunk_token_start * static_cast<int64_t>(H) +
                          static_cast<int64_t>(head_idx)) *
                             static_cast<int64_t>(HiddenSize) +
                         static_cast<int64_t>(vid) * HalfChunk *
                             static_cast<int64_t>(BSND_V_STRIDE);

      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows;
        _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_V_STRIDE, 1>>
            _gm(O_handle + o_offset, _gs);
        UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC> _st(local_rows,
                                                                HiddenSize);
        TASSIGN(_st, OHalfUbAddr);
        TSTORE(_gm, _st);
      }

      // Vec→Cube: done with this chunk (flag 3)
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
    }
  } else {
    // ── Variable-length sequence path (cu_seqlens != nullptr) ──────────
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;
            int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
            int32_t local_rows = valid_rows - row_offset;
            if (local_rows < 0) local_rows = 0;
            if (local_rows > HalfChunk) local_rows = HalfChunk;

            if (local_rows > 0) {
              // Load G
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = 1;
                _gs.shape[4] = valid_rows;
                GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
                    G_handle + static_cast<int64_t>(head_idx) * total_tokens +
                        chunk_token_start,
                    _gs);
                UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(
                    1, valid_rows);
                TASSIGN(_ld, GUbAddr);
                TLOAD(_ld, _gm);
                if (valid_rows != ChunkSize) {
                  UbND<float, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
                  TASSIGN(_pd, GUbAddr);
                  TFILLPAD_INPLACE(_pd, _ld);
                }
              }
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

              // Compute gating coefficients
              UbND<float, 1, HalfChunk> g_ub_temp_v;
              TASSIGN(g_ub_temp_v,
                      GUbAddr + static_cast<int32_t>(vid) * HalfChunk *
                                    static_cast<int32_t>(sizeof(float)));
              TMOV(g_v_ub, g_ub_temp_v);

              UbND<float, HalfChunk, ChunkSize> g_r_2d_v;
              TASSIGN(g_r_2d_v, QSUbAddr);
              UbDN<float, HalfChunk, 1> g_v_col_v;
              TASSIGN(g_v_col_v, GvUbAddr);
              TROWEXPAND(g_r_2d_v, g_v_col_v);
              TCOLEXPAND(coeff_ub, g_ub);
              TSUB(coeff_ub, g_r_2d_v, coeff_ub);
              pipe_barrier(PIPE_V);
              TMINS(coeff_ub, coeff_ub, 0.0f);
              pipe_barrier(PIPE_V);
              TEXP(coeff_ub, coeff_ub);
              pipe_barrier(PIPE_V);
              TMUL(coeff_ub, coeff_ub, msk_ub);
              pipe_barrier(PIPE_V);
              TEXP(g_v_ub, g_v_ub);
            }

            wait_flag_dev(0);
            if (local_rows == 0) {
              ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
              wait_flag_dev(2);
              ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
            } else {
              // Load QK from workspace
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows;
                _gs.shape[4] = ChunkSize;
                GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>>
                    _gm(workspace_qk_handle +
                            static_cast<int64_t>(cid) * WsQKSize +
                            static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                        _gs);
                UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC,
                     PadValue::Zero>
                    _ld(local_rows, ChunkSize);
                TASSIGN(_ld, QKHalfUbAddr);
                TLOAD(_ld, _gm);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(qk_ub_half, _ld);
                }
              }

              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);

              set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

              // Load QS from workspace
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows;
                _gs.shape[4] = HiddenSize;
                GlobalTensor<half, decltype(_gs),
                             Stride<1, 1, 1, HiddenSize, 1>>
                    _gm(workspace_qs_qkv_handle +
                            static_cast<int64_t>(cid) * WsQSSize +
                            static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                        _gs);
                UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC,
                     PadValue::Zero>
                    _ld(local_rows, HiddenSize);
                TASSIGN(_ld, QSHalfUbAddr);
                TLOAD(_ld, _gm);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(qs_ub_half, _ld);
                }
              }

              TMUL(qk_ub, qk_ub, coeff_ub);
              TCVT(qk_ub_half, qk_ub, pto::RoundMode::CAST_NONE);

              // Store QK_gated → workspace
              set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows;
                _gs.shape[4] = ChunkSize;
                GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>>
                    _gm(workspace_qk_gated_handle +
                            static_cast<int64_t>(cid) * WsGatedSize +
                            static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                        _gs);
                UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(
                    local_rows, ChunkSize);
                TASSIGN(_st, QKHalfUbAddr);
                TSTORE(_gm, _st);
              }
              // Vec→Cube: QK_gated ready (flag 1)
              ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));

              // Scale QS by exp(g)
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              TCVT(qs_ub, qs_ub_half, pto::RoundMode::CAST_NONE);

              UbND<float, HalfChunk, HiddenSize> g_exp_2d_v;
              TASSIGN(g_exp_2d_v, CoeffUbAddr);
              UbDN<float, HalfChunk, 1> g_v_col2_v;
              TASSIGN(g_v_col2_v, GvUbAddr);
              TROWEXPAND(g_exp_2d_v, g_v_col2_v);
              pipe_barrier(PIPE_V);
              TMUL(qs_ub, qs_ub, g_exp_2d_v);

              wait_flag_dev(2);

              // Load QKV from workspace
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows;
                _gs.shape[4] = HiddenSize;
                GlobalTensor<half, decltype(_gs),
                             Stride<1, 1, 1, HiddenSize, 1>>
                    _gm(workspace_qs_qkv_handle +
                            static_cast<int64_t>(cid) * WsQSSize +
                            static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                        _gs);
                UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC,
                     PadValue::Zero>
                    _ld(local_rows, HiddenSize);
                TASSIGN(_ld, OHalfUbAddr);
                TLOAD(_ld, _gm);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(o_ub_half, _ld);
                }
              }

              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

              // O = QS_gated + QKV
              TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);
              TADD(o_ub, qs_ub, o_ub);
              TCVT(o_ub_half, o_ub, pto::RoundMode::CAST_NONE);

              // Store O → GM
              set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

              int64_t o_offset = (chunk_token_start * static_cast<int64_t>(H) +
                                  static_cast<int64_t>(head_idx)) *
                                     static_cast<int64_t>(HiddenSize) +
                                 static_cast<int64_t>(vid) * HalfChunk *
                                     static_cast<int64_t>(BSND_V_STRIDE);

              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows;
                _gs.shape[4] = HiddenSize;
                GlobalTensor<half, decltype(_gs),
                             Stride<1, 1, 1, BSND_V_STRIDE, 1>>
                    _gm(O_handle + o_offset, _gs);
                UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC> _st(
                    local_rows, HiddenSize);
                TASSIGN(_st, OHalfUbAddr);
                TSTORE(_gm, _st);
              }

              // Vec→Cube: done with this chunk (flag 3)
              ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
            }
          }
          gi++;
        }
      }
    }
  }
#endif
}

// ── Device kernel entry point ─────────────────────────────────────────
extern "C" __global__ AICORE void gdn_chunk_o(
    __gm__ uint8_t* Q_handle, __gm__ uint8_t* K_handle,
    __gm__ uint8_t* V_handle, __gm__ uint8_t* S_handle,
    __gm__ uint8_t* G_handle, __gm__ uint8_t* Msk_handle,
    __gm__ uint8_t* workspace_qk, __gm__ uint8_t* workspace_qs_qkv,
    __gm__ uint8_t* workspace_qk_gated, __gm__ uint8_t* O_handle,
    __gm__ uint8_t* cu_seqlens, int64_t batch_size, int64_t seq_len,
    int64_t total_tokens) {
  chunk_o_kernel<GDN_H, GDN_HG, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half*>(Q_handle),
      reinterpret_cast<__gm__ half*>(K_handle),
      reinterpret_cast<__gm__ half*>(V_handle),
      reinterpret_cast<__gm__ half*>(S_handle),
      reinterpret_cast<__gm__ float*>(G_handle),
      reinterpret_cast<__gm__ float*>(Msk_handle),
      reinterpret_cast<__gm__ half*>(workspace_qk),
      reinterpret_cast<__gm__ half*>(workspace_qs_qkv),
      reinterpret_cast<__gm__ half*>(workspace_qk_gated),
      reinterpret_cast<__gm__ half*>(O_handle),
      reinterpret_cast<__gm__ int32_t*>(cu_seqlens), batch_size, seq_len,
      total_tokens);
}
