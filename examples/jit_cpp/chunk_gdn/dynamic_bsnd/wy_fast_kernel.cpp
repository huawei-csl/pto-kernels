// ============================================================================
// wy_fast_kernel.cpp — WY representation for GatedDeltaNet chunk recurrence
//
// Computes the WY update matrices U and W for each chunk of C tokens:
//   U = A2 @ V     where A2 = A * beta_2d        (beta-scaled attention)
//   W = A1 @ K     where A1 = A * (exp(g)*beta)_2d (gate+beta-scaled attention)
//
// beta is the decay factor, g is the gate value, A is the triangular attention
// matrix (from the kkt kernel).  The column-broadcast notation x_2d means
// expanding a 1xC vector into a C/2 x C matrix by replicating across rows.
//
// Architecture: Vec+Cube cooperative kernel using cross-core synchronization.
//
//  Vec core (two sub-blocks for upper/lower C/2 rows):
//    For each chunk:
//      1. Load beta [H,T] and A [B,S,H,C], compute A2 = A * beta_2d -> ws
//      2. Load G [H,T], compute A1 = A * (exp(g)*beta)_2d -> ws
//      3. Signal Cube via cross-core flags when workspaces are ready
//
//  Cube core (waits for Vec signals):
//    For each chunk:
//      1. Load K, V from BSND layout into L1
//      2. Load A2 from workspace -> GEMM: U = A2 @ V
//      3. Load A1 from workspace -> GEMM: W = A1 @ K
//      4. Store U, W back to BSND layout
//
// NPU memory hierarchy used:
//   GM -> UB (Vec), GM -> L1 -> L0A/L0B -> L0C -> GM (Cube)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This kernel uses BOTH the Cube engine (matrix multiply) and Vec engine
// (SIMD element-wise ops), running on SEPARATE physical cores that
// communicate via Global Memory (GM) + cross-core flags (FFTS).
//
// Execution flow:
//   Vec core:  load A,beta,G → compute A2,A1 → store to GM workspace
//   Cube core: wait for workspace → load A2/A1 + K/V → GEMM → store U,W
//
// Key PTO APIs (with numpy/torch equivalents):
//   TLOAD(ub_tile, gm)      — ub_tile = gm[...]          (DMA: GM→UB, async MTE2)
//   TSTORE(gm, ub_tile)     — gm[...] = ub_tile          (DMA: UB→GM, async MTE3)
//   TCVT(dst, src, mode)    — dst = src.float() or .half() (type conversion)
//   TMOV(dst, src)          — dst = src.clone()
//   TMUL(d, a, b)           — d = a * b                   (element-wise)
//   TEXP(d, s)              — d = torch.exp(s)
//   TCOLEXPAND(2d, row)     — 2d[i,j] = row[j]  (broadcast row across all rows)
//   TEXTRACT(l0, l1, r, c)  — L1 sub-block → L0A/L0B     (MTE1 for Cube GEMM)
//   TMATMUL(C, A, B)        — C = A @ B in Cube engine    (fp16→fp32 accumulate)
//   set_flag / wait_flag    — sync between pipes on SAME core
//   ffts_cross_core_sync    — signal ACROSS Cube↔Vec cores
//   wait_flag_dev(flag)     — wait for cross-core signal
// ============================================================================

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

// Compile-time constants for head count, hidden size, and chunk size.
// These are set via -D flags at JIT compilation time to specialize the kernel.
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
// UbND<T, R, C, RV, CV, P>: A tile in UB (on-chip SRAM) with row-major layout.
//   Like torch.empty((R, C), dtype=T) in fast on-chip memory.
//   T=dtype, R×C=static shape, RV×CV=valid sub-region (handles partial/tail chunks).
//   P = pad value for TLOAD (PadValue::Zero fills outside valid region with 0).
//   Used by Vec engine for element-wise computation.
#ifdef __CCE_AICORE__
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;

// L1Mat<T, R, C>: A tile in L1 cache, NZ (column-major) fractal format,
//   for Cube GEMM input.
//   Think of it as a matrix staged in L1 cache, ready for matrix multiplication.
//   TLOAD(l1_tile, gm_tensor) loads data from GM → L1.
//   TEXTRACT(l0_tile, l1_tile, row, col) copies from L1 → L0A or L0B
//   (the Cube engine's register files).
//   T=dtype, R×C=static shape, RV×CV=valid region. Zero-padded on TLOAD.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::ColMajor,
                        RV, CV, pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;
#endif

// ── Kernel function (runs on each AI core) ────────────────────────────
// Template params: NumHeads (H), HiddenSize (D), ChunkSize (C).
// __gm__ pointers: Global Memory addresses passed from the host.
//   K, V:         key/value tensors [B, S, N, D] (BSND layout)
//   Beta, G:      decay/gate vectors [H, total_tokens] (pre-transposed)
//   A:            triangular attention matrix [B, S, H, C] (from kkt kernel)
//   workspace_a1/a2: GM scratch space for Vec→Cube data transfer
//   W, U:         output matrices [B, S, N, D] (BSND layout)
//   cu_seqlens:   cumulative seq lengths (nullptr for fixed-length batches)
//   ffts_addr:    cross-core synchronization control address
template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void wy_fast_kernel(
    __gm__ half *K_handle, __gm__ half *V_handle,
    __gm__ half *Beta_handle, __gm__ float *G_handle,
    __gm__ half *A_handle,
    __gm__ half *workspace_a1_handle, __gm__ half *workspace_a2_handle,
    __gm__ half *W_handle, __gm__ half *U_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens,
    uint64_t ffts_addr)
{
  // Each Vec sub-block processes half the chunk rows (C/2).
  constexpr int32_t HalfChunk = ChunkSize / 2;
  // KTail handles the last partial 128-element block of HiddenSize (for alignment).
  constexpr uint32_t KTail =
      (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);

  // ── UB Memory Layout (manual memory management) ─────────────────────
  // On NPU, there is NO dynamic memory allocator for on-chip buffers.
  // We manually assign each tile a fixed byte address in UB, like a C union.
  // The compiler verifies these don't overlap (or we manage it ourselves).
  // Think of it as: ub = bytearray(256*1024)  # 256KB UB
  //   beta_ub_half = ub[0:256]       # half[1, C]
  //   a1_ub_half   = ub[256:16640]   # half[C/2, C]
  //   beta_ub      = ub[16640:17152] # float[1, C]
  //   beta_r_ub    = ub[17152:17664] # float[1, C]  (copy for TCOLEXPAND)
  //   beta_2d_ub   = ub[17664:50432] # float[C/2, C] (broadcast result)
  //   tmp_ub       = ub[50432:75008] # scratch space
  //   a1_ub        = ub[75008:107776]  # float[C/2, C]
  //   a2_ub        = ub[107776:140544] # float[C/2, C]
  //   a2_ub_half   = ub[140544:156928] # half[C/2, C]
  //   g_ub         = ub[156928:157440] # float[1, C]
  //   g_r_ub       = ub[157440:157952] # float[1, C]  (copy for TCOLEXPAND)
  //   g_2d_ub      = ub[157952:...]    # float[C/2, C] (broadcast result)
  constexpr int32_t BetaHalfUbAddr = 0;
  constexpr int32_t A1HalfUbAddr   = 256;
  constexpr int32_t BetaUbAddr     = 16640;
  constexpr int32_t BetaRUbAddr    = 17152;
  constexpr int32_t Beta2dUbAddr   = 17664;
  constexpr int32_t TmpUbAddr      = 50432;
  constexpr int32_t A1UbAddr       = 75008;
  constexpr int32_t A2UbAddr       = 107776;
  constexpr int32_t A2HalfUbAddr   = 140544;
  constexpr int32_t GUbAddr        = 156928;
  constexpr int32_t GRUbAddr       = 157440;
  constexpr int32_t G2dUbAddr      = 157952;

  // Workspace sizes (in elements) for A1 and A2 in Global Memory.
  // Each core gets its own workspace slice so cores don't collide.
  constexpr int32_t WsA1Size = ChunkSize * ChunkSize;
  constexpr int32_t WsA2Size = ChunkSize * ChunkSize;

  // Initialize cross-core synchronization base address for this kernel launch.
  set_ffts_base_addr(ffts_addr);
  // cid = this AI core's index (like CUDA blockIdx.x)
  auto cid = get_block_idx();
  // block_num = total number of AI cores running this kernel (like CUDA gridDim.x)
  auto block_num = get_block_num();
  // vid = Vec sub-block ID (0 or 1). Each Vec core has 2 sub-blocks that
  // process the upper (vid=0) and lower (vid=1) C/2 rows of A in parallel.
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;

  // ── UB tile declarations (Vec sub-blocks) ─────────────────────────────
  // Each UbND tile is "assigned" a fixed byte address in UB via TASSIGN.
  // This is how we map logical tile names to physical on-chip memory regions.
  UbND<half, 1, ChunkSize> beta_ub_half;
  TASSIGN(beta_ub_half, BetaHalfUbAddr);
  UbND<half, HalfChunk, ChunkSize> a1_ub_half;
  TASSIGN(a1_ub_half, A1HalfUbAddr);
  UbND<float, 1, ChunkSize> beta_ub;
  TASSIGN(beta_ub, BetaUbAddr);
  UbND<float, 1, ChunkSize> beta_r_ub;
  TASSIGN(beta_r_ub, BetaRUbAddr);
  UbND<float, HalfChunk, ChunkSize> beta_2d_ub;
  TASSIGN(beta_2d_ub, Beta2dUbAddr);
  UbND<uint8_t, 1, 24576> tmp_ub;
  TASSIGN(tmp_ub, TmpUbAddr);
  UbND<float, HalfChunk, ChunkSize> a1_ub;
  TASSIGN(a1_ub, A1UbAddr);
  UbND<float, HalfChunk, ChunkSize> a2_ub;
  TASSIGN(a2_ub, A2UbAddr);
  UbND<half, HalfChunk, ChunkSize> a2_ub_half;
  TASSIGN(a2_ub_half, A2HalfUbAddr);
  UbND<float, 1, ChunkSize> g_ub;
  TASSIGN(g_ub, GUbAddr);
  UbND<float, 1, ChunkSize> g_r_ub;
  TASSIGN(g_r_ub, GRUbAddr);
  UbND<float, HalfChunk, ChunkSize> g_2d_ub;
  TASSIGN(g_2d_ub, G2dUbAddr);

  // ── L1 / L0C tile declarations (Cube engine) ─────────────────────────
  // L1 holds data loaded from GM, waiting to be fed into the Cube.
  // L0A / L0B are the Cube engine's input register files (left/right operands).
  // L0C (TileAcc) is the Cube accumulator — always float32 for precision.
  L1Mat<half, ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 0);
  L1Mat<half, ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 32768);
  L1Mat<half, ChunkSize, ChunkSize> a2_l1;
  TASSIGN(a2_l1, 65536);
  // TileAcc<float, C, D>: Cube accumulator in L0C (float32).
  // GEMM always accumulates in fp32 for numerical precision.
  // When TSTORE writes TileAcc to a half GlobalTensor, automatic fp32→fp16 cast.
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> u_l0;
  TASSIGN(u_l0, 0);
  L1Mat<half, ChunkSize, ChunkSize> a1_l1;
  TASSIGN(a1_l1, 98304);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> w_l0;
  TASSIGN(w_l0, 65536);

  // ── Work distribution ─────────────────────────────────────────────────
  // total_work = num_seqs × chunks_per_seq × NumHeads
  // Each AI core processes work items in a grid-stride loop:
  //   for (work_idx = cid; work_idx < total_work; work_idx += block_num)
  // This is the NPU equivalent of CUDA's grid-stride loop pattern.
  int64_t total_work = 0;
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    total_work = num_seqs * chunks_per_seq * NumHeads;
  }

  // ════════════════════════════════════════════════════════════════════════
  // Vec phase: compute A2 = A*beta_2d and A1 = A*(exp(g)*beta)_2d
  // Two Vec sub-blocks (vid=0,1) handle upper/lower C/2 rows in parallel.
  // ════════════════════════════════════════════════════════════════════════
#if defined(__DAV_C220_VEC__)
  // set_mask_norm / set_vector_mask: configure the Vec engine's SIMD lanes.
  // -1, -1 means "enable all 128 lanes" — full-width SIMD operation.
  set_mask_norm();
  set_vector_mask(-1, -1);

  // ── Fixed-length sequence path ────────────────────────────────────────
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    // first_iter: On the very first iteration, there's no previous cross-core
    // signal to wait for (the "done" flag from Cube hasn't been set yet).
    // So we skip wait_flag_dev() on the first iteration only.
    bool first_iter = true;
    for (int64_t work_idx = static_cast<int64_t>(cid);
         work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      int32_t head_idx = static_cast<int32_t>(work_idx % NumHeads);
      int64_t chunk_head_idx = work_idx / NumHeads;
      int64_t seq_idx = chunk_head_idx / chunks_per_seq;
      int64_t ci = chunk_head_idx % chunks_per_seq;

      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);
      int64_t chunk_token_start = bos + chunk_start;

      // Load beta (pre-transposed [H, total_tokens]) -> UB, zero-pad tail
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = 1; _gs.shape[4] = valid_rows;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
            Beta_handle + static_cast<int64_t>(head_idx) * total_tokens
                        + chunk_token_start, _gs);
        UbND<half, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, valid_rows);
        TASSIGN(_ld, BetaHalfUbAddr);
        TLOAD(_ld, _gm);
        if (valid_rows != ChunkSize) {
          UbND<half, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
          TASSIGN(_pd, BetaHalfUbAddr);
          TFILLPAD_INPLACE(_pd, _ld);
        }
      }

      // Load A [B,S,H,C] — this sub-block's C/2 rows
      int64_t a_gm_offset =
          ((chunk_token_start +
            static_cast<int64_t>(vid) * HalfChunk) *
           NumHeads + head_idx) *
          static_cast<int64_t>(ChunkSize);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * ChunkSize, 1>> _gm(
            A_handle + a_gm_offset, _gs);
        UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfChunk, ChunkSize);
        TASSIGN(_ld, A1HalfUbAddr);
        TLOAD(_ld, _gm);
      }

      // Sync: wait for TLOAD (MTE2 pipe) to finish before Vec engine reads data.
      // set_flag(PIPE_MTE2, PIPE_V) signals that DMA loads are complete;
      // wait_flag(PIPE_MTE2, PIPE_V) blocks the Vec pipe until that signal.
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      // ── A2 = A * beta_2d (numpy pseudocode) ──────────────────────────────
      // # beta is [1, C] — one scalar per token in this chunk
      // beta_f32 = beta.float()                              # TCVT half→float
      // beta_2d = np.tile(beta_f32, (C/2, 1))                # TCOLEXPAND
      // A_f32 = A[my_rows].float()                           # TCVT half→float
      // A2 = A_f32 * beta_2d                                 # TMUL element-wise
      // A2_f16 = A2.half()                                   # TCVT float→half

      // A2 = A * beta_2d: column-broadcast beta then elementwise multiply
      TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TMOV(beta_r_ub, beta_ub);
      pipe_barrier(PIPE_V);
      TCOLEXPAND(beta_2d_ub, beta_r_ub);

      TCVT(a1_ub, a1_ub_half, pto::RoundMode::CAST_NONE);
      TMUL(a2_ub, a1_ub, beta_2d_ub);
      TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);

      // ── Store A2 to GM workspace for Cube ─────────────────────────────────
      // After Vec computes A2, it must be accessible by the Cube core.
      // Since Cube and Vec are on DIFFERENT physical cores, they share data
      // through Global Memory (GM). The workflow is:
      //   1. Vec: TSTORE(workspace, A2)  — write to GM (MTE3 pipe)
      //   2. Vec: ffts_cross_core_sync(flag 2)  — signal Cube "A2 is ready"
      //   3. Cube: wait_flag_dev(2)  — wait for Vec's signal
      //   4. Cube: TLOAD(l1, workspace)  — read A2 from GM into L1

      // Store A2 -> workspace GM, signal Cube (cross-core flag 2)
      if (!first_iter) wait_flag_dev(3);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_a2_handle +
                static_cast<int64_t>(cid) * WsA2Size +
                static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
        UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(HalfChunk, ChunkSize);
        TASSIGN(_st, A2HalfUbAddr);
        TSTORE(_gm, _st);
      }
      // ffts_cross_core_sync encodes: pipe | (dest_core_type << 4) | (flag_id << 8)
      //   1 = current pipe done, 2<<4 = target is Cube core, 2<<8 = flag ID 2
      //   Cube will call wait_flag_dev(2) to receive this signal.
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));

      // Load G (pre-transposed [H, total_tokens]) -> UB, zero-pad tail
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = 1; _gs.shape[4] = valid_rows;
        GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
            G_handle + static_cast<int64_t>(head_idx) * total_tokens
                     + chunk_token_start, _gs);
        UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, valid_rows);
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

      // ── A1 = A * (exp(g) * beta)_2d (numpy pseudocode) ──────────────────
      // # g is [1, C] float — cumulative gate values for this chunk
      // g_exp = np.exp(g)                                    # TEXP
      // g_exp_beta = g_exp * beta_f32                        # TMUL
      // g_exp_beta_2d = np.tile(g_exp_beta, (C/2, 1))        # TCOLEXPAND
      // A1 = A_f32 * g_exp_beta_2d                           # TMUL
      // A1_f16 = A1.half()                                   # TCVT float→half

      // A1 = A * (exp(g) * beta)_2d: gate modulation before column-broadcast
      TEXP(g_ub, g_ub);
      pipe_barrier(PIPE_V);
      TMUL(g_ub, g_ub, beta_ub);
      pipe_barrier(PIPE_V);
      TMOV(g_r_ub, g_ub);
      pipe_barrier(PIPE_V);
      TCOLEXPAND(g_2d_ub, g_r_ub);
      TMUL(a1_ub, a1_ub, g_2d_ub);
      TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);

      // Store A1 -> workspace GM, signal Cube (cross-core flag 1)
      if (!first_iter) wait_flag_dev(4);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_a1_handle +
                static_cast<int64_t>(cid) * WsA1Size +
                static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
        UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(HalfChunk, ChunkSize);
        TASSIGN(_st, A1HalfUbAddr);
        TSTORE(_gm, _st);
      }
      // Signal Cube: flag ID 1 means "A1 is ready in workspace GM"
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
      first_iter = false;
    }
  }
  // ── Variable-length sequence path (Vec) ───────────────────────────────
  // When cu_seqlens is provided, sequences have different lengths.
  // cu_seqlens = [0, len0, len0+len1, ...] — cumulative sequence boundaries.
  // We iterate over (sequence, chunk, head) and use round-robin assignment
  // to distribute work across AI cores.
  else {
    int64_t gi = 0;
    bool first_iter_v = true;
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

            // Load beta -> UB
            {
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = 1; _gs.shape[4] = valid_rows;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
                  Beta_handle + static_cast<int64_t>(head_idx) * total_tokens
                              + chunk_token_start, _gs);
              UbND<half, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, valid_rows);
              TASSIGN(_ld, BetaHalfUbAddr);
              TLOAD(_ld, _gm);
              if (valid_rows != ChunkSize) {
                UbND<half, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
                TASSIGN(_pd, BetaHalfUbAddr);
                TFILLPAD_INPLACE(_pd, _ld);
              }
            }

            // Load A -> UB
            int64_t a_gm_offset =
                ((chunk_token_start +
                  static_cast<int64_t>(vid) * HalfChunk) *
                 NumHeads + head_idx) *
                static_cast<int64_t>(ChunkSize);
            {
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * ChunkSize, 1>> _gm(
                  A_handle + a_gm_offset, _gs);
              UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfChunk, ChunkSize);
              TASSIGN(_ld, A1HalfUbAddr);
              TLOAD(_ld, _gm);
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // A2 = A * beta_2d
            TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);
            TMOV(beta_r_ub, beta_ub);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(beta_2d_ub, beta_r_ub);

            TCVT(a1_ub, a1_ub_half, pto::RoundMode::CAST_NONE);
            TMUL(a2_ub, a1_ub, beta_2d_ub);
            TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);

            // Store A2 -> workspace, signal Cube (flag 2)
            if (!first_iter_v) wait_flag_dev(3);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
                  workspace_a2_handle +
                      static_cast<int64_t>(cid) * WsA2Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
              UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(HalfChunk, ChunkSize);
              TASSIGN(_st, A2HalfUbAddr);
              TSTORE(_gm, _st);
            }
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));

            // Load G -> UB
            {
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = 1; _gs.shape[4] = valid_rows;
              GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
                  G_handle + static_cast<int64_t>(head_idx) * total_tokens
                           + chunk_token_start, _gs);
              UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, valid_rows);
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

            // A1 = A * (exp(g) * beta)_2d
            TEXP(g_ub, g_ub);
            pipe_barrier(PIPE_V);
            TMUL(g_ub, g_ub, beta_ub);
            pipe_barrier(PIPE_V);
            TMOV(g_r_ub, g_ub);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(g_2d_ub, g_r_ub);
            TMUL(a1_ub, a1_ub, g_2d_ub);
            TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);

            // Store A1 -> workspace, signal Cube (flag 1)
            if (!first_iter_v) wait_flag_dev(4);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
                  workspace_a1_handle +
                      static_cast<int64_t>(cid) * WsA1Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
              UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(HalfChunk, ChunkSize);
              TASSIGN(_st, A1HalfUbAddr);
              TSTORE(_gm, _st);
            }
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
            first_iter_v = false;
          }
          gi++;
        }
      }
    }
  }
#endif

  // ════════════════════════════════════════════════════════════════════════
  // Cube phase: GEMM  U = A2 @ V  and  W = A1 @ K
  // Waits for Vec cross-core flags before loading workspace matrices.
  // Single L0 split (K=ChunkSize=128 fits in one 64KB L0 block).
  // ════════════════════════════════════════════════════════════════════════
#if defined(__DAV_C220_CUBE__)
  // ── Fixed-length sequence path (Cube) ─────────────────────────────────
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    for (int64_t work_idx = static_cast<int64_t>(cid);
         work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      int32_t head_idx = static_cast<int32_t>(work_idx % NumHeads);
      int64_t chunk_head_idx = work_idx / NumHeads;
      int64_t seq_idx = chunk_head_idx / chunks_per_seq;
      int64_t ci = chunk_head_idx % chunks_per_seq;

      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);
      int64_t chunk_token_start = bos + chunk_start;

      int64_t kv_offset =
          (chunk_token_start * NumHeads + head_idx) *
          static_cast<int64_t>(HiddenSize);

      // Load K [B,S,N,D] -> L1, zero-pad if tail chunk
      {
        L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
        TASSIGN(_l1, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
            K_handle + kv_offset, _gs);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }
      // Load V [B,S,N,D] -> L1
      {
        L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
        TASSIGN(_l1, 32768);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
            V_handle + kv_offset, _gs);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }

      // Wait for Vec's A2 workspace (cross-core flag 2) -> load A2 -> L1
      wait_flag_dev(2);
      {
        L1Mat<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(ChunkSize, ChunkSize);
        TASSIGN(_l1, 65536);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_a2_handle +
                static_cast<int64_t>(cid) * WsA2Size, _gs);
        TLOAD(_l1, _gm);
      }

      // ── Cube GEMM: U = A2 @ V ────────────────────────────────────────────
      // numpy equivalent: U = A2.half() @ V.half()  # result accumulated in float32
      //
      // NPU Cube pipeline:
      //   1. A2 is already in L1 (a2_l1). V is in L1 (v_l1).
      //   2. TEXTRACT copies them to L0A and L0B (the Cube's register files).
      //   3. TMATMUL computes C×D = (C×C) @ (C×D), accumulating in float32 L0C.
      //   4. TSTORE writes L0C → GM (with implicit float32→float16 conversion).
      //
      // WAR (Write-After-Read) sync before TEXTRACT:
      //   MTE2→MTE1: ensure L1 data from TLOAD is ready before TEXTRACT reads it
      //   M→MTE1: ensure previous TMATMUL has read L0A/L0B before overwriting

      // GEMM: U = A2 @ V  (L1 -> L0A/L0B -> L0C)
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
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
        TEXTRACT(_l0a, a2_l1, 0, 0);
        TEXTRACT(_l0b, v_l1, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we);
        wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(u_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we);
        wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we);
        wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // Store U from L0C -> GM (fp32->fp16 cast handled by TSTORE)
      {
        TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(valid_rows, HiddenSize);
        TASSIGN(_l0, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
            U_handle + kv_offset, _gs);
        TSTORE(_gm, _l0);
      }
      // Signal Vec: flag ID 3 tells Vec "Cube is done reading A2 workspace,
      // safe to overwrite it next iteration". Vec waits on this via wait_flag_dev(3).
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (3 << 8));

      // Wait for Vec's A1 workspace (cross-core flag 1) -> load A1 -> L1
      wait_flag_dev(1);
      {
        L1Mat<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(ChunkSize, ChunkSize);
        TASSIGN(_l1, 98304);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_a1_handle +
                static_cast<int64_t>(cid) * WsA1Size, _gs);
        TLOAD(_l1, _gm);
      }

      // ── Cube GEMM: W = A1 @ K ────────────────────────────────────────────
      // Same pipeline as U = A2 @ V above, but with A1 as left operand
      // and K as right operand. Result W is also accumulated in fp32 L0C.

      // GEMM: W = A1 @ K  (L1 -> L0A/L0B -> L0C)
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
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
        TEXTRACT(_l0a, a1_l1, 0, 0);
        TEXTRACT(_l0b, k_l1, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we);
        wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(w_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we);
        wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we);
        wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // Store W from L0C -> GM
      {
        TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(valid_rows, HiddenSize);
        TASSIGN(_l0, 65536);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
            W_handle + kv_offset, _gs);
        TSTORE(_gm, _l0);
      }
      // Signal Vec: flag ID 4 tells Vec "Cube is done reading A1 workspace,
      // safe to overwrite it next iteration". Vec waits on this via wait_flag_dev(4).
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (4 << 8));
    }
  }
  // ── Variable-length sequence path (Cube) ──────────────────────────────
  // Same logic as fixed-length but iterates over cu_seqlens boundaries.
  // Round-robin work assignment: gi % block_num == cid.
  else {
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

            int64_t kv_offset =
                (chunk_token_start * NumHeads + head_idx) *
                static_cast<int64_t>(HiddenSize);

            // Load K -> L1
            {
              L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
              TASSIGN(_l1, 0);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
                  K_handle + kv_offset, _gs);
              TLOAD(_l1, _gm);
              if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
            }
            // Load V -> L1
            {
              L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
              TASSIGN(_l1, 32768);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
                  V_handle + kv_offset, _gs);
              TLOAD(_l1, _gm);
              if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
            }

            // Wait for A2, load -> L1
            wait_flag_dev(2);
            {
              L1Mat<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(ChunkSize, ChunkSize);
              TASSIGN(_l1, 65536);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
                  workspace_a2_handle +
                      static_cast<int64_t>(cid) * WsA2Size, _gs);
              TLOAD(_l1, _gm);
            }

            // GEMM: U = A2 @ V
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
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
              TEXTRACT(_l0a, a2_l1, 0, 0);
              TEXTRACT(_l0b, v_l1, 0, 0);
              set_flag(PIPE_MTE1, PIPE_M, _we);
              wait_flag(PIPE_MTE1, PIPE_M, _we);
              TMATMUL(u_l0, _l0a, _l0b);
              set_flag(PIPE_MTE1, PIPE_MTE2, _we);
              wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
              set_flag(PIPE_M, PIPE_FIX, _we);
              wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            // Store U
            {
              TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(valid_rows, HiddenSize);
              TASSIGN(_l0, 0);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
                  U_handle + kv_offset, _gs);
              TSTORE(_gm, _l0);
            }
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (3 << 8));

            // Wait for A1, load -> L1
            wait_flag_dev(1);
            {
              L1Mat<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(ChunkSize, ChunkSize);
              TASSIGN(_l1, 98304);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
                  workspace_a1_handle +
                      static_cast<int64_t>(cid) * WsA1Size, _gs);
              TLOAD(_l1, _gm);
            }

            // GEMM: W = A1 @ K
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
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
              TEXTRACT(_l0a, a1_l1, 0, 0);
              TEXTRACT(_l0b, k_l1, 0, 0);
              set_flag(PIPE_MTE1, PIPE_M, _we);
              wait_flag(PIPE_MTE1, PIPE_M, _we);
              TMATMUL(w_l0, _l0a, _l0b);
              set_flag(PIPE_MTE1, PIPE_MTE2, _we);
              wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
              set_flag(PIPE_M, PIPE_FIX, _we);
              wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            // Store W
            {
              TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(valid_rows, HiddenSize);
              TASSIGN(_l0, 65536);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * HiddenSize, 1>> _gm(
                  W_handle + kv_offset, _gs);
              TSTORE(_gm, _l0);
            }
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (4 << 8));
          }
          gi++;
        }
      }
    }
  }
#endif
}

// ── Device kernel entry point ─────────────────────────────────────────
// extern "C" __global__ AICORE: NPU kernel function, callable from the host.
// All pointer args are uint8_t* (type-erased) and reinterpret_cast'd to their
// actual types inside. This is the standard pattern for NPU kernel launch
// interfaces — similar to how CUDA kernels receive void* from the launcher.
extern "C" __global__ AICORE void launch_wy_fast(
    __gm__ uint8_t *K_handle, __gm__ uint8_t *V_handle,
    __gm__ uint8_t *Beta_handle, __gm__ uint8_t *G_handle,
    __gm__ uint8_t *A_handle,
    __gm__ uint8_t *workspace_a1_handle, __gm__ uint8_t *workspace_a2_handle,
    __gm__ uint8_t *W_handle, __gm__ uint8_t *U_handle,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens,
    uint64_t ffts_addr)
{
  wy_fast_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(V_handle),
      reinterpret_cast<__gm__ half *>(Beta_handle),
      reinterpret_cast<__gm__ float *>(G_handle),
      reinterpret_cast<__gm__ half *>(A_handle),
      reinterpret_cast<__gm__ half *>(workspace_a1_handle),
      reinterpret_cast<__gm__ half *>(workspace_a2_handle),
      reinterpret_cast<__gm__ half *>(W_handle),
      reinterpret_cast<__gm__ half *>(U_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, ffts_addr);
}

// ── Host launcher (called from Python ctypes) ─────────────────────────
// call_kernel: launches the NPU kernel on `block_dim` AI cores.
// rtGetC2cCtrlAddr: retrieves the FFTS cross-core control address that
//   enables Cube↔Vec synchronization at runtime.
// <<<block_dim, nullptr, stream>>>: NPU kernel launch syntax, analogous
//   to CUDA's <<<grid, block, stream>>> but for AI cores.
extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *k, uint8_t *v, uint8_t *beta, uint8_t *g_sum, uint8_t *A,
    uint8_t *workspace_a1, uint8_t *workspace_a2,
    uint8_t *w, uint8_t *u,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_wy_fast<<<block_dim, nullptr, stream>>>(
      k, v, beta, g_sum, A,
      workspace_a1, workspace_a2,
      w, u,
      cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
