#include <pto/pto-inst.hpp>

using namespace pto;

#if defined(__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

// L1 memory layout (512KB total):
//   L1_A_PING: 0x00000 .. 0x40000  (8 * 32KB = 256KB)  slot 0 doubles as B
//   staging L1_A_PONG: 0x40000 .. 0x80000  (8 * 32KB = 256KB) Total: 512KB,
//   fully utilised
namespace detail {

constexpr uint32_t DEPTH = 8;

// L0A: 64KB = 2 x 32KB (ping/pong for matmul)
// L0C: 128KB = 2 x 64KB (ping/pong for matmul)

template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void SetFlag(uint32_t id) {
  set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}
template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void WaitFlag(uint32_t id) {
  wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

// Internal helper: load up to DEPTH A-tiles from GM into one L1 half-buffer.
template <uint32_t TILE_ELEMS, typename GlobalIn, typename TileL1>
AICORE uint32_t loadBatch(__gm__ half* a, TileL1* a_l1_slots, uint32_t buf,
                          uint32_t total_batches, uint32_t num_cores,
                          uint32_t* next_load, uint32_t* buf_batch_base) {
  buf_batch_base[buf] = *next_load;
  uint32_t count = 0;
  for (uint32_t i = 0; i < DEPTH && *next_load < total_batches;
       ++i, *next_load += num_cores) {
    GlobalIn a_gm(a + (*next_load) * TILE_ELEMS);
    TLOAD(a_l1_slots[i], a_gm);
    count++;
  }
  SetFlag<PIPE_MTE2, PIPE_MTE1>(buf);
  return count;
}

// Internal kernel body with L1 and L0 double-buffering.
template <uint32_t M_TILE, uint32_t N, uint32_t K>
AICORE void runBlockRotate(__gm__ half* a, __gm__ half* b, __gm__ half* c,
                           uint32_t m) {
  constexpr uint32_t TILE_ELEMS = M_TILE * K;
  constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(half);

  using TensorShape = TileShape2D<half, M_TILE, K, Layout::ND>;
  using TensorStrides = BaseShape2D<half, M_TILE, K, Layout::ND>;
  using GlobalIn = GlobalTensor<half, TensorShape, TensorStrides, Layout::ND>;
  using TileL1 = Tile<TileType::Mat, half, M_TILE, K, BLayout::ColMajor, M_TILE,
                      K, SLayout::RowMajor, 512>;

  const uint32_t total_batches = m / M_TILE;
  const uint32_t core_id = get_block_idx();
  const uint32_t num_cores = block_num;

  // Interleaved: core i owns tiles i, i+num_cores, i+2*num_cores, ...
  if (core_id >= total_batches) return;

  const uint32_t base_tiles = total_batches / num_cores;
  const uint32_t remainder = total_batches % num_cores;
  const uint32_t batches_per_core = base_tiles + (core_id < remainder ? 1 : 0);

  using TensorShapeOut = TileShape2D<half, M_TILE, N, Layout::ND>;
  using TensorStridesOut = BaseShape2D<half, M_TILE, N, Layout::ND>;
  using GlobalOut =
      GlobalTensor<half, TensorShapeOut, TensorStridesOut, Layout::ND>;

  using TileL0A = TileLeft<half, M_TILE, K>;
  using TileL0B = TileRight<half, K, N>;
  using TileL0C = TileAcc<float, M_TILE, N>;

  // --- L1 tiles ---
  TileL1 b_l1;
  TileL1 a_l1_slots[2][DEPTH];
  TASSIGN(b_l1, 0x0);
  for (uint32_t buf = 0; buf < 2; ++buf) {
    for (uint32_t i = 0; i < DEPTH; ++i) {
      TASSIGN(a_l1_slots[buf][i], (buf * DEPTH + i) * TILE_BYTES);
    }
  }

  // --- L0A tiles: ping/pong ---
  TileL0A a_l0[2];
  TASSIGN(a_l0[0], 0x0);
  TASSIGN(a_l0[1], TILE_BYTES);

  // --- L0B tile ---
  TileL0B b_l0;
  TASSIGN(b_l0, 0x0);

  // --- L0C tiles: ping/pong ---
  TileL0C c_l0[2];
  TASSIGN(c_l0[0], 0x0);
  TASSIGN(c_l0[1], M_TILE * N * sizeof(float));

  // === Load B once: GM -> L1 -> L0B ===
  GlobalIn b_global(b);
  TLOAD(b_l1, b_global);
  SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
  WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
  TEXTRACT(b_l0, b_l1);
  SetFlag<PIPE_MTE1, PIPE_M>(0);
  WaitFlag<PIPE_MTE1, PIPE_M>(0);

  // NOTE: Without this it doesn't work. IDK why
  SetFlag<PIPE_MTE1, PIPE_MTE2>(0);
  WaitFlag<PIPE_MTE1, PIPE_MTE2>(0);

  // Interleaved: each core starts at core_id, strides by num_cores
  uint32_t next_load = core_id;
  uint32_t next_proc = core_id;
  uint32_t load_buf = 0;

  const uint32_t end_proc =
      core_id + batches_per_core * num_cores;  // virtual end for this core

  uint32_t buf_tile_count[2] = {0, 0};
  uint32_t buf_batch_base[2] = {0, 0};

  // --- Prefetch first batch into ping ---
  buf_tile_count[0] = loadBatch<TILE_ELEMS, GlobalIn, TileL1>(
      a, a_l1_slots[0], 0, total_batches, num_cores, &next_load,
      buf_batch_base);
  load_buf = 1;

  // --- Main double-buffer loop ---
  while (next_proc < end_proc) {
    const uint32_t proc_buf = load_buf ^ 1;
    const uint32_t tile_count = buf_tile_count[proc_buf];
    const uint32_t batch_base =
        buf_batch_base[proc_buf];  // first tile index in GM for this batch

    // Fire DMA for next batch while we process current
    if (next_load < total_batches) {
      buf_tile_count[load_buf] = loadBatch<TILE_ELEMS, GlobalIn, TileL1>(
          a, a_l1_slots[load_buf], load_buf, total_batches, num_cores,
          &next_load, buf_batch_base);
      load_buf ^= 1;
    }

    WaitFlag<PIPE_MTE2, PIPE_MTE1>(proc_buf);

    // --- Pipelined tile processing with L0A/L0C ping-pong ---
    // Timeline for a given parity buffer (curr = r&1):
    //   iter r-2: TMATMUL -> TSTORE uses buffer curr
    //   iter r  : wait until r-2 consumed/stored, then reuse same buffer curr
    //   iter r+2: may reuse again if it exists (hence conditional SetFlag)
    //
    // Handshakes by parity ID `curr`:
    //   M -> MTE1 : "a_l0[curr] consumed, safe to refill for r+2"
    //   M -> FIX  : "c_l0[curr] produced, safe to store"
    //   FIX -> M  : "c_l0[curr] store done, safe to overwrite for r+2"
    for (uint32_t r = 0; r < tile_count; ++r) {
      const uint32_t curr = r & 1;

      // Buffers are ping/pong by parity: curr=0/1 is reused every 2 iterations.
      // Before MTE1 writes a_l0[curr] for iteration r, wait until M has
      // consumed that same a_l0[curr] from iteration r-2.
      if (r >= 2) WaitFlag<PIPE_M, PIPE_MTE1>(curr);

      // L1 -> L0A (producer: MTE1), then notify M pipe that a_l0[curr] is
      // ready.
      TEXTRACT(a_l0[curr], a_l1_slots[proc_buf][r]);
      SetFlag<PIPE_MTE1, PIPE_M>(curr);
      WaitFlag<PIPE_MTE1, PIPE_M>(curr);

      // c_l0[curr] is also reused every 2 iterations.
      // Before starting TMATMUL into c_l0[curr], wait until FIX has finished
      // storing c_l0[curr] produced by iteration r-2.
      if (r >= 2) WaitFlag<PIPE_FIX, PIPE_M>(curr);

      TMATMUL(c_l0[curr], a_l0[curr], b_l0);

      // If iteration r+2 will exist, permit MTE1 to refill a_l0[curr] after
      // this matmul has consumed current a_l0[curr].
      if (r + 2 < tile_count) SetFlag<PIPE_M, PIPE_MTE1>(curr);

      // Matmul result in c_l0[curr] is ready for FIX pipe to store.
      SetFlag<PIPE_M, PIPE_FIX>(curr);
      WaitFlag<PIPE_M, PIPE_FIX>(curr);

      // Output address is the original GM tile index for this tile
      // batch_base + r * num_cores gives the interleaved GM tile index
      GlobalOut c_gm(c + (batch_base + r * num_cores) * TILE_ELEMS);
      TSTORE(c_gm, c_l0[curr]);

      // If iteration r+2 will reuse c_l0[curr], allow M to proceed only after
      // FIX completes this store and the accumulator buffer is safe to
      // overwrite.
      if (r + 2 < tile_count) SetFlag<PIPE_FIX, PIPE_M>(curr);
    }

    pipe_barrier(PIPE_ALL);
    next_proc += tile_count * num_cores;
  }
}

}  // namespace detail

#endif

/*
 * @brief External C ABI kernel entrypoint pinned to 128x128x128 tiles.
 * @param a Pointer to A in global memory.
 * @param b Pointer to B in global memory.
 * @param c Pointer to C in global memory.
 * @param m Total rows of A (and C) processed.
 */
extern "C" __global__ AICORE void block_rotate_fp16(__gm__ void* a,
                                                    __gm__ void* b,
                                                    __gm__ void* c,
                                                    uint32_t m) {
#if defined(__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))
  detail::runBlockRotate<128, 128, 128>((__gm__ half*)a, (__gm__ half*)b,
                                        (__gm__ half*)c, m);
#endif
}

/*
 * @brief External host launcher for the 128x128x128 kernel variant.
 * @param blockDim Number of kernel blocks (cores) to launch.
 * @param stream Execution stream handle.
 * @param a Pointer to A in global memory.
 * @param b Pointer to B in global memory.
 * @param c Pointer to C in global memory.
 * @param m Total rows of A (and C) processed.
 */
extern "C" void call_kernel(uint32_t blockDim, void* stream, uint8_t* a,
                            uint8_t* b, uint8_t* c, uint32_t m) {
  block_rotate_fp16<<<blockDim, nullptr, stream>>>(a, b, c, m);
}
