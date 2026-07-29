// copy_ref_256_a5 — the copy-floor reference for fast_hadamard_256_a5.
//
// A plain GM -> UB -> GM round trip over the *same* tiling as the transform it
// is compared against, and nothing else: no vector-execute work at all, just a
// ping/pong DMA pipeline. That makes it the DMA ceiling for the shape, which is
// the honest yardstick for a memory-bound transform — a kernel running at this
// bandwidth has nothing left to win.
//
// It is deliberately a separate translation unit from the transform so the
// transform stays standalone; the only thing the two must agree on is the
// tiling (ROWS_PER_TILE x COPY_N), which the benchmark passes to both.
#include <pto/pto-inst.hpp>
using namespace pto;

#ifndef COPY_N
#define COPY_N 256  // elements per row; must match the transform's HAD_N
#endif
#ifndef ROWS_PER_TILE
#define ROWS_PER_TILE 64
#endif

#ifdef __CCE_AICORE__
constexpr unsigned FLAT = ROWS_PER_TILE * COPY_N;  // f16 elems/tile
constexpr unsigned X_BYTES = FLAT * sizeof(half);
constexpr unsigned aln(unsigned b) { return (b + 511u) & ~511u; }
#define XOFF(i) ((unsigned)(i) * ((X_BYTES + 511u) & ~511u))

// Fixed at 2 by design: a ping/pong is enough to keep MTE2 and MTE3 both busy,
// which is all this kernel needs to reach the DMA ceiling. Deeper buffering
// would change the reference the transform is measured against.
#define COPY_NBUF 2
#ifndef UB_USABLE_BYTES
#define UB_USABLE_BYTES (192u * 1024u)
#endif
// Without this the two buffers silently overrun UB at large ROWS_PER_TILE
// (e.g. ROWS_PER_TILE=256 needs 2 x 128 KB = 256 KB), which corrupts the
// measured floor instead of failing.
static_assert(COPY_NBUF * aln(X_BYTES) <= UB_USABLE_BYTES,
              "UB overflow: ROWS_PER_TILE too large for the ping/pong copy");
#endif  // __CCE_AICORE__

__global__ AICORE void copy256(__gm__ void *x_gm, uint32_t batch) {
#ifdef __DAV_VEC__
  using Sh = pto::Shape<1, 1, 1, 1, FLAT>;
  using St = pto::Stride<1, 1, 1, FLAT, 1>;
  using T = Tile<TileType::Vec, half, 1, FLAT, BLayout::RowMajor, 1, FLAT>;
  const event_t ev[COPY_NBUF] = {EVENT_ID0, EVENT_ID1};
  const uint32_t cid = get_block_idx(), nc = get_block_num();
  const uint32_t tiles = batch / ROWS_PER_TILE;
  for (int i = 0; i < COPY_NBUF; ++i) set_flag(PIPE_MTE3, PIPE_MTE2, ev[i]);
  uint32_t it = 0;
  for (uint32_t tb = cid; tb < tiles; tb += nc, ++it) {
    const int pp = it % COPY_NBUF;
    const uint64_t off = (uint64_t)tb * FLAT;
    wait_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);
    T xt;
    TASSIGN(xt, XOFF(pp));
    GlobalTensor<half, Sh, St> g((__gm__ half *)x_gm + off, Sh());
    TLOAD(xt, g);
    set_flag(PIPE_MTE2, PIPE_MTE3, ev[pp]);
    wait_flag(PIPE_MTE2, PIPE_MTE3, ev[pp]);
    TSTORE(g, xt);
    set_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);
  }
  for (int i = 0; i < COPY_NBUF; ++i) wait_flag(PIPE_MTE3, PIPE_MTE2, ev[i]);
#else
  (void)x_gm;
  (void)batch;
#endif
}

extern "C" void call_copy256(uint32_t bd, void *s, uint8_t *x, uint32_t b) {
  copy256<<<bd, nullptr, s>>>(x, b);
}
