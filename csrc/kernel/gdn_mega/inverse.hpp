#pragma once

// Four independent 16x16 unit-triangular solves share one vector instruction
// stream. The gather repeats each block's elimination coefficient over that
// block's columns. This is forward substitution, never an inverse power series.
#if defined(__DAV_VEC__)
template <typename D, typename S>
AICORE inline void mega_copy_block16(D& dst, S& src) {
#if __CCE_AICORE__ == 220
  set_vector_mask(0, 65535);
  vadds(dst.data(), src.data(), 0.0f, 16, 1, 1, D::Cols / 8, S::Cols / 8);
  PipeBarrierVec();
  set_vector_mask(-1, -1);
#else
  TADDS(dst, src, 0.0f);
#endif
}

template <typename Stripe, typename Identity>
AICORE inline void mega_diagonal_inverses(Stripe& a, Identity& identity,
                                          int stripe) {
  constexpr int B = 16, R = 64, Base = 128 * 1024,
                Bytes = B * R * sizeof(float);
  using Batch = Tile<TileType::Vec, float, B, R>;
  Batch inverse, coefficients;
  TASSIGN(inverse, Base);
  TASSIGN(coefficients, Base + 3 * Bytes);
  for (int b = 0; b < R / B; ++b) {
    using Source = Tile<TileType::Vec, float, B, 128, BLayout::RowMajor, B, B>;
    using Dest = Tile<TileType::Vec, float, B, R, BLayout::RowMajor, B, B>;
    Source unit;
    Dest idst;
    TASSIGN(unit, (uint64_t)identity.data() +
                      (b * B * 128 + stripe + b * B) * sizeof(float));

    TASSIGN(idst, Base + b * B * sizeof(float));
    mega_copy_block16(idst, unit);
    PipeBarrierVec();
  }
  Tile<TileType::Vec, uint32_t, 1, B * R> indices;
  TASSIGN(indices, 176 * 1024);
  for (int i = 0; i + 1 < B; ++i) {
    Tile<TileType::Vec, float, 1, R> solved, source;
    TASSIGN(solved, Base + i * R * sizeof(float));
    TASSIGN(source, (uint64_t)a.data() + i * sizeof(float));
#if __CCE_AICORE__ == 220
    set_mask_count();
    set_vector_mask(0, B * R);
    vgather(reinterpret_cast<__ubuf__ uint32_t*>(coefficients.data()),
            indices.data(), reinterpret_cast<uintptr_t>(source.data()), 8, 1);
    PipeBarrierVec();
    set_mask_norm();
    set_vector_mask(-1, -1);
    // The solved row is reused by every destination row directly by the
    // vector instruction. No materialized 16-row broadcast is necessary.
    vmla(inverse.data(), coefficients.data(), solved.data(), B, 1, 1, 1, R / 8,
         R / 8, 0);
#else
    // A5 uses element indices and register-based vector instructions.
    Tile<TileType::Vec, uint32_t, B, R> gather_indices;
    TASSIGN(gather_indices, 176 * 1024);
    Batch broadcast;
    TASSIGN(broadcast, Base + Bytes);
    TGATHER(coefficients, source, gather_indices, broadcast);
    TCOLEXPAND(broadcast, solved);
    TMULADDDST(inverse, coefficients, broadcast);
#endif
    PipeBarrierVec();
  }
  for (int b = 0; b < R / B; ++b) {
    using Source = Tile<TileType::Vec, float, B, R, BLayout::RowMajor, B, B>;
    using Dest = Tile<TileType::Vec, float, B, 128, BLayout::RowMajor, B, B>;
    Source src;
    Dest dst;
    TASSIGN(src, Base + b * B * sizeof(float));
    TASSIGN(dst, (uint64_t)a.data() +
                     (b * B * 128 + stripe + b * B) * sizeof(float));
    mega_copy_block16(dst, src);
    PipeBarrierVec();
  }
}
#endif
