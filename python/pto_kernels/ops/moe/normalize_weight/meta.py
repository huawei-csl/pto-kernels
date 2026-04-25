from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='moe',
    name='normalize_weight',
    builders=['normalize_weight'],
    configs=[{'case': 'normalize_weight',
  'configs': [{'num_topk': 1},
              {'num_topk': 2},
              {'num_topk': 6},
              {'num_topk': 7},
              {'num_topk': 8},
              {'num_topk': 9}]}],
    source='tilekernels/moe/normalize_weight',
    notes='PTO-DSL uses row_sum, tadds with TileKernels 1e-20 denominator sentinel, row_expand, and row_expand_div. Generated validation assets include host runner and NumPy golden/compare for a deterministic runtime shape.',
)
