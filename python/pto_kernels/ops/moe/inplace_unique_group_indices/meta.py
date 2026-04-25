from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='moe',
    name='inplace_unique_group_indices',
    builders=['inplace_unique_group_indices'],
    configs=[{'case': 'inplace_unique_group_indices',
  'configs': [{'num_groups': 8, 'num_topk': 1},
              {'num_groups': 8, 'num_topk': 2},
              {'num_groups': 8, 'num_topk': 6},
              {'num_groups': 16, 'num_topk': 7},
              {'num_groups': 16, 'num_topk': 8},
              {'num_groups': 72, 'num_topk': 9}]}],
    source='tilekernels/moe/inplace_unique_group_indices',
    notes="Correctness-first per-token scalar scan removes later duplicate non-negative group ids by writing TileKernels' -1 sentinel.",
)
