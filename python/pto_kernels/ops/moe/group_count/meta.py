from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='moe',
    name='group_count',
    builders=['group_count'],
    configs=[{'case': 'group_count',
  'configs': [{'num_groups': 1, 'num_topk': 1},
              {'num_groups': 4, 'num_topk': 2},
              {'num_groups': 9, 'num_topk': 6},
              {'num_groups': 9, 'num_topk': 7},
              {'num_groups': 32, 'num_topk': 8},
              {'num_groups': 32, 'num_topk': 9}]}],
    source='tilekernels/moe/group_count',
    notes='Correctness-first scalar PTO scan avoids TileLang atomics by using one block and scanning per expert. This is real validation code, not the final high-throughput algorithm.',
)
