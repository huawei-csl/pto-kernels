from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='moe',
    name='mask_indices_by_tp',
    builders=['mask_indices_by_tp'],
    configs=[{'case': 'mask_indices_by_tp',
  'configs': [{'num_ep_ranks': 1, 'num_experts': 1, 'num_topk': 1, 'num_tp_ranks': 2},
              {'num_ep_ranks': 8, 'num_experts': 9, 'num_topk': 2, 'num_tp_ranks': 2},
              {'num_ep_ranks': 8, 'num_experts': 9, 'num_topk': 6, 'num_tp_ranks': 4},
              {'num_ep_ranks': 64, 'num_experts': 4, 'num_topk': 7, 'num_tp_ranks': 4},
              {'num_ep_ranks': 8, 'num_experts': 32, 'num_topk': 8, 'num_tp_ranks': 8},
              {'num_ep_ranks': 8, 'num_experts': 32, 'num_topk': 9, 'num_tp_ranks': 8}]}],
    source='tilekernels/moe/mask_indices_by_tp',
    notes='Scalar PTO remaps int64 expert indices for the current TP rank and preserves TileKernels -1 sentinel behavior.',
)
