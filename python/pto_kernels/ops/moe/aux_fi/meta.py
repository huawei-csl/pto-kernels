from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='moe',
    name='aux_fi',
    builders=['aux_fi'],
    configs=[{'case': 'aux_fi',
  'configs': [{'num_experts': 1, 'num_topk': 1},
              {'num_experts': 4, 'num_topk': 2},
              {'num_experts': 9, 'num_topk': 6},
              {'num_experts': 9, 'num_topk': 7},
              {'num_experts': 32, 'num_topk': 8},
              {'num_experts': 32, 'num_topk': 9}]}],
    source='tilekernels/moe/aux_fi',
    notes='Correctness-first scalar PTO scan computes per-expert counts and scales by num_experts / (num_tokens * num_aux_topk).',
)
