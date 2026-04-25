from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='mhc',
    name='head_compute_mix',
    builders=['head_compute_mix_fwd', 'head_compute_mix_bwd'],
    configs=[{'case': 'head_compute_mix_fwd', 'configs': [{'eps': 1e-06, 'mhc_mult': 4}]},
 {'case': 'head_compute_mix_bwd', 'configs': [{'mhc_mult': 4}]}],
    source='tilekernels/mhc/head_compute_mix',
    notes='Tile f32 sigmoid head mix forward over num_tokens={0,4001}; uses PTO texp/trecip instead of scalar math.exp. Correctness-first single-block backward computes input gradient plus one partial scale gradient and one partial base-gradient row.',
)
