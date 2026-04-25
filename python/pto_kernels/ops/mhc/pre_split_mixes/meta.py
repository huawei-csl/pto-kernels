from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='mhc',
    name='pre_split_mixes',
    builders=['pre_split_mixes_fwd', 'pre_split_mixes_bwd'],
    configs=[{'case': 'pre_split_mixes_fwd',
  'configs': [{'mhc_mult': 4, 'mhc_post_mult_value': 2.0, 'mhc_pre_eps': 0.01}]},
 {'case': 'pre_split_mixes_bwd', 'configs': [{'mhc_mult': 4, 'mhc_post_mult_value': 2.0}]}],
    source='tilekernels/mhc/pre_split_mixes',
    notes='Tile f32 forward splits 24-wide mix rows into pre/post/comb outputs using PTO tile sigmoid and row arithmetic. Correctness-first single-block backward writes input gradients plus one partial 3-value scale gradient and 24-value base-gradient row.',
)
