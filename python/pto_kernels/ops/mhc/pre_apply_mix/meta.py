from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='mhc',
    name='pre_apply_mix',
    builders=['pre_apply_mix_fwd', 'pre_apply_mix_bwd'],
    configs=[{'case': 'pre_apply_mix_fwd', 'configs': [{'mhc_mult': 4}]},
 {'case': 'pre_apply_mix_bwd', 'configs': [{'mhc_mult': 4}]}],
    source='tilekernels/mhc/pre_apply_mix',
    notes='bf16 MHC pre-apply forward computes the hidden-axis weighted sum over mhc lanes with f32 accumulation. Correctness-first bf16 backward updates x_grad and writes per-token f32 mix_grad via hidden-axis dot products.',
)
