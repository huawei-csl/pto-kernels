from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='mhc',
    name='post',
    builders=['post_fwd', 'post_bwd'],
    configs=[{'case': 'post_fwd', 'configs': [{'mhc_mult': 4}]},
 {'case': 'post_bwd', 'configs': [{'mhc_mult': 4}]}],
    source='tilekernels/mhc/post',
    notes='bf16 MHC post forward combines post-layer x and comb-residual branches with f32 accumulation. Correctness-first bf16 backward writes gradients for x, residual, post_layer_mix, and comb_res_mix.',
)
