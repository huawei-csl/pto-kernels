from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='mhc',
    name='expand_to_mhc',
    builders=['expand_to_mhc_fwd', 'expand_to_mhc_bwd'],
    configs=[{'case': 'expand_to_mhc_fwd', 'configs': [{'mhc_mult': 2}, {'mhc_mult': 4}, {'mhc_mult': 8}]},
 {'case': 'expand_to_mhc_bwd', 'configs': [{'mhc_mult': 2}, {'mhc_mult': 4}, {'mhc_mult': 8}]}],
    source='tilekernels/mhc/expand_to_mhc',
    notes='Forward copy expands bf16 input rows across the MHC axis for the deterministic n0/n1/hidden manifest grid. Backward path reduces bf16 gradients across the MHC axis using f32 tile accumulation before conversion back to bf16.',
)
