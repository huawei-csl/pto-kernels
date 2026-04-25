from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='engram',
    name='fused_weight',
    builders=['fused_weight'],
    configs=[{'case': 'fused_weight', 'configs': [{'hc': 4, 'input_dtype': 'bf16', 'output_dtype': 'f32'}]}],
    source='tilekernels/engram/fused_weight',
    notes='Ports the bf16 weight_hidden * weight_embed path with f32 accumulation/output over the TileKernels hidden-size grid.',
)
