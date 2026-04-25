from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='engram',
    name='grad_w_reduce',
    builders=['grad_w_reduce'],
    configs=[{'case': 'grad_w_reduce',
  'configs': [{'hc_mult': 4, 'num_persistent_blocks': 4},
              {'hc_mult': 4, 'num_persistent_blocks': 8}]}],
    source='tilekernels/engram/grad_w_reduce',
    notes='Tile f32 reduction over persistent partial gradients, bf16-to-f32 weight conversion, and in-place accumulation into hidden/embed weight gradients.',
)
