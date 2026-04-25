from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='engram',
    name='gate',
    builders=['engram_gate_fwd', 'engram_gate_bwd'],
    configs=[{'case': 'engram_gate_fwd',
  'configs': [{'hc_mult': 4, 'hidden_size': 2048},
              {'hc_mult': 4, 'hidden_size': 4096},
              {'hc_mult': 4, 'hidden_size': 7168}]},
 {'case': 'engram_gate_bwd',
  'configs': [{'hc_mult': 4, 'hidden_size': 2048, 'num_persistent_blocks': 4},
              {'hc_mult': 4, 'hidden_size': 4096, 'num_persistent_blocks': 4},
              {'hc_mult': 4, 'hidden_size': 7168, 'num_persistent_blocks': 4}]}],
    source='tilekernels/engram/gate',
    notes='Fixed-hidden forward save path using 1024-wide row tiles, f32 RMS reductions, signed-sqrt sigmoid gating, bf16 output, and saved dot/gate/rstd intermediates. Correctness-first backward computes grad_x, grad_k, grad_v, and per-persistent-block grad_w_partial with regular tile reductions and stores.',
)
