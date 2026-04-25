from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='mhc',
    name='norm_fn',
    builders=['pre_norm_fn_fwd', 'fn_normw_merge_fwd', 'fn_normw_merge_bwd'],
    configs=[{'case': 'pre_norm_fn_fwd',
  'configs': [{'eps': 1e-06, 'hidden_size': 1280, 'mhc_mult': 4},
              {'eps': 1e-06, 'hidden_size': 2560, 'mhc_mult': 4},
              {'eps': 1e-06, 'hidden_size': 7168, 'mhc_mult': 4}]},
 {'case': 'fn_normw_merge_fwd',
  'configs': [{'hidden_size': 1280, 'mhc_mult': 4},
              {'hidden_size': 2560, 'mhc_mult': 4},
              {'hidden_size': 7168, 'mhc_mult': 4}]},
 {'case': 'fn_normw_merge_bwd',
  'configs': [{'hidden_size': 1280, 'mhc_mult': 4},
              {'hidden_size': 2560, 'mhc_mult': 4},
              {'hidden_size': 7168, 'mhc_mult': 4}]}],
    source='tilekernels/mhc/norm_fn',
    notes='Baseline forward without optional norm-weight merge; computes RMS-normalized residual/FN projection with f32 output. Optional norm-weight merge for pre_norm_fn; computes out_fn = fn * normw. Optional norm-weight merge backward; accumulates fn_grad and normw_grad.',
)
