from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='mhc',
    name='sinkhorn_normalize',
    builders=['sinkhorn_normalize_fwd', 'sinkhorn_normalize_bwd'],
    configs=[{'case': 'sinkhorn_normalize_fwd', 'configs': [{'eps': 1e-06, 'mhc': 4, 'repeat': 10}]},
 {'case': 'sinkhorn_normalize_bwd', 'configs': [{'eps': 1e-06, 'mhc': 4, 'repeat': 10}]}],
    source='tilekernels/mhc/sinkhorn',
    notes='Tile f32 Sinkhorn forward for n0={1,2}, n1={1,1024,4096}; uses PTO texp, row/column reductions, row/column expansion, and tile division. Tile f32 Sinkhorn backward for n0={1,2}, n1={1,1024,4096}; stores forward stage tiles and applies row/column normalization adjoints in reverse.',
)
