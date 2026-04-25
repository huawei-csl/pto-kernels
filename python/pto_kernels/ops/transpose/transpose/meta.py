from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='transpose',
    name='transpose',
    builders=['transpose', 'batched_transpose'],
    configs=[{'case': 'transpose', 'configs': [{'dtype': 'bf16'}, {'dtype': 'f32'}]},
 {'case': 'batched_transpose', 'configs': [{'dtype': 'bf16'}, {'dtype': 'f32'}]}],
    source='tilekernels/transpose/transpose',
    notes='Dynamic rows/cols port using PTO TTrans tiles. Runtime shape grid mirrors TileKernels deterministic correctness rows in the manifest. fp8_e4m3 remains compile-only/deferred. Dynamic batches/rows/cols port using PTO TTrans tiles. Generated validation assets include host runner and NumPy golden/compare for one deterministic TileKernels-grid shape.',
)
