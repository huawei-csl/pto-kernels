from pto_kernels.ops.common import tilekernels_meta


META = tilekernels_meta(
    family='engram',
    name='engram_hash',
    builders=['engram_hash'],
    configs=[{'case': 'engram_hash',
  'configs': [{'max_ngram_size': 3, 'num_embed_table_per_ngram': 8, 'num_ngram_layers': 2}]}],
    source='tilekernels/engram/hash',
    notes='Scalar/token-parallel int hash over n-gram token ids with int64 multipliers, per-layer vocab modulo, and offset addition.',
)
