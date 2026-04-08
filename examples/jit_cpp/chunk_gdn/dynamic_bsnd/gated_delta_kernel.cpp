// The original scalar fallback prototype has been retired.
//
// `dynamic_bsnd` is being ported stage-by-stage onto PTO vector/tile kernels,
// following the same structure as `static_baseline` and the dynamic BSND
// metadata style from `linear_attention.cpp`.
//
// Implemented stages live in dedicated translation units such as
// `chunk_cumsum_kernel.cpp`. The full chained forward kernel will be restored
// only after each stage is ported and validated independently for both
// correctness and performance.
