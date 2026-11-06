# How mega GDN solves the chunk inverse

Mega GDN processes each sequence in chunks of up to 128 tokens. Within a chunk, a token's residual depends on earlier tokens' residuals. That creates a lower-triangular system: row 0 can be solved immediately, row 1 depends on row 0, and so on. The kernel solves small pieces on Vec, then combines them using Cube matrix multiplications.

The explanation below follows execution in `kernel_gdn_mega.cpp`, `fused_pipeline.h`, `inverse.h`, and `recurrence_pipeline.h`. Cube and Vec run concurrently, so where their work overlaps, both sides are described together.

**1. Prepare the information shared by all chunks.**

Before the recurrent loop, `mega_gate_scan` computes gate prefixes and rearranges gates and beta values for chunk processing. Each sequence starts its own chunks; a short sequence tail is not combined with the next sequence.

At the same time, `mega_shared_grams` computes key–key and query–key dot products. The key–key product supplies the interactions between tokens needed by the triangular solve. Grouped value heads reuse the products for their shared key head.

A global barrier finishes this phase before the recurrent pipeline reads its results.

**2. Vec forms the first chunk's triangular coefficients.**

Before entering the chunk loop, `mega_prepare_inverse` calls `mega_prepare_weights`, then `mega_finish_inverse`, for the first nonempty chunk of the sequence/head being processed.

Write `G_i` for the cumulative log gate inside the chunk, `beta_i` for its update strength, and `k_i` for its key. The strictly lower-triangular interaction matrix is

```text
L[i,j] = beta_i * exp(G_i - G_j) * dot(k_i, k_j),  j < i
L[i,j] = 0,                                      j >= i
```

The system we will eventually solve is

```text
(I + L) R = B
```

Here `R` contains the corrected value residuals, one row per token. `B` will be formed later, once the current recurrent state has been projected through the keys.

`mega_prepare_weights` constructs `N = -L`. The negative sign matters: it lets the later elimination and joins add contributions instead of subtracting them. At this point, the tile holds coefficients, not an inverse.

There are two Vec sub-blocks. One owns rows 0–63 and the other owns rows 64–127. Each owns a 64×128 stripe, including interactions with earlier rows outside that stripe.

The code also computes gated query–key products for output generation. Both computations reuse the same gate-decay matrix. Gate prefixes are stored as a leading FP32 value and a correction; the code uses both when forming the decay.

**3. Vec solves the eight diagonal 16×16 blocks.**

`mega_finish_inverse` obtains an identity stripe by subtracting the strictly lower mask from the inclusive lower mask. It then calls `mega_diagonal_inverses`.

Each Vec sub-block takes the four diagonal 16×16 blocks within its stripe and places four identity matrices side by side in a 16×64 working tile. Across both sub-blocks, that is eight independent small solves.

For each small block, the goal is

```text
X = (I - N_block)^(-1)
```

The routine starts with `X = I`. For pivots `i = 0,...,14`, it updates destination rows using

```text
X[row,:] += N_block[row,i] * X[i,:]
```

Only rows below the pivot have a nonzero coefficient. By the time row `i` is used, all earlier rows have already contributed to it, so it is solved. This is forward substitution applied to all columns of the identity at once. The final row needs no pivot step because there are no rows below it.

For a tiny example:

```text
I - N = [ 1   0   0 ]          inverse = [ 1       0   0 ]
        [-a   1   0 ]                    [ a       1   0 ]
        [-b  -c   1 ]                    [ b+c*a   c   1 ]
```

The `c*a` contribution appears naturally when the solved second row is added to the third row. There is no truncated inverse series or convergence loop.

A gather selects each block's pivot coefficient and repeats it across that block's 16 columns. The multiply-add performs the row updates for all four blocks together. A2/A3 uses the raw `vgather`/`vmla` instructions; the A5 path uses `TGATHER`, `TCOLEXPAND`, and `TMULADDDST` for the same calculation.

The eight inverses are copied back into their original diagonal positions. Entries below and between those blocks still contain the original `N` coefficients. The resulting 128×128 object is therefore a mixture of finished diagonal inverses and coupling coefficients.

**4. Vec exports the stripes and initializes the recurrent state.**

The two stripes are stored in FP32 workspace buffers named `first` and `second`. Despite the half-typed workspace pointers used by the interface, these stores and subsequent Cube loads use FP32 data.

After exporting the stripes, `mega_finish_inverse` also prepares decay-weighted keys for the later state update. This is separate from constructing the inverse.

For the first chunk, `mega_initialize_state` then initializes the recurrent state from the supplied initial state or zeros. Vec signals readiness. Cube loads its working state representation and releases Vec to continue.

**5. Cube starts the current chunk while Vec prepares the next one.**

Inside the chunk loop, Cube computes the current chunk's key/state and query/state projections, `K*S` and `Q*S`.

Concurrently, Vec calls `mega_prepare_weights` for the next chunk, if one exists. Its coefficients remain in Vec memory; Vec does not overwrite the exported inverse buffers yet.

Cube next loads the current chunk's two exported stripes into its 128×128 matrix tile. Once both loads finish, it signals flag 5. Vec can now form the current right-hand side, and Cube can start joining the current diagonal inverses.

**6. Cube joins 16×16 inverses into 32×32 inverses.**

`mega_float_inverse` first calls `mega_float_join_level<16,128>`.

Consider any pair of adjacent diagonal blocks. Their original system has this form:

```text
M = [ A    0 ]
    [-N21  D ]
```

Vec has already supplied `F = A^(-1)` and `E = D^(-1)`. The inverse of the pair is

```text
M^(-1) = [ F          0 ]
         [ E*N21*F    E ]
```

Intuitively, `N21` describes how the first block influences the second. Multiplying by `F` expresses that influence in terms of a solved first block; multiplying by `E` solves its effect inside the second block.

The code evaluates this in two Cube products:

```text
T = E * N21
coupling = T * F
```

It writes the new coupling into the lower-left block. The existing diagonal inverses stay in place. Four independent pairs are joined at this level, producing four complete 32×32 inverses.

Intermediate products pass through FP32 GM scratch and are loaded back into Cube memory. The scratch is a staging area for matrix results; it does not change the block-inverse formula.

**7. Cube joins 32×32 inverses into 64×64 inverses.**

`mega_float_join_level<32,128>` repeats the same two-product operation for two pairs. The tile now contains two complete 64×64 diagonal inverses. Their lower-left 64×64 coupling still holds the original `N21` coefficients between the two halves.

While Cube performs these joins, Vec forms the current chunk's right-hand side:

```text
B_i = beta_i * (V_i - exp(G_i) * (K*S)_i)
```

Vec exports `B` and signals flag 6. It then finishes the next chunk's eight small inverses and exports them. This reuse is safe because Cube has already loaded the current stripes before signalling flag 5.

**8. Cube stops the final join halfway.**

The last call is `mega_float_join_level<64,128,true>`. The `true` argument is `FirstOnly`: compute only

```text
T = E * N21
```

and skip `T * F`.

The stored object is now

```text
[ F   0 ]
[ T   E ]
```

This is deliberately not the full 128×128 inverse. The missing multiplication can be folded into applying the inverse to `B`, saving the work of materializing `T*F`.

**9. Cube applies the factored inverse to the right-hand side.**

After loading attention data and waiting for Vec's right-hand-side signal, Cube calls `mega_factored_solve`. Split `B` and `R` into top and bottom 64-row halves:

```text
R_top    = F * B_top
R_bottom = E * B_bottom + T * R_top
```

Since `T = E*N21`, this is exactly the block solve

```text
R_bottom = E * (B_bottom + N21 * R_top)
```

The implementation computes the top result, stores it to GM, and reloads it for the bottom correction. It also computes `E*B_bottom`, then accumulates `T*R_top`. All of these inverse-join and solve products use native FP32 operands and FP32 accumulation.

**10. Use the solved residuals and advance to the next chunk.**

Cube uses `R` in the attention/output product and in the decay-weighted key product for the state update. Vec finishes the output, updates its authoritative FP32 state, and exports the state representation that Cube will load for the next chunk.

The next chunk's small inverses have already been prepared during this work. The loop repeats with Cube joining those blocks while Vec works ahead again.

For a short final chunk, the physical solve still uses 128-row tiles. Invalid interactions and right-hand-side rows are zero-padded, and the diagonal identities remain present, so the padded rows do not affect valid rows. An empty sequence bypasses the chunk solve entirely.
