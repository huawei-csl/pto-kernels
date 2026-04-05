# Linear Attention Optimization Lessons

This note records the optimization lessons learned from the self-contained PTO-ISA examples in this directory. It is meant to be a practical reference for future work on other PTO-ISA kernels, not just for `linear_attention`.

The file name intentionally matches the requested spelling: `optimization_lession.md`.

## How To Use This Note

Use this file as:
- a checklist before starting a new kernel optimization task
- a reminder of which changes gave real speedups here
- a warning list of common correctness failures and deadlock traps
- a template for planning and recording future experiments

If you want the concrete runnable history behind these lessons, read `optimize_step_by_step/README.md` and the numbered tutorial directories beside it.

## Current Reference Point

The current directory gives you two complementary references:
- `linear_attention.cpp`: the current optimized kernel
- `optimize_step_by_step/`: the local optimization ladder from naive code to the current fast path

Current kernel shape:
- dynamic `B` and `L`
- compile-time `H`, `D`, and `C`
- fixed `block_dim = num_cores`
- persistent-kernel style work loop inside the kernel

Current fast configuration:
- `C=128, D=128`
- precomputed causal mask
- shared L0C reuse
- cube-side L0 ping-pong
- 2-slot cube/vector staging
- in-place mask apply
- dual `H`-state L1 buffers

Current validated performance class in this directory:
- roughly `77 TFLOP/s` on large enough benchmark shapes

## Core Lessons

### 1. Start With The Simplest Correct Structure

The early tutorial steps were useful because they made the dataflow obvious:
- load `Q`, `K`, `V`
- form `QK^T`
- form `K^T V`
- apply the causal rule
- accumulate the running hidden state
- finish `O = masked_scores @ V + Q @ H`

That simple structure is the right starting point even when it is slow. Optimization was much easier once the kernel had a clear baseline and a matching NumPy/PyTorch explanation.

Rule for future kernels:
- get one small, readable, correctness-checked version working first
- only then start adding buffering, staging, and flag choreography

### 2. Keep Hot Dimensions Compile-Time Specialized

The biggest stability and codegen wins came from keeping the inner tile shape fixed at compile time.

In this kernel that meant:
- `H`, `D`, and `C` as compile-time constants
- `B` and `L` as runtime dimensions

Why it helped:
- fewer dynamic branches in inner loops
- simpler on-chip allocation
- more predictable tile lowering and instruction scheduling

Rule for future kernels:
- keep the dimensions that determine tile shape and on-chip layout compile-time if you can
- push only outer problem-size dimensions to runtime

### 3. Fixed Launch Shape Plus In-Kernel Work Loop Is A Good Default

Switching to:
- fixed `block_dim = num_cores`
- logical work mapping inside the kernel

was the right dynamic-shape structure.

The key pattern is:
- `work_id = work_idx * block_num + cid`
- skip when `work_id >= total_work`

Why it helped:
- host launch stays stable
- runtime shape changes do not require changing launch geometry
- the kernel becomes more like a persistent worker loop

Rule for future kernels:
- if the logical workload varies but the per-core kernel structure stays the same, prefer a fixed launch plus in-kernel work assignment

### 4. Budget L1, L0C, And UB Explicitly

The kernel only became robust once memory use was treated as a first-class design constraint.

Practices that helped:
- explicit byte accounting for L1, L0C, and UB
- `static_assert` guards for invalid tile choices
- separating "one-slot" and "two-slot" workspace footprints
- designing tile sizes around real on-chip budget, not just around the math

Rule for future kernels:
- write the memory budget down in bytes
- fail early at compile time when a tile choice cannot fit
- treat on-chip memory planning as part of the algorithm design

### 5. Remove Scalar Work From The Vector Path Early

One of the biggest speedups came from deleting the scalar causal-mask loop and replacing it with:
- a precomputed triangular mask tensor
- a vector `TMUL`

Why it helped:
- removed per-element scalar control flow
- let the vector unit handle masking as a tile operation
- made the vector side much simpler to pipeline later

Rule for future kernels:
- whenever you see elementwise scalar loops on the vector side, first ask whether they can become tile-vector operations

### 6. Reuse On-Chip Storage Aggressively

Another important step was changing the L0C layout from:
- separate regions for score, state, and output

to:
- one shared region reused across serialized cube stages

Why it helped:
- larger tile choices became legal
- `C=128, D=128` fit without changing the math
- arithmetic intensity improved

Rule for future kernels:
- if two stages never need the same buffer live at the same time, consider aliasing them onto the same on-chip region

### 7. Optimize The Cube Microkernel Before Redesigning The Whole Kernel

The first major structural speedup came from improving the local cube helper:
- split `K=128` into `2 x 64`
- ping-pong two L0 buffers
- overlap extract with cube compute

Why it helped:
- the inner GEMM path stopped looking like a single serial block
- the outer algorithm stayed unchanged

Rule for future kernels:
- before rewriting the whole kernel, inspect the most repeated GEMM-like helper and see whether load/compute overlap can be introduced there first

### 8. Inter-Core Producer/Consumer Pipelines Give Large Wins

The next big jump came from moving from:
- one chunk of cube work, then one chunk of vector work

to:
- cube producing chunk `i + 1` while vector consumes chunk `i`

The working version used:
- two workspace slots
- stage-aware cross-core flags
- an explicit end-of-work-item acknowledgment

Why it helped:
- reduced chunk-to-chunk bubbles between cube and vector
- let both sides stay busier on long sequences

Rule for future kernels:
- if cube and vector naturally form a producer/consumer pair, a small staged workspace is often worth more than another tiny local instruction tweak

### 9. Reduce Temporary Tiles When UB Is Tight

Applying the mask in-place on `acc_ub` removed one extra UB tile.

Why it helped:
- lowered UB pressure
- made mask preload possible again
- cut some unnecessary data motion

Rule for future kernels:
- once the functional structure is stable, inspect temporary tiles and ask which ones can safely become in-place updates

### 10. Prefetch The Next Recurrent State Early

The final major improvement here was adding a second `H`-state L1 buffer so the next prefix-state tile could be loaded while the current chunk still had work left.

Why it helped:
- hid part of the recurrent-state load cost
- reduced bubbles in the output stage

Rule for future kernels:
- in recurrent or iterative kernels, the next state load is often a good prefetch target once the main pipeline exists

### 11. Compiler Flags Matter, But Only Measured Ones Count

This directory also showed that not every seemingly stronger compiler option helps.

The currently proven settings keep:
- stack sizing flags
- overflow-record flags
- `-cce-aicore-dcci-insert-for-scalar=false`

The local sweep showed this kernel was faster without:
- `-cce-aicore-addr-transform`
- `-DL2_CACHE_HINT`

Rule for future kernels:
- treat compiler flags as experiments, not assumptions
- keep only settings that survive correctness and benchmark comparison

## Common Failure Modes

These were recurring problems during the work here:

### Deadlocks

Typical causes:
- reusing a staged buffer before the peer core released it
- narrowing dependencies too aggressively
- forgetting an end-of-work-item handshake when the same physical core later serves a different logical job

Guardrail:
- if you add or change a pipeline stage, re-check all producer/consumer ownership transitions explicitly

### Silent Numerical Regressions

Typical causes:
- wrong byte offsets
- wrong aliasing assumptions in L0C or UB
- reordered accumulation without checking tolerance impact

Guardrail:
- keep full correctness sweeps, not just one smoke shape

### Overfitting To Benchmark Noise

Typical cause:
- keeping a change because one run was slightly faster

Guardrail:
- compare repeated measurements on the same shape set
- revert marginal gains if they do not repeat cleanly

### Complexity Without Throughput Gain

Typical cause:
- adding a pipeline or microkernel that looks more advanced but does not improve the dominant bottleneck

Guardrail:
- only keep structural complexity when the measured benefit is clear

## Practical Optimization Order

For a new PTO-ISA kernel, a good order is:

1. Get a small, direct, correctness-checked baseline.
2. Move runtime variability out of the hot inner tile logic.
3. Remove scalar work from vector code.
4. Revisit tile shape and on-chip memory reuse.
5. Improve the local cube microkernel.
6. Add staged producer/consumer overlap between pipelines.
7. Reduce temporary buffers and prefetch recurrent state.
8. Sweep compiler flags only after the kernel structure is stable.
9. Tune benchmark shapes to expose steady-state throughput.

## What To Measure

For each experiment, keep the same checklist:

- correctness on a full shape sweep
- at least one small smoke benchmark
- at least one larger steady-state benchmark
- best TFLOP/s shape
- best GiB/s shape
- whether bandwidth includes or excludes workspace traffic

If possible, keep:
- one fixed quick shape set for iteration
- one fixed large-shape table for decisions

## Experiment Template

Record each attempt with:

- `ID`: short experiment name
- `Goal`: the bottleneck being targeted
- `Hypothesis`: why it might help
- `Change`: exact implementation change
- `Check`: correctness and benchmark commands
- `Status`: `todo`, `doing`, `done`, `reverted`, `dropped`
- `Result`: measured outcome and short conclusion

Recommended workflow:

1. Pick one experiment only.
2. Record the benchmark shapes before editing.
3. Run correctness first.
4. Run the same benchmark set before and after.
5. Keep or drop the change based on repeated evidence.

## Local Progression Summary

The local tutorial ladder in `optimize_step_by_step/` also captures the high-level progression:

1. naive static shape
2. dynamic work mapping
3. cached causal mask
4. larger chunk size
5. cube L0 ping-pong
6. two-slot cube/vector pipeline
7. L1 hidden-state prefetch

That sequence is a useful default mental model for future optimization tasks:
- first remove obvious scalar waste
- then improve tile size and memory reuse
- then overlap local stages
- then overlap whole pipelines

## Closing Thought

The biggest gains in this directory did not come from changing the algorithm. They came from:

- reducing scalar work
- specializing the hot path
- planning on-chip memory explicitly
- reusing buffers aggressively
- overlapping cube, vector, and memory movement
- keeping only measured improvements

For future PTO-ISA kernels, the main takeaway is simple: start from a clear baseline, optimize one bottleneck at a time, and only keep structural complexity that earns its place in the benchmark table.
