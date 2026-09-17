---
name: porting-a2a3-kernels-to-a5
description: Port PTO kernels written for Ascend A2/A3 (DAV_2201, dav-c220) so the same source also runs on A5 (DAV_3510, dav-c310). Use when converting a kernel to A5, replacing __DAV_C220_* guards, fixing Cube/Vec handshakes that hang on A5, debugging wrong matmul results traced to L0A fractal layout, or switching a build from MEMORY_BASE to REGISTER_BASE.
---

# Porting PTO Kernels From A2/A3 To A5

Every difference below can be `#if`-guarded, so keep one source that builds for both targets
rather than forking the kernel.

| | A2/A3 | A5 |
|---|---|---|
| AI core arch | `dav-c220` | `dav-c310` |
| `__CCE_AICORE__` | `220` | not `220` |
| Memory model | `-DMEMORY_BASE` | `-DREGISTER_BASE` |
| `--npu-arch=` | `dav-2201` | `dav-3510` |
| L0A fractal layout | ZZ | NZ |
| `pipe_barrier(PIPE_V)` | required | no-op |
| Cube ↔ Vec | separate cores, FFTS cross-core flags | one core, intra-block flags |

Three arch-detection macros, don't mix them up: **`__CCE_AICORE__ == 220`** is true on A2/A3
only (use for sync and barriers); **`__DAV_C310__`** is defined on A5 only (use for layout);
**`__DAV_CUBE__` / `__DAV_VEC__`** say which compile pass this is and are defined on *both*.

## Do this

**1. Rename the compile-pass macros.** `__DAV_C220_CUBE__` → `__DAV_CUBE__`,
`__DAV_C220_VEC__` → `__DAV_VEC__`; the arch-specific spellings no longer exist. Catch them in
comments and `#endif //` trailers too. The three-pass model is otherwise unchanged: pass 1
defines `__DAV_VEC__`, pass 2 `__DAV_CUBE__`, pass 3 neither (host launcher).

**2. Fix the build flags.** `--npu-arch=` replaces `--cce-aicore-arch=` and also
`--cce-soc-version=` / `--cce-soc-core-type=`; the driver expands it to the AI core arch
(`dav-2201` → `dav-c220`, `dav-3510` → `dav-c310`) and keeps mix mode, so both passes still run.
Two consequences: the memory model must come from the command line only — a kernel containing
`#ifndef MEMORY_BASE / #define MEMORY_BASE` can never be built register-base — and compiled
artifacts and JIT caches must be keyed on the target, since the two builds are different SoCs
and not interchangeable.

**3. Guard `pipe_barrier(PIPE_V)` behind `__CCE_AICORE__ == 220`.** A5's vector pipe handles
that dependency itself. An inline wrapper that compiles to nothing on A5 keeps call sites
readable. Still needed on both targets: `pipe_barrier(PIPE_ALL)`, `pipe_barrier` on any other
pipe, and every cross-pipe `set_flag`/`wait_flag` pair.

**4. Flip L0A's outer layout.** L0A goes ZZ → NZ, so its outer `BLayout` becomes `ColMajor`
under `__DAV_C310__`; **L0B stays ZN / `RowMajor`**. Inner `SLayout` (RowMajor for L0A,
ColMajor for L0B) and L1 tiles are unchanged, so for tiles declared through `Tile<>` this is a
one-parameter change. If you compute L0A fractal addresses by hand, fix those too: ZZ walks
the fractal grid row-major, NZ column-major, so the two strides swap — the index that was
multiplied by the full matrix dimension gets `FractalSize` instead, and vice versa. L0B
arithmetic is unchanged.
Details: <https://pto-isa.github.io/docs/isa/cube/nz-fractal-layout/#per-buffer-nz-layouts>

**5. Convert every Cube↔Vec handoff to intra-block flags.** This is the bulk of the work. On
A2/A3 the Cube (AIC) and Vec (AIV) are separate cores and hand off through FFTS —
`ffts_cross_core_sync(pipe, msg)` to signal, `wait_flag_dev(flag)` to wait, after
`set_ffts_base_addr()`. On A5 the Cube and *both* Vec sub-blocks share one core, so those
become `set_intra_block(pipe, flag)` / `wait_intra_block(pipe, flag)`.

The asymmetry that bites: there are two Vec sub-blocks, and the flag ID a sub-block touches is
offset by **16** for sub-block 1 — so the two land on different flags even though they execute
the same source line. One logical event therefore has two waiters: the Cube side signals and
waits the pair explicitly (fan-out down, join up), the Vec side writes a single plain flag.
Getting this backwards deadlocks, or half-releases.

| Direction | Flags | Pipe |
|---|---|---|
| Cube signals Vec | set both `F` and `F+16` | `PIPE_FIX` |
| Cube waits on Vec | wait on both `F` and `F+16` | `PIPE_MTE2` |
| Vec signals Cube | set `F` only, no `+16` | `PIPE_MTE3` |
| Vec waits on Cube | wait on `F` only, no `+16` | `PIPE_MTE3` |

`wait_flag_dev` took no pipe, so you have to pick one: Cube waits on `PIPE_MTE2` because the op
consuming the awaited data is the GM→L1 `TLOAD`, and that's the pipe the wait must block.
Intra-block flags also don't carry the ordering the FFTS path implied, so add
`pipe_barrier(PIPE_ALL)` before every signal and after every wait unless the surrounding code
already has one there. Flag IDs themselves don't change between targets — you only gain the
`+16` companion — so keep base IDs in 0..15 to stop `base+16` colliding with another pair.
Finally, only handoffs *within a block* become intra-block: device-wide and cross-block
barriers still go through FFTS on both targets, so audit any all-core barrier helper
separately — it may work as-is, or may now deadlock because Cube and Vec share a core.

**6. Qualify `Stride` as `pto::Stride`.** It becomes ambiguous under the A5 headers even with
`using namespace pto;`. Other PTO type names (`Shape`, `GlobalTensor`, `BaseShape2D`,
`TileShape2D`, …) are unaffected.

## Pitfalls

- **Guard the whole conditional, not the call.** `if (cond) wait_flag_dev(F);` becomes a
  different number of statements per target — wrap the entire `if` body.
- **Early-out paths must still run the full handshake.** A chunk with no work that skips its
  signal/wait sequence desynchronizes the other core's flag counts.
- **Drain outstanding flags before kernel exit.** A wait gated on `if (!first_iter)` leaves the
  last signal unconsumed, and the stale count corrupts the next launch.
- **Don't convert intra-core pipeline sync.** `set_flag`/`wait_flag` between pipes on one core
  is unrelated to Cube↔Vec sync and still required.
- If you factor the arch guards into a shared header *and* `#include` kernels into namespaces,
  include that header at global scope first — `#pragma once` otherwise traps it inside
  whichever namespace pulled it in first.

## Verifying the port

A ported kernel is not done until it has been compiled and run on A5 hardware — sync bugs from step 5
hang rather than return wrong numbers, so a clean compile proves nothing.
