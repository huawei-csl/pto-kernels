#!/usr/bin/env python3
"""Generate the CCE-PARITY fast Walsh-Hadamard VMI .pto, PARAMETRIC on N
(N in {32,64,128,256,512,1024,2048}): MULTI-CORE + 4-buffer deep software pipeline
(prefetch depth 2) with a 32 KB UB tile, the TILE LOOP UNROLLED at generate-time.

This is the VMI/ptodsl reproduction of the CCE PR #221 kernel structure
(fast_hadamard/cce/csrc/fast_hadamard_a5.cpp), generalized to any supported N by
lifting gen_had.py's per-N LAYOUT into gen_had_full.py's PIPELINE:

  * MULTI-CORE batch partition (from gen_had_rtmc.py): host launches grid=G blocks;
    block b in [0,G) reads bidx=get_block_idx()/gnum=get_block_num() and streams its
    contiguous row band [b*rpb,(b+1)*rpb) (rpb=batch/G). CCE block_dim=G scheme.
  * DEEP per-block pipeline: the block streams its rpb rows in CR-row, 32 KB tiles
    through a 4-buffer UB ring (CCE DEF_BUFFERS=4), 2 tiles in flight ahead of the
    compute (CCE DEF_PREFETCH=2), fenced by set_flag/wait_flag (no PIPE_ALL/tile).
    Unrolled-by-4 (chunk 4q+k -> buffer k, compile-time event id), peeled
    prologue/steady/epilogue for the depth-2 prefetch.
  * PER-N LAYOUT (lifted from gen_had.py / CCE KernelShape):
      - N < 256  PACKING: R = 256/N rows share one 256-window so every stage drives
        all 128 lanes; the window emerges index-rotated-right by log2(N), fixed by
        ROT = 8 - log2(N) register-to-register `pto.vdintlv` FUSED into the peeled
        final stage. N=32 needs ROT=3 and WORKS (rel_err 7.6e-4 on cannsim): an
        isolated probe (scratchpad/probe_vdintlv_*) proved pto.vdintlv composes
        the CORRECT deinterleave to depth 3 -- there is NO net-rotation cap. The
        earlier "N=32 miscomputes ~1.5" was the dangling-pointer driver bug (the
        same artifact that made N=512 look like rel_err 591); with the fixed
        driver N=32 is correct. N=16 (ROT=4) is untested here and left blocked.
      - N == 256 : one 256-window/row, log2(N)=8 deinterleave stages, ROT=0.
      - N > 256  CHUNKING: a row spans chunks=N/256 windows, all loads issued before
        any store (the in-place aliasing rule).

THE PARITY FIX (preserved for every N): the per-chunk TILE loop is UNROLLED at
generate-time (Python `for t in range(TILES)`), so each tile's UB byte offset
(buf*32768 + t*tilestride) is a COMPILE-TIME CONSTANT. castptr then folds to an
immediate and NO per-tile scalar address arithmetic (index_cast/muli/addi ->
RV_SADD/RV_SMOV) leaks into the vector pipe -- which is what lets vector-pipe issue
throughput reach CCE's ~1.6 IPC instead of the ~1.1 the runtime scf.for tile loop
gave (it serialized every VLDI behind its address add). CCE's bisheng gets the same
immediate addressing by strength-reducing its iter loop; unrolling reproduces it.

THE UNIFORM-TILE INVARIANT (why one pipeline serves every N): CR = 32KB/(N*2) =
16384/N rows/chunk (== CCE RowsFor<N>), so CHUNK_ELEMS = CR*N = 16384 and
CHUNK_BYTES = 32 KB for EVERY N; 4 buffers = 128 KB UB. A tile is a fixed
TILE_ELEMS=2048 elems (4 KB), TILES = 16384/2048 = 8 tiles/chunk, and every tile
carries EXACTLY 8 butterfly units for every N (8 pack-groups, or
rows_per_tile*chunks = (2048/N)*(N/256) = 8 chunk-windows). So the ring, the
prologue/steady/epilogue and the DMA geometry are byte-for-byte N-independent; only
the in-tile offset pattern (pack vs chunk), the rotation, log2(N) and the DMA burst
length clen=N*2 change with N.

DIVISIBILITY: batch % (G * 4*CR) == 0  (rpb=batch/G a multiple of 4*CR so
nchunk_block=rpb/CR is a multiple of 4 -> nquad>=1, no tail). 4*CR = 65536/N.
Per N the smallest single-core (G=1) batch is 65536/N:
  N= 32 ->2048, N=64 ->1024, N=128 ->512, N=256 ->256, N=512 ->128, N=1024 ->64, N=2048 ->32.

Usage: gen_had_full.py [N] [BATCH_HINT] [OUT.pto]  (BATCH_HINT is annotation only;
the kernel reads batch AND grid at launch). Every N (default 256) writes
fast_hadamard_vmi_full_n{N}.pto with func @fast_hadamard_vmi_full_n{N} -- uniform
across all widths.
"""
import os
import sys
import math

# Emit .pto beside this script by default -- the same directory full_n_mlir.py's
# @pto.jit(source=...) resolves against. Override the output path with argv[3].
MLDIR = os.path.dirname(os.path.abspath(__file__))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
BATCH_HINT = int(sys.argv[2]) if len(sys.argv) > 2 else 256

WINDOW = 256
LOG2W = 8
LANES = 128
MAX_WORKING_ROT = 3               # pto.vdintlv verified correct to depth 3 (N=32 OK); N=16 (ROT=4) untested
NBUF = 4                          # 4-buffer UB ring (CCE DEF_BUFFERS)
PREFETCH = 2                      # tiles in flight ahead of compute (CCE DEF_PREFETCH)
TILE_BYTES = 32 * 1024            # CCE TILE_BYTES: one UB buffer = 32 KB

# ---- reject unsupported N (power-of-two in [32,2048]) ----
if not (32 <= N <= 2048 and (N & (N - 1)) == 0):
    sys.exit(f"gen_had_full: N={N} unsupported (need power-of-two in [32,2048]; "
             f"N=4096 needs 16 chunks/row and is rejected, matching CCE)")

LOG2N = int(round(math.log2(N)))
R = WINDOW // N if N < WINDOW else 1            # rows per packed 256-window (N<256)
group = N * R                                   # 256 if N<=256 else N
rotations = LOG2W - min(LOG2N, LOG2W)           # >0 only for N < 256
upper = group // 2
lanes = min(upper, LANES)                       # == 128 for every supported N
chunks = upper // lanes                          # 1 for N<=256, N/256 for N>256
plain = LOG2N - (1 if rotations else 0)          # normal stages; last peeled if ROT
assert lanes * chunks == upper
assert R == 1 or chunks == 1, "pack and chunk are mutually exclusive"

if rotations > MAX_WORKING_ROT:
    sys.exit(
        f"gen_had_full: N={N} needs ROT={rotations} vdintlv window rotations, above "
        f"the depth ({MAX_WORKING_ROT}) verified correct on this toolchain (N=32 "
        f"ROT=3 is proven; see scratchpad/probe_vdintlv_*). N<=16 (ROT>=4) is "
        f"untested here and rejected conservatively, NOT because vdintlv is known to "
        f"fail -- the probe found no rotation cap; re-run the probe at higher depth "
        f"to lift this.")

FUNC = f"fast_hadamard_vmi_full_n{N}"
DEFAULT_OUT = f"{MLDIR}/fast_hadamard_vmi_full_n{N}.pto"
OUT = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUT

# ---- uniform-tile geometry (N-independent: 32KB chunk, 4KB tile, 8 tiles) -----
CR = TILE_BYTES // (N * 2)         # rows/chunk == CCE RowsFor<N>
CHUNK_ELEMS = CR * N               # == 16384 for every N
CHUNK_BYTES = CHUNK_ELEMS * 2      # == 32768 (32 KB) for every N
assert NBUF * CHUNK_BYTES == 131072, "4 buffers must be 128 KB UB"
TILE_ELEMS = 2048                  # one immediate tile (== GPT*group for N<=256)
assert CHUNK_ELEMS % TILE_ELEMS == 0
TILES = CHUNK_ELEMS // TILE_ELEMS  # == 8 tiles/chunk for every N
GPT = TILE_ELEMS // group          # pack-groups per tile (N<=256); == rows/tile (N>256)
rows_per_tile = TILE_ELEMS // N    # rows per tile
UNITS = GPT if chunks == 1 else rows_per_tile   # butterfly units per tile (rows for chunk)
clen = N * 2                       # DMA burst length (bytes) per row
tilestride_bytes = TILE_ELEMS * 2  # 4096 bytes -- constant across N
FOUR_CR = 4 * CR                   # batch % (G*FOUR_CR) == 0


# One event id per buffer, reused across free/loaded/computed edges (CCE style).
def ev(b):
    return f"EVENT_ID{b}"


L = []
def emit(s=""):
    L.append(s)

U = [0]
def nx():
    U[0] += 1
    return U[0]

VF16 = "!pto.vreg<128xf16>"


def emit_tile(base, rot, tag):
    """Emit the GPT (=8) 256-window butterfly units of one tile at compile-time UB
    pointer `base`, BATCHED BY OP TYPE (all vldsx2, then all vadd, then all vsub,
    then the ROT vdintlv fuse, then all vsts). Allocation discipline matches the
    proven N=256 kernel: offset consts, then ALL `lo` regs, then ALL `hi` regs, so
    vldsx2's two outputs land in SEPARATE ranges (lo[g], hi[g] = G apart).
    Adjacent vldsx2 output SSA regs in the no-rotation pack case miscompile under
    this ptoas (verified: interleaved-pair N=256 -> rel_err 52; separated -> PASS).
    `rot` deinterleave-window rotations (N<256 packing) are fused before the store,
    batched across the 8 groups (each group's vdintlv is independent)."""
    oL = [nx() for _ in range(GPT)]
    oH = [nx() for _ in range(GPT)]
    for g in range(GPT):
        off = g * group
        emit(f"          %e{oL[g]} = arith.constant {off} : index")
        emit(f"          %e{oH[g]} = arith.constant {off + upper} : index")
    lo = [nx() for _ in range(GPT)]
    hi = [nx() for _ in range(GPT)]
    for g in range(GPT):
        emit(f'          %v{lo[g]}, %v{hi[g]} = pto.vldsx2 %{base}[%e{oL[g]}], "DINTLV_B16"')
        emit(f"              : !pto.ptr<f16, ub>, index -> {VF16}, {VF16}")
    s = [nx() for _ in range(GPT)]
    for g in range(GPT):
        emit(f"          %v{s[g]} = pto.vadd %v{lo[g]}, %v{hi[g]}, %mask{tag} : {VF16}, {VF16}, !pto.mask<b16> -> {VF16}")
    d = [nx() for _ in range(GPT)]
    for g in range(GPT):
        emit(f"          %v{d[g]} = pto.vsub %v{lo[g]}, %v{hi[g]}, %mask{tag} : {VF16}, {VF16}, !pto.mask<b16> -> {VF16}")
    finlo, finhi = list(s), list(d)
    for _ in range(rot):
        newlo = [nx() for _ in range(GPT)]
        newhi = [nx() for _ in range(GPT)]
        for g in range(GPT):
            emit(f"          %v{newlo[g]}, %v{newhi[g]} = pto.vdintlv %v{finlo[g]}, %v{finhi[g]} : {VF16}, {VF16} -> {VF16}, {VF16}")
        finlo, finhi = newlo, newhi
    for g in range(GPT):
        emit(f'          pto.vsts %v{finlo[g]}, %{base}[%e{oL[g]}], %mask{tag} {{dist = "NORM_B16"}} : {VF16}, !pto.ptr<f16, ub>, !pto.mask<b16>')
    for g in range(GPT):
        emit(f'          pto.vsts %v{finhi[g]}, %{base}[%e{oH[g]}], %mask{tag} {{dist = "NORM_B16"}} : {VF16}, !pto.ptr<f16, ub>, !pto.mask<b16>')


IND = "          "  # 10-space body indent inside scf.for


def _tile_units():
    """The 8 butterfly units of ONE chunk tile: rows_per_tile rows x chunks
    256-windows, as (load_off, store_lo_off, store_hi_off) ELEMENT offsets WITHIN
    the tile. The `chunks` units of a row are load/store COUPLED in place (unit c's
    store aliases unit c'!=c's load), so all loads of a tile precede all stores."""
    u = []
    for r in range(rows_per_tile):
        ro = r * group
        for c in range(chunks):
            u.append((ro + c * 2 * lanes, ro + c * lanes, ro + upper + c * lanes))
    assert len(u) == 8, f"expected 8 units/tile, got {len(u)}"
    return u


def emit_compute_chunk(sfx, buf):
    """CHUNK path (N>256): batched, generate-time UNROLLED tiles -- immediate UB
    offsets, no scalar address leak. Each tile's 8 units (rows_per_tile x chunks)
    are batched by op type (all vldsx2, then vadd, then vsub, then vsts).

    This is the in-place chunk kernel AT ITS ptodsl FLOOR (RVEC ~1.71x CCE for
    N=512); see compare.md for the full root-cause writeup. Summary: the butterfly
    is in place, so within a tile every load must precede every store (unit c's
    store aliases sibling unit c''s load window). cannsim is offset-aware (that is
    why N<=256, whose 8 units are DISJOINT 256-windows, saturates the store subpipe
    at ~8% idle from this exact same batched-unrolled form). But for N>256 the
    intra-tile aliasing serializes each tile into a load phase then a store phase,
    so only ONE tile's 16 stores are ever available to overlap a tile's load+compute
    phase -- not enough to keep the store subpipe (the store-throughput-bound
    critical resource) full, leaving it ~40% idle. Two ways to saturate it both
    exceed the ptodsl budget here and were rejected after measuring:
      * ROLL the tile loop (CCE `sweep`/`iters`, bisheng strength-reduces the base):
        cross-iteration overlap, but ptodsl cannot strength-reduce, so the per-tile
        base arithmetic leaks RV_SADD/RV_SMOV that serialize the load pipe -> 2.45x
        (LOAD-bound, measured RVEC 70694).
      * INTERLEAVE a tile's stores into the next tile's loads: cannsim serializes
        fine-grained load/store interleaving to one buffer -> 2.6x (RVEC 128806).
      * DEPTH-2 software pipeline (hold two tiles' results) needs ~48 vregs > 32;
        PING-PONG (separate read/write UB regions to remove aliasing) needs 2x UB,
        breaking the 4-buffer/128KB geometry. Both out of scope for this fix.
    So batched-unrolled (this) is the best correct in-place kernel; N>256 ships at
    its ~1.71x floor, honestly reported, not faked."""
    bufbase = buf * CHUNK_BYTES
    tag = f"_{sfx}"
    units = _tile_units()
    offs = sorted({o for w in units for o in w})

    emit(f"      pto.vecscope {{")
    emit(f"        %mask_{sfx}, %so_{sfx} = pto.plt_b16 %c128_i32 : i32 -> !pto.mask<b16>, i32")
    emit(f"        %cP_{sfx} = arith.constant {plain} : index")
    emit(f"        scf.for %stage_{sfx} = %c0 to %cP_{sfx} step %c1 {{")
    # in-tile offset constants (immediate, loop-invariant -> hoisted, reused/tile)
    offid = {}
    for off in offs:
        oi = nx()
        emit(f"{IND}%e{oi}_{sfx} = arith.constant {off} : index")
        offid[off] = oi
    for t in range(TILES):
        tbb = bufbase + t * tilestride_bytes  # compile-time UB byte offset (immediate)
        emit(f"{IND}%tbb_{sfx}_{t} = arith.constant {tbb} : i64")
        emit(f"{IND}%ub_{sfx}_{t} = pto.castptr %tbb_{sfx}_{t} : i64 -> !pto.ptr<f16, ub>")
        ub = f"ub_{sfx}_{t}"
        # 8 units batched; lo[]/hi[] pre-allocated in SEPARATE ranges (vldsx2's two
        # adjacent outputs miscompile under this ptoas).
        lo = [nx() for _ in range(8)]
        hi = [nx() for _ in range(8)]
        for k in range(8):
            emit(f'{IND}%v{lo[k]}, %v{hi[k]} = pto.vldsx2 %{ub}[%e{offid[units[k][0]]}_{sfx}], "DINTLV_B16"')
            emit(f"{IND}    : !pto.ptr<f16, ub>, index -> {VF16}, {VF16}")
        s = [nx() for _ in range(8)]
        for k in range(8):
            emit(f"{IND}%v{s[k]} = pto.vadd %v{lo[k]}, %v{hi[k]}, %mask{tag} : {VF16}, {VF16}, !pto.mask<b16> -> {VF16}")
        d = [nx() for _ in range(8)]
        for k in range(8):
            emit(f"{IND}%v{d[k]} = pto.vsub %v{lo[k]}, %v{hi[k]}, %mask{tag} : {VF16}, {VF16}, !pto.mask<b16> -> {VF16}")
        for k in range(8):
            emit(f'{IND}pto.vsts %v{s[k]}, %{ub}[%e{offid[units[k][1]]}_{sfx}], %mask{tag} {{dist = "NORM_B16"}} : {VF16}, !pto.ptr<f16, ub>, !pto.mask<b16>')
        for k in range(8):
            emit(f'{IND}pto.vsts %v{d[k]}, %{ub}[%e{offid[units[k][2]]}_{sfx}], %mask{tag} {{dist = "NORM_B16"}} : {VF16}, !pto.ptr<f16, ub>, !pto.mask<b16>')
    emit(f'          pto.mem_bar "VST_VLD"')
    emit(f"        }}")
    emit(f"      }}")


def emit_compute(sfx, buf):
    """log2(N)-stage butterfly on the CR rows resident in buffer `buf` (UB byte
    offset buf*CHUNK_BYTES). The TILES tiles are UNROLLED at generate-time (offsets
    compile-time -> immediate addressing, the parity fix). `plain` normal stages run
    in an scf.for; if rotations>0 (N<256) the final stage is PEELED and carries the
    ROT vdintlv fuse (faithful CCE port). N>256 (chunking) dispatches to the
    software-pipelined chunk emitter (see emit_compute_chunk)."""
    if chunks > 1:
        return emit_compute_chunk(sfx, buf)
    bufbase = buf * CHUNK_BYTES
    emit(f"      pto.vecscope {{")
    emit(f"        %mask_{sfx}, %so_{sfx} = pto.plt_b16 %c128_i32 : i32 -> !pto.mask<b16>, i32")
    emit(f"        %cP_{sfx} = arith.constant {plain} : index")
    emit(f"        scf.for %stage_{sfx} = %c0 to %cP_{sfx} step %c1 {{")
    for t in range(TILES):
        tbb = bufbase + t * tilestride_bytes
        emit(f"          %tbb_{sfx}_{t} = arith.constant {tbb} : i64")
        emit(f"          %ubt_{sfx}_{t} = pto.castptr %tbb_{sfx}_{t} : i64 -> !pto.ptr<f16, ub>")
        emit_tile(f"ubt_{sfx}_{t}", 0, f"_{sfx}")
    emit(f'          pto.mem_bar "VST_VLD"')
    emit(f"        }}")
    if rotations:
        emit(f"        // peeled final stage: ROT={rotations} vdintlv window rotation fused")
        for t in range(TILES):
            tbb = bufbase + t * tilestride_bytes
            emit(f"        %rtbb_{sfx}_{t} = arith.constant {tbb} : i64")
            emit(f"        %rubt_{sfx}_{t} = pto.castptr %rtbb_{sfx}_{t} : i64 -> !pto.ptr<f16, ub>")
            emit_tile(f"rubt_{sfx}_{t}", rotations, f"_{sfx}")
        emit(f'        pto.mem_bar "VST_VLD"')
    emit(f"      }}")


def emit_gx(ci_reg, sfx):
    """gx_<sfx> = x + (bbase + ci*CHUNK_ELEMS) elems  (ci_reg: index)."""
    emit(f"      %coff_{sfx} = arith.muli %{ci_reg}, %crn_idx : index")
    emit(f"      %cbase_{sfx} = arith.addi %bbase, %coff_{sfx} : index")
    emit(f"      %gx_{sfx} = pto.addptr %x, %cbase_{sfx} : !pto.ptr<f16, gm> -> !pto.ptr<f16, gm>")


def emit_load(gx, buf):
    """wait buffer free, load CR rows GM->UB into buffer `buf`, mark loaded."""
    ub = f"ubuf{buf}"
    emit(f'      pto.wait_flag["PIPE_MTE3", "PIPE_MTE2", "{ev(buf)}"]')
    emit(f"      pto.mte_gm_ub %{gx}, %{ub}, %c0_i64, %clen_i64")
    emit(f"        nburst(%rows_i64, %clen_i64, %clen_i64)")
    emit(f"        : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64")
    emit(f'      pto.set_flag["PIPE_MTE2", "PIPE_V", "{ev(buf)}"]')


def emit_store(buf, gx):
    ub = f"ubuf{buf}"
    emit(f"      pto.mte_ub_gm %{ub}, %{gx}, %clen_i64")
    emit(f"        nburst(%rows_i64, %clen_i64, %clen_i64)")
    emit(f"        : !pto.ptr<f16, ub>, !pto.ptr<f16, gm>, i64, i64, i64, i64")


def emit_substep(ci_c_reg, buf, ci_f_reg, buf_f, set_free):
    """Compute chunk ci_c in buffer `buf`, first forward-prefetching chunk ci_f into
    buffer `buf_f` (skip the prefetch if ci_f_reg is None)."""
    sfx = f"b{buf}"
    emit_gx(ci_c_reg, f"{sfx}c")
    if ci_f_reg is not None:
        emit_gx(ci_f_reg, f"{sfx}f")
        emit(f"      // prefetch chunk into buf{buf_f} (overlaps compute of buf{buf})")
        emit_load(f"gx_{sfx}f", buf_f)
    emit(f'      pto.wait_flag["PIPE_MTE2", "PIPE_V", "{ev(buf)}"]')
    emit_compute(f"{sfx}c", buf)
    emit(f'      pto.set_flag["PIPE_V", "PIPE_MTE3", "{ev(buf)}"]')
    emit(f'      pto.wait_flag["PIPE_V", "PIPE_MTE3", "{ev(buf)}"]')
    emit_store(buf, f"gx_{sfx}c")
    if set_free:
        emit(f'      pto.set_flag["PIPE_MTE3", "PIPE_MTE2", "{ev(buf)}"]')


emit(f"""// Fast Walsh-Hadamard transform (VMI raw .pto), N={N}, CCE-PARITY kernel:
// MULTI-CORE + {NBUF}-buffer deep pipeline (prefetch depth {PREFETCH}), 32KB UB
// tiles, TILE LOOP UNROLLED (immediate addressing = the CCE-parity fix). fp16,
// in-place y = x @ H (unnormalized). AUTO-GENERATED by gen_had_full.py.
// PER-N LAYOUT: R(pack)={R} chunks={chunks} rot={rotations} group={group}
//   plain={plain} log2N={LOG2N} lanes={lanes} (packing N<256 / chunking N>256).
// Reproduces CCE (DEF_BUFFERS={NBUF}, DEF_PREFETCH={PREFETCH}, TILE_BYTES={CHUNK_BYTES}=32KB,
// block_dim=G). GRID-AGNOSTIC multi-core: block b streams row band [b*rpb,(b+1)*rpb)
// (rpb=batch/G) in CR={CR}-row chunks (CHUNK={CHUNK_ELEMS} elems) through a {NBUF}-buffer
// UB ring, unrolled-by-{NBUF}. TILES={TILES}/chunk, 8 units/tile (UNITS={UNITS}).
// REQUIRES batch % (G*{FOUR_CR}) == 0. batch_hint={BATCH_HINT} (annotation only).
module attributes {{
  pto.target_arch = "a5",
  pto.kernel_kind = #pto.kernel_kind<vector>
}} {{
  func.func @{FUNC}(
      %x: !pto.ptr<f16, gm>, %batch: i32, %n: i32, %log2n: i32)
      attributes {{pto.kernel}} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c0_i64 = arith.constant 0 : i64
    %clen_i64 = arith.constant {clen} : i64
    %c128_i32 = arith.constant 128 : i32
    %cr_idx = arith.constant {CR} : index
    %n_idx = arith.constant {N} : index
    %crn_idx = arith.constant {CHUNK_ELEMS} : index
    %rows_i64 = arith.constant {CR} : i64
    %buf0_i64 = arith.constant 0 : i64
    %buf1_i64 = arith.constant {1 * CHUNK_BYTES} : i64
    %buf2_i64 = arith.constant {2 * CHUNK_BYTES} : i64
    %buf3_i64 = arith.constant {3 * CHUNK_BYTES} : i64
    %ubuf0 = pto.castptr %buf0_i64 : i64 -> !pto.ptr<f16, ub>
    %ubuf1 = pto.castptr %buf1_i64 : i64 -> !pto.ptr<f16, ub>
    %ubuf2 = pto.castptr %buf2_i64 : i64 -> !pto.ptr<f16, ub>
    %ubuf3 = pto.castptr %buf3_i64 : i64 -> !pto.ptr<f16, ub>

    // ---- runtime block geometry (a5 multi-core idiom, PA split-KV style) ----
    %bidx_i64 = pto.get_block_idx
    %gnum_i64 = pto.get_block_num
    %bidx = arith.index_cast %bidx_i64 : i64 to index
    %gnum = arith.index_cast %gnum_i64 : i64 to index
    %batch_idx = arith.index_cast %batch : i32 to index

    // rpb = batch / G   (REQUIRES batch % (G*{FOUR_CR}) == 0)
    %rpb = arith.divui %batch_idx, %gnum : index
    // nchunk_block = rpb / CR   (multiple of 4)
    %nchunk = arith.divui %rpb, %cr_idx : index
    // nquad = nchunk_block / 4 ; steady-state trip = nquad - 1
    %nquad = arith.divui %nchunk, %c4 : index
    %trip = arith.subi %nquad, %c1 : index
    // block base (elements) = bidx * rpb * N
    %brow = arith.muli %bidx, %rpb : index
    %bbase = arith.muli %brow, %n_idx : index
    // epilogue chunk indices (last 4 chunks of the block)
    %nchunk_m1 = arith.subi %nchunk, %c1 : index
    %nchunk_m2 = arith.subi %nchunk, %c2 : index
    %nchunk_m3 = arith.subi %nchunk, %c3 : index
    %nchunk_m4 = arith.subi %nchunk, %c4 : index

    // ---- prologue: mark all {NBUF} buffers free, prefetch chunk0->buf0, chunk1->buf1 ----
    pto.set_flag["PIPE_MTE3", "PIPE_MTE2", "{ev(0)}"]
    pto.set_flag["PIPE_MTE3", "PIPE_MTE2", "{ev(1)}"]
    pto.set_flag["PIPE_MTE3", "PIPE_MTE2", "{ev(2)}"]
    pto.set_flag["PIPE_MTE3", "PIPE_MTE2", "{ev(3)}"]""")

emit("    // gx for prologue chunks 0 and 1")
emit(f"    %pg0 = pto.addptr %x, %bbase : !pto.ptr<f16, gm> -> !pto.ptr<f16, gm>")
emit(f"    %p1off = arith.constant {CHUNK_ELEMS} : index")
emit(f"    %p1base = arith.addi %bbase, %p1off : index")
emit(f"    %pg1 = pto.addptr %x, %p1base : !pto.ptr<f16, gm> -> !pto.ptr<f16, gm>")
emit(f'    pto.wait_flag["PIPE_MTE3", "PIPE_MTE2", "{ev(0)}"]')
emit(f"    pto.mte_gm_ub %pg0, %ubuf0, %c0_i64, %clen_i64")
emit(f"      nburst(%rows_i64, %clen_i64, %clen_i64)")
emit(f"      : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64")
emit(f'    pto.set_flag["PIPE_MTE2", "PIPE_V", "{ev(0)}"]')
emit(f'    pto.wait_flag["PIPE_MTE3", "PIPE_MTE2", "{ev(1)}"]')
emit(f"    pto.mte_gm_ub %pg1, %ubuf1, %c0_i64, %clen_i64")
emit(f"      nburst(%rows_i64, %clen_i64, %clen_i64)")
emit(f"      : !pto.ptr<f16, gm>, !pto.ptr<f16, ub>, i64, i64, i64, i64, i64")
emit(f'    pto.set_flag["PIPE_MTE2", "PIPE_V", "{ev(1)}"]')
emit("")
emit("    // ---- steady state: nquad-1 quads, each processes chunks 4q..4q+3 ----")
emit("    scf.for %q = %c0 to %trip step %c1 {")
emit("      %q4 = arith.muli %q, %c4 : index")
emit("      %q4p1 = arith.addi %q4, %c1 : index")
emit("      %q4p2 = arith.addi %q4, %c2 : index")
emit("      %q4p3 = arith.addi %q4, %c3 : index")
emit("      %q4p4 = arith.addi %q4, %c4 : index")
emit("      %q4p5 = arith.addi %q4, %c5 : index")
emit_substep("q4",   0, "q4p2", 2, set_free=True)
emit_substep("q4p1", 1, "q4p3", 3, set_free=True)
emit_substep("q4p2", 2, "q4p4", 0, set_free=True)
emit_substep("q4p3", 3, "q4p5", 1, set_free=True)
emit("    }")
emit("")
emit("    // ---- epilogue: final quad (chunks nchunk-4..nchunk-1), no forward prefetch past end ----")
emit_substep("nchunk_m4", 0, "nchunk_m2", 2, set_free=False)
emit_substep("nchunk_m3", 1, "nchunk_m1", 3, set_free=False)
emit_substep("nchunk_m2", 2, None, None, set_free=False)
emit_substep("nchunk_m1", 3, None, None, set_free=False)
emit("")
emit("    pto.barrier #pto.pipe<PIPE_ALL>")
emit("    return")
emit("  }")
emit("}")

open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT}: func={FUNC} N={N} CCE-PARITY {NBUF}-buf prefetch{PREFETCH} "
      f"CR={CR} CHUNK={CHUNK_ELEMS}({CHUNK_BYTES//1024}KB/buf,{NBUF*CHUNK_BYTES//1024}KB UB) "
      f"TILES={TILES} units/tile=8 group={group} R={R} chunks={chunks} rot={rotations} "
      f"plain={plain} log2N={LOG2N}; REQUIRES batch%(G*{FOUR_CR})==0")
