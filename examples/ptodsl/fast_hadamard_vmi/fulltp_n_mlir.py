"""Parametric source-JIT wrapper for the `fulltp` Walsh-Hadamard kernel.

`fulltp` is the flexible-and-fast variant: it keeps `full`'s contiguous per-core
row bands + 4-buffer software pipeline (which beats CCE's grid-stride on device),
but relaxes the batch-alignment from `full`'s `batch % (G*4*CR) == 0` to
`batch % (G*CR) == 0` (4x finer, any per-band chunk count nchunk >= 1) via a
partial-quad tail. The tail uses an OVERLAPPED ring free[0] handshake (no PIPE_ALL
drain), so it stays at CCE cycle-parity even on the remainder chunks.

gen_had_fulltp.py emits one .pto per N, uniformly named
fast_hadamard_vmi_fulltp_n{N}.pto with func @fast_hadamard_vmi_fulltp_n{N}.
make_fulltp(N) returns the matching @pto.jit handle (cached). ptodsl caches on the
.pto content digest and resolves the relative `source=` path against this file's
directory (so the .pto must sit beside this module).
"""

from functools import lru_cache

from ptodsl import pto

# gm-space fp16 pointer type; hoisted to a name so the "gm" string is not in
# annotation position (pyflakes would treat it as a forward reference).
_GM_F16 = pto.ptr(pto.f16, "gm")

# All supported transform widths. N=32..2048 are correctness-verified (rel_err
# ~1e-3). N=32 needs ROT=3 vdintlv window rotations and works (there is NO
# net-rotation cap); N=16 (ROT=4) is untested and left out.
SUPPORTED_NS = (32, 64, 128, 256, 512, 1024, 2048)


@lru_cache(maxsize=None)
def make_fulltp(n: int):
    """Return the compiled-on-demand @pto.jit handle for the fulltp kernel at N=n."""
    if n not in SUPPORTED_NS:
        raise ValueError(
            f"fulltp: N={n} unsupported (choose {SUPPORTED_NS}; N=16 (ROT=4) is "
            f"untested and N>2048 needs >=8 chunks/row -- see gen_had_fulltp.py)"
        )
    name = f"fast_hadamard_vmi_fulltp_n{n}"
    src = f"fast_hadamard_vmi_fulltp_n{n}.pto"

    @pto.jit(
        name=name,
        target="a5",
        backend="vpto",
        mode="explicit",
        insert_sync=False,
        source=src,
    )
    def _fulltp(
        x: _GM_F16, batch: pto.i32, n: pto.i32, log2n: pto.i32
    ):  # pylint: disable=unused-argument
        pass

    return _fulltp
