"""Parametric source-JIT wrapper for the parametric-N `full` Walsh-Hadamard kernel.

gen_had_full.py emits one .pto per N, uniformly named
fast_hadamard_vmi_full_n{N}.pto with func @fast_hadamard_vmi_full_n{N}.
make_full(N) returns the matching @pto.jit handle (cached), so the runner can
build/launch the per-N kernel through one path. ptodsl caches on the .pto content
digest and resolves the relative `source=` path against this file's directory (so
the .pto must sit beside this module)."""
from functools import lru_cache

from ptodsl import pto

# All supported transform widths. N=32..2048 are correctness-verified under
# cannsim (rel_err ~1e-3). N=32 needs ROT=3 vdintlv window rotations and works
# (there is NO net-rotation cap); N=16 (ROT=4) is untested and left out.
SUPPORTED_NS = (32, 64, 128, 256, 512, 1024, 2048)


@lru_cache(maxsize=None)
def make_full(n: int):
    """Return the compiled-on-demand @pto.jit handle for the full kernel at N=n."""
    if n not in SUPPORTED_NS:
        raise ValueError(
            f"full_n: N={n} unsupported (choose {SUPPORTED_NS}; N=16 (ROT=4) is "
            f"untested and N>2048 needs >=8 chunks/row -- see gen_had_full.py)")
    name = f"fast_hadamard_vmi_full_n{n}"
    src = f"fast_hadamard_vmi_full_n{n}.pto"

    @pto.jit(name=name, target="a5", backend="vpto", mode="explicit",
             insert_sync=False, source=src)
    def _full(x: pto.ptr(pto.f16, "gm"), batch: pto.i32, n: pto.i32, log2n: pto.i32):
        pass

    return _full
