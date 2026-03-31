# Import torch is required to avoid "libc10.so: cannot open shared object file: No such file or directory"
# See https://github.com/facebookresearch/pytorch3d/issues/1531#issuecomment-1538198217
import torch  # noqa
from collections import namedtuple

from .benchmarking import do_bench  # noqa
from .pto_kernels_ops import *  # noqa

# Named-tuple matching torch.return_types.histogram
_HistogramResult = namedtuple("histogram", ["hist", "bin_edges"])

# Tile length imposed by the PTO-ISA histogram kernel (must stay in sync with
# HISTOGRAM_TILE_LEN defined in csrc/kernel/kernel_histogram.cpp).
_HISTOGRAM_TILE_LEN = 64


def pto_histogram(input, bins=100, *, range=None, weight=None, density=False):
    """Compute a histogram of ``input`` on Ascend NPU.

    The interface matches :func:`torch.histogram` so that the two functions
    can be used interchangeably (subject to the hardware constraints listed
    below).

    Args:
        input (Tensor): Input tensor of dtype ``torch.float16`` or
            ``torch.float32``.  May be multi-dimensional; it is flattened
            internally.
        bins (int or Tensor): If an ``int``, specifies the number of
            equal-width bins.  If a 1-D ``Tensor``, its values are used
            directly as the monotonically increasing bin edges (the first and
            last values define ``range_min`` and ``range_max``).
        range (tuple[float, float], optional): ``(min, max)`` of the histogram
            range.  Values that fall outside this range are clamped into the
            nearest edge bin.  If *None*, the range is inferred from
            ``input.min()`` and ``input.max()``.  Ignored when *bins* is a
            Tensor.
        weight (Tensor, optional): Not yet supported; raises
            :class:`NotImplementedError` if provided.
        density (bool): If ``True``, the returned histogram is normalised so
            that the integral over the range equals 1.

    Returns:
        histogram (namedtuple): A named tuple with fields

        - **hist** (*Tensor*) – 1-D float32 tensor of length ``bins``
          containing the (optionally normalised) element counts.
        - **bin_edges** (*Tensor*) – 1-D float32 tensor of length
          ``bins + 1`` containing the bin edges.

    Note:
        The PTO-ISA kernel requires ``input.numel()`` to be a multiple of
        ``64``.  Inputs whose element count is not a multiple of 64 are
        zero-padded to the next multiple before the kernel is called; the
        extra counts introduced by padding are subtracted afterwards.
    """
    if weight is not None:
        raise NotImplementedError(
            "pto_histogram does not currently support the 'weight' parameter."
        )

    # ---- Resolve bins and range ----
    if isinstance(bins, torch.Tensor):
        bin_edges_tensor = bins.cpu().float()
        if bin_edges_tensor.dim() != 1 or bin_edges_tensor.numel() < 2:
            raise ValueError(
                "When 'bins' is a Tensor it must be a 1-D tensor with at "
                "least 2 elements."
            )
        n_bins = int(bin_edges_tensor.numel()) - 1
        range_min = float(bin_edges_tensor[0])
        range_max = float(bin_edges_tensor[-1])
    else:
        n_bins = int(bins)
        if range is None:
            range_min = float(input.min())
            range_max = float(input.max())
            # Widen the range by a small epsilon so the maximum value falls
            # strictly inside the last bin – same heuristic used by NumPy /
            # torch.histogram.
            if range_min == range_max:
                range_min -= 0.5
                range_max += 0.5
            else:
                range_max = range_max + (range_max - range_min) * 1e-6
        else:
            range_min, range_max = float(range[0]), float(range[1])

    # ---- Pad input to a multiple of the tile length ----
    flat = input.flatten()
    n = int(flat.numel())
    remainder = n % _HISTOGRAM_TILE_LEN
    pad_len = (_HISTOGRAM_TILE_LEN - remainder) if remainder != 0 else 0

    if pad_len > 0:
        # Pad with range_min - 1.0 so pad elements are clamped to bin 0 and
        # can be corrected afterwards.
        pad_val = range_min - 1.0
        padding = torch.full(
            (pad_len,), fill_value=pad_val, dtype=flat.dtype, device=flat.device
        )
        flat = torch.cat([flat, padding])

    # ---- Run NPU kernel ----
    hist, bin_edges = pto_histogram_op(flat, n_bins, range_min, range_max)

    # Subtract counts introduced by padding (all land in bin 0 due to clamping)
    if pad_len > 0:
        hist = hist.clone()
        hist[0] = hist[0] - pad_len

    # ---- Handle bins-as-Tensor: return the user-supplied bin edges ----
    if isinstance(bins, torch.Tensor):
        bin_edges = bins.to(dtype=torch.float32, device=hist.device)

    # ---- Optional density normalisation ----
    if density:
        total = float(n)
        bin_widths = (bin_edges[1:] - bin_edges[:-1]).to(hist.device)
        hist = hist / (total * bin_widths)

    return _HistogramResult(hist=hist, bin_edges=bin_edges)
