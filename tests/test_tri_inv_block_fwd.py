"""Standalone extraction of mega-GDN's 128x128 blocked inverse."""

import pytest
import torch
from pto_kernels import pto_tri_inv_block_fwd


@pytest.mark.parametrize("bsnd", [False, True])
@pytest.mark.parametrize("tiles,heads", [(1, 1), (3, 2), (25, 4)])
@pytest.mark.parametrize("scale", [0.0, 0.1, 0.3])
def test_block_fwd(npu_device, bsnd, tiles, heads, scale):
    torch.manual_seed(1024)
    x = (scale * torch.randn(tiles, heads, 128, 128)).half()
    # Nonzero diagonal/upper entries must be ignored.
    matrix = torch.eye(128) + torch.tril(x.float(), diagonal=-1)
    reference = torch.linalg.solve_triangular(
        matrix.double(),
        torch.eye(128).double().expand_as(matrix),
        upper=False,
        unitriangular=True,
    ).float()
    inp = (
        x.transpose(1, 2).contiguous().reshape(1, tiles * 128, heads, 128)
        if bsnd
        else x
    )
    result = pto_tri_inv_block_fwd(inp.to(npu_device), is_bsnd_format=bsnd).cpu()
    if bsnd:
        result = result.reshape(tiles, 128, heads, 128).transpose(1, 2)
    # Use per-matrix absolute scaling for ill-conditioned triangular inputs.
    error = (result.float() - reference).abs()
    assert torch.all(error <= 1e-3 * reference.abs().amax((-2, -1), keepdim=True))
    assert torch.equal(
        result.diagonal(dim1=-2, dim2=-1), torch.ones_like(result[..., 0])
    )
    assert torch.count_nonzero(torch.triu(result, diagonal=1)) == 0


def test_block_fwd_validation(npu_device):
    for shape, dtype, bsnd in [
        ((16, 16), torch.float16, False),
        ((128, 128), torch.float32, False),
        ((1, 129, 2, 128), torch.float16, True),
    ]:
        with pytest.raises(RuntimeError, match="tri_inv_block_fwd"):
            pto_tri_inv_block_fwd(
                torch.empty(shape, dtype=dtype, device=npu_device), bsnd
            )
    empty = torch.empty((0, 128, 128), dtype=torch.float16, device=npu_device)
    assert pto_tri_inv_block_fwd(empty).shape == empty.shape
    with pytest.raises(RuntimeError, match="contiguous"):
        pto_tri_inv_block_fwd(
            torch.empty((128, 128), dtype=torch.float16, device=npu_device).T
        )
