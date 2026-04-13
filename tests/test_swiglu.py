import pytest
import torch
import torch_npu  # noqa

from pto_kernels import pto_swiglu


DTYPE = torch.float16


def swiglu_ref(x):
    gate, up = torch.chunk(x.float(), 2, dim=-1)
    return (gate * torch.sigmoid(gate) * up).to(x.dtype)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("batch", [1, 8, 128])
@pytest.mark.parametrize("hidden_dim", [128, 1024, 4096, 16384])
def test_pto_swiglu_matches_reference_and_torch_npu(
    npu_device,
    seed,
    batch,
    hidden_dim,
):
    torch.manual_seed(seed)
    x = torch.randn(batch, hidden_dim * 2, device=npu_device, dtype=DTYPE)

    actual = pto_swiglu(x)
    expected = swiglu_ref(x.cpu())
    torch_npu_expected = torch_npu.npu_swiglu(x, dim=-1)

    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-5)
    torch.testing.assert_close(actual, torch_npu_expected, rtol=1e-2, atol=1e-5)


def test_pto_swiglu_rejects_non_last_dim(npu_device):
    x = torch.randn(2, 256, device=npu_device, dtype=DTYPE)

    with pytest.raises(RuntimeError, match="dim=-1"):
        pto_swiglu(x, dim=0)
