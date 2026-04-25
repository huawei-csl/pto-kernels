import pytest
import torch
import torch_npu  # noqa

from pto_kernels import pto_swiglu

pytestmark = pytest.mark.npu

DTYPE = torch.float16
TEST_SEEDS = [0, 1]
TEST_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
TEST_HIDDEN_DIMS = [128, 256, 512, 1024, 4096, 6144, 16384, 32768, 65536, 262144]
TEST_LARGE_HIDDEN_DIM_MIN = 16384
TEST_LARGE_HIDDEN_DIM_MAX_BATCH = 64
TEST_CASES = [
    (batch, hidden_dim)
    for batch in TEST_BATCHES
    for hidden_dim in TEST_HIDDEN_DIMS
    if hidden_dim < TEST_LARGE_HIDDEN_DIM_MIN
    or batch <= TEST_LARGE_HIDDEN_DIM_MAX_BATCH
]


def swiglu_ref(x):
    gate, up = torch.chunk(x.float(), 2, dim=-1)
    return (gate * torch.sigmoid(gate) * up).to(x.dtype)


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize("batch,hidden_dim", TEST_CASES)
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
