import pytest
import torch
import torch_npu  # noqa: F401

from chunk_o import chunk_o, ref_chunk_o


DTYPE = torch.float16
RTOL = 1e-2


def make_inputs(b: int, h: int, l: int, d: int):
    q = torch.randn((b, h, l, d), device="npu", dtype=DTYPE)
    k = torch.randn((b, h, l, d), device="npu", dtype=DTYPE)
    v = torch.randn((b, h, l, d), device="npu", dtype=DTYPE)
    q = q / (q.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    k = k / (k.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)
    return q, k, v


def pick_atol(seq_len: int) -> float:
    if seq_len >= 4096:
        return 4e-2
    if seq_len >= 2048:
        return 2e-2
    return 1e-2


@pytest.mark.parametrize(
    ("b", "h", "l", "d", "c"),
    [
        (1, 2, 64, 128, 64),
        (1, 2, 256, 128, 64),
        (4, 2, 128, 128, 64),
        (8, 2, 512, 128, 64),
        (1, 2, 300, 128, 64),
        (2, 2, 153, 64, 64),
        (4, 4, 257, 128, 128),
    ],
)
def test_chunk_o_forward_matches_reference(b: int, h: int, l: int, d: int, c: int):
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")

    q, k, v = make_inputs(b, h, l, d)
    out = chunk_o(q, k, v, chunk_size=c)
    ref = ref_chunk_o(q, k, v, chunk_size=c)

    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=RTOL, atol=pick_atol(l))


@pytest.mark.parametrize(
    ("b", "h", "l", "d", "c"),
    [
        (1, 2, 65, 64, 64),
        (2, 4, 192, 128, 64),
    ],
)
def test_chunk_o_final_state_matches_reference(
    b: int, h: int, l: int, d: int, c: int
):
    torch.manual_seed(1)
    torch.npu.set_device("npu:0")

    q, k, v = make_inputs(b, h, l, d)
    h0 = torch.randn((b, h, d, d), device="npu", dtype=torch.float32)

    out, final_state = chunk_o(
        q, k, v, chunk_size=c, initial_state=h0, output_final_state=True
    )
    ref_out, ref_final_state = ref_chunk_o(
        q,
        k,
        v,
        chunk_size=c,
        initial_state=h0,
        output_final_state=True,
    )

    torch.npu.synchronize()
    atol = pick_atol(l)
    torch.testing.assert_close(out.cpu(), ref_out.cpu(), rtol=RTOL, atol=atol)
    torch.testing.assert_close(
        final_state.cpu(), ref_final_state.cpu(), rtol=RTOL, atol=atol
    )
