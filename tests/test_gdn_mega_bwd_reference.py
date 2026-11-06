# Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
"""CPU backward-reference checks.

Run: python -m pytest --noconftest tests/test_gdn_mega_bwd_reference.py -q
"""

import pytest
import torch

from gdn_mega_bwd_reference import recurrent_gdn_reference


def make_inputs(tokens=4, sequences=1, dtype=torch.float64):
    rng = torch.Generator().manual_seed(42)

    def randn(*shape):
        return torch.randn(*shape, generator=rng, dtype=dtype) * 0.2

    # Small unequal K/V dimensions and shared Q/K heads make gradcheck cheap.
    values = (
        randn(1, tokens, 1, 2),
        randn(1, tokens, 1, 2),
        randn(1, tokens, 2, 3),
        -torch.rand(1, tokens, 2, generator=rng, dtype=dtype),
        torch.rand(1, tokens, 2, generator=rng, dtype=dtype),
        randn(sequences, 2, 2, 3),
    )
    return tuple(x.requires_grad_() for x in values)


@pytest.mark.parametrize("with_initial_state", [False, True])
def test_gradcheck(with_initial_state):
    inputs = make_inputs(sequences=4)
    cu = torch.tensor([0, 0, 1, 4, 4])
    if not with_initial_state:
        inputs = inputs[:5]
    # Finite differences independently validate every input and both outputs.
    assert torch.autograd.gradcheck(
        lambda *xs: recurrent_gdn_reference(*xs, cu_seqlens=cu, scale=0.7),
        inputs,
    )


@pytest.mark.parametrize("loss_kind", ["output", "state", "both"])
def test_backward_across_call_boundary(loss_kind):
    whole_inputs = make_inputs(tokens=5)
    split_inputs = tuple(x.detach().clone().requires_grad_() for x in whole_inputs)
    whole_o, whole_h = recurrent_gdn_reference(*whole_inputs)
    q, k, v, g, beta, h0 = split_inputs
    first_o, first_h = recurrent_gdn_reference(
        *(x[:, :2] for x in (q, k, v, g, beta)), h0
    )
    second_o, split_h = recurrent_gdn_reference(
        *(x[:, 2:] for x in (q, k, v, g, beta)), first_h
    )
    split_o = torch.cat((first_o, second_o), dim=1)
    torch.testing.assert_close(split_o, whole_o)
    torch.testing.assert_close(split_h, whole_h)

    rng = torch.Generator().manual_seed(17)
    do = torch.randn(whole_o.shape, generator=rng, dtype=torch.float64)
    dh = torch.randn(whole_h.shape, generator=rng, dtype=torch.float64)
    for out, state in ((whole_o, whole_h), (split_o, split_h)):
        loss = 0
        if loss_kind != "state":
            loss = loss + (out * do).sum()
        if loss_kind != "output":
            loss = loss + (state * dh).sum()
        loss.backward()
    for whole, split in zip(whole_inputs, split_inputs):
        if whole.grad is None:  # Queries have no effect on the final state.
            assert split.grad is None
        else:
            torch.testing.assert_close(split.grad, whole.grad)
    assert torch.count_nonzero(whole_inputs[1].grad[:, :2]) > 0


def test_packed_sequences_are_independent():
    inputs = make_inputs(tokens=5, sequences=3)
    q, k, v, g, beta, h0 = inputs
    cu = torch.tensor([0, 2, 2, 5])
    out, ht = recurrent_gdn_reference(*inputs, cu_seqlens=cu)
    for seq, (start, end) in enumerate(zip(cu.tolist(), cu.tolist()[1:])):
        seq_out, seq_ht = recurrent_gdn_reference(
            *(x[:, start:end] for x in (q, k, v, g, beta)), h0[seq : seq + 1]
        )
        torch.testing.assert_close(out[:, start:end], seq_out)
        torch.testing.assert_close(ht[seq : seq + 1], seq_ht)
    # A loss on the last sequence must not leak into preceding sequences.
    out[:, 2:].square().sum().backward()
    for x in inputs[:5]:
        assert torch.count_nonzero(x.grad[:, :2]) == 0
    assert torch.count_nonzero(h0.grad[:2]) == 0


def test_empty_sequences():
    inputs = make_inputs(tokens=0, sequences=2)
    out, ht = recurrent_gdn_reference(*inputs, cu_seqlens=torch.tensor([0, 0, 0]))
    assert out.shape == (1, 0, 2, 3)
    torch.testing.assert_close(ht, inputs[-1])
    (out.sum() + ht.sum()).backward()
    for x in inputs[:5]:
        assert x.grad is not None and x.grad.numel() == 0
    torch.testing.assert_close(inputs[-1].grad, torch.ones_like(inputs[-1]))


def test_conversions_preserve_gradients_and_inputs():
    inputs = make_inputs(dtype=torch.float32)
    snapshots = [x.detach().clone() for x in inputs]
    out, ht = recurrent_gdn_reference(*inputs, scale=2**-0.5)
    assert out.dtype == ht.dtype == torch.float64
    (out.square().sum() + ht.square().sum()).backward()
    for x, snapshot in zip(inputs, snapshots):
        torch.testing.assert_close(x, snapshot, rtol=0, atol=0)
        assert x.grad is not None and x.grad.dtype == x.dtype
        assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("tokens", [0, 5])
def test_reordered_reference_matches_equations(tokens):
    inputs = make_inputs(tokens=tokens, sequences=3)
    cu = torch.tensor([0, 0, tokens // 2, tokens])
    golden = recurrent_gdn_reference(*inputs, cu_seqlens=cu, scale=0.7)
    reordered = recurrent_gdn_reference(
        *inputs, cu_seqlens=cu, scale=0.7, reorder_ops=True
    )
    upstream = tuple(torch.randn_like(x) for x in golden)
    for a, b in zip(golden, reordered):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    for a, b in zip(
        torch.autograd.grad(golden, inputs, upstream),
        torch.autograd.grad(reordered, inputs, upstream),
    ):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
