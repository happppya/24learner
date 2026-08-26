import pytest
import torch

from learner.network import SetTransformer24, top_k_action_mask


@pytest.mark.parametrize("num_elems", [2, 5, 10])
def test_forward_handles_variable_set_sizes(num_elems):
    torch.manual_seed(0)
    net = SetTransformer24(dim=32, num_inducing=4, num_layers=1)
    values = torch.randn(3, num_elems)
    depths = torch.randint(0, 4, (3, num_elems))
    target = torch.tensor([24.0, 7.5, -13.0])
    out = net(values, depths, target)
    assert out.binary_logits.shape == (3, num_elems, num_elems, 5)
    assert out.unary_logits.shape == (3, num_elems, 3)
    assert out.value.shape == (3,)
    assert out.value.abs().max() <= 1.0


def test_padded_positions_are_masked():
    torch.manual_seed(0)
    net = SetTransformer24(dim=32, num_inducing=4, num_layers=1)
    values = torch.zeros(1, 6)
    depths = torch.zeros(1, 6, dtype=torch.long)
    pad_mask = torch.tensor([[False] * 3 + [True] * 3])
    out = net(values, depths, torch.tensor([24.0]), pad_mask=pad_mask)
    assert torch.isinf(out.binary_logits[0, :3, 3:, :]).all()
    assert torch.isinf(out.unary_logits[0, 3:, :]).all()
    assert torch.isfinite(out.binary_logits[0, :3, :3, :]).all()


def test_top_k_mask_selects_exactly_k_when_available():
    binary = torch.zeros(2, 4, 4, 5)
    unary = torch.zeros(2, 4, 3)
    mask_bin, mask_un = top_k_action_mask(binary, unary, k=16)
    assert int(mask_bin.sum()) + int(mask_un.sum()) == 32


def test_top_k_mask_respects_finite_entries_only():
    binary = torch.full((1, 3, 3, 5), float("-inf"))
    unary = torch.full((1, 3, 3), float("-inf"))
    binary[0, 0, 1, 2] = 1.0
    mask_bin, mask_un = top_k_action_mask(binary, unary, k=16)
    assert int(mask_bin.sum()) + int(mask_un.sum()) == 1
    assert bool(mask_bin[0, 0, 1, 2])


def test_backward_pass_populates_gradients():
    torch.manual_seed(0)
    net = SetTransformer24(dim=16, num_inducing=4, num_layers=1)
    out = net(torch.randn(2, 4), torch.zeros(2, 4, dtype=torch.long), torch.tensor([24.0, 5.0]))
    loss = (
        out.value.sum()
        + out.binary_logits.flatten(1).logsumexp(dim=-1).sum()
        + out.unary_logits.flatten(1).logsumexp(dim=-1).sum()
    )
    loss.backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing
