import math

import pytest
import torch

from learner.network import BINARY_OPS, UNARY_OPS, SetTransformer24, top_k_action_mask, top_k_priors


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


def test_padding_does_not_perturb_real_positions():
    torch.manual_seed(0)
    net = SetTransformer24(dim=32, num_inducing=4, num_layers=1)
    values = torch.tensor([[1.0, 2.0, 3.0]])
    depths = torch.zeros(1, 3, dtype=torch.long)
    target = torch.tensor([24.0])
    clean = net(values, depths, target)

    padded = net(
        torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0]]),
        torch.zeros(1, 5, dtype=torch.long),
        target,
        pad_mask=torch.tensor([[False, False, False, True, True]]),
    )
    assert torch.allclose(padded.binary_logits[0, :3, :3, :], clean.binary_logits[0], atol=1e-6)
    assert torch.allclose(padded.unary_logits[0, :3, :], clean.unary_logits[0], atol=1e-6)
    assert torch.allclose(padded.value, clean.value, atol=1e-6)


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


def reference_priors(bin_logits, un_logits, legal, k):
    scores = []
    for action in legal:
        kind, i, j, op = action
        if kind == "binary":
            scores.append(bin_logits[i, j, BINARY_OPS.index(op)].item())
        else:
            scores.append(un_logits[i, UNARY_OPS.index(op)].item())
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
    selected = [scores[i] for i in order]
    shifted = [math.exp(s - max(selected)) for s in selected]
    total = sum(shifted)
    return [(legal[i], shifted[idx] / total) for idx, i in enumerate(order)]


def test_top_k_priors_match_reference():
    rng = torch.Generator().manual_seed(7)
    bin_logits = torch.randn(4, 4, 5, generator=rng)
    un_logits = torch.randn(4, 3, generator=rng)
    legal = [
        ("binary", 0, 1, "add"),
        ("binary", 0, 1, "mul"),
        ("binary", 2, 3, "pow"),
        ("binary", 1, 2, "div"),
        ("unary", 0, 0, "sqrt"),
        ("unary", 1, 0, "ln"),
        ("unary", 2, 0, "neg"),
    ]
    for k in (1, 3, 16):
        got = top_k_priors(bin_logits, un_logits, legal, k=k)
        want = reference_priors(bin_logits, un_logits, legal, k=k)
        assert [a for a, _ in got] == [a for a, _ in want]
        assert [p for _, p in got] == pytest.approx([p for _, p in want], abs=1e-6)


def test_top_k_priors_normalized_and_ranked():
    bin_logits = torch.zeros(3, 3, 5)
    un_logits = torch.zeros(3, 3)
    legal = [("unary", i, 0, op) for i in range(3) for op in UNARY_OPS]
    priors = top_k_priors(bin_logits, un_logits, legal, k=2)
    assert len(priors) == 2
    assert sum(p for _, p in priors) == pytest.approx(1.0, abs=1e-6)
    assert all(p >= 0.0 for _, p in priors)
    assert priors[0][1] >= priors[1][1]


def test_top_k_priors_dominant_action_wins():
    bin_logits = torch.zeros(3, 3, 5)
    un_logits = torch.zeros(3, 3)
    un_logits[0, 2] = 50.0  # ln on elem 0 is the clear favorite
    legal = [("binary", 0, 1, "add"), ("unary", 0, 0, "ln")]
    priors = top_k_priors(bin_logits, un_logits, legal, k=2)
    assert priors[0][0] == ("unary", 0, 0, "ln")
    assert priors[0][1] == pytest.approx(1.0, abs=1e-6)


def test_top_k_priors_drop_nonfinite_scores():
    bin_logits = torch.full((3, 3, 5), float("-inf"))
    un_logits = torch.full((3, 3), float("-inf"))
    bin_logits[0, 1, 0] = 1.0
    legal = [("binary", 0, 1, "add"), ("binary", 1, 0, "add"), ("unary", 0, 0, "neg")]
    priors = top_k_priors(bin_logits, un_logits, legal, k=16)
    assert [a for a, _ in priors] == [("binary", 0, 1, "add")]
    assert all(math.isfinite(p) for _, p in priors)
    assert sum(p for _, p in priors) == pytest.approx(1.0)


def test_top_k_priors_empty_legal_returns_empty():
    assert top_k_priors(torch.zeros(2, 2, 5), torch.zeros(2, 3), [], k=16) == []


def test_top_k_priors_integration_with_model_and_bindings():
    pycore24 = pytest.importorskip("pycore24")
    torch.manual_seed(0)
    net = SetTransformer24(dim=32, num_inducing=4, num_layers=1)
    game = pycore24.GameState([8.0, 3.0, 3.0, 10.0], 24.0)
    legal = [tuple(a) for a in game.legal_actions()]
    values = torch.tensor([e[0] for e in game.elems]).unsqueeze(0)
    depths = torch.tensor([e[1] for e in game.elems], dtype=torch.long).unsqueeze(0)
    out = net(values, depths, torch.tensor([24.0]))
    priors = top_k_priors(out.binary_logits[0], out.unary_logits[0], legal, k=16)
    assert len(priors) == min(16, len(legal))
    assert all(math.isfinite(p) for _, p in priors)
    assert sum(p for _, p in priors) == pytest.approx(1.0, abs=1e-6)
    assert {a for a, _ in priors} <= set(legal)


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
