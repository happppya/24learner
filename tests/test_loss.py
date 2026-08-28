import pytest
import torch

from learner.loss import alphazero_loss, collate_steps
from learner.network import BINARY_OPS, UNARY_OPS, SetTransformer24, action_logit
from learner.selfplay import Step

ADD = ("binary", 0, 1, "add")
MUL = ("binary", 0, 1, "mul")
SQRT = ("unary", 0, 0, "sqrt")


def test_loss_matches_manual_computation():
    binary = torch.zeros(1, 2, 2, 5)
    unary = torch.zeros(1, 2, 3)
    binary[0, 0, 1, BINARY_OPS.index("add")] = 1.0
    binary[0, 0, 1, BINARY_OPS.index("mul")] = 0.0
    value = torch.tensor([0.4])
    targets = [{"visits": {ADD: 2, MUL: 1}, "outcome": 0.5}]

    total, policy, value_loss = alphazero_loss(binary, unary, value, targets)

    logits = torch.stack([action_logit(binary[0], unary[0], a) for a in (ADD, MUL)])
    target_probs = torch.tensor([2.0 / 3.0, 1.0 / 3.0])
    expected_policy = -(target_probs * logits.log_softmax(dim=-1)).sum()
    expected_value = (0.4 - 0.5) ** 2
    assert policy == pytest.approx(expected_policy.item())
    assert value_loss == pytest.approx(expected_value)
    assert total == pytest.approx(expected_policy.item() + expected_value)


def test_perfect_prediction_reaches_entropy_and_zero_mse():
    target_probs = torch.tensor([0.5, 0.3, 0.2])
    counts = [5, 3, 2]
    actions = [ADD, MUL, SQRT]
    binary = torch.full((1, 2, 2, 5), float("-inf"))
    unary = torch.full((1, 2, 3), float("-inf"))
    binary[0, 0, 1, BINARY_OPS.index("add")] = target_probs[0].log()
    binary[0, 0, 1, BINARY_OPS.index("mul")] = target_probs[1].log()
    unary[0, 0, UNARY_OPS.index("sqrt")] = target_probs[2].log()
    value = torch.tensor([0.7])
    targets = [{"visits": {a: c for a, c in zip(actions, counts, strict=True)}, "outcome": 0.7}]

    total, policy, value_loss = alphazero_loss(binary, unary, value, targets)
    assert value_loss == pytest.approx(0.0, abs=1e-6)
    entropy = -(target_probs * target_probs.log()).sum()
    assert policy == pytest.approx(entropy.item(), abs=1e-5)
    assert total == pytest.approx(entropy.item(), abs=1e-5)


def test_sample_without_visits_contributes_value_only():
    binary = torch.zeros(2, 2, 2, 5)
    unary = torch.zeros(2, 2, 3)
    value = torch.tensor([0.3, -0.2])
    targets = [
        {"visits": {ADD: 3}, "outcome": 1.0},
        {"visits": {}, "outcome": 0.0},
    ]
    total, policy, value_loss = alphazero_loss(binary, unary, value, targets)
    assert policy == pytest.approx(0.0, abs=1e-6)  # single visited action → CE 0
    expected_value = ((0.3 - 1.0) ** 2 + (-0.2 - 0.0) ** 2) / 2
    assert value_loss == pytest.approx(expected_value)
    assert total == pytest.approx(expected_value, abs=1e-6)


def test_collate_pads_and_aligns_targets():
    steps = [
        Step(
            elems=[(8.0, 0), (3.0, 0)],
            target=24.0,
            action=MUL,
            priors=[],
            visits={MUL: 7},
            root_value=0.0,
        ),
        Step(
            elems=[(2.0, 0)],
            target=2.0,
            action=("unary", 0, 0, "neg"),
            priors=[],
            visits={("unary", 0, 0, "neg"): 3},
            root_value=0.0,
        ),
    ]
    values, depths, targets, pad_mask, loss_targets = collate_steps(steps, [1.0, 0.0])
    assert values.shape == (2, 2)
    assert values[0, 0].item() == 8.0
    assert values[1, 1].item() == 0.0  # padded
    assert bool(pad_mask[1, 1]) and not bool(pad_mask[0].any())
    assert loss_targets[0]["visits"] == {MUL: 7}
    assert loss_targets[0]["outcome"] == 1.0
    assert loss_targets[1]["outcome"] == 0.0


def test_collate_rejects_length_mismatch():
    with pytest.raises(ValueError):
        collate_steps([Step([(1.0, 0)], 1.0, ADD, [], {ADD: 1}, 0.0)], [1.0, 2.0])


def test_full_training_step_backward():
    torch.manual_seed(0)
    net = SetTransformer24(dim=16, heads=2, num_inducing=4, num_layers=1)
    steps = [
        Step(
            elems=[(8.0, 0), (3.0, 0), (3.0, 0)],
            target=24.0,
            action=MUL,
            priors=[],
            visits={MUL: 6, SQRT: 4},
            root_value=0.2,
        ),
        Step(
            elems=[(4.0, 1)],
            target=2.0,
            action=SQRT,
            priors=[],
            visits={SQRT: 10},
            root_value=0.1,
        ),
    ]
    values, depths, targets, pad_mask, loss_targets = collate_steps(steps, [1.0, 0.0])
    assert values.shape == (2, 3)
    assert bool(pad_mask[1, 1]) and bool(pad_mask[1, 2])
    out = net(values, depths, targets, pad_mask=pad_mask)
    total, policy, value_loss = alphazero_loss(out.binary_logits, out.unary_logits, out.value, loss_targets)
    assert policy.item() > 0.0
    assert value_loss.item() >= 0.0
    total.backward()
    missing = [name for name, param in net.named_parameters() if param.grad is None]
    assert not missing
