import random

import pytest
import torch

from learner.loss import alphazero_loss
from learner.network import SetTransformer24
from learner.replay import ReplayBuffer
from learner.selfplay import Episode, Step

MUL = ("binary", 0, 1, "mul")
NEG = ("unary", 0, 0, "neg")


def make_step(value=8.0, visits=None):
    return Step(
        elems=[(value, 0), (3.0, 0)],
        target=24.0,
        action=MUL,
        priors=[],
        visits=visits or {MUL: 7, NEG: 3},
        root_value=0.0,
    )


def test_add_and_items_keep_insertion_order():
    buffer = ReplayBuffer(capacity=5)
    for i in range(3):
        buffer.add(make_step(value=float(i)), 1.0)
    assert len(buffer) == 3
    assert [step.elems[0][0] for step, _ in buffer.items()] == [0.0, 1.0, 2.0]
    assert [outcome for _, outcome in buffer.items()] == [1.0, 1.0, 1.0]


def test_ring_evicts_oldest_when_full():
    buffer = ReplayBuffer(capacity=2)
    for i in range(4):
        buffer.add(make_step(value=float(i)), 0.0)
    assert len(buffer) == 2
    assert [step.elems[0][0] for step, _ in buffer.items()] == [2.0, 3.0]


def test_count_never_exceeds_capacity():
    buffer = ReplayBuffer(capacity=3)
    for i in range(20):
        buffer.add(make_step(value=float(i)), 0.0)
    assert len(buffer) == 3


def test_add_episode_flattens_steps_with_outcome():
    episode = Episode(steps=[make_step(value=1.0), make_step(value=2.0)], outcome=1.0)
    buffer = ReplayBuffer(capacity=4)
    buffer.add_episode(episode)
    assert len(buffer) == 2
    assert all(outcome == 1.0 for _, outcome in buffer.items())


def test_sample_returns_collated_batch():
    buffer = ReplayBuffer(capacity=8)
    for i in range(4):
        buffer.add(make_step(value=float(i), visits={MUL: 10 - i, NEG: i}), float(i % 2))
    values, depths, targets, pad_mask, loss_targets = buffer.sample(3, rng=random.Random(0))
    assert values.shape == (3, 2)
    assert depths.shape == (3, 2)
    assert targets.shape == (3,)
    assert pad_mask.shape == (3, 2)
    assert len(loss_targets) == 3
    for target in loss_targets:
        assert set(target["visits"]) == {MUL, NEG}
        assert target["outcome"] in (0.0, 1.0)


def test_sample_is_deterministic_with_seeded_rng():
    buffer = ReplayBuffer(capacity=8)
    for i in range(4):
        buffer.add(make_step(value=float(i)), float(i % 2))

    def run():
        return buffer.sample(3, rng=random.Random(5))

    first, second = run(), run()
    for ta, tb in zip(first[:4], second[:4], strict=True):
        assert torch.equal(ta, tb)
    assert first[4] == second[4]


def test_sample_with_replacement_handles_batch_larger_than_buffer():
    buffer = ReplayBuffer(capacity=4)
    buffer.add(make_step(), 0.0)
    values, _, _, _, _ = buffer.sample(5, rng=random.Random(0))
    assert values.shape == (5, 2)


def test_sampled_batch_feeds_joint_loss():
    torch.manual_seed(0)
    net = SetTransformer24(dim=16, heads=2, num_inducing=4, num_layers=1)
    buffer = ReplayBuffer(capacity=8)
    for i in range(4):
        buffer.add(make_step(value=float(i + 1)), float(i % 2))
    values, depths, targets, pad_mask, loss_targets = buffer.sample(2, rng=random.Random(1))
    out = net(values, depths, targets, pad_mask=pad_mask)
    total, policy, value_loss = alphazero_loss(out.binary_logits, out.unary_logits, out.value, loss_targets)
    assert policy.item() >= 0.0
    assert value_loss.item() >= 0.0
    total.backward()
    missing = [name for name, param in net.named_parameters() if param.grad is None]
    assert not missing


def test_empty_buffer_rejects_sampling():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=4).sample(1)


def test_invalid_capacity_rejected():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)


def test_invalid_batch_size_rejected():
    buffer = ReplayBuffer(capacity=4)
    buffer.add(make_step(), 0.0)
    with pytest.raises(ValueError):
        buffer.sample(0)
