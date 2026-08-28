import csv

import torch

from learner.network import SetTransformer24
from learner.train import evaluate_solve_rate, load_checkpoint, save_checkpoint, train

WIN = ("unary", 0, 0, "neg")


class EvalGame:
    """Any action solves on the next state."""

    def __init__(self, values, target, solved=False):
        self._values = list(values)
        self.elems = [(float(v), 0) for v in self._values]
        self.target = float(target)
        self._solved = solved

    def legal_actions(self):
        return [WIN]

    def apply(self, action):
        return EvalGame(self._values, self.target, solved=True)

    def solved(self):
        return self._solved


class StuckGame(EvalGame):
    """Never solves."""

    def apply(self, action):
        return StuckGame(self._values, self.target)


def make_eval_game(values, target):
    return EvalGame(list(values), target)


def make_stuck_game(values, target):
    return StuckGame(list(values), target)


def test_solve_rate_perfect_policy_solves_all():
    def policy_value(game):
        return [(WIN, 1.0)], 1.0

    instances = [([2.0, 3.0], 6.0), ([1.0, 5.0], 5.0)]
    rate = evaluate_solve_rate(make_eval_game, policy_value, instances, simulations=2, max_steps=4, seed=0)
    assert rate == 1.0


def test_solve_rate_stuck_policy_solves_none():
    def policy_value(game):
        return [(WIN, 1.0)], 0.0

    rate = evaluate_solve_rate(
        make_stuck_game, policy_value, [([2.0, 3.0], 6.0)], simulations=2, max_steps=3, seed=0
    )
    assert rate == 0.0


def test_solve_rate_empty_instances_is_zero():
    assert evaluate_solve_rate(make_eval_game, lambda g: ([], 0.0), [], simulations=2, max_steps=4) == 0.0


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    net = SetTransformer24(dim=16, heads=2, num_inducing=4, num_layers=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    optimizer.zero_grad()
    net(torch.randn(1, 3), torch.zeros(1, 3, dtype=torch.long), torch.tensor([24.0])).value.sum().backward()
    optimizer.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(net, optimizer, path, step=7, solve_rate=0.5)

    fresh = SetTransformer24(dim=16, heads=2, num_inducing=4, num_layers=1)
    fresh_optimizer = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    meta = load_checkpoint(fresh, fresh_optimizer, path)
    assert meta["step"] == 7
    assert meta["solve_rate"] == 0.5
    for (_, p1), (_, p2) in zip(net.named_parameters(), fresh.named_parameters(), strict=True):
        assert torch.equal(p1, p2)


def tiny_net():
    torch.manual_seed(0)
    return SetTransformer24(dim=8, heads=2, num_inducing=2, num_layers=1)


def test_train_loop_end_to_end(tmp_path, capsys):
    log = tmp_path / "log.csv"
    train(
        make_game=make_eval_game,
        net_factory=tiny_net,
        seed=0,
        workers=1,
        simulations=2,
        episodes_per_iter=1,
        train_steps_per_iter=1,
        batch_size=4,
        inference_batch=4,
        top_k=2,
        max_steps=3,
        replay_capacity=32,
        steps=2,
        lr=1e-3,
        dirichlet_alpha=None,
        eval_every=1,
        eval_instances=2,
        eval_simulations=2,
        checkpoint_dir=tmp_path,
        log_file=log,
    )
    checkpoints = sorted(tmp_path.glob("step-*.pt"))
    assert len(checkpoints) == 2
    rows = list(csv.reader(open(log, encoding="utf-8")))
    assert rows[0] == ["step", "solve_rate", "loss", "policy", "value", "buffer"]
    assert len(rows) == 3
    assert float(rows[1][1]) == 1.0  # every instance solves in one step here
    assert float(rows[1][2]) >= 0.0
    assert int(rows[1][5]) >= 1  # buffer is non-empty
    assert "solve_rate=" in capsys.readouterr().out
