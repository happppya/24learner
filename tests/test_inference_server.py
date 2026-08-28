import pytest
import torch

from learner.inference import InferenceServer
from learner.mcts import MCTS, PyCore24Game, make_policy_value, make_server_policy_value
from learner.network import SetTransformer24


class FakeGame:
    """Minimal protocol stand-in: fixed legal actions, optional rejections."""

    def __init__(self, actions, reject=(), solved=False):
        self._actions = list(actions)
        self._reject = set(reject)
        self._solved = solved

    def legal_actions(self):
        return self._actions

    def apply(self, action):
        if action in self._reject:
            return None
        return FakeGame(self._actions, self._reject, self._solved)

    def solved(self):
        return self._solved


SQRT = ("unary", 0, 0, "sqrt")
NEG = ("unary", 0, 0, "neg")
ADD = ("binary", 0, 1, "add")


def test_search_returns_normalized_visit_target():
    game = FakeGame([SQRT, NEG, ADD])

    def policy_value(g):
        return [(SQRT, 0.8), (NEG, 0.2)], 0.5

    visits, value = MCTS(policy_value).search(game, simulations=50)
    assert sum(visits.values()) == 50
    assert set(visits) == {SQRT, NEG}
    assert visits[SQRT] > visits[NEG]
    assert value == 0.5


def test_unapplicable_actions_are_dropped_and_priors_renormalized():
    game = FakeGame([ADD, SQRT], reject={SQRT})

    def policy_value(g):
        return [(ADD, 0.3), (SQRT, 0.7)], 0.0

    mcts = MCTS(policy_value)
    visits, _ = mcts.search(game, simulations=10)
    assert set(visits) == {ADD}
    assert sum(visits.values()) == 10

    # Renormalized prior on the surviving child is 1.0.
    assert mcts.root.children[ADD].prior == pytest.approx(1.0)


def test_solved_state_is_terminal_without_policy_calls():
    game = FakeGame([NEG], solved=True)

    def policy_value(g):
        raise AssertionError("policy_value must not be called on a solved state")

    visits, value = MCTS(policy_value).search(game, simulations=5)
    assert visits == {}
    assert value == 1.0


def test_depth_cap_terminates_unary_chains():
    game = FakeGame([NEG])
    calls = {"n": 0}

    def policy_value(g):
        calls["n"] += 1
        return [(NEG, 1.0)], 0.0

    mcts = MCTS(policy_value, max_depth=3)
    visits, _ = mcts.search(game, simulations=4)
    assert sum(visits.values()) == 4
    assert calls["n"] >= 4


def test_dirichlet_noise_perturbs_root_priors_and_stays_normalized():
    game = FakeGame([SQRT, NEG, ADD])

    def policy_value(g):
        return [(SQRT, 0.8), (NEG, 0.1), (ADD, 0.1)], 0.0

    torch.manual_seed(0)
    noisy = MCTS(policy_value, dirichlet_alpha=0.5, noise_weight=0.25)
    noisy.search(game, simulations=5)
    noisy_priors = {a: n.prior for a, n in noisy.root.children.items()}

    clean = MCTS(policy_value)
    clean.search(game, simulations=5)
    clean_priors = {a: n.prior for a, n in clean.root.children.items()}

    assert noisy_priors != clean_priors
    assert sum(noisy_priors.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= prior <= 1.0 for prior in noisy_priors.values())


def test_dirichlet_noise_is_deterministic_with_seed():
    def policy_value(g):
        return [(SQRT, 0.8), (NEG, 0.2)], 0.0

    def run():
        torch.manual_seed(7)
        return MCTS(policy_value, dirichlet_alpha=0.5).search(FakeGame([SQRT, NEG]), 3)

    assert run()[0] == run()[0]


def test_server_policy_value_matches_direct_forward():
    pycore24 = pytest.importorskip("pycore24")
    torch.manual_seed(0)
    net = SetTransformer24(dim=16, heads=2, num_inducing=4, num_layers=1)
    game = PyCore24Game(pycore24.GameState([8.0, 3.0, 3.0], 24.0))
    direct = make_policy_value(net, top_k=8)(game)
    with InferenceServer(net, device="cpu", precision="fp32", max_batch=4, timeout_s=0.001) as server:
        via_server = make_server_policy_value(server, top_k=8)(game)
    assert via_server[0] == direct[0]
    assert via_server[1] == pytest.approx(direct[1], abs=1e-6)


def test_search_never_returns_actions_outside_legal_set():
    game = FakeGame([ADD, NEG])

    def policy_value(g):
        return [(ADD, 1.0)], 0.0

    visits, _ = MCTS(policy_value).search(game, simulations=20)
    assert set(visits) <= set(game.legal_actions())


def test_rollout_with_real_network_and_bindings():
    pycore24 = pytest.importorskip("pycore24")
    torch.manual_seed(0)
    net = SetTransformer24(dim=32, num_inducing=4, num_layers=1)
    game = PyCore24Game(pycore24.GameState([8.0, 3.0, 3.0], 24.0))
    mcts = MCTS(make_policy_value(net, top_k=8))
    visits, value = mcts.search(game, simulations=30)
    assert sum(visits.values()) == 30
    assert set(visits) <= set(game.legal_actions())
    assert all(v >= 0 for v in visits.values())
    assert -1.0 <= value <= 1.0
