import random

import pytest
import torch

from learner.mcts import MCTS, PyCore24Game, make_policy_value
from learner.network import SetTransformer24
from learner.selfplay import (
    SCHEMA_VERSION,
    Episode,
    SelfPlayPool,
    Step,
    append_episode,
    episode_from_json,
    episode_to_json,
    play_episode,
    read_episodes,
)

ADD = ("binary", 0, 1, "add")
SQRT = ("unary", 0, 0, "sqrt")
NEG = ("unary", 0, 0, "neg")


class TerminatingGame:
    """Applies any action once, then reports solved."""

    def __init__(self, actions, solved=False):
        self._actions = list(actions)
        self._solved = solved
        self.elems = [(24.0, 0)] if solved else [(8.0, 0), (3.0, 0)]
        self.target = 24.0

    def legal_actions(self):
        return self._actions

    def apply(self, action):
        if self._solved:
            return None
        return TerminatingGame(self._actions, solved=True)

    def solved(self):
        return self._solved


def test_play_episode_plays_to_terminal_and_records_step():
    game = TerminatingGame([ADD])

    def policy_value(g):
        return [(ADD, 1.0)], 0.5

    episode = play_episode(game, MCTS(policy_value), simulations=10, rng=random.Random(0))
    assert len(episode.steps) == 1
    step = episode.steps[0]
    assert step.action == ADD
    assert step.visits == {ADD: 10}
    assert step.priors == [(ADD, 1.0)]
    assert step.root_value == 1.0  # search backs up the solved-leaf terminal value
    assert step.elems == [(8.0, 0), (3.0, 0)]
    assert step.target == 24.0
    assert episode.outcome == 1.0


def test_zero_temperature_picks_argmax_deterministically():
    def policy_value(g):
        return [(SQRT, 0.9), (NEG, 0.1)], 0.0

    episodes = [
        play_episode(
            TerminatingGame([SQRT, NEG]),
            MCTS(policy_value),
            simulations=20,
            temperature=0.0,
            rng=random.Random(seed),
        )
        for seed in range(5)
    ]
    assert all(ep.steps[0].action == SQRT for ep in episodes)


class StuckGame:
    """Never solves; apply returns an identical unsolved game."""

    def __init__(self, actions):
        self._actions = list(actions)
        self.elems = [(8.0, 0), (3.0, 0)]
        self.target = 24.0

    def legal_actions(self):
        return self._actions

    def apply(self, action):
        return StuckGame(self._actions)

    def solved(self):
        return False


def test_unsolved_episode_has_zero_outcome():
    def policy_value(g):
        return [(SQRT, 1.0)], 0.0

    episode = play_episode(
        StuckGame([SQRT]),
        MCTS(policy_value),
        simulations=5,
        max_steps=3,
        rng=random.Random(0),
    )
    assert len(episode.steps) == 3
    assert episode.outcome == 0.0


def test_episode_json_round_trip():
    episode = Episode(
        steps=[
            Step(
                elems=[(8.0, 0), (3.0, 0)],
                target=24.0,
                action=("binary", 0, 1, "mul"),
                priors=[(("binary", 0, 1, "mul"), 0.7), (SQRT, 0.3)],
                visits={("binary", 0, 1, "mul"): 7, SQRT: 3},
                root_value=0.25,
            )
        ],
        outcome=1.0,
    )
    assert episode_from_json(episode_to_json(episode)) == episode


def test_episode_json_is_version_tagged():
    episode = Episode([], outcome=1.0)
    assert episode_to_json(episode)["version"] == SCHEMA_VERSION


def test_legacy_untagged_episode_parses():
    payload = {"outcome": 1.0, "steps": []}  # pre-versioning line
    episode = episode_from_json(payload)
    assert episode.outcome == 1.0 and episode.steps == []


def test_unsupported_schema_version_raises():
    payload = {"version": SCHEMA_VERSION + 1, "outcome": 1.0, "steps": []}
    with pytest.raises(ValueError, match=str(SCHEMA_VERSION)):
        episode_from_json(payload)


def test_read_episodes_handles_mixed_legacy_and_current(tmp_path):
    path = tmp_path / "mixed.jsonl"
    current = Episode([], outcome=1.0)
    append_episode(path, current)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"outcome": 0.0, "steps": []}\n')  # legacy, untagged
    episodes = list(read_episodes(path))
    assert episodes == [current, Episode([], outcome=0.0)]


def test_read_episodes_raises_on_unsupported_version(tmp_path):
    path = tmp_path / "bad.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f'{{"version": {SCHEMA_VERSION + 1}, "outcome": 1.0, "steps": []}}\n')
    with pytest.raises(ValueError):
        list(read_episodes(path))


def test_append_and_read_episodes(tmp_path):
    path = tmp_path / "episodes.jsonl"
    first = Episode(
        [Step([(1.0, 0)], 24.0, NEG, [(NEG, 1.0)], {NEG: 5}, 0.1)],
        outcome=0.0,
    )
    second = Episode([], outcome=1.0)
    append_episode(path, first)
    append_episode(path, second)
    assert list(read_episodes(path)) == [first, second]


def make_real_game():
    pycore24 = pytest.importorskip("pycore24")
    return PyCore24Game(pycore24.GameState([8.0, 3.0, 3.0], 24.0))


def make_real_net():
    torch.manual_seed(0)
    return SetTransformer24(dim=16, heads=2, num_inducing=4, num_layers=1)


def test_pool_runs_parallel_episodes(tmp_path):
    with SelfPlayPool(
        make_real_game,
        make_real_net(),
        workers=3,
        simulations=10,
        max_steps=5,
        top_k=8,
        seed=1,
    ) as pool:
        episodes = pool.run(7)
    assert len(episodes) == 7
    for episode in episodes:
        assert 0 < len(episode.steps) <= 5
        assert episode.outcome in (0.0, 1.0)

    path = tmp_path / "pool.jsonl"
    for episode in episodes:
        append_episode(path, episode)
    assert len(list(read_episodes(path))) == 7


def test_pool_single_worker_is_deterministic():
    def run():
        with SelfPlayPool(
            make_real_game,
            make_real_net(),
            workers=1,
            simulations=10,
            max_steps=4,
            top_k=8,
            seed=0,
        ) as pool:
            return pool.run(3)

    assert run() == run()


def test_pool_with_dirichlet_noise_produces_valid_episodes():
    with SelfPlayPool(
        make_real_game,
        make_real_net(),
        workers=2,
        simulations=10,
        max_steps=5,
        top_k=8,
        dirichlet_alpha=0.3,
        seed=2,
    ) as pool:
        episodes = pool.run(4)
    assert len(episodes) == 4
    assert all(ep.outcome in (0.0, 1.0) for ep in episodes)


def test_pool_rejects_invalid_worker_count():
    with pytest.raises(ValueError):
        SelfPlayPool(make_real_game, make_real_net(), workers=0)


def test_pool_zero_episodes_returns_empty():
    with SelfPlayPool(make_real_game, make_real_net(), workers=2, simulations=5, seed=0) as pool:
        assert pool.run(0) == []


def test_play_episode_with_real_network_and_bindings(tmp_path):
    pycore24 = pytest.importorskip("pycore24")
    torch.manual_seed(0)
    net = SetTransformer24(dim=16, heads=2, num_inducing=4, num_layers=1)
    game = PyCore24Game(pycore24.GameState([8.0, 3.0, 3.0], 24.0))
    mcts = MCTS(make_policy_value(net, top_k=8))
    episode = play_episode(game, mcts, simulations=10, max_steps=6, rng=random.Random(0))
    assert 0 < len(episode.steps) <= 6
    assert episode.outcome in (0.0, 1.0)
    assert all(len(step.visits) <= 8 for step in episode.steps)
    assert all(sum(step.visits.values()) == 10 for step in episode.steps)

    path = tmp_path / "ep.jsonl"
    append_episode(path, episode)
    assert list(read_episodes(path)) == [episode]
