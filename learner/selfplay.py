"""Self-play episode generation and logging.

Plays games by sampling from the MCTS visit distribution, recording per-step
(states, priors, visit counts, root value) plus the episode outcome, and
serializes episodes as NDJSON (one compact JSON object per episode) for the
Phase 5 trainer. Policy target for a step is the normalized visit counts; the
value target is the retrospective episode outcome. Each line is tagged with
`SCHEMA_VERSION`; untagged legacy lines parse silently, unknown versions raise.
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass

from learner.inference import InferenceServer
from learner.mcts import MCTS, make_server_policy_value

SCHEMA_VERSION = 1


@dataclass(slots=True)
class Step:
    elems: list[tuple[float, int]]
    target: float
    action: tuple
    priors: list[tuple[tuple, float]]
    visits: dict[tuple, int]
    root_value: float


@dataclass(slots=True)
class Episode:
    steps: list[Step]
    outcome: float  # 1.0 if the episode solved, else 0.0


def play_episode(
    game, mcts, *, simulations: int, rng=None, max_steps: int = 64, temperature: float = 1.0
) -> Episode:
    """Play one episode, sampling actions from the MCTS visit distribution.

    `temperature` scales the sampling weights (counts ** (1/T)); T = 0 is
    deterministic argmax. Pass a seeded `rng` for reproducible playouts.
    """
    if rng is None:
        rng = random.Random()
    steps: list[Step] = []
    for _ in range(max_steps):
        if game.solved():
            break
        visits, root_value = mcts.search(game, simulations)
        if not visits:
            break
        action = _sample_action(rng, visits, temperature)
        priors = [(action_spec, node.prior) for action_spec, node in mcts.root.children.items()]
        steps.append(
            Step(
                elems=list(game.elems),
                target=game.target,
                action=action,
                priors=priors,
                visits=visits,
                root_value=root_value,
            )
        )
        nxt = game.apply(action)
        if nxt is None:
            break
        game = nxt
    outcome = 1.0 if game.solved() else 0.0
    return Episode(steps=steps, outcome=outcome)


def _sample_action(rng: random.Random, visits: dict[tuple, int], temperature: float) -> tuple:
    if temperature == 0.0:
        return max(visits, key=visits.get)
    actions = list(visits)
    weights = [visits[a] ** (1.0 / temperature) for a in actions]
    return rng.choices(actions, weights=weights, k=1)[0]


def episode_to_json(episode: Episode) -> dict:
    return {
        "version": SCHEMA_VERSION,
        "outcome": episode.outcome,
        "steps": [
            {
                "elems": [[v, d] for v, d in step.elems],
                "target": step.target,
                "action": list(step.action),
                "priors": [[list(a), p] for a, p in step.priors],
                "visits": [[list(a), v] for a, v in step.visits.items()],
                "root_value": step.root_value,
            }
            for step in episode.steps
        ],
    }


def episode_from_json(payload: dict) -> Episode:
    version = payload.get("version")
    if version is not None and version != SCHEMA_VERSION:
        raise ValueError(f"unsupported episode schema version {version}; expected {SCHEMA_VERSION}")
    steps = [
        Step(
            elems=[(float(v), int(d)) for v, d in step["elems"]],
            target=float(step["target"]),
            action=tuple(step["action"]),
            priors=[(tuple(a), float(p)) for a, p in step["priors"]],
            visits={tuple(a): int(v) for a, v in step["visits"]},
            root_value=float(step["root_value"]),
        )
        for step in payload["steps"]
    ]
    return Episode(steps=steps, outcome=float(payload["outcome"]))


def append_episode(path, episode: Episode) -> None:
    """Append one episode as an NDJSON line."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(episode_to_json(episode)) + "\n")


def read_episodes(path):
    """Yield episodes from an NDJSON file, skipping blank lines."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield episode_from_json(json.loads(line))


class SelfPlayPool:
    """Runs parallel self-play workers sharing one batched InferenceServer.

    Each worker thread owns its own MCTS (thread-local tree and RNG) and
    evaluates nodes through the shared server's `infer`, so concurrent node
    evaluations gather into batches. Games come from `make_game`, called
    fresh per episode.
    """

    def __init__(
        self,
        make_game,
        net,
        *,
        workers: int = 8,
        simulations: int = 100,
        max_steps: int = 64,
        top_k: int = 16,
        temperature: float = 1.0,
        dirichlet_alpha: float | None = None,
        noise_weight: float = 0.25,
        seed: int | None = None,
        max_batch: int = 64,
        timeout_s: float = 0.005,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.make_game = make_game
        self.workers = workers
        self.simulations = simulations
        self.max_steps = max_steps
        self.top_k = top_k
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_weight = noise_weight
        self.seed = seed
        self.server = InferenceServer(net, max_batch=max_batch, timeout_s=timeout_s)

    def run(self, episodes: int) -> list[Episode]:
        if episodes < 1:
            return []
        per_worker = (episodes + self.workers - 1) // self.workers
        results: list[Episode] = []
        lock = threading.Lock()

        def worker(index: int) -> None:
            rng = random.Random(self.seed + index) if self.seed is not None else random.Random()
            mcts = MCTS(
                make_server_policy_value(self.server, top_k=self.top_k),
                dirichlet_alpha=self.dirichlet_alpha,
                noise_weight=self.noise_weight,
            )
            for _ in range(per_worker):
                episode = play_episode(
                    self.make_game(),
                    mcts,
                    simulations=self.simulations,
                    rng=rng,
                    max_steps=self.max_steps,
                    temperature=self.temperature,
                )
                with lock:
                    results.append(episode)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(self.workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return results[:episodes]

    def close(self) -> None:
        self.server.close()

    def __enter__(self) -> SelfPlayPool:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
