"""Minimal MCTS rollout consuming `top_k_priors`.

Reference implementation for the Phase 4 Rust-hosted workers (see
notes/design.md): PUCT selection, expansion from priors, value backup,
visit-distribution output. Games are accessed through a small protocol
(legal_actions/apply/solved/elems/target) so the search loop itself is
transport-agnostic; `PyCore24Game` adapts the in-process bindings.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from learner.network import top_k_priors


@dataclass(slots=True)
class Node:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    children: dict[tuple, Node] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class PyCore24Game:
    """Immutable adapter over a `pycore24.GameState` for tree traversal."""

    def __init__(self, game) -> None:
        self.game = game

    @property
    def elems(self) -> list[tuple[float, int]]:
        return [(float(v), int(d)) for v, d in self.game.elems]

    @property
    def target(self) -> float:
        return float(self.game.target)

    def legal_actions(self) -> list[tuple]:
        return [tuple(a) for a in self.game.legal_actions()]

    def apply(self, action: tuple):
        kind, i, j, op = action
        nxt = self.game.cloned()
        if not nxt.step(kind, i, j, op):
            return None
        return PyCore24Game(nxt)

    def solved(self) -> bool:
        return self.game.solved()


def make_policy_value(net, *, top_k: int = 16):
    """Build a policy_value(game) -> (priors, value) callable for one state.

    Priors come from `top_k_priors` over the game's legal actions; value is
    the network's value head. Runs a direct forward pass (no batching); the
    batched path is `InferenceServer`'s job once workers exist.
    """

    def policy_value(game):
        values = torch.tensor([e[0] for e in game.elems]).unsqueeze(0)
        depths = torch.tensor([e[1] for e in game.elems], dtype=torch.long).unsqueeze(0)
        target = torch.tensor([game.target])
        with torch.no_grad():
            out = net(values, depths, target)
        priors = top_k_priors(out.binary_logits[0], out.unary_logits[0], game.legal_actions(), k=top_k)
        return priors, out.value[0].item()

    return policy_value


def make_server_policy_value(server, *, top_k: int = 16):
    """Build a policy_value(game) routed through an InferenceServer batch queue.

    The production path for parallel workers: many threads submit through the
    shared server, so node evaluations gather into GPU/CPU batches. Results
    are identical to `make_policy_value` (same logits, fp32).
    """

    def policy_value(game):
        pred = server.infer([e[0] for e in game.elems], [e[1] for e in game.elems], game.target)
        priors = top_k_priors(
            torch.from_numpy(pred.binary_logits),
            torch.from_numpy(pred.unary_logits),
            game.legal_actions(),
            k=top_k,
        )
        return priors, pred.value

    return policy_value


class MCTS:
    """Single-threaded PUCT search; `search` returns visit counts + root value."""

    def __init__(
        self,
        policy_value: Callable[[object], tuple[list[tuple[tuple, float]], float]],
        *,
        c_puct: float = 1.25,
        max_depth: int = 64,
        dirichlet_alpha: float | None = None,
        noise_weight: float = 0.25,
    ) -> None:
        self.policy_value = policy_value
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_weight = noise_weight

    def search(self, game, simulations: int) -> tuple[dict[tuple, int], float]:
        root = Node(prior=0.0)
        if game.solved():
            self.root = root
            return {}, 1.0
        priors, _ = self.policy_value(game)
        self._expand(game, root, priors)
        if self.dirichlet_alpha is not None and root.children:
            self._apply_root_noise(root)
        for _ in range(simulations):
            self._simulate(game, root, 0)
        self.root = root
        visits = {action: node.visits for action, node in root.children.items()}
        return visits, root.value

    def _simulate(self, game, node: Node, depth: int) -> None:
        path: list[tuple[object, Node]] = [(game, node)]
        while node.children and depth < self.max_depth:
            action = self._select(node)
            nxt = game.apply(action)
            if nxt is None:
                break
            game = nxt
            node = node.children[action]
            path.append((game, node))
            depth += 1
        if game.solved():
            value = 1.0
        elif not node.children and depth < self.max_depth:
            priors, value = self.policy_value(game)
            self._expand(game, node, priors)
        else:
            _, value = self.policy_value(game)
        for _, visited in path:
            visited.visits += 1
            visited.value_sum += value

    def _select(self, node: Node) -> tuple:
        best_action: tuple | None = None
        best_score = float("-inf")
        for action, child in node.children.items():
            exploration = self.c_puct * child.prior * math.sqrt(node.visits + 1.0) / (1.0 + child.visits)
            score = child.value + exploration
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _apply_root_noise(self, root: Node) -> None:
        alpha = torch.full((len(root.children),), self.dirichlet_alpha)
        noise = torch.distributions.Dirichlet(alpha).sample()
        for (_, child), xi in zip(root.children.items(), noise.tolist(), strict=True):
            child.prior = (1.0 - self.noise_weight) * child.prior + self.noise_weight * xi

    def _expand(self, game, node: Node, priors: list[tuple[tuple, float]]) -> None:
        children: dict[tuple, Node] = {}
        total = 0.0
        for action, prior in priors:
            if game.apply(action) is not None:
                children[action] = Node(prior=prior)
                total += prior
        if total > 0.0:
            for child in children.values():
                child.prior /= total
        node.children = children
