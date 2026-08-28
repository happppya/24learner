"""Shared-RAM replay buffer for self-play training.

Ring buffer over logged steps (episodes flattened with their retrospective
outcome) bounded by a step capacity; `sample` draws a uniform minibatch with
replacement and collates it via `collate_steps` straight into the joint-loss
input format.
"""

from __future__ import annotations

import random

from learner.loss import collate_steps
from learner.selfplay import Episode, Step


class ReplayBuffer:
    """Capacity-bounded ring buffer of (step, outcome) training examples.

    `capacity` is in steps; eviction is oldest-first. To map the 32 GB target
    onto a capacity, estimate bytes per step for your curriculum (a few KB for
    N ~ 6, top-K 16) and size accordingly — byte-accurate accounting is a
    later refinement.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._slots: list[tuple[Step, float] | None] = [None] * capacity
        self._head = 0
        self._count = 0
        self._rng = random.Random()

    def __len__(self) -> int:
        return self._count

    def add(self, step: Step, outcome: float) -> None:
        index = (self._head + self._count) % self.capacity
        self._slots[index] = (step, outcome)
        if self._count == self.capacity:
            self._head = (self._head + 1) % self.capacity
        else:
            self._count += 1

    def add_episode(self, episode: Episode) -> None:
        for step in episode.steps:
            self.add(step, episode.outcome)

    def add_episodes(self, episodes) -> None:
        for episode in episodes:
            self.add_episode(episode)

    def items(self) -> list[tuple[Step, float]]:
        """(step, outcome) pairs in insertion order, oldest first."""
        return [self._slots[(self._head + i) % self.capacity] for i in range(self._count)]

    def sample(self, batch_size: int, rng: random.Random | None = None):
        """Uniform minibatch collated for `alphazero_loss`.

        Returns (values, depths, targets, pad_mask, loss_targets). Draws with
        replacement, so `batch_size` steps are returned even when the buffer
        holds fewer.
        """
        if self._count == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        rng = rng or self._rng
        steps: list[Step] = []
        outcomes: list[float] = []
        for _ in range(batch_size):
            index = (self._head + rng.randrange(self._count)) % self.capacity
            step, outcome = self._slots[index]
            steps.append(step)
            outcomes.append(outcome)
        return collate_steps(steps, outcomes)
