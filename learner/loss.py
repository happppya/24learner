"""AlphaZero joint loss and training-batch collation.

Policy term: cross-entropy between the normalized MCTS visit distribution and
the network's softmax over the *same* (top-K visited) actions, so the loss
only supervises the actions the search actually explored. Value term: MSE
against the retrospective episode outcome. `collate_steps` turns logged
self-play steps into padded network inputs plus per-sample loss targets.
"""

from __future__ import annotations

import torch

from learner.network import action_logit


def alphazero_loss(
    binary_logits: torch.Tensor,
    unary_logits: torch.Tensor,
    value: torch.Tensor,
    targets: list[dict],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Joint loss over a padded batch; returns (total, policy, value) means.

    `binary_logits` (B, N, N, 5), `unary_logits` (B, N, 3), `value` (B,).
    Each target dict has "visits" ({action: count}) and "outcome" (float).
    Samples with no visits contribute only the value term.
    """
    policy_losses: list[torch.Tensor] = []
    value_losses: list[torch.Tensor] = []
    for i, target in enumerate(targets):
        visits = target["visits"]
        if visits:
            actions = list(visits)
            n = max(max(i, j) for _, i, j, _ in actions) + 1
            counts = torch.tensor(
                [visits[a] for a in actions], dtype=torch.float32, device=binary_logits.device
            )
            target_probs = counts / counts.sum()
            logits = torch.stack(
                [action_logit(binary_logits[i, :n, :n, :], unary_logits[i, :n, :], a) for a in actions]
            )
            policy_losses.append(-(target_probs * logits.log_softmax(dim=-1)).sum())
        value_losses.append((value[i] - target["outcome"]) ** 2)
    policy = torch.stack(policy_losses).mean() if policy_losses else torch.tensor(0.0)
    value_loss = torch.stack(value_losses).mean()
    return policy + value_loss, policy, value_loss


def collate_steps(steps: list, outcomes: list[float]):
    """Batch logged steps into padded network inputs plus loss targets.

    Returns (values, depths, targets, pad_mask, loss_targets) where
    `loss_targets` aligns 1:1 with the batch samples and each entry carries
    the step's visits plus its retrospective episode outcome.
    """
    if len(steps) != len(outcomes):
        raise ValueError("steps and outcomes must have the same length")
    batch = len(steps)
    max_n = max(len(step.elems) for step in steps)
    values = torch.zeros(batch, max_n)
    depths = torch.zeros(batch, max_n, dtype=torch.long)
    targets = torch.zeros(batch)
    pad_mask = torch.zeros(batch, max_n, dtype=torch.bool)
    loss_targets: list[dict] = []
    for i, step in enumerate(steps):
        n = len(step.elems)
        values[i, :n] = torch.tensor([v for v, _ in step.elems])
        depths[i, :n] = torch.tensor([d for _, d in step.elems], dtype=torch.long)
        targets[i] = step.target
        pad_mask[i, n:] = True
        loss_targets.append({"visits": dict(step.visits), "outcome": float(outcomes[i])})
    return values, depths, targets, pad_mask, loss_targets
