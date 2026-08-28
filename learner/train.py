"""First training loop: self-play -> replay buffer -> Adam on the joint loss.

Each iteration the `SelfPlayPool` generates episodes (actor net), the steps
land in the `ReplayBuffer`, sampled minibatches drive Adam on the AlphaZero
joint loss (critic net), and the actor is re-synced. Periodically the current
solve rate is measured on held-out instances and logged (CSV) with a
checkpoint. Checkpoints and logs are never committed (see .gitignore).
"""

from __future__ import annotations

import argparse
import csv
import random
import threading
from pathlib import Path

import torch

from learner.device import resolve_device
from learner.loss import alphazero_loss
from learner.mcts import MCTS, PyCore24Game, make_policy_value
from learner.network import SetTransformer24
from learner.replay import ReplayBuffer
from learner.selfplay import SelfPlayPool, play_episode


def default_instance(rng: random.Random, n_min: int = 2, n_max: int = 5):
    """One random (values, target) instance in the initial curriculum range."""
    n = rng.randint(n_min, n_max)
    values = [round(rng.uniform(-99.0, 99.0), 3) for _ in range(n)]
    target = round(rng.uniform(-100.0, 100.0), 3)
    return values, target


def make_pycore24_game(values, target):
    import pycore24

    return PyCore24Game(pycore24.GameState(values, target))


def evaluate_solve_rate(make_game, policy_value, instances, *, simulations, max_steps, seed=0):
    """Fraction of instances solved by playing the MCTS policy (temperature 0)."""
    if not instances:
        return 0.0
    solved = 0
    for index, (values, target) in enumerate(instances):
        mcts = MCTS(policy_value)
        episode = play_episode(
            make_game(values, target),
            mcts,
            simulations=simulations,
            rng=random.Random(seed + index),
            max_steps=max_steps,
            temperature=0.0,
        )
        solved += episode.outcome == 1.0
    return solved / len(instances)


def save_checkpoint(net, optimizer, path, **meta) -> None:
    torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(), **meta}, path)


def load_checkpoint(net, optimizer, path) -> dict:
    payload = torch.load(path, weights_only=True)
    net.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload


def train(
    *,
    make_game,
    device: str = "auto",
    precision: str = "auto",
    seed: int = 0,
    workers: int = 8,
    simulations: int = 100,
    episodes_per_iter: int = 8,
    train_steps_per_iter: int = 4,
    batch_size: int = 64,
    inference_batch: int = 32,
    top_k: int = 16,
    max_steps: int = 64,
    replay_capacity: int = 100_000,
    steps: int = 100,
    lr: float = 3e-4,
    dirichlet_alpha: float | None = 0.3,
    eval_every: int = 10,
    eval_instances: int = 32,
    eval_simulations: int = 64,
    checkpoint_dir: str = "checkpoints",
    log_file: str | None = None,
    net_factory=SetTransformer24,
    instance_factory=default_instance,
    progress=print,
):
    torch.manual_seed(seed)
    device = resolve_device(device)
    trainer_net = net_factory().to(device)
    actor_net = net_factory().to(device)
    actor_net.load_state_dict(trainer_net.state_dict())
    optimizer = torch.optim.Adam(trainer_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=replay_capacity)
    sample_rng = random.Random(seed + 3)

    instance_rng = random.Random(seed + 1)
    instance_lock = threading.Lock()

    def new_game():
        with instance_lock:
            values, target = instance_factory(instance_rng)
        return make_game(values, target)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_file) if log_file else None
    if log_path:
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["step", "solve_rate", "loss", "policy", "value", "buffer"])

    eval_pool = [instance_factory(random.Random(seed + 2 + i)) for i in range(eval_instances)]

    with SelfPlayPool(
        new_game,
        actor_net,
        workers=workers,
        simulations=simulations,
        max_steps=max_steps,
        top_k=top_k,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed + 4,
        max_batch=inference_batch,
    ) as pool:
        for step in range(1, steps + 1):
            buffer.add_episodes(pool.run(episodes_per_iter))
            loss_total = loss_policy = loss_value = 0.0
            for _ in range(train_steps_per_iter):
                batch = buffer.sample(batch_size, rng=sample_rng)
                values, depths, targets, pad_mask = (t.to(device) for t in batch[:4])
                optimizer.zero_grad()
                out = trainer_net(values, depths, targets, pad_mask=pad_mask)
                total, policy, value = alphazero_loss(
                    out.binary_logits, out.unary_logits, out.value, batch[4]
                )
                total.backward()
                optimizer.step()
                loss_total += total.item()
                loss_policy += policy.item()
                loss_value += value.item()
            actor_net.load_state_dict(trainer_net.state_dict())
            if eval_instances and (step % eval_every == 0 or step == steps):
                rate = evaluate_solve_rate(
                    make_game,
                    make_policy_value(actor_net, top_k=top_k),
                    eval_pool,
                    simulations=eval_simulations,
                    max_steps=max_steps,
                    seed=seed + 5,
                )
                loss_total /= train_steps_per_iter
                loss_policy /= train_steps_per_iter
                loss_value /= train_steps_per_iter
                progress(
                    f"step {step}: solve_rate={rate:.3f} loss={loss_total:.4f} "
                    f"policy={loss_policy:.4f} value={loss_value:.4f} buffer={len(buffer)}"
                )
                if log_path:
                    with open(log_path, "a", newline="", encoding="utf-8") as fh:
                        csv.writer(fh).writerow(
                            [step, rate, loss_total, loss_policy, loss_value, len(buffer)]
                        )
                save_checkpoint(
                    trainer_net, optimizer, checkpoint_dir / f"step-{step}.pt", step=step, solve_rate=rate
                )
    return trainer_net


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learner", description="Self-play trainer for the generalized 24 game"
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda[:index]")
    parser.add_argument("--precision", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--episodes-per-iter", type=int, default=8)
    parser.add_argument("--train-steps-per-iter", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inference-batch", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-instances", type=int, default=32)
    parser.add_argument("--eval-simulations", type=int, default=64)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-file", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(make_game=make_pycore24_game, **vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
