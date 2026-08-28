"""Minimal training run: N=2 only, small model, modest settings for CPU."""

import random
import time

from learner.train import default_instance, make_pycore24_game, train


def n2_instance(rng: random.Random):
    """Instance with exactly 2 elements."""
    return default_instance(rng, n_min=2, n_max=2)


def main():
    t0 = time.time()
    print(f"Starting training at {time.strftime('%H:%M:%S')}")
    print("Config: N=2, 50 steps, 4 workers, 50 sims, 4 episodes/iter")
    print("=" * 60)

    train(
        make_game=make_pycore24_game,
        device="cpu",
        seed=42,
        workers=4,
        simulations=50,
        episodes_per_iter=4,
        train_steps_per_iter=4,
        batch_size=32,
        inference_batch=16,
        top_k=16,
        max_steps=32,
        replay_capacity=50_000,
        steps=50,
        lr=3e-4,
        dirichlet_alpha=0.3,
        eval_every=10,
        eval_instances=16,
        eval_simulations=50,
        checkpoint_dir="checkpoints/small_run",
        log_file="runs/small_run.csv",
        instance_factory=n2_instance,
    )

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
