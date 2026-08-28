"""Training run with solvable instances: N=2 only, 100 steps."""
import random
import time
from learner.instances import solvable_instance
from learner.train import make_pycore24_game, train


def solvable_factory(rng: random.Random):
    return solvable_instance(rng, n=2, val_range=50.0)


def main():
    t0 = time.time()
    print(f"Starting training at {time.strftime('%H:%M:%S')}")
    print("Config: N=2 solvable instances, 100 steps, 4 workers, 50 sims")
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
        steps=100,
        lr=3e-4,
        dirichlet_alpha=0.3,
        eval_every=10,
        eval_instances=20,
        eval_simulations=50,
        checkpoint_dir="checkpoints/solvable_run",
        log_file="runs/solvable_run.csv",
        instance_factory=solvable_factory,
    )

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
