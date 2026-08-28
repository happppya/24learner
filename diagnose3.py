"""Deeper diagnosis: check if self-play episodes solve, and why loss spikes."""
import random
import torch
from learner.network import SetTransformer24
from learner.mcts import MCTS, make_policy_value
from learner.selfplay import SelfPlayPool, play_episode
from learner.loss import alphazero_loss, collate_steps
from learner.replay import ReplayBuffer
from learner.train import make_pycore24_game
import pycore24


def n2_instance(rng):
    n = 2
    values = [round(rng.uniform(-10.0, 10.0), 3) for _ in range(n)]
    target = round(rng.uniform(-10.0, 10.0), 3)
    return values, target


# 1. Play a few episodes and check outcomes
print("=== Self-play episodes (untrained net, N=2) ===")
net = SetTransformer24()
pv = make_policy_value(net, top_k=16)
for seed in range(10):
    rng = random.Random(seed)
    values, target = n2_instance(rng)
    g = make_pycore24_game(values, target)
    mcts = MCTS(pv)
    ep = play_episode(g, mcts, simulations=50, rng=random.Random(seed+100), max_steps=32, temperature=1.0)
    print(f"  seed={seed}: {values}->{target}, outcome={ep.outcome}, steps={len(ep.steps)}")

# 2. Run the actual SelfPlayPool and check outcomes
print("\n=== SelfPlayPool batch (4 episodes) ===")
def new_game():
    return make_pycore24_game(*n2_instance(random.Random(42)))

with SelfPlayPool(
    new_game, net,
    workers=2, simulations=50, max_steps=32, top_k=16,
    dirichlet_alpha=0.3, seed=0, max_batch=8,
) as pool:
    episodes = pool.run(4)

solved_count = sum(1 for ep in episodes if ep.outcome == 1.0)
total_steps = sum(len(ep.steps) for ep in episodes)
print(f"  {solved_count}/{len(episodes)} solved, total steps={total_steps}")

# 3. Collate steps and check the loss
print("\n=== Loss check on batch ===")
all_steps = []
all_outcomes = []
for ep in episodes:
    all_steps.extend(ep.steps)
    all_outcomes.extend([ep.outcome] * len(ep.steps))

if all_steps:
    batch = collate_steps(all_steps, all_outcomes)
    values, depths, targets, pad_mask = batch[:4]
    print(f"  Batch shape: values={values.shape}, depths={depths.shape}")
    print(f"  Pad mask sum per sample: {pad_mask.sum(dim=1).tolist()}")

    with torch.no_grad():
        out = net(values, depths, targets, pad_mask=pad_mask)
    print(f"  Binary logits: min={out.binary_logits.min():.4f}, max={out.binary_logits.max():.4f}, mean={out.binary_logits.mean():.4f}")
    print(f"  Value: min={out.value.min():.4f}, max={out.value.max():.4f}, mean={out.value.mean():.4f}")

    total, policy, value = alphazero_loss(out.binary_logits, out.unary_logits, out.value, batch[4])
    print(f"  Loss: total={total.item():.4f}, policy={policy.item():.4f}, value={value.item():.4f}")

    # Check loss targets
    targets_info = batch[4]
    print(f"  Loss targets type: {type(targets_info)}")
    if isinstance(targets_info, dict):
        for k, v in targets_info.items():
            if hasattr(v, 'shape'):
                print(f"    {k}: shape={v.shape}")
            else:
                print(f"    {k}: {type(v)} len={len(v) if hasattr(v, '__len__') else 'N/A'}")
