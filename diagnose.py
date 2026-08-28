"""Diagnostic: can the solver solve N=2 instances? Does MCTS find solutions?"""
import random
import pycore24
from learner.mcts import MCTS, PyCore24Game, make_policy_value
from learner.network import SetTransformer24

# 1. Check solver on N=2
print("=== Solver check (N=2) ===")
for seed in range(20):
    rng = random.Random(seed)
    n = 2
    values = [round(rng.uniform(-10.0, 10.0), 3) for _ in range(n)]
    target = round(rng.uniform(-10.0, 10.0), 3)
    s = pycore24.GameState(values, target)
    actions = list(s.legal_actions())
    print(f"  {values} -> {target}: {len(actions)} legal actions, solved={s.solved()}")

# 2. Check MCTS + network on one instance
print("\n=== MCTS check (N=2, untrained net) ===")
net = SetTransformer24()
pv = make_policy_value(net, top_k=16)
s = pycore24.GameState([3.0, 5.0], 8.0)
g = PyCore24Game(s)
mcts = MCTS(pv)
from learner.selfplay import play_episode
ep = play_episode(g, mcts, simulations=50, rng=random.Random(0), max_steps=32, temperature=1.0)
print(f"  Episode outcome={ep.outcome}, steps={len(ep.steps)}")
for i, step in enumerate(ep.steps):
    print(f"    step {i}: action={step.action}, root_value={step.root_value:.4f}")

# 3. Check N=2 solve rate via brute-force solver
print("\n=== Brute solve rate (N=2, 50 random instances) ===")
solved = 0
for seed in range(50):
    rng = random.Random(seed)
    values = [round(rng.uniform(-10.0, 10.0), 3) for _ in range(2)]
    target = round(rng.uniform(-10.0, 10.0), 3)
    s = pycore24.GameState(values, target)
    if s.solved():
        solved += 1
print(f"  Solved {solved}/50 by initial state")

# Try with solver
print("\n=== Solver check via engine binary ===")
import subprocess, json
for seed in range(10):
    rng = random.Random(seed)
    values = [round(rng.uniform(-10.0, 10.0), 3) for _ in range(2)]
    target = round(rng.uniform(-10.0, 10.0), 3)
    s = pycore24.GameState(values, target)
    actions_before = len(list(s.legal_actions()))
    # Just apply first legal action and check
    s2 = pycore24.GameState(values, target)
    for a in s2.legal_actions():
        ok = s2.step(*a[1:])
        if ok:
            new_elems = list(s2.elems)
            print(f"  {values}->{target}: applied {a}, elems now {new_elems}, solved={s2.solved()}")
            break
