"""Check MCTS solve rate on N=3 solvable instances."""
import random
import pycore24
from learner.instances import solvable_instance
from learner.mcts import MCTS, PyCore24Game, make_policy_value
from learner.selfplay import play_episode
from learner.network import SetTransformer24

net = SetTransformer24()
pv = make_policy_value(net, top_k=16)
rng = random.Random(77)

print("=== MCTS solve check on N=3 solvable instances ===")
solved = 0
total = 15
for i in range(total):
    values, target = solvable_instance(rng, n=3)
    g = PyCore24Game(pycore24.GameState(values, target))
    mcts = MCTS(pv)
    ep = play_episode(g, mcts, simulations=80, rng=random.Random(i+2000), max_steps=32, temperature=0.0)
    if ep.outcome == 1.0:
        solved += 1
    print(f"  {values} -> {target}: outcome={ep.outcome}, steps={len(ep.steps)}")

print(f"\nSolved {solved}/{total} with 100 sims (untrained net)")
