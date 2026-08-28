"""Reverse-engineered instance generator: guaranteed-solvable by construction.

For N=2, pick two random values and one binary operation; the result is the
target.  Every instance is solvable in exactly 1 step (the inverse operation
recovers the initial values from the target).  We also include the identity
case (one element already equals the target) for variety.

The generator scales to larger N by chaining operations, but we start with
N=2 for the initial curriculum.
"""

from __future__ import annotations

import math
import random

# Binary operations that are closed over the reals (ignoring domain guards).
# We avoid div-by-zero and pow domain traps by clamping inputs.
_BINARY_OPS = [
    ("add", lambda a, b: a + b),
    ("sub", lambda a, b: a - b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / b if abs(b) > 1e-12 else None),
]


def _safe_generate(rng: random.Random, n: int, val_range: float = 50.0):
    """Generate (values, target) by forward-simulation: build a short expression tree
    from n random leaves and one root operation.

    Returns (values, target) where values has length n and target is the result
    of combining all values via a tree of binary operations.  The instance is
    always solvable because replaying the operations in reverse recovers the
    initial state.

    For n == 2, this is a single binary op.  For n > 2 we chain pairwise.
    """
    # Generate n random values in [-val_range, val_range], avoiding near-zero
    # for division safety
    values = []
    for _ in range(n):
        v = rng.uniform(-val_range, val_range)
        while abs(v) < 0.5:
            v = rng.uniform(-val_range, val_range)
        values.append(round(v, 3))

    # Build target by chaining binary operations left-to-right:
    # result = ((v0 op1 v1) op2 v2) op3 v3 ...
    result = values[0]
    ops_used = []
    for i in range(1, n):
        op_name, op_fn = rng.choice(_BINARY_OPS)
        new_val = op_fn(result, values[i])
        if new_val is None or not math.isfinite(new_val) or abs(new_val) > 1e8:
            # Retry with add (always safe)
            new_val = result + values[i]
            op_name = "add"
        result = new_val
        ops_used.append((op_name, i))

    return values, round(result, 6)


def solvable_instance(rng: random.Random, n: int = 2, val_range: float = 50.0, retries: int = 10):
    """Generate a guaranteed-solvable instance with n elements.

    Tries up to `retries` times to produce a finite, well-bounded target.
    """
    for _ in range(retries):
        values, target = _safe_generate(rng, n, val_range)
        if math.isfinite(target) and abs(target) < 1e8:
            return values, target
    # Fallback: simple addition
    a = round(rng.uniform(1.0, 20.0), 3)
    b = round(rng.uniform(1.0, 20.0), 3)
    return [a, b], round(a + b, 3)


def make_instance_factory(n_min: int = 2, n_max: int = 2, val_range: float = 50.0):
    """Return a factory function compatible with `train(..., instance_factory=...)`.

    Generates solvable instances by forward-construction: the target is the
    result of chaining binary operations over n random values, so the solver
    can always recover the initial state.
    """
    def factory(rng: random.Random):
        n = rng.randint(n_min, n_max)
        return solvable_instance(rng, n=n, val_range=val_range)
    return factory


# --- Quick sanity check ---
if __name__ == "__main__":
    import pycore24

    rng = random.Random(42)
    print("=== Solvable instance generator (N=2) ===")
    for _i in range(20):
        values, target = solvable_instance(rng, n=2)
        s = pycore24.GameState(values, target)
        actions = list(s.legal_actions())
        print(f"  {values} -> {target}: {len(actions)} legal actions, solved={s.solved()}")

    print("\n=== Solvable instance generator (N=3) ===")
    for _i in range(10):
        values, target = solvable_instance(rng, n=3)
        s = pycore24.GameState(values, target)
        print(f"  {values} -> {target}: {len(list(s.legal_actions()))} legal actions")
