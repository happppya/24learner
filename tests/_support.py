import random

from learner.env import Binary, Elem, State, Unary

SEED = 20260825


def encode_action(action):
    if isinstance(action, Binary):
        return {"kind": "binary", "i": action.i, "j": action.j, "op": action.op}
    return {"kind": "unary", "i": action.i, "op": action.op}


def decode_action(payload):
    if payload["kind"] == "binary":
        return Binary(i=payload["i"], j=payload["j"], op=payload["op"])
    return Unary(i=payload["i"], op=payload["op"])


def action_key(action):
    if isinstance(action, Binary):
        return ("binary", action.i, action.j, action.op)
    return ("unary", action.i, None, action.op)


def rust_key(payload):
    if payload["kind"] == "binary":
        return ("binary", payload["i"], payload["j"], payload["op"])
    return ("unary", payload["i"], None, payload["op"])


def random_state(rng, n_min=2, n_max=8):
    n = rng.randint(n_min, n_max)
    values = [round(rng.uniform(-99.0, 99.0), 3) for _ in range(n)]
    depths = [rng.randint(0, 3) for _ in range(n)]
    target = round(rng.uniform(-100.0, 100.0), 3)
    pairs = [[v, d] for v, d in zip(values, depths, strict=True)]
    state = State(tuple(Elem(v, d) for v, d in zip(values, depths, strict=True)), target)
    return pairs, state


def py_solve(state, budget):
    failed = set()

    def dfs(s, b):
        if s.solved():
            return True
        if b == 0:
            return False
        key = tuple(sorted((e.value.hex(), e.depth) for e in s.elems))
        if key in failed:
            return False
        for action in s.legal_actions():
            nxt = s.step(action)
            if nxt is not None and dfs(nxt, b - 1):
                return True
        failed.add(key)
        return False

    return dfs(state, budget)


def reference_random(seed):
    return random.Random(seed)
