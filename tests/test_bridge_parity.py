import json
import math
import random
import subprocess
from pathlib import Path

import pytest
from _support import (
    SEED,
    action_key,
    decode_action,
    encode_action,
    py_solve,
    random_state,
    rust_key,
)

from learner.env import Binary, Elem, State, shaped_reward

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "core" / "target" / "release" / "engine.exe"


def build_engine():
    build = subprocess.run(
        ["cargo", "build", "--release", "--bin", "engine"],
        cwd=ROOT / "core",
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise RuntimeError(f"engine build failed:\n{build.stderr[-2000:]}")


@pytest.fixture(scope="module")
def engine():
    build_engine()
    proc = subprocess.Popen(
        [str(ENGINE)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    yield proc
    proc.terminate()


def rpc(proc, payload):
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line, "engine closed the pipe"
    return json.loads(line)


def test_legal_actions_multiset_agreement(engine):
    rng = random.Random(SEED)
    for _ in range(120):
        pairs, state = random_state(rng)
        resp = rpc(engine, {"cmd": "actions", "elems": pairs})
        rust = {rust_key(a) for a in resp["actions"]}
        mine = {action_key(a) for a in state.legal_actions()}
        assert rust == mine, f"diverged on elems={pairs}"


def test_apply_agreement_including_rejections(engine):
    rng = random.Random(SEED + 1)
    accepted = rejected = 0
    for _ in range(500):
        pairs, state = random_state(rng)
        action = rng.choice(state.legal_actions())
        expected = state.step(action)
        resp = rpc(
            engine,
            {
                "cmd": "apply",
                "elems": pairs,
                "target": state.target,
                "action": encode_action(action),
            },
        )
        if expected is None:
            rejected += 1
            assert resp == {"ok": False}, f"rust accepted what python rejected: {action}"
        else:
            accepted += 1
            assert resp["ok"] is True, f"rust rejected what python accepted: {action}"
            value, depth = resp["elem"]
            idx = min(action.i, action.j) if isinstance(action, Binary) else action.i
            assert value == expected.elems[idx].value
            assert depth == expected.elems[idx].depth
    assert accepted > 300
    assert rejected > 20


def test_out_of_range_indices_rejected_by_both_sides(engine):
    with pytest.raises(ValueError):
        State.from_values([1.0, 2.0], 24.0).step(Binary(i=0, j=5, op="add"))
    resp = rpc(
        engine,
        {
            "cmd": "apply",
            "elems": [[1.0, 0], [2.0, 0]],
            "target": 24.0,
            "action": {"kind": "binary", "i": 0, "j": 5, "op": "add"},
        },
    )
    assert "error" in resp


def test_shaped_reward_agreement(engine):
    rng = random.Random(SEED + 2)
    for _ in range(150):
        pairs, state = random_state(rng, n_min=1, n_max=10)
        lam = rng.choice([0.0, 1e-4, 1e-2, 0.5])
        resp = rpc(
            engine,
            {"cmd": "reward", "elems": pairs, "target": state.target, "lambda": lam},
        )
        assert math.isclose(resp["reward"], shaped_reward(state, lam), rel_tol=1e-12, abs_tol=1e-15)


@pytest.mark.parametrize("trial", range(14))
def test_solver_agreement_on_small_instances(engine, trial):
    rng = random.Random(SEED + 3 + trial)
    pool = [-6, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8]
    n = rng.randint(2, 3)
    values = [float(rng.choice(pool)) for _ in range(n)]
    target = float(rng.choice([-30, -5, 0, 1, 6, 7, 13, 24, 36]))
    budget = 3

    state = State(tuple(Elem(v) for v in values), target)
    expected = py_solve(state, budget)
    resp = rpc(
        engine,
        {"cmd": "solve", "elems": [[v, 0] for v in values], "target": target, "budget": budget},
    )
    assert resp["solvable"] == expected, f"trial {trial}: values={values} target={target}"
    if resp["solvable"]:
        assert resp["verified"]
        replay = state
        for payload in resp["plan"]:
            replay = replay.step(decode_action(payload))
            assert replay is not None
        assert replay.solved()
