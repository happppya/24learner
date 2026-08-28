"""Transport microbenchmark: in-process PyO3 vs subprocess NDJSON.

Compares the two candidate bridges for MCTS policy-inference calls (engine
state query + network prediction) at single-state and small-batch sizes.
Everything is seeded; medians are reported to stay robust to scheduler noise.

Run:  python notes/bench_transport.py
"""

from __future__ import annotations

import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import pycore24
import torch

from learner.inference import InferenceServer
from learner.network import SetTransformer24

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "core" / "target" / "release" / "engine.exe"

SEED = 20260826
N_ELEMS = 6  # mid-curriculum set size
BATCH = 32  # MCTS -> GPU batch target
QUERY_REPEATS = 500  # engine state-query repeats (fast path)
FULL_REPEATS = 60  # full policy-call repeats (torch forward dominates)
BATCH_TRIALS = 5


def build_engine() -> None:
    if ENGINE.exists():
        return
    build = subprocess.run(
        ["cargo", "build", "--release", "--bin", "engine"],
        cwd=ROOT / "core",
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise RuntimeError(f"engine build failed:\n{build.stderr[-2000:]}")


def make_states(rng: random.Random, count: int) -> list[tuple[list[float], list[int], float]]:
    states = []
    for _ in range(count):
        values = [round(rng.uniform(-99.0, 99.0), 3) for _ in range(N_ELEMS)]
        depths = [rng.randint(0, 3) for _ in range(N_ELEMS)]
        target = round(rng.uniform(-100.0, 100.0), 3)
        states.append((values, depths, target))
    return states


class EngineProc:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [str(ENGINE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def rpc(self, payload: dict) -> dict:
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("engine closed the pipe")
        return json.loads(line)

    def close(self) -> None:
        self.proc.terminate()
        self.proc.wait()


def med_p90(ns_times: list[int]) -> tuple[float, float]:
    return statistics.median(ns_times) / 1e3, statistics.quantiles(ns_times, n=100)[89] / 1e3


def timed(fn) -> list[int]:
    """Run a callable repeatedly, returning per-call wall times in ns."""
    times = []
    fn()  # warmup (spawns nothing for PyO3; warms engine line buffer)
    for _ in range(QUERY_REPEATS):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    return times


def main() -> None:
    build_engine()
    rng = random.Random(SEED)
    states = make_states(rng, count=BATCH)
    values, depths, target = states[0]
    elems = [[v, d] for v, d in zip(values, depths, strict=True)]

    engine = EngineProc()
    try:
        # One-time cost: spawning the engine subprocess.
        t0 = time.perf_counter_ns()
        engine.rpc({"cmd": "actions", "elems": elems, "target": target})
        spawn_ns = time.perf_counter_ns() - t0

        # --- Engine state query: legal_actions in-process vs over the pipe. ---
        pyo3_q = timed(lambda: pycore24.GameState(values, target, depths=depths).legal_actions())
        ndjson_q = timed(lambda: engine.rpc({"cmd": "actions", "elems": elems, "target": target}))

        # --- Network-only cost (identical in both transports). ---
        torch.manual_seed(SEED)
        model = SetTransformer24(dim=64, heads=4, num_inducing=16, num_layers=2)
        server = InferenceServer(model, device="cpu", precision="auto", max_batch=BATCH)
        try:
            net_single = []
            for _ in range(FULL_REPEATS):
                t0 = time.perf_counter_ns()
                server.infer(values, depths, target)
                net_single.append(time.perf_counter_ns() - t0)
            net_single = net_single[5:]  # drop warmup

            net_batch = []
            for _ in range(BATCH_TRIALS + 1):
                t0 = time.perf_counter_ns()
                futures = [server.submit(*s) for s in states[:BATCH]]
                for f in futures:
                    f.result()
                net_batch.append(time.perf_counter_ns() - t0)
            net_batch = net_batch[1:]  # drop first trial (torch one-time warmup)

            # --- Full policy-inference call, single state. ---
            def full_pyo3(k):
                v, d, t = states[k % BATCH]
                pycore24.GameState(v, t, depths=d).legal_actions()
                server.infer(v, d, t)

            def full_ndjson(k):
                v, d, t = states[k % BATCH]
                engine.rpc(
                    {"cmd": "actions", "elems": [[x, y] for x, y in zip(v, d, strict=True)], "target": t}
                )
                server.infer(v, d, t)

            full_pyo3_t = []
            for k in range(FULL_REPEATS):
                t0 = time.perf_counter_ns()
                full_pyo3(k)
                full_pyo3_t.append(time.perf_counter_ns() - t0)
            full_ndjson_t = []
            for k in range(FULL_REPEATS):
                t0 = time.perf_counter_ns()
                full_ndjson(k)
                full_ndjson_t.append(time.perf_counter_ns() - t0)

            # --- Full policy-inference call, batch of 32 states. ---
            def batch_pyo3():
                futures = []
                for v, d, t in states[:BATCH]:
                    pycore24.GameState(v, t, depths=d).legal_actions()
                    futures.append(server.submit(v, d, t))
                for f in futures:
                    f.result()

            def batch_ndjson():
                futures = []
                for v, d, t in states[:BATCH]:
                    engine.rpc(
                        {"cmd": "actions", "elems": [[x, y] for x, y in zip(v, d, strict=True)], "target": t}
                    )
                    futures.append(server.submit(v, d, t))
                for f in futures:
                    f.result()

            batch_pyo3_t = []
            for _ in range(BATCH_TRIALS):
                t0 = time.perf_counter_ns()
                batch_pyo3()
                batch_pyo3_t.append(time.perf_counter_ns() - t0)
            batch_ndjson_t = []
            for _ in range(BATCH_TRIALS):
                t0 = time.perf_counter_ns()
                batch_ndjson()
                batch_ndjson_t.append(time.perf_counter_ns() - t0)
        finally:
            server.close()
    finally:
        engine.close()

    q_pyo3 = med_p90(pyo3_q)
    q_ndjson = med_p90(ndjson_q)
    f_pyo3 = med_p90(full_pyo3_t)
    f_ndjson = med_p90(full_ndjson_t)
    b_pyo3 = statistics.mean(batch_pyo3_t) / 1e6
    b_ndjson = statistics.mean(batch_ndjson_t) / 1e6
    ns = statistics.median(net_single) / 1e6
    nb = statistics.mean(net_batch) / 1e6

    print(f"engine spawn (first rpc): {spawn_ns / 1e6:.1f} ms")
    print(
        f"network-only: single {ns:.2f} ms/call, batch-{BATCH} {nb:.2f} ms total "
        f"({nb / BATCH * 1000:.0f} us/call)"
    )
    print()
    print("legal_actions query (N=6):")
    print(f"  in-process  GameState.legal_actions()  median {q_pyo3[0]:8.1f} us   p90 {q_pyo3[1]:8.1f} us")
    print(
        f"  subprocess  engine.exe rpc 'actions'   median {q_ndjson[0]:8.1f} us   p90 {q_ndjson[1]:8.1f} us"
    )
    print(f"  ratio subprocess/in-process: {q_ndjson[0] / q_pyo3[0]:.1f}x")
    print()
    f_pyo3_ms = f_pyo3[0] / 1e3
    f_ndjson_ms = f_ndjson[0] / 1e3
    print("full policy call (state query + infer), single state:")
    print(f"  in-process : {f_pyo3_ms:7.2f} ms median")
    print(f"  subprocess : {f_ndjson_ms:7.2f} ms median   (+{f_ndjson_ms - f_pyo3_ms:.2f} ms)")
    print()
    print(f"full policy call, batch of {BATCH}:")
    print(f"  in-process : {b_pyo3:7.2f} ms mean")
    print(f"  subprocess : {b_ndjson:7.2f} ms mean   (+{b_ndjson - b_pyo3:.2f} ms)")
    overhead = (b_ndjson - b_pyo3) / b_pyo3 * 100
    print(f"  subprocess adds {overhead:.1f}% wall time; network is {nb:.2f} ms of the total")


if __name__ == "__main__":
    main()
