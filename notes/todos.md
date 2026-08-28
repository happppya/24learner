# Roadmap / Todos

Living checklist. Mark items `[x]` when done; append new work under the right phase.

## Phase 0 — Project Bootstrap

- [x] Verify toolchains (cargo 1.93, Python 3.14, torch 2.13 cp314 wheels)
- [x] Repo layout, `.gitignore`, README, AGENTS.md, notes/
- [x] Rust crate skeleton with rule-engine invariants + unit tests
- [x] Python package skeleton (reference env, Set Transformer, train stub) + tests
- [x] Virtualenv with CUDA torch 2.13.0+cu130 installed
- [x] Device abstraction: auto CUDA→CPU selection, bf16/fp16 autocast policy (`learner/device.py`)

## Phase 1 — Rust Core Engine ✅ (2026-08-25)

- [x] Exhaustive DFS solver over the full action space (`core24::solver::solve_plan`,
      failure-memoized by value/depth fingerprints, node-budget bounded, plan verification)
- [x] Property/fuzz parity Rust vs Python: `tests/test_parity.py` drives the NDJSON
      `engine` binary — action sets, apply outcomes (incl. rejections), rewards,
      and solver verdicts all agree on seeded random instances
- [x] Shaped reward ported to Rust (`min_distance`, `shaped_reward`, `terminal_reward`)
- [x] Benchmarks (`cargo run --release --bin bench`, this workstation, release profile):

| N     | 2        | 4        | 8        | 16       | 32       | 64       | 100      |
| ----- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| act/s | 30.6 M   | 30.5 M   | 30.0 M   | 25.2 M   | 20.6 M   | 16.9 M   | 13.8 M   |

## Phase 2 — Bridge & Batched Inference ✅ (2026-08-26)

- [x] PyO3 bindings exposing `State`, `legal_actions`, `apply` (`bindings/pycore24`,
      `GameState` with `legal_actions`/`step`, verified by `tests/test_bridge_parity.py`)
- [x] Central inference queue: MCTS workers batch requests (32–64) to GPU
      (`learner/inference.py`: `BatchingQueue` + `InferenceServer` — threaded consumer,
      padded mixed-size batches, autocast, `submit`/`infer` API)
- [x] Padding is inference-safe: `SetTransformer24` now threads `pad_mask` through the
      encoder ISABs and PMA pooling (regression-tested; mixed-size batches match
      single-state forward exactly)
- [x] Decide transport: in-process PyO3 vs subprocess IPC — `notes/bench_transport.py`
      microbenchmark: PyO3 3–4× cheaper per state query, single calls within noise,
      batch-32 +25–35% wall time (upper bound; shrinks if only the network crosses
      the pipe). **Decision: in-process PyO3 primary**, NDJSON engine kept as parity
      oracle/fallback (see design.md)

## Phase 3 — Network

- [ ] Validate ISAB scaling to N=100 (memory + latency profile)
- [x] Top-K (K=16) masked softmax policy head wired to MCTS priors
      (`learner.network.top_k_priors`: engine-legal actions → ranked top-K
      candidates with softmax-normalized priors in canonical transport form,
      non-finite scores dropped;      reference-parity + model/bindings integration
      tests in `tests/test_network.py`)
- [x] MCTS rollout with priors (`learner/mcts.py`: PUCT search consuming `top_k_priors`,
      root pre-expansion so the visit target sums exactly to simulations, priors
      renormalized over applicable actions, depth cap, `PyCore24Game` adapter; tested in
      `tests/test_mcts.py`) — Python reference for the Phase 4 Rust-hosted workers
- [x] Self-play episode logging (`learner/selfplay.py`: `play_episode` samples the MCTS
      visit policy with temperature, records per-step states/priors/visits/root value +
      outcome, serializes episodes as NDJSON — one compact object per line; tested in
      `tests/test_selfplay.py`)
- [x] FP16/BF16 autocast parity checks vs FP32 outputs — CPU done in
      `notes/check_autocast.py`: bf16 drifts up to ~4 abs on binary logits and 0.32 on
      policy priors, and is 7.7× slower; `auto` now defaults to fp32 on CPU. CUDA-side
      bf16/fp16 parity still to verify on the 3090 rig (re-run the script there)
- [ ] Re-baseline throughput on RTX 3090 rig (iGPU box runs CPU fallback meanwhile)

## Phase 4 — Self-Play Infrastructure ✅ (2026-08-26)

- [x] Parallel MCTS workers with PUCT + Dirichlet root noise
      (`learner/selfplay.py` `SelfPlayPool`: per-worker MCTS + RNG sharing one
      `InferenceServer` for batched inference; Dirichlet root noise in
      `learner/mcts.py`, mixed at root pre-expansion, seedable; worker-count,
      determinism, and noise tests) — Python-side reference, Rust workers later
- [x] Shared-RAM replay buffer (target 32 GB ring buffer)
      (`learner/replay.py` `ReplayBuffer`: capacity-bounded ring over flattened
      (step, outcome) examples, oldest-first eviction, seeded uniform minibatch
      sampling collated straight into `alphazero_loss`; tests in `tests/test_replay.py`)
      — capacity in steps; 32 GB target sized via bytes-per-step estimate (see design.md)
- [x] Episode logging (NDJSON via `learner/selfplay.py`; landed in Phase 3)
- [x] Dataset versioning (`SCHEMA_VERSION` per NDJSON episode line: legacy
      untagged lines parse silently, unknown versions raise with a migration
      message; no file header, so concurrent appends stay readable)

## Phase 5 — Training Loop

- [x] AlphaZero joint loss (CE on visit distributions + MSE on outcomes)
      (`learner/loss.py`: `alphazero_loss` supervises only the top-K visited actions,
      `collate_steps` batches logged steps + retrospective outcomes; math pinned by
      hand-computation and perfect-prediction tests in `tests/test_loss.py`)
- [x] First training loop (`learner/train.py`: `SelfPlayPool` → `ReplayBuffer` → Adam on
      the joint loss, actor/critic net split with periodic re-sync, solve-rate eval on
      held-out instances, CSV solve-rate log + checkpoints; `evaluate_solve_rate` and
      `save/load_checkpoint` helpers; tests in `tests/test_train.py`)
- [ ] Curriculum: start N ∈ [2, 5], widen toward N = 100
- [ ] Evaluator: fresh-challenger gauntlet between checkpoints
- [ ] Track solve-rate vs number of simulations (scaling curve)

## Phase 6 — Hardening

- [ ] Numerical edge-case suite (denormals, ±1e8 boundary, pow domain traps)
- [ ] Reproducibility: pinned seeds + config snapshots per run
- [x] Local validation: cargo test + pytest + ruff on restored source/tests (2026-08-28); CI automation remains open
