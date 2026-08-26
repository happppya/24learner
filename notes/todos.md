# Roadmap / Todos

Living checklist. Mark items `[x]` when done; append new work under the right phase.

## Phase 0 — Project Bootstrap

- [x] Verify toolchains (cargo 1.93, Python 3.14, torch 2.13 cp314 wheels)
- [x] Repo layout, `.gitignore`, README, AGENTS.md, notes/
- [x] Rust crate skeleton with rule-engine invariants + unit tests
- [x] Python package skeleton (reference env, Set Transformer, train stub) + tests
- [x] Virtualenv with CUDA torch 2.13.0+cu130 installed
- [x] Device abstraction: auto CUDA→CPU selection, bf16/fp16 autocast policy (`learner/device.py`)

## Phase 1 — Rust Core Engine

- [ ] Exhaustive DFS solver over binary ops (baseline correctness oracle)
- [ ] Property/fuzz tests: Rust vs Python reference env agreement on random states
- [ ] Shaped-reward function ported to Rust (`-ln(1+min dist) - λt`)
- [ ] Benchmarks: actions/sec for N ∈ {2, 4, 8, 16, 32, 64, 100}

## Phase 2 — Bridge & Batched Inference

- [ ] PyO3 bindings exposing `State`, `legal_actions`, `apply`
- [ ] Central inference queue: MCTS workers batch requests (32–64) to GPU
- [ ] Decide transport: in-process PyO3 vs subprocess IPC (see design.md open questions)

## Phase 3 — Network

- [ ] Validate ISAB scaling to N=100 (memory + latency profile)
- [ ] Top-K (K=16) masked softmax policy head wired to MCTS priors
- [ ] FP16/BF16 autocast parity checks vs FP32 outputs
- [ ] Re-baseline throughput on RTX 3090 rig (iGPU box runs CPU fallback meanwhile)

## Phase 4 — Self-Play Infrastructure

- [ ] Parallel MCTS workers with PUCT + Dirichlet root noise
- [ ] Shared-RAM replay buffer (target 32 GB ring buffer)
- [ ] Episode logging, dataset versioning

## Phase 5 — Training Loop

- [ ] AlphaZero joint loss (CE on visit distributions + MSE on outcomes)
- [ ] Curriculum: start N ∈ [2, 5], widen toward N = 100
- [ ] Evaluator: fresh-challenger gauntlet between checkpoints
- [ ] Track solve-rate vs number of simulations (scaling curve)

## Phase 6 — Hardening

- [ ] Numerical edge-case suite (denormals, ±1e8 boundary, pow domain traps)
- [ ] Reproducibility: pinned seeds + config snapshots per run
- [ ] CI: cargo test + pytest + ruff + clippy on push
