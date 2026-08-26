# Design Notes — 24learner

Capture of the project formulation plus a running decision log. Update this file whenever an
architectural choice is made or revisited.

## Problem Formulation

Given an initial multiset $S_0 = \{v_1, \dots, v_N\}$, $N \ge 2$, and target $T \in \mathbb{R}$,
construct a sequence of unary/binary algebraic operations reducing $S_0$ to a state containing a
value within $\epsilon = 10^{-6}$ of $T$.

Search challenges:

- Reducing size-$N$ to one element needs exactly $N-1$ binary operations plus unbounded unary ops.
- Unary loops prevented by per-element depth tags $d(v_i) \in \{0,1,2,3\}$; unary increments,
  binary resets to 0, $d = 3$ forbids further unary application.
- Action space at step $t$: $O(N^2 + N)$ — unguided search is intractable for large $N$.

## Operator Set (fixed in Phase 0)

- Binary: `add`, `sub`, `mul`, `div`, `pow` (ordered ops generated in both argument orders)
- Unary: `neg`, `sqrt`, `ln`
- Deferred: `gamma` / factorial — expensive and numerically hairy; revisit if curriculum stalls.

## Guardrails

- Reject non-finite intermediates; prune $|v| > 10^8$.
- Domain rules: `sqrt(x)` requires $x \ge 0$; `ln(x)` requires $x > 0$; division by zero rejected;
  `pow` traps NaN (e.g. negative base with fractional exponent).
- Fourth consecutive unary on one element is invalid.

## Neural Architecture

- Element embedding: scalar value + learned depth embedding → linear projection.
- Encoder: stacked ISAB blocks, $O(N \cdot M)$ with $M$ inducing points (default $M = 16$).
- Target conditioning: target encoded as a query token via cross-attention over set tokens.
- Policy: pairwise bilinear scoring against per-operator embeddings over symmetric pair features
  $[h_i + h_j,\ h_i \odot h_j,\ |h_i - h_j|]$; unary head is a per-token linear projection.
  Top-$K$ ($K = 16$) filtering feeds MCTS.
- Value: PMA pooling (single seed) → MLP → scalar in $[-1, 1]$.

## Reward

$$R(S_t) = -\ln(1 + \min_{v \in S_t}|v - T|) - \lambda t$$

Terminal: $\min_v |v - T| < 10^{-6}$ pays $+100$ and ends the rollout. Lambda kept small so the
distance term dominates shaping.

## Decision Log

### 2026-08-25 — Core language: Rust (not C++)

Machine has cargo 1.93 but no C++ compiler (no cmake/msvc/clang). Rust gives memory-safe parallel
MCTS workers without a build-toolchain detour. Crate named `core24` (`core` clashes with the
language prelude crate).

### 2026-08-25 — Python 3.14 + full CUDA torch 2.13

cp314 wheels confirmed on PyPI. User opted for the full CUDA build despite iGPU-only hardware;
CPU/DirectML paths still work with this wheel. Revisit if disk footprint becomes a problem.

### 2026-08-25 — Monorepo layout

Root `pyproject.toml` (package `learner`) beside `core/` Rust crate. Single venv, single test
entry point; bridge code later lives in `core/` behind PyO3 feature flag.

### 2026-08-25 — Device-portable execution (iGPU dev box → RTX 3090 target)

Dev workstation has an integrated GPU only; the training rig carries an RTX 3090 (sm_86,
supported by CUDA 13). Chose `torch 2.13.0+cu130` from the PyTorch index over the accidental
PyPI CPU wheel so one environment serves both machines. `learner.device` implements the contract:
`resolve_device("auto")` prefers CUDA and falls back to CPU; bf16 autocast on both backends;
fp16 gated to CUDA. MCTS workers stay CPU threads regardless of inference backend.

## Open Questions

- Bridge transport: in-process PyO3 module vs subprocess with batched IPC. PyO3 likely wins on
  latency; decide in Phase 2 with a microbenchmark.
- Inducing-point count $M$ and Top-K $K$: defaults 16/16 pending scaling experiments.
- Whether `pow` earns its combinatorial cost for small $N$ curricula or should be gated off early.
