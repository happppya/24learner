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

### 2026-08-25 — Phase 1: solver scope, memo soundness, parity transport

- **Solver spans the full action space** (binary + unary), not binary-only: depth tags cap unary
  chains between merges and merges bound resets, so the reachable space is finite and exhaustive
  DFS terminates. Failures memoized on sorted `(value.to_bits(), depth)` fingerprints.
- **Memo soundness:** identical fingerprints imply identical element count, which pins search
  depth (merges strictly shrink sets; no cycle returns to an exact value+depth multiset), so
  remaining budget is depth-determined and caching failures across branches is safe.
- **Budget semantics:** solved-state detection precedes budget accounting (terminal recognition
  costs nothing); every child at `budget == 1` must still be visited because any could solve at
  zero further cost. An early-break optimization here silently dropped solutions (caught by the
  `{2,3,5}→30` unit test).
- **Parity transport pre-PyO3:** NDJSON subprocess bridge (`core/src/bin/engine.rs`) driven by
  pytest. Python owns case generation (single RNG authority); Rust evaluates. This doubles as
  the IPC-fallback prototype for the Phase 2 transport decision.
- **Reward mirrored verbatim:** $R = -\ln(1+\min_v|v-T|) - \lambda t$, default $\lambda = 10^{-2}$
  both sides; terminal bonus `SUCCESS_BONUS = 100`.

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
`resolve_device("auto")` prefers CUDA and falls back to CPU; autocast bf16/fp16 on CUDA, fp32
by default on CPU (bf16 on CPU measured both slower and drift-prone — see 2026-08-26 log). MCTS workers stay CPU threads regardless of inference backend.

### 2026-08-26 — Phase 2: PyO3 bindings, batched inference queue, padding fix

- **Bindings landed:** `pycore24` exposes `GameState` (constructor with optional depths,
  `elems`/`target`/`steps` getters, `solved`, rewards, `cloned`, `legal_actions`, `step`).
  Rejections return `False` and leave state untouched; API misuse raises `ValueError`.
  Parity vs the Python reference is randomized in `tests/test_bridge_parity.py`.
- **Inference queue semantics:** `BatchingQueue.drain` returns a batch immediately when
  anything is pending (up to `max_size`), else blocks up to `timeout_s` and returns `None`
  when idle. Consumers must distinguish idle-`None` from closed-`None` via `is_closed()`
  (an earlier draft looped forever on idle queues — that hang is fixed and regression-tested).
- **`InferenceServer`:** one background consumer thread batches `submit` requests into padded
  tensors (pad mask per sample), runs `model` in eval + no_grad under `torch.autocast`
  (bf16/fp16 CUDA, fp32 CPU via `learner.device`), and resolves each future with
  numpy `Prediction` (binary/unary logits sliced to the sample's own N, plus value).
  Autostarts by default; `close()` drains pending work then joins.
- **Padding leak fixed:** `SetTransformer24` previously never passed `pad_mask` to the
  encoder ISAB stack or PMA pooling, so zero-valued padded tokens perturbed real positions
  through the inducing points and the value pool — mixed-size batches disagreed with
  single-state inference. Encoder is now a `ModuleList` threaded with `pad_mask`, and PMA
  gets an extended mask. Regression test pins exact parity.

### 2026-08-26 — Phase 2: transport decision (in-process PyO3 vs subprocess NDJSON)

Settled with `notes/bench_transport.py` (seeded, N=6 states, dim=64 Set Transformer,
CPU, batch 32; re-run with `python notes/bench_transport.py`). Median/p90 per call,
means for batches; ranges span repeated runs (scheduler noise on this workstation):

| path | in-process PyO3 | subprocess NDJSON | delta |
| --- | ---: | ---: | ---: |
| engine spawn (first rpc) | — | ~2.5–3 ms (one-time) | — |
| `legal_actions` query | ~30 µs (p90 ~30 µs) | ~100 µs (p90 ~120–130 µs) | 3–4× |
| full policy call, single state | ~4.7–5.3 ms | ~4.8–5.3 ms | within noise (~0–0.2 ms) |
| full policy call, batch-32 | ~27 ms | ~33–37 ms | +6–10 ms (~25–35%) |
| network only (both paths) | single ~4.7 ms · batch-32 ~25–27 ms (~0.8 ms/call) | | |

Reading: the network forward dominates every call (~4.7 ms single, ~0.8 ms/call batched), so
the transport delta matters mainly when state queries cross the boundary. In-process PyO3 is
3–4× cheaper per state query and adds no serialization. The measured batch-32 delta is an
**upper bound**: it assumes every `legal_actions` crosses the pipe; if MCTS is hosted in Rust
(the design intent), state queries are native in both transports and only the batched network
round trip crosses, whose cost is bounded by the ~100 µs/call pipe figure above.

**Decision: in-process PyO3 is the primary transport** — bindings are already built and
parity-tested (`tests/test_bridge_parity.py`), latency is lowest, and torch releases the GIL
during kernels so Python-hosted MCTS threads can run while inference executes. Keep the NDJSON
`engine` binary as the parity oracle and crash-isolation fallback. Revisit only if Phase 4
profiling shows GIL contention from Python-hosted orchestration; the subprocess layout with
batched IPC is the escape hatch and its cost is bounded by these numbers.

### 2026-08-26 — Phase 3: top-K policy priors for MCTS

`learner.network.top_k_priors(binary_logits, unary_logits, legal_actions, k=16)` turns the
network logits into the candidate set MCTS expands. Contract:

- **Input actions** are in the canonical transport form from
  `pycore24.GameState.legal_actions()`: `("binary", i, j, op)` / `("unary", i, 0, op)`, the
  same shape the NDJSON engine and the bridge speak — no env-dataclass dependency in the
  network module.
- Scores are logit lookups: `binary_logits[i, j, op]` / `unary_logits[i, op]` (op indexed by
  the fixed `BINARY_OPS`/`UNARY_OPS` order shared with the engine). Top `k` by score are
  kept and softmaxed, so priors sum to 1 and come back ranked, as `[(action, prior), ...]`.
- **Non-finite guard:** scores that are not finite (padding/self-pair `-inf` from the network
  mask) are dropped before selection, so MCTS never ingests NaN priors even if a caller
  passes a sloppy action list; engine-legal actions are always finite in practice.
- Per-state, not batched: MCTS expands one node at a time; batching the mask path remains
  `top_k_action_mask`'s job for any future GPU-side pipeline.

Parity is pinned two ways in `tests/test_network.py`: a brute-force reference softmax must
match selection order and values, and a real `SetTransformer24` forward over `pycore24`
legal actions must yield finite, normalized priors contained in the legal set.

### 2026-08-26 — Phase 5: first training loop

`learner/train.py` replaces the stub with a working loop:

- **Actor/critic split:** the `SelfPlayPool` plays with a separate actor net while Adam
  updates the critic; the actor re-syncs every iteration (`load_state_dict`). Keeps the
  inference server's eval-mode forwards separate from training-mode updates and mirrors
  the eventual distributed layout.
- **Per iteration:** pool runs `episodes_per_iter` episodes into the `ReplayBuffer`,
  `train_steps_per_iter` minibatches drive Adam on `alphazero_loss` (batch moved to the
  resolved device), then the actor syncs. Uniform instances come from `instance_factory`
  (default `default_instance`, N ∈ [2, 5] — the initial curriculum range).
- **Solve-rate log:** every `eval_every` steps (and the final step), `evaluate_solve_rate`
  plays each held-out instance with the actor policy at temperature 0 and records the
  fraction solved; printed, appended to a CSV (`step, solve_rate, loss, policy, value,
  buffer`), and carried into the checkpoint meta (`step-N.pt` via `save_checkpoint`,
  `load_checkpoint` restores net + optimizer + meta). Checkpoints/logs land in
  gitignored `checkpoints/` / `runs/`.
- Self-play uses Dirichlet root noise (default α = 0.3) for exploration; eval is
  noise-free. On a first smoke run (3 steps, N ∈ [2, 5], untrained net) the joint loss
  decreased 5.09 → 4.77 with solve rate 0.0 — expected before the curriculum teaches
  solvable instances.

### 2026-08-26 — Phase 4: dataset versioning

- **Per-line schema version:** every NDJSON episode line carries `"version": SCHEMA_VERSION`
  (currently 1), written by `episode_to_json`/`append_episode`. Versioning per line rather
  than via a file header keeps concurrent appends from multiple writers readable and lets
  a single file mix generations.
- **Read-side handling** (`episode_from_json`/`read_episodes`): untagged lines are legacy
  v0 and parse silently (the v0 shape is byte-compatible with v1 — the format has never
  changed); an explicit unknown version raises `ValueError` naming both the found and
  expected versions, so a future format change fails loudly instead of corrupting the
  replay silently.

### 2026-08-26 — Phase 4: shared-RAM replay buffer

`learner/replay.py` `ReplayBuffer` is the trainer's memory:

- **Stores steps, not episodes** — episodes are flattened into `(step, outcome)` examples
  on add, so minibatch sampling draws uniformly over positions (standard AlphaZero). The
  retrospective outcome rides along with each step as the value target.
- **Ring semantics:** capacity in steps, oldest-first eviction via head pointer; `items()`
  exposes insertion order for inspection/tests. Uniform sampling draws **with replacement**,
  so a minibatch is always full even when the buffer is sparsely populated.
- **`sample(batch_size, rng)`** returns the collated batch `(values, depths, targets,
  pad_mask, loss_targets)` directly in the `alphazero_loss` input format — the trainer
  forwards and calls the joint loss with zero glue. Seeded `rng` makes sampling
  reproducible.
- **32 GB sizing:** capacity is in steps; a step is on the order of a few KB in RAM for
  N ≈ 6 / top-K 16, so the 32 GB target corresponds to roughly 10 M steps. Byte-accurate
  accounting is deferred until the buffer holds real curriculum data.

### 2026-08-26 — Phase 4: parallel self-play workers + Dirichlet root noise

- **Dirichlet root noise** (`MCTS(dirichlet_alpha=..., noise_weight=0.25)`): one draw per
  search, mixed into root child priors as `(1-ε)p + ε·ξ` at pre-expansion, so all
  simulations of that search share the perturbed priors (standard AlphaZero variant).
  Seedable through `torch.manual_seed`; defaults off so existing searches are unchanged.
- **Server-backed policy/value** (`make_server_policy_value`): routes a single node's
  evaluation through the shared `InferenceServer` instead of a direct forward. Returns
  byte-identical priors and the same value as `make_policy_value` (same fp32 logits);
  this is what makes batching pay off when many threads evaluate concurrently.
- **`SelfPlayPool`** (`learner/selfplay.py`): N worker threads, each with its own MCTS
  instance and seeded RNG (thread-local trees), all submitting through one shared
  `InferenceServer`. Episodes are split ceil(N/workers) per worker and returned in a
  lock-protected list. Context-managed (`close()` joins the server thread).
- Consistent with the rollout decision, this is the **Python-side reference** for
  parallelism; the eventual Rust-hosted workers (design intent) will be parity-checked
  against it. On this workstation (CPU, tiny model) 4 workers ran 8 episodes ~1.9× faster
  than 1 worker; the batching win scales with model size and GPU.

### 2026-08-26 — Phase 5: AlphaZero joint loss

`learner/loss.py` implements the training objective recorded in the roadmap:

- **Policy term:** CE between the normalized MCTS visit distribution and the network's
  softmax over the *same* actions — i.e. only the top-K actions the search actually
  expanded (the visit target's support). Non-visited actions' logits are ignored, matching
  the Top-K convention; single-action visits contribute zero CE.
- **Value term:** MSE against the retrospective episode outcome (+1 solved / 0 unsolved,
  per the episode-logging decision). Equal weighting: `total = policy + value`.
- **`collate_steps(steps, outcomes)`** batches logged self-play steps into padded network
  inputs (values/depths/targets/pad_mask) plus per-sample loss targets, exactly like the
  inference-server collation; samples without visits contribute only the value term.
- Logit lookup shares `learner.network.action_logit` with `top_k_priors` (single source of
  op ordering). Loss math is pinned by a hand-computed case and a perfect-prediction test
  (CE reaches the target distribution's entropy, MSE → 0); a full collate→forward→backward
  test exercises the whole path. Wiring into the training loop is deferred to the Phase
  4/5 loop (replay buffer, curriculum, evaluator).

### 2026-08-26 — Phase 3: self-play episode logging

`learner/selfplay.py` turns the rollout into logged episodes the trainer can consume:

- **`play_episode(game, mcts, simulations, ...)`** plays one game by sampling the MCTS
  visit distribution. Temperature scales sampling weights as `counts ** (1/T)` (T = 0 →
  deterministic argmax; default T = 1 → proportional). Pass a seeded `rng` for
  reproducible playouts.
- **Per-step record** (`Step`): state features (`elems` values+depths, `target`), the
  played `action`, the root's priors, raw `visits` (the policy target once normalized),
  and `root_value`. The value target is retrospective: the episode `outcome`, +1.0 if
  solved else 0.0 (single-player puzzle, consistent with the search's terminal value and
  the value head's tanh scale).
- **Format:** NDJSON — one compact JSON object per episode (`outcome` + `steps[]`), action
  tuples encoded as lists. `append_episode`/`read_episodes` round-trip losslessly; the
  trainer batches these into the Phase 4 replay buffer. Dataset versioning (ids, schema
  headers) is deferred to Phase 4.

### 2026-08-26 — Phase 3: MCTS rollout with priors (Python reference)

`learner/mcts.py` implements the minimal search step that consumes `top_k_priors`:
PUCT selection, expansion from priors, value backup, and a visit-distribution target.

- **Where it lives:** Python. The priors and network are Python-side, and Phase 1 set the
  pattern of a Python reference implementation validated by parity before the Rust worker
  lands — the Phase 4 Rust-hosted workers (`core/src/mcts.rs` already has the PUCT
  mechanics) will be parity-tested against this rollout.
- **Game protocol:** the search loop is transport-agnostic over `legal_actions`/`apply`/
  `solved`/`elems`/`target`; `PyCore24Game` adapts the in-process bindings, so actions stay
  in the canonical tuple form end-to-end (`("binary", i, j, op)` / `("unary", i, 0, op)`).
- **Rejection handling:** `legal_actions` is an unfiltered generator (domain/depth checks
  happen in `apply`), so expansion drops unapplicable candidates and **renormalizes** the
  surviving priors — the prior sum over root children stays 1.
- **Root pre-expansion:** the root is expanded with network priors before the first
  simulation (Leela-style), so every simulation descends through exactly one root child and
  the visit target sums exactly to `simulations` (naive expand-on-first-visit leaves
  `simulations - 1`).
- **Terminals:** solved pays `+1.0` (matches the value head's tanh scale; the shaped-reward
  bonus is a training signal, not a search signal); a `max_depth` (64) cap bounds unary
  chains. `make_policy_value(net)` runs a direct forward pass; the batched path remains
  `InferenceServer`'s job once Phase 4 workers exist.

### 2026-08-26 — Phase 3: autocast fidelity check → CPU defaults to fp32

Measured with `notes/check_autocast.py` on the default model (dim=128, heads=4,
num_inducing=16, num_layers=3, CPU): fp32 vs bf16 autocast, seeds {0,1,2} × batch
{1,16,64} plus a server-style mixed-N padded batch-16. Worst-case drift over valid
(real) positions:

| metric | batch-1 | batch-16 | batch-64 | mixed-N padded 16 |
| --- | ---: | ---: | ---: | ---: |
| binary logits max abs | 1.69 | 3.52 | 4.20 | 3.29 |
| binary logits mean abs | 0.26 | 0.23 | 0.25 | — |
| unary logits max abs | 0.10 | 0.14 | 0.20 | 0.15 |
| value max abs | 0.002 | 0.06 | 0.06 | 0.01 |
| policy prior max |p−p'| | 0.010 | 0.32 | 0.29 | 0.25 |

Timing (batch-64): fp32 11.6 ms median vs bf16 89.5 ms — bf16 is **7.7× slower** on this
CPU (oneDNN bf16 without hardware bf16 acceleration falls back to emulation).

Reading: bf16's 8-bit mantissa error compounds through the 3 ISAB layers; the pairwise
features (hi·hj) amplify it, so binary logits drift up to ~4 and the induced policy prior
moves by up to 0.32 — that would silently change MCTS priors between fp32 and bf16 runs,
i.e. material. And on CPU it pays nothing.

**Decision:** `learner.device.resolve_precision("auto")` now maps to fp32 on CPU and keeps
bf16 (fp16 explicit) on CUDA — autocast only where it pays. Explicit `precision="bf16"`
on CPU remains allowed for experiments. CUDA-side parity (bf16/fp16 vs fp32 on the 3090)
is still open and should reuse this script on the rig.

## Open Questions

- ~~Bridge transport: in-process PyO3 module vs subprocess with batched IPC.~~ Resolved
  2026-08-26 — in-process PyO3 wins; see decision log above.
- ~~Autocast fidelity on CPU.~~ Resolved 2026-08-26 — CPU defaults to fp32; CUDA bf16/fp16
  parity still to be verified on the 3090 rig (re-run `notes/check_autocast.py` there).
- Inducing-point count $M$ and Top-K $K$: defaults 16/16 pending scaling experiments.
- Whether `pow` earns its combinatorial cost for small $N$ curricula or should be gated off early.
