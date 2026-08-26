# 24learner

AlphaZero-style reinforcement learning for the **generalized 24 game**: reduce a multiset of reals
$S_0 = \{v_1, \dots, v_N\}$ ($N \ge 2$) to a value within $\epsilon = 10^{-6}$ of a target $T$ using
binary operators $(+,\ -, \times,\ \div,\ \hat{})$ and unary operators $(\mathrm{neg},\ \sqrt{\cdot},\ \ln)$
under strict numerical guardrails.

A **Rust core** implements the validated state/rule engine and hosts parallel MCTS workers.
A **PyTorch Set Transformer** supplies policy priors and value estimates through a batched
CPU-iGPU inference pipeline.

## Architecture

```
+---------------------------------------------------------------------------------+
|                                 Rust Core (core/)                               |
|  +---------------------------+                   +---------------------------+  |
|  |     State & Rule Engine   |                   |     Parallel MCTS Workers |  |
|  |  - Math validation        | <---------------- |  - Tree expansion         |  |
|  |  - Unary depth tracking   |                   |  - PUCT selection         |  |
|  |  - Magnitude pruning      |                   |  - Top-K action filtering |  |
|  +---------------------------+                   +---------------------------+  |
+---------------------------------------------------------------------------------+
                                         | (Batched Inference Requests)
                                         v
+---------------------------------------------------------------------------------+
|                          Python + PyTorch (learner/)                            |
|  Set Transformer:  ISAB encoder -> target cross-attention                       |
|                    Policy head (pairwise bilinear scoring)                      |
|                    Value head  (PMA pooling -> V(s) in [-1, 1])                 |
+---------------------------------------------------------------------------------+
```

## Key Mechanics

- **Unary depth tags:** every element carries $d(v_i) \in \{0..3\}$ counting consecutive unary
  applications; a 4th is pruned. Binary operations reset depth to 0.
- **Numerical guards:** domain violations rejected (`sqrt(x<0)`, `ln(x<=0)`, `x/0`, NaN/Inf),
  values with $|v| > 10^8$ pruned.
- **Shaped reward:** $R(S_t) = -\ln(1 + \min_{v \in S_t}|v-T|) - \lambda t$, with a $+100$
  terminal bonus when $\min_v |v-T| < 10^{-6}$.
- **Top-K policy filter:** the $O(N^2 + N)$ action space is scored by the policy head; only the
  best $K = 16$ candidates enter MCTS expansion.
- **Set Transformer:** ISAB encoder scales as $O(N \cdot M)$ for sets up to $N = 100$; PMA pools
  the set into a fixed-length global vector for the value head.

## Tech Stack

| Layer               | Choice                                        |
| ------------------- | --------------------------------------------- |
| Rule engine/search  | Rust (crate `core24` in `core/`)              |
| Policy-value net    | Python 3.14 + PyTorch 2.x (CUDA 13.0 build)   |
| Training precision  | BF16 (CPU + CUDA) / FP16 (CUDA only)          |
| Hardware            | Dev: iGPU workstation · Target: RTX 3090      |
| Bridge (planned)    | PyO3 bindings + batched inference queue       |

Training code is device-agnostic: `learner.device.resolve_device("auto")` picks CUDA when a GPU +
driver are present and falls back to CPU otherwise, so the same environment runs on both machines.

## Quickstart (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
cargo test --manifest-path core\Cargo.toml
.venv\Scripts\python.exe -m pytest
```

## Repository Layout

```
24learner/
├── AGENTS.md          # conventions and commands for coding agents
├── notes/             # design decisions and living roadmap
├── core/              # Rust crate: state, ops, guards, MCTS
├── learner/           # Python package: env reference impl, network, training
└── tests/             # Python test suite
```

## Status

Pre-alpha scaffold. The roadmap lives in `notes/todos.md`; design decisions and open questions
are recorded in `notes/design.md`. See `AGENTS.md` before contributing.
