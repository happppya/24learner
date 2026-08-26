# AGENTS.md

Guidance for humans and AI agents working in this repository.

## Mission

Build an AlphaZero-style solver for the generalized 24 game: a Rust rule/search engine guided by a
Set Transformer policy-value network trained via self-play. Read `notes/design.md` for the full
formulation before making architectural changes.

## Repository Layout

```
core/           Rust crate `core24`: state, operations, guardrails, MCTS
learner/        Python package: reference environment, Set Transformer, training entry point
tests/          Python tests (pytest)
notes/          design.md (decisions, open questions) and todos.md (roadmap)
```

## Environment

- Windows, PowerShell 5.1. There is no `&&`; chain with `;` or `if ($?) { cmd2 }`.
- Rust stable via cargo (edition 2024). No C++ toolchain is installed; do not add C/C++ sources.
- Python 3.14 virtualenv at `.venv\`. PyTorch is the full CUDA build (cu130); reinstall with
  `.venv\Scripts\python.exe -m pip install "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130`.
- **Two machines:** this workstation has an integrated GPU only (torch runs on CPU here);
  the training rig has an RTX 3090. Keep all learner code device-agnostic through
  `learner.device`; never hardcode `cuda` or `cpu`.
- Quote paths containing spaces; prefer absolute or `workdir`-relative invocations.

## Commands

| Action                  | Command                                                        | Workdir    |
| ----------------------- | -------------------------------------------------------------- | ---------- |
| Python setup (once)     | `python -m venv .venv; .venv\Scripts\python.exe -m pip install -e ".[dev]"` | root |
| Python tests            | `.venv\Scripts\python.exe -m pytest`                           | root       |
| Lint Python             | `.venv\Scripts\python.exe -m ruff check .`                     | root       |
| Format Python           | `.venv\Scripts\python.exe -m ruff format .`                    | root       |
| Build + test Rust       | `cargo test`                                                   | `core`     |
| Lint Rust               | `cargo clippy --all-targets`                                   | `core`     |
| Format Rust             | `cargo fmt`                                                    | `core`     |

Run `cargo test`, `pytest`, and `ruff check` before claiming any change is complete.

## Conventions

- **Single source of truth:** the Rust crate owns all game semantics (operators, guards,
  constants). The Python `learner.env` module is a reference mirror used for tests and
  prototyping until PyO3 bindings land; keep the two semantically identical.
- **Invariant constants** (do not change without updating `notes/design.md`):
  - `EPSILON = 1e-6` (terminal tolerance)
  - `MAX_UNARY_DEPTH = 3` (consecutive unary applications per element)
  - `MAGNITUDE_LIMIT = 1e8` (prune anything larger in absolute value)
  - Terminal success bonus `+100`, step penalty coefficient lambda small positive.
- **Style:** no comments unless explaining non-obvious math; prefer expressive names. Rust
  errors as `Option`/`Result`, Python returns `None` for invalid transitions and reserves
  exceptions for API misuse.
- **Determinism:** thread seeds through every experiment; record seed choices in `notes/design.md`.
- **Never commit:** secrets, `.venv/`, `core/target/`, model checkpoints, datasets.
- **Git:** do not commit unless explicitly asked. Do not rewrite history.

## Working Agreement

- Update `notes/todos.md` as items complete; append newly discovered work there rather than
  leaving it implicit.
- Record non-obvious decisions (and rejected alternatives) in `notes/design.md` under Decision Log
  with the date.
