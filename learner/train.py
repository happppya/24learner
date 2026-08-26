from __future__ import annotations

import argparse

import torch

from learner.device import resolve_device, resolve_precision
from learner.env import State, shaped_reward


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learner", description="Self-play trainer for the generalized 24 game"
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda[:index]")
    parser.add_argument("--precision", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inference-batch", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--replay-cap-gb", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype = resolve_precision(args.precision, device)
    state = State.from_values([2.0, 3.0, 4.0], 24.0)
    print(f"device={device.type} amp_dtype={amp_dtype} seed={args.seed}")
    print(f"sanity: reward({list(state.elems[0].value for _ in [0])}...) ", end="")
    print(f"{shaped_reward(state):.4f}")
    print("training loop pending Phase 4/5 (see notes/todos.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
