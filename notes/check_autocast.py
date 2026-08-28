"""Autocast fidelity check: fp32 vs bf16 autocast on the default SetTransformer24.

Measures max-abs / relative drift of binary and unary logits plus value, and
the drift of the induced policy prior (softmax over the valid action space),
across seeds and batch sizes. Batches are padded to the max N exactly like
`InferenceServer._forward`. A quick timing comparison tells whether bf16 pays
on this backend.

Run:  python notes/check_autocast.py
"""

from __future__ import annotations

import statistics
import time

import torch

from learner.network import SetTransformer24

SEEDS = (0, 1, 2)
BATCH_SIZES = (1, 16, 64)
N_ELEMS = 6
MIXED_N = (3, 4, 5, 6, 7, 8, 9, 10)  # server-style padded batch
TIMING_REPEATS = 5


def make_inputs(batch: int, ns: list[int], seed: int):
    torch.manual_seed(seed)
    max_n = max(ns)
    values = torch.rand(batch, max_n) * 200.0 - 100.0
    depths = torch.randint(0, 4, (batch, max_n))
    target = torch.rand(batch) * 200.0 - 100.0
    pad_mask = torch.zeros(batch, max_n, dtype=torch.bool)
    for i, n in enumerate(ns):
        pad_mask[i, n:] = True
    return values, depths, target, pad_mask


def forward(net, values, depths, target, pad_mask):
    with torch.no_grad():
        return net(values, depths, target, pad_mask=pad_mask)


def valid_slices(out, ns):
    """Yield per-sample (binary, unary, value) slices over real positions only."""
    for i, n in enumerate(ns):
        yield out.binary_logits[i, :n, :n, :], out.unary_logits[i, :n, :], out.value[i]


def logit_drift(a, b):
    """a = fp32 (float32), b = autocast (converted to float32)."""
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    scale = a.abs().max().item()
    max_rel = (diff / a.abs().clamp_min(1e-8)).max().item()
    return max_abs, mean_abs, max_rel / scale if scale > 0 else 0.0


def policy_prior_drift(bin_a, bin_b, un_a, un_b):
    """Max |p_fp32 - p_bf16| over the softmax of the joint action space."""
    logits_a = torch.cat([bin_a.flatten(), un_a.flatten()])
    logits_b = torch.cat([bin_b.flatten(), un_b.flatten()])
    pa = logits_a.softmax(dim=-1)
    pb = logits_b.softmax(dim=-1)
    return (pa - pb).abs().max().item()


def main() -> None:
    torch.manual_seed(0)
    net = SetTransformer24()  # defaults: dim=128, heads=4, num_inducing=16, num_layers=3
    net.eval()

    print("model: SetTransformer24(dim=128, heads=4, num_inducing=16, num_layers=3), CPU\n")

    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = SetTransformer24()
        net.eval()
        for batch in BATCH_SIZES:
            ns = [N_ELEMS] * batch
            values, depths, target, pad_mask = make_inputs(batch, ns, seed)
            fp32 = forward(net, values, depths, target, pad_mask)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                cast = forward(net, values, depths, target, pad_mask)
            bin_ma = bin_me = un_ma = un_me = val_ma = pol = 0.0
            for (ba, ua, va), (bb, ub, vb) in zip(
                valid_slices(fp32, ns), valid_slices(cast, ns), strict=True
            ):
                ma, me, _ = logit_drift(ba, bb)
                bin_ma = max(bin_ma, ma)
                bin_me += me
                ma, me, _ = logit_drift(ua, ub)
                un_ma = max(un_ma, ma)
                un_me += me
                val_ma = max(val_ma, (va - vb).abs().item())
                pol = max(pol, policy_prior_drift(ba, bb, ua, ub))
            rows.append((seed, batch, bin_ma, bin_me / batch, un_ma, un_me / batch, val_ma, pol))

    # Mixed-N padded batch, server-style, one seed.
    seed, batch = 0, 16
    ns = [MIXED_N[i % len(MIXED_N)] for i in range(batch)]
    values, depths, target, pad_mask = make_inputs(batch, ns, seed)
    fp32 = forward(net, values, depths, target, pad_mask)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        cast = forward(net, values, depths, target, pad_mask)
    mixed = [0.0, 0.0, 0.0, 0.0]
    for (ba, ua, va), (bb, ub, vb) in zip(valid_slices(fp32, ns), valid_slices(cast, ns), strict=True):
        ma, _, _ = logit_drift(ba, bb)
        mixed[0] = max(mixed[0], ma)
        ma, _, _ = logit_drift(ua, ub)
        mixed[1] = max(mixed[1], ma)
        mixed[2] = max(mixed[2], (va - vb).abs().item())
        mixed[3] = max(mixed[3], policy_prior_drift(ba, bb, ua, ub))

    print(
        f"{'seed':<5} {'batch':<6} {'bin max_abs':<11} {'bin mean_abs':<12} "
        f"{'un max_abs':<11} {'un mean_abs':<12} {'val max_abs':<11} {'policy max':<10}"
    )
    for seed, batch, bma, bme, uma, ume, vma, pol in rows:
        print(
            f"{seed:<5} {batch:<6} {bma:<11.5f} {bme:<12.5f} {uma:<11.5f} "
            f"{ume:<12.5f} {vma:<11.5f} {pol:<10.5f}"
        )
    print(
        f"mixed-N padded batch-16 (seed 0): bin {mixed[0]:.5f} un {mixed[1]:.5f} "
        f"val {mixed[2]:.5f} policy {mixed[3]:.5f}"
    )

    # Timing: does bf16 pay on this backend?
    torch.manual_seed(0)
    net = SetTransformer24()
    net.eval()
    values, depths, target, pad_mask = make_inputs(64, [N_ELEMS] * 64, 0)
    for label, fn in (
        ("fp32  ", lambda: forward(net, values, depths, target, pad_mask)),
        (
            "bf16  ",
            lambda: torch.autocast(device_type="cpu", dtype=torch.bfloat16)(
                lambda: forward(net, values, depths, target, pad_mask)
            )(),
        ),
    ):
        fn()  # warmup
        times = []
        for _ in range(TIMING_REPEATS):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        print(f"batch-64 forward ({label}): median {statistics.median(times) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
