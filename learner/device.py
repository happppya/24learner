from __future__ import annotations

import torch

_PRECISIONS = ("bf16", "fp16", "fp32")


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    preference = preference.strip()
    if preference.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(preference)


def resolve_precision(precision: str, device: torch.device) -> torch.dtype | None:
    if precision == "auto":
        # bf16 pays on CUDA tensor cores; on CPU it is slower than fp32 and its
        # 8-bit mantissa compounds to material logit/prior drift (see notes/check_autocast.py).
        precision = "bf16" if device.type == "cuda" else "fp32"
    if precision == "fp32":
        return None
    if precision == "bf16":
        if device.type == "cpu":
            return torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None
    if precision == "fp16":
        if device.type == "cuda":
            return torch.float16
        raise RuntimeError("fp16 autocast requires CUDA")
    raise ValueError(f"unknown precision: {precision}; expected one of {_PRECISIONS}")
