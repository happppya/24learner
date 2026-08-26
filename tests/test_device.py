import pytest
import torch

from learner.device import resolve_device, resolve_precision


def test_auto_falls_back_to_available_backend():
    device = resolve_device("auto")
    assert device.type in {"cpu", "cuda"}


def test_explicit_cpu_roundtrip():
    assert resolve_device("cpu").type == "cpu"


def test_cuda_without_gpu_raises():
    if torch.cuda.is_available():
        pytest.skip("CUDA present on this machine")
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_precision_rules_on_cpu():
    cpu = torch.device("cpu")
    assert resolve_precision("bf16", cpu) == torch.bfloat16
    assert resolve_precision("fp32", cpu) is None
    assert resolve_precision("auto", cpu) == torch.bfloat16


def test_fp16_rejected_off_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA present on this machine")
    with pytest.raises(RuntimeError):
        resolve_precision("fp16", torch.device("cpu"))
