from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np
import torch

from learner.device import resolve_device, resolve_precision


@dataclass(slots=True)
class Batch:
    items: list[object]
    futures: list[Future]


@dataclass(slots=True)
class InferenceRequest:
    values: list[float]
    depths: list[int]
    target: float


@dataclass(slots=True)
class Prediction:
    binary_logits: np.ndarray
    unary_logits: np.ndarray
    value: float


def resolve_batch(batch: Batch, fn: Callable[[object], object]) -> None:
    for item, future in zip(batch.items, batch.futures, strict=True):
        try:
            future.set_result(fn(item))
        except Exception as exc:
            future.set_exception(exc)


class BatchingQueue:
    """Collects submissions and hands them to a consumer in batches.

    `drain` immediately returns everything buffered (up to `max_size`), so a
    busy pipeline naturally gathers large GPU batches while keeping latency
    low; an idle consumer blocks up to `timeout_s` waiting for arrivals.
    """

    def __init__(self, max_size: int, timeout_s: float = 0.005) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self.timeout_s = timeout_s
        self._condition = threading.Condition()
        self._pending: list[tuple[object, Future]] = []
        self._closed = False

    def submit(self, request: object) -> Future:
        future: Future = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("queue is closed")
            self._pending.append((request, future))
            self._condition.notify_all()
        return future

    def drain(self, blocking: bool = True) -> Batch | None:
        with self._condition:
            while True:
                if self._pending:
                    taken = self._pending[: self.max_size]
                    del self._pending[: self.max_size]
                    return Batch(
                        items=[item for item, _ in taken],
                        futures=[future for _, future in taken],
                    )
                if self._closed or not blocking:
                    return None
                self._condition.wait(self.timeout_s)
                if not self._pending:
                    return None

    def is_closed(self) -> bool:
        with self._condition:
            return self._closed

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class InferenceServer:
    """Batched policy/value inference backed by a BatchingQueue.

    A single background consumer thread drains requests from the queue and
    runs the model forward on padded batches (up to `max_batch`), so MCTS
    workers can call `submit` from many threads and the GPU/CPU sees large
    contiguous batches instead of one sample at a time. The model is moved to
    `device` and put in eval mode; forward passes run under `torch.no_grad`
    with autocast when the resolved precision is not fp32.
    """

    def __init__(
        self,
        model,
        *,
        max_batch: int = 64,
        timeout_s: float = 0.005,
        device: str | torch.device | None = None,
        precision: str = "auto",
        autostart: bool = True,
    ) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else resolve_device("auto")
        self.model.to(self.device)
        self.model.eval()
        dtype = resolve_precision(precision, self.device)
        self._autocast = dtype is not None
        self._dtype = dtype
        self._queue = BatchingQueue(max_size=max_batch, timeout_s=timeout_s)
        self._thread: threading.Thread | None = None
        if autostart:
            self.start()

    def submit(self, values, depths, target) -> Future:
        request = InferenceRequest(list(values), list(depths), float(target))
        return self._queue.submit(request)

    def infer(self, values, depths, target) -> Prediction:
        return self.submit(values, depths, target).result()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("server already started")
        self._thread = threading.Thread(target=self._serve, name="inference-server", daemon=True)
        self._thread.start()

    def close(self, timeout: float = 30.0) -> None:
        self._queue.close()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def __enter__(self) -> InferenceServer:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _serve(self) -> None:
        while True:
            batch = self._queue.drain(blocking=True)
            if batch is None:
                if self._queue.is_closed():
                    break
                continue
            self._run_batch(batch)

    def _run_batch(self, batch: Batch) -> None:
        try:
            predictions = self._forward(batch.items)
        except Exception as exc:
            for future in batch.futures:
                future.set_exception(exc)
            return
        for future, prediction in zip(batch.futures, predictions, strict=True):
            future.set_result(prediction)

    def _forward(self, requests: list[InferenceRequest]) -> list[Prediction]:
        batch = len(requests)
        max_n = max(len(r.values) for r in requests)
        values = torch.zeros(batch, max_n, dtype=torch.float32)
        depths = torch.zeros(batch, max_n, dtype=torch.int64)
        targets = torch.zeros(batch, dtype=torch.float32)
        pad_mask = torch.zeros(batch, max_n, dtype=torch.bool)
        for i, request in enumerate(requests):
            n = len(request.values)
            values[i, :n] = torch.tensor(request.values, dtype=torch.float32)
            depths[i, :n] = torch.tensor(request.depths, dtype=torch.int64)
            targets[i] = request.target
            pad_mask[i, n:] = True
        values = values.to(self.device)
        depths = depths.to(self.device)
        targets = targets.to(self.device)
        pad_mask = pad_mask.to(self.device)
        with torch.no_grad():
            if self._autocast:
                with torch.autocast(device_type=self.device.type, dtype=self._dtype):
                    out = self.model(values, depths, targets, pad_mask=pad_mask)
            else:
                out = self.model(values, depths, targets, pad_mask=pad_mask)
        predictions: list[Prediction] = []
        for i, request in enumerate(requests):
            n = len(request.values)
            predictions.append(
                Prediction(
                    binary_logits=out.binary_logits[i, :n, :n, :].float().cpu().numpy(),
                    unary_logits=out.unary_logits[i, :n, :].float().cpu().numpy(),
                    value=float(out.value[i].float().cpu()),
                )
            )
        return predictions
