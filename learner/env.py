from __future__ import annotations

import math
from dataclasses import dataclass

EPSILON = 1e-6
MAX_UNARY_DEPTH = 3
MAGNITUDE_LIMIT = 1e8
SUCCESS_BONUS = 100.0

BINARY_OPS = ("add", "sub", "mul", "div", "pow")
UNARY_OPS = ("neg", "sqrt", "ln")


@dataclass(frozen=True, slots=True)
class Elem:
    value: float
    depth: int = 0


@dataclass(frozen=True, slots=True)
class Unary:
    i: int
    op: str


@dataclass(frozen=True, slots=True)
class Binary:
    i: int
    j: int
    op: str


Action = Unary | Binary


def _eval_binary(a: float, b: float, op: str) -> float | None:
    if op == "add":
        out = a + b
    elif op == "sub":
        out = a - b
    elif op == "mul":
        out = a * b
    elif op == "div":
        if b == 0.0:
            return None
        out = a / b
    elif op == "pow":
        try:
            out = math.pow(a, b)
        except (OverflowError, ValueError):
            return None
    else:
        raise ValueError(f"unknown binary op: {op}")
    if not math.isfinite(out) or abs(out) > MAGNITUDE_LIMIT:
        return None
    return out


def _eval_unary(value: float, depth: int, op: str) -> float | None:
    if depth >= MAX_UNARY_DEPTH:
        return None
    if op == "neg":
        out = -value
    elif op == "sqrt":
        if value < 0.0:
            return None
        out = math.sqrt(value)
    elif op == "ln":
        if value <= 0.0:
            return None
        out = math.log(value)
    else:
        raise ValueError(f"unknown unary op: {op}")
    if not math.isfinite(out) or abs(out) > MAGNITUDE_LIMIT:
        return None
    return out


@dataclass(frozen=True, slots=True)
class State:
    elems: tuple[Elem, ...]
    target: float
    steps: int = 0

    @staticmethod
    def from_values(values, target) -> State:
        return State(tuple(Elem(float(v)) for v in values), float(target))

    def solved(self) -> bool:
        return any(abs(e.value - self.target) < EPSILON for e in self.elems)

    def min_distance(self) -> float:
        return min(abs(e.value - self.target) for e in self.elems)

    def legal_actions(self) -> list[Action]:
        n = len(self.elems)
        actions: list[Action] = [Unary(i=i, op=op) for i in range(n) for op in UNARY_OPS]
        for i in range(n):
            for j in range(i + 1, n):
                for op in ("add", "mul"):
                    actions.append(Binary(i=i, j=j, op=op))
                for op in ("sub", "div", "pow"):
                    actions.append(Binary(i=i, j=j, op=op))
                    actions.append(Binary(i=j, j=i, op=op))
        return actions

    def step(self, action: Action) -> State | None:
        n = len(self.elems)
        match action:
            case Unary(i=i, op=op):
                if not 0 <= i < n:
                    raise ValueError("unary index out of range")
                out = _eval_unary(self.elems[i].value, self.elems[i].depth, op)
                if out is None:
                    return None
                elems = list(self.elems)
                elems[i] = Elem(out, elems[i].depth + 1)
                return State(tuple(elems), self.target, self.steps + 1)
            case Binary(i=i, j=j, op=op):
                if i == j or not (0 <= i < n and 0 <= j < n):
                    raise ValueError("binary indices invalid")
                out = _eval_binary(self.elems[i].value, self.elems[j].value, op)
                if out is None:
                    return None
                kept = [e for k, e in enumerate(self.elems) if k not in (i, j)]
                kept.insert(min(i, j), Elem(out, 0))
                return State(tuple(kept), self.target, self.steps + 1)
            case _:
                raise TypeError(f"unsupported action: {action!r}")


def shaped_reward(state: State, lam: float = 1e-2) -> float:
    return -math.log1p(state.min_distance()) - lam * state.steps


def terminal_reward(state: State) -> float:
    if state.solved():
        return SUCCESS_BONUS
    raise ValueError("terminal_reward called on unsolved state")
