from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

BINARY_OPS = ("add", "sub", "mul", "div", "pow")
UNARY_OPS = ("neg", "sqrt", "ln")
NEG_INF = float("-inf")


@dataclass(slots=True)
class PolicyOutput:
    binary_logits: torch.Tensor
    unary_logits: torch.Tensor
    value: torch.Tensor


class MAB(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_h = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h, _ = self.attn(
            self.ln_q(q),
            kv,
            kv,
            key_padding_mask=pad_mask,
            need_weights=False,
        )
        x = q + h
        return x + self.ff(self.ln_h(x))


class ISAB(nn.Module):
    def __init__(self, dim: int, heads: int, num_inducing: int) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(num_inducing, dim).normal_(std=0.02))
        self.mab_in = MAB(dim, heads)
        self.mab_out = MAB(dim, heads)

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        seeds = self.inducing.unsqueeze(0).expand(tokens.size(0), -1, -1)
        h = self.mab_in(seeds, tokens, pad_mask)
        return self.mab_out(tokens, h)


class PMA(nn.Module):
    def __init__(self, dim: int, heads: int, num_seeds: int = 1) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.empty(num_seeds, dim).normal_(std=0.02))
        self.mab = MAB(dim, heads)

    def forward(self, tokens: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        seeds = self.seeds.unsqueeze(0).expand(tokens.size(0), -1, -1)
        return self.mab(seeds, tokens, pad_mask)


class SetTransformer24(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        heads: int = 4,
        num_inducing: int = 16,
        num_layers: int = 3,
        top_k: int = 16,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.depth_embed = nn.Embedding(4, 8)
        self.elem_proj = nn.Linear(9, dim)
        self.target_proj = nn.Linear(3, dim)
        self.encoder = nn.ModuleList([ISAB(dim, heads, num_inducing) for _ in range(num_layers)])
        self.condense = MAB(dim, heads)
        self.pair_weight = nn.Parameter(torch.empty(len(BINARY_OPS), 3 * dim).normal_(std=0.02))
        self.pair_bias = nn.Parameter(torch.zeros(len(BINARY_OPS)))
        self.unary_head = nn.Linear(dim, len(UNARY_OPS))
        self.pma = PMA(dim, heads)
        self.value_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(
        self,
        values: torch.Tensor,
        depths: torch.Tensor,
        target: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> PolicyOutput:
        batch, num = values.shape
        if pad_mask is None:
            pad_mask = torch.zeros(batch, num, dtype=torch.bool, device=values.device)
        token_in = torch.cat([values.unsqueeze(-1), self.depth_embed(depths.clamp(max=3))], dim=-1)
        tokens = self.elem_proj(token_in)
        for block in self.encoder:
            tokens = block(tokens, pad_mask)
        target_in = torch.stack([target, torch.log1p(target.abs()), torch.sign(target)], dim=-1)
        target_token = self.condense(self.target_proj(target_in).unsqueeze(1), tokens, pad_mask)
        value_mask = torch.cat(
            [pad_mask, torch.zeros(batch, 1, dtype=torch.bool, device=values.device)], dim=1
        )
        pooled = self.pma(torch.cat([tokens, target_token], dim=1), value_mask)
        value = torch.tanh(self.value_head(pooled.squeeze(1))).squeeze(-1)

        binary_logits = self._pair_logits(tokens)
        unary_logits = self.unary_head(tokens)

        valid = ~pad_mask
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        binary_logits = binary_logits.masked_fill(~pair_valid.unsqueeze(-1), NEG_INF)
        unary_logits = unary_logits.masked_fill(pad_mask.unsqueeze(-1), NEG_INF)
        return PolicyOutput(binary_logits=binary_logits, unary_logits=unary_logits, value=value)

    def _pair_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        hi = tokens.unsqueeze(2)
        hj = tokens.unsqueeze(1)
        feats = torch.cat([hi + hj, hi * hj, (hi - hj).abs()], dim=-1)
        return torch.einsum("bijc,oc->bijo", feats, self.pair_weight) + self.pair_bias


def top_k_action_mask(
    binary_logits: torch.Tensor, unary_logits: torch.Tensor, k: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = binary_logits.size(0)
    flat_bin = binary_logits.reshape(batch, -1)
    flat_un = unary_logits.reshape(batch, -1)
    flat = torch.cat([flat_bin, flat_un], dim=-1)
    finite = torch.isfinite(flat)
    scores = flat.masked_fill(~finite, NEG_INF)
    topk = scores.topk(min(k, scores.size(-1)), dim=-1).indices
    selected = torch.zeros_like(finite).scatter_(dim=-1, index=topk, value=True) & finite
    mask_bin = selected[:, : flat_bin.size(1)].view_as(binary_logits)
    mask_un = selected[:, flat_bin.size(1) :].view_as(unary_logits)
    return mask_bin, mask_un


def action_logit(binary_logits: torch.Tensor, unary_logits: torch.Tensor, action: tuple) -> torch.Tensor:
    """Logit for one canonical action tuple: ("binary", i, j, op) or ("unary", i, 0, op)."""
    kind, i, j, op = action
    if kind == "binary":
        return binary_logits[i, j, BINARY_OPS.index(op)]
    return unary_logits[i, UNARY_OPS.index(op)]


def top_k_priors(
    binary_logits: torch.Tensor,
    unary_logits: torch.Tensor,
    legal_actions: list[tuple],
    k: int = 16,
) -> list[tuple[tuple, float]]:
    """Softmax-normalized priors over the top-K legal actions for MCTS.

    `legal_actions` uses the canonical transport form returned by
    `pycore24.GameState.legal_actions()`: ("binary", i, j, op) or
    ("unary", i, 0, op). Scores are the corresponding logits; the top `k` by
    score are kept and softmaxed, so priors sum to 1. Non-finite scores are
    dropped before selection (engine-legal actions should never hit them), so
    MCTS never receives NaN priors. Returns [(action, prior), ...] ranked by
    descending score.
    """
    scores = torch.empty(len(legal_actions), dtype=torch.float32, device=binary_logits.device)
    for idx, action in enumerate(legal_actions):
        scores[idx] = action_logit(binary_logits, unary_logits, action)
    order = torch.isfinite(scores).nonzero().squeeze(-1)
    order = order[scores[order].argsort(descending=True)][:k]
    top = [legal_actions[i] for i in order.tolist()]
    priors = scores[order].softmax(dim=-1)
    return [(action, float(prior)) for action, prior in zip(top, priors.tolist(), strict=True)]
