"""Intended-vs-realized bfloat16 update statistics, lightened from exp_01m1tdvatwe1rrek9g34p75vje
(src/realization.py there): no shadow curve, no histogram.

Per-step pooled statistics (over a set of coordinates, i = intent, r = realized change, both fp32):
    overshoot rho        sum r^2 / sum i*r
    length ratio         ||r|| / ||i||
    cosine               <i, r> / (||i|| ||r||)
    unmoved fraction     #(r == 0) / N
    exact write agreement #(bf16_rne(p_pre + i) - p_pre == r) / N   (the one-write model)

Cumulative delivery over a whole run (I = sum of intents, R = p_final - p_initial, fp32 per coord):
    cumulative length ratio  ||R|| / ||I||
    cumulative rho           ||R||^2 / <I, R>
    cumulative cosine        <I, R> / (||I|| ||R||)
    never-moved fraction     #(R == 0) / N
All reductions in float64.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def bf16_spacing(p: torch.Tensor) -> torch.Tensor:
    """bfloat16 lattice spacing at each float32 value p (0 where p == 0)."""
    _, e = torch.frexp(p)
    u = torch.ldexp(torch.ones_like(p), (e - 1 - 7).to(torch.int32))
    return torch.where(p == 0, torch.zeros_like(u), u)


def write_model(p_pre: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    """Realized change predicted by one round-to-nearest bfloat16 write of p_pre + i."""
    return (p_pre + i).to(torch.bfloat16).float() - p_pre


@dataclass
class Acc:
    n: int = 0
    n_zero: int = 0
    n_exact: int = 0
    r2: float = 0.0
    ir: float = 0.0
    i2: float = 0.0

    @torch.no_grad()
    def add(self, i: torch.Tensor, r: torch.Tensor, p_pre: torch.Tensor | None = None) -> None:
        i = i.reshape(-1)
        r = r.reshape(-1)
        self.n += i.numel()
        self.n_zero += int((r == 0).sum())
        self.r2 += float((r.double() ** 2).sum())
        self.ir += float((i.double() * r.double()).sum())
        self.i2 += float((i.double() ** 2).sum())
        if p_pre is not None:
            r_hat = write_model(p_pre.reshape(-1), i)
            self.n_exact += int((r_hat == r).sum())

    def merge(self, other: "Acc") -> None:
        for f in ("n", "n_zero", "n_exact", "r2", "ir", "i2"):
            setattr(self, f, getattr(self, f) + getattr(other, f))

    def summary(self) -> dict:
        nan = float("nan")
        return {
            "n": self.n,
            "overshoot": self.r2 / self.ir if self.ir else nan,
            "length_ratio": math.sqrt(self.r2 / self.i2) if self.i2 else nan,
            "cosine": self.ir / math.sqrt(self.r2 * self.i2) if self.r2 and self.i2 else nan,
            "unmoved_fraction": self.n_zero / self.n if self.n else nan,
            "exact_write_agreement": self.n_exact / self.n if self.n else nan,
            "intent_norm": math.sqrt(self.i2),
            "realized_norm": math.sqrt(self.r2),
        }


def cumulative_summary(acc: Acc) -> dict:
    """Same sums, named for the cumulative (whole-run) reading."""
    s = acc.summary()
    return {
        "n": s["n"],
        "cumulative_length_ratio": s["length_ratio"],
        "cumulative_overshoot": s["overshoot"],
        "cumulative_cosine": s["cosine"],
        "never_moved_fraction": s["unmoved_fraction"],
        "intent_norm": s["intent_norm"],
        "realized_norm": s["realized_norm"],
    }
