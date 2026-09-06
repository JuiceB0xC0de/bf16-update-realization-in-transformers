"""Intended-vs-realized bfloat16 update statistics (torch, device-agnostic).

Definitions follow the reviewed Mamba work (workspace analysis/percoord.py):

    intent   i = -(lr / bc1) * m / (sqrt(v) / sqrt(bc2) + eps) - lr * wd * p_pre
    spacing  u = bfloat16 lattice spacing at p_pre  (2^(floor(log2|p|) - 7); 0 at p == 0)
    realized r = p_post - p_pre
    x = |i| / u,  n = r / u

    pooled overshoot  rho = sum r^2 / sum i*r
    length ratio      ||r|| / ||i||
    cosine            <i, r> / (||i|| ||r||)
    unmoved fraction  #(r == 0) / N
    multi-spacing     #(|n| >= 1.5) / #(r != 0)      ("movers travelling more than one spacing")
    round agreement   #(| |n| - round(x) | < 0.51) / #(u > 0)   (percoord.py definition)

Write model: r_hat = bf16_rtn(p_pre + i) - p_pre, applied once with round-to-nearest-even.
    predicted overshoot rho_hat = sum r_hat^2 / sum i*r_hat
    exact write agreement       #(r_hat == r) / N
Shadow curve: i_k = k * i for k in 2^-4 .. 2^4, same write model, same trajectory.

All arithmetic on float32 tensors; reductions in float64.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

SHADOW_FACTORS = [2.0 ** k for k in range(-4, 5)]  # 1/16 .. 16, nine factors
HIST_LOG2_MIN, HIST_LOG2_MAX = -16, 8  # log2(x) bins, one per octave


def bf16_spacing(p: torch.Tensor) -> torch.Tensor:
    """bfloat16 lattice spacing at each float32 value p (0 where p == 0)."""
    _, e = torch.frexp(p)  # |p| = m * 2^e with m in [0.5, 1) -> floor(log2|p|) = e - 1
    u = torch.ldexp(torch.ones_like(p), (e - 1 - 7).to(torch.int32))
    return torch.where(p == 0, torch.zeros_like(u), u)


def intent(m: torch.Tensor, v: torch.Tensor, p_pre: torch.Tensor,
           lr: float, wd: float, eps: float, bc1: float, bc2: float) -> torch.Tensor:
    denom = v.float().sqrt() / math.sqrt(bc2) + eps
    return -(lr / bc1) * m.float() / denom - (lr * wd) * p_pre


def write_model(p_pre: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    """Realized change predicted by one round-to-nearest bfloat16 write of p_pre + i."""
    return (p_pre + i).to(torch.bfloat16).float() - p_pre


@dataclass
class Acc:
    """Pooled sums over a set of coordinates (float64 python scalars)."""
    n: int = 0
    n_zero: int = 0
    n_mover: int = 0
    n_multi: int = 0
    n_lattice: int = 0          # coordinates with u > 0
    n_round_ok: int = 0
    n_exact: int = 0            # r_hat == r
    r2: float = 0.0
    ir: float = 0.0
    i2: float = 0.0
    rhat2: float = 0.0
    irhat: float = 0.0
    shadow_r2: list = field(default_factory=lambda: [0.0] * len(SHADOW_FACTORS))
    shadow_ir: list = field(default_factory=lambda: [0.0] * len(SHADOW_FACTORS))
    shadow_zero: list = field(default_factory=lambda: [0] * len(SHADOW_FACTORS))
    hist: list = field(default_factory=lambda: [0] * (HIST_LOG2_MAX - HIST_LOG2_MIN))

    def add(self, i: torch.Tensor, r: torch.Tensor, p_pre: torch.Tensor,
            shadow: bool = True, hist: bool = False) -> None:
        i = i.reshape(-1).float()
        r = r.reshape(-1).float()
        p_pre = p_pre.reshape(-1).float()
        u = bf16_spacing(p_pre)
        lat = u > 0
        n = torch.where(lat, r / torch.where(lat, u, torch.ones_like(u)), torch.zeros_like(r))
        x = torch.where(lat, i.abs() / torch.where(lat, u, torch.ones_like(u)), torch.zeros_like(r))
        na = n.abs()
        moved = r != 0
        r_hat = write_model(p_pre, i)

        self.n += i.numel()
        self.n_zero += int((~moved).sum())
        self.n_mover += int(moved.sum())
        self.n_multi += int((moved & (na >= 1.5)).sum())
        self.n_lattice += int(lat.sum())
        self.n_round_ok += int((lat & ((na - torch.round(x)).abs() < 0.51)).sum())
        self.n_exact += int((r_hat == r).sum())
        self.r2 += float((r.double() ** 2).sum())
        self.ir += float((i.double() * r.double()).sum())
        self.i2 += float((i.double() ** 2).sum())
        self.rhat2 += float((r_hat.double() ** 2).sum())
        self.irhat += float((i.double() * r_hat.double()).sum())
        if shadow:
            for k, f in enumerate(SHADOW_FACTORS):
                ik = i * f
                rk = write_model(p_pre, ik)
                self.shadow_r2[k] += float((rk.double() ** 2).sum())
                self.shadow_ir[k] += float((ik.double() * rk.double()).sum())
                self.shadow_zero[k] += int((rk == 0).sum())
        if hist:
            xl = x[lat & (x > 0)]
            if xl.numel():
                b = torch.floor(torch.log2(xl)).clamp(HIST_LOG2_MIN, HIST_LOG2_MAX - 1) - HIST_LOG2_MIN
                counts = torch.bincount(b.to(torch.int64), minlength=len(self.hist))
                for j, c in enumerate(counts.tolist()):
                    self.hist[j] += c

    def merge(self, other: "Acc") -> None:
        for f in ("n", "n_zero", "n_mover", "n_multi", "n_lattice", "n_round_ok", "n_exact",
                  "r2", "ir", "i2", "rhat2", "irhat"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        for f in ("shadow_r2", "shadow_ir", "shadow_zero", "hist"):
            setattr(self, f, [a + b for a, b in zip(getattr(self, f), getattr(other, f))])

    def summary(self) -> dict:
        nan = float("nan")
        rho = self.r2 / self.ir if self.ir else nan
        rho_hat = self.rhat2 / self.irhat if self.irhat else nan
        out = {
            "n": self.n,
            "overshoot": rho,
            "length_ratio": math.sqrt(self.r2 / self.i2) if self.i2 else nan,
            "cosine": self.ir / math.sqrt(self.r2 * self.i2) if self.r2 and self.i2 else nan,
            "unmoved_fraction": self.n_zero / self.n if self.n else nan,
            "multi_spacing_share_of_movers": self.n_multi / self.n_mover if self.n_mover else nan,
            "round_agreement": self.n_round_ok / self.n_lattice if self.n_lattice else nan,
            "exact_write_agreement": self.n_exact / self.n if self.n else nan,
            "predicted_overshoot": rho_hat,
            "predicted_minus_stored": rho_hat - rho,
            "intent_norm": math.sqrt(self.i2),
            "realized_norm": math.sqrt(self.r2),
        }
        if any(self.shadow_ir):
            out["shadow"] = [
                {"factor": f,
                 "overshoot": (self.shadow_r2[k] / self.shadow_ir[k]) if self.shadow_ir[k] else nan,
                 "unmoved_fraction": self.shadow_zero[k] / self.n if self.n else nan}
                for k, f in enumerate(SHADOW_FACTORS)
            ]
        if any(self.hist):
            out["x_hist_log2"] = {"log2_min": HIST_LOG2_MIN, "counts": list(self.hist)}
        return out


def analyze_bundle(arrays: dict, lr: float, wd: float, eps: float, bc1: float, bc2: float,
                   include=None, exclude_regions=("D", "dt_bias")) -> dict:
    """Run the pooled analysis over a detail bundle laid out like the Mamba step-100 npz:
    keys '<region_path>/{param_pre, exp_avg_post, exp_avg_sq_post, delta}'.
    `arrays` maps key -> torch float32 tensor (or numpy array). Returns per-tensor and pooled summaries."""
    bases = sorted({k[: -len("/delta")] for k in arrays if k.endswith("/delta")})
    pooled = Acc()
    per_tensor = {}
    for base in bases:
        region = base.split("/")[1] if "/" in base else base
        if region in exclude_regions:
            continue
        if include is not None and not include(base):
            continue
        try:
            p_pre = torch.as_tensor(arrays[base + "/param_pre"]).float().reshape(-1)
            m = torch.as_tensor(arrays[base + "/exp_avg_post"]).float().reshape(-1)
            v = torch.as_tensor(arrays[base + "/exp_avg_sq_post"]).float().reshape(-1)
            r = torch.as_tensor(arrays[base + "/delta"]).float().reshape(-1)
        except KeyError:
            continue
        i = intent(m, v, p_pre, lr, wd, eps, bc1, bc2)
        a = Acc()
        a.add(i, r, p_pre, shadow=True, hist=True)
        per_tensor[base] = a.summary()
        pooled.merge(a)
    return {"pooled": pooled.summary(), "per_tensor": per_tensor}
