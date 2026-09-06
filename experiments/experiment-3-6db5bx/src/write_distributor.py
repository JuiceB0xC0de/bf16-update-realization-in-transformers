"""WriteDistributor: a keno-style write rule that sits between an optimizer's fp32 request and a
bfloat16 weight.

Every `sticky_interval` optimizer steps (a "block") the distributor draws a fresh random mask over
every coordinate of every parameter. Within the block the mask is fixed ("sticky"). Per coordinate it
keeps a bfloat16 tab c. Given the fp32 request i and the bfloat16 weight p at one step:

    drawn        t = p + i + c;   p <- bf16_rne(t);   c <- bf16(t - p)      exact delivery with memory
    undrawn/drop nothing written, i discarded, c unchanged
    undrawn/tab  nothing written, c <- bf16(c + i); paid when the coordinate is next drawn

The draw fraction is either fixed (e.g. 0.25) or, for the "range" variant, a fresh U(lo, hi) draw per
block (defaults 0.30..0.70, after the researcher's deep-chaos-scheduler). Draws use one seeded
torch.Generator per device so a (seed, writer) pair is reproducible.

The distributor is optimizer-agnostic: `begin_step(step)` once per optimizer step, then `write(p,
p_pre32, i32, state)` once per parameter with a state dict created by `init_state(p)`.
"""
from __future__ import annotations

from typing import Optional

import torch

MODES = ("drop", "tab")


def bf16_rne(t32: torch.Tensor) -> torch.Tensor:
    return t32.to(torch.bfloat16)


class WriteDistributor:
    def __init__(self, mode: str, fraction: Optional[float], sticky_interval: int = 50,
                 range_lo: float = 0.30, range_hi: float = 0.70, seed: int = 0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if fraction is not None and not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if sticky_interval < 1:
            raise ValueError("sticky_interval must be >= 1")
        self.mode = mode
        self.fraction = None if fraction is None else float(fraction)
        self.sticky_interval = int(sticky_interval)
        self.range_lo, self.range_hi = float(range_lo), float(range_hi)
        self.seed = int(seed)
        self._gens: dict = {}
        self.step = 0
        self.block = -1
        self.block_fraction = float("nan")
        # per-block bookkeeping: realised mask fraction over every coordinate written in that block
        self.block_log: list[dict] = []
        self._cur: Optional[dict] = None

    # ------------------------------------------------------------------ randomness
    def _gen(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._gens:
            g = torch.Generator(device=device)
            g.manual_seed(self.seed)
            self._gens[key] = g
        return self._gens[key]

    def _draw_block_fraction(self, device: torch.device) -> float:
        if self.fraction is not None:
            return self.fraction
        u = torch.rand(1, generator=self._gen(device), device=device).item()
        return self.range_lo + (self.range_hi - self.range_lo) * u

    # ------------------------------------------------------------------ lifecycle
    def begin_step(self, step: int, device: torch.device) -> None:
        """Call once per optimizer step (steps are 1-based). Opens a new block every sticky_interval steps."""
        self.step = int(step)
        block = (self.step - 1) // self.sticky_interval
        if block != self.block:
            self.block = block
            self.block_fraction = self._draw_block_fraction(device)
            if self._cur is not None:
                self._close_block()
            self._cur = {"block": block, "first_step": self.step, "fraction_target": self.block_fraction,
                         "n_drawn": 0, "n": 0}

    def _close_block(self) -> None:
        c = self._cur
        if c is not None and c["n"] > 0:
            c["fraction_realized"] = c["n_drawn"] / c["n"]
            self.block_log.append(c)
        self._cur = None

    def finish(self) -> None:
        """Close the open block so block_log is complete."""
        self._close_block()

    def init_state(self, p: torch.Tensor) -> dict:
        if p.dtype != torch.bfloat16:
            raise TypeError(f"WriteDistributor expects bfloat16 parameters, got {p.dtype}")
        return {"comp": torch.zeros_like(p), "mask": torch.zeros(p.shape, dtype=torch.bool, device=p.device),
                "mask_block": -1}

    def _refresh_mask(self, st: dict, p: torch.Tensor) -> None:
        if st["mask_block"] != self.block:
            r = torch.rand(p.shape, generator=self._gen(p.device), device=p.device)
            st["mask"] = r < self.block_fraction
            st["mask_block"] = self.block
            if self._cur is not None:
                self._cur["n_drawn"] += int(st["mask"].sum())
                self._cur["n"] += p.numel()

    # ------------------------------------------------------------------ the write
    @torch.no_grad()
    def write(self, p: torch.Tensor, p_pre: torch.Tensor, i: torch.Tensor, st: dict) -> None:
        """p: bf16 weight (written in place); p_pre: fp32 copy of p before the write; i: fp32 request."""
        self._refresh_mask(st, p)
        mask, c = st["mask"], st["comp"]
        c32 = c.float()
        t = p_pre + i + c32                       # drawn coordinates settle their tab now
        p_new = bf16_rne(t)
        resid = t - p_new.float()
        p.copy_(torch.where(mask, p_new, p))
        if self.mode == "drop":
            c.copy_(torch.where(mask, resid, c32))
        else:  # tab
            c.copy_(torch.where(mask, resid, c32 + i))

    # ------------------------------------------------------------------ reporting
    def summary(self) -> dict:
        self.finish()
        fr = [b["fraction_realized"] for b in self.block_log]
        ft = [b["fraction_target"] for b in self.block_log]
        return {"mode": self.mode, "fraction": self.fraction, "sticky_interval": self.sticky_interval,
                "range": [self.range_lo, self.range_hi] if self.fraction is None else None, "seed": self.seed,
                "n_blocks": len(self.block_log),
                "fraction_target_mean": sum(ft) / len(ft) if ft else None,
                "fraction_realized_mean": sum(fr) / len(fr) if fr else None,
                "fraction_realized_min": min(fr) if fr else None, "fraction_realized_max": max(fr) if fr else None,
                "max_abs_target_gap": max(abs(a - b) for a, b in zip(fr, ft)) if fr else None,
                "blocks": self.block_log}


# ------------------------------------------------------------------ writer name parsing
KENO_FRACTIONS = {"25": 0.25, "50": 0.50, "75": 0.75, "range": None}
KENO_WRITERS = tuple(f"keno_{m}_{f}" for m in MODES for f in KENO_FRACTIONS)


def parse_keno_writer(name: str) -> tuple[str, Optional[float]]:
    """'keno_drop_25' -> ('drop', 0.25); 'keno_tab_range' -> ('tab', None)."""
    parts = name.split("_")
    if len(parts) != 3 or parts[0] != "keno" or parts[1] not in MODES or parts[2] not in KENO_FRACTIONS:
        raise ValueError(f"not a keno writer name: {name!r}")
    return parts[1], KENO_FRACTIONS[parts[2]]


def make_distributor(name: str, seed: int, sticky_interval: int = 50) -> WriteDistributor:
    mode, frac = parse_keno_writer(name)
    return WriteDistributor(mode, frac, sticky_interval=sticky_interval, seed=seed)
