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

exp_01m1v355jafe8sqbzmwmmxc06t adds `ThresholdDistributor` (writers `thr_drop_{15,25,50}`): the same exact
write and drop rule, but the coordinates are picked the way the bfloat16 format picks them, by request size
relative to the weight's bfloat16 spacing (|i| / spacing(p) >= theta), re-evaluated every step. theta is
calibrated per learning rate and level so the cumulative delivery matches the corresponding keno_drop level.
The format's own rule is theta = 1/2 with quantised writes. Also adds the keno_drop_15 level (fraction 0.15).
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
# "15" appended last so CONDITIONS.index (the mask seed) of the previous run's writers is unchanged
KENO_FRACTIONS = {"25": 0.25, "50": 0.50, "75": 0.75, "range": None, "15": 0.15}
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


# ------------------------------------------------------------------ magnitude-based selection
def bf16_spacing(p32: torch.Tensor) -> torch.Tensor:
    """bfloat16 lattice spacing at each float32 value p: 2^(exponent - 7), i.e. the largest power of two not above |p|
    times 2^-7 (0 where p is zero or subnormal). Equal to realization.bf16_spacing for every normal float32; computed
    from the exponent bits so it costs three elementwise ops per step."""
    pow2 = (p32.contiguous().view(torch.int32) & 0x7F800000).view(torch.float32)
    return pow2 * (2.0 ** -7)


def threshold_mask(p_pre: torch.Tensor, i: torch.Tensor, theta: float) -> torch.Tensor:
    """Coordinates whose request is at least theta bfloat16 spacings of the current weight.
    Where the weight is zero (or subnormal, spacing 0) any non-zero request is selected: the format would move it too."""
    sp = bf16_spacing(p_pre)
    ai = i.abs()
    return torch.where(sp > 0, ai >= theta * sp, ai > 0)


class ThresholdDistributor:
    """Magnitude-based write gate: every step select |i| / spacing(p) >= theta; selected coordinates get the exact
    write t = p + i + c, p <- bf16_rne(t), c <- bf16(t - p); unselected requests are discarded, tab unchanged.
    Same interface as WriteDistributor (begin_step / init_state / write / finish / summary). No randomness.
    The selected count is accumulated on the device and read back once per step."""

    mode = "drop"

    def __init__(self, theta: float, level: Optional[str] = None):
        if not (theta > 0):
            raise ValueError(f"theta must be positive, got {theta}")
        self.theta = float(theta)
        self.level = level
        self.step = 0
        self.block = 0
        self.block_fraction = float("nan")
        self._n_sel: Optional[torch.Tensor] = None
        self._n = 0
        self.step_fraction: list[float] = []   # realised selected fraction per step, whole model

    def begin_step(self, step: int, device=None) -> None:
        self._flush()
        self.step = int(step)

    def _flush(self) -> None:
        if self._n > 0 and self._n_sel is not None:
            self.block_fraction = float(self._n_sel) / self._n
            self.step_fraction.append(self.block_fraction)
        self._n_sel = None
        self._n = 0

    def finish(self) -> None:
        self._flush()

    def init_state(self, p: torch.Tensor) -> dict:
        if p.dtype != torch.bfloat16:
            raise TypeError(f"ThresholdDistributor expects bfloat16 parameters, got {p.dtype}")
        return {"comp": torch.zeros_like(p)}

    @torch.no_grad()
    def write(self, p: torch.Tensor, p_pre: torch.Tensor, i: torch.Tensor, st: dict) -> None:
        mask = threshold_mask(p_pre, i, self.theta)
        c = st["comp"]
        c32 = c.float()
        t = p_pre + i + c32
        p_new = bf16_rne(t)
        resid = t - p_new.float()
        p.copy_(torch.where(mask, p_new, p))
        c.copy_(torch.where(mask, resid, c32))
        n_sel = mask.sum()
        self._n_sel = n_sel if self._n_sel is None else self._n_sel + n_sel
        self._n += p.numel()

    def summary(self) -> dict:
        self.finish()
        f = self.step_fraction
        return {"mode": "threshold_drop", "theta": self.theta, "level": self.level, "n_steps": len(f),
                "fraction_realized_mean": sum(f) / len(f) if f else None,
                "fraction_realized_min": min(f) if f else None, "fraction_realized_max": max(f) if f else None,
                "fraction_realized_first100_mean": sum(f[:100]) / len(f[:100]) if f else None,
                "fraction_realized_last100_mean": sum(f[-100:]) / len(f[-100:]) if f else None,
                "step_fraction": f}


THR_LEVELS = ("15", "25", "50")
THR_WRITERS = tuple(f"thr_drop_{lv}" for lv in THR_LEVELS)


def parse_thr_writer(name: str) -> str:
    """'thr_drop_25' -> '25' (the keno_drop level it is calibrated to)."""
    parts = name.split("_")
    if len(parts) != 3 or parts[0] != "thr" or parts[1] != "drop" or parts[2] not in THR_LEVELS:
        raise ValueError(f"not a threshold writer name: {name!r}")
    return parts[2]


def make_threshold_distributor(name: str, theta: float) -> ThresholdDistributor:
    return ThresholdDistributor(theta, level=parse_thr_writer(name))
