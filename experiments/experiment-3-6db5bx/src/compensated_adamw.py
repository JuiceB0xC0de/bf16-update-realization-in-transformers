"""CompensatedAdamW: AdamW with bfloat16 parameters and moments, one fp32 intent path,
and five interchangeable weight writers.

Every writer shares the same state update and intent computation; only the final write
of the bfloat16 parameter differs.

    moments  m <- bf16(lerp(m, g, 1-b1)),  v <- bf16(bf16(v*b2) + (1-b2)*g^2)
             (fp32 arithmetic per op, stored back in bfloat16 after each op: the same op order
             and rounding points as torch's foreach AdamW, so the moments match torch bit for bit)
    intent   i = -(lr / bc1) * m / (sqrt(v) / sqrt(bc2) + eps) - lr * wd * p_ref
             computed in fp32 from the STORED bfloat16 moments, bc1 = 1-b1^t, bc2 = 1-b2^t

    writers
      rne          p <- bf16_rne(p + i)                       baseline, torch's write up to arithmetic order
      kahan        t = p + i + c;  p <- bf16_rne(t);  c <- bf16(t - p)      c: bf16 buffer per parameter
      selective    kahan, but c <- 0 wherever the write moved the coordinate
      sr           p <- bf16_sr(p + i)                        stochastic rounding, seeded, no buffer
      fp32_master  M <- M + i;  p <- bf16_rne(M)              fp32 master M per parameter (p_ref = M)

`step(hook=fn)` calls fn(param, p_pre_fp32, intent_fp32, p_post_fp32) once per parameter after
the write, so an observer can accumulate intended and realized updates without a second pass.

keno writers (exp_01m1tvjx4ee3xshh9djt6db5bx): `keno_{drop,tab}_{25,50,75,range}` hand the write to a
`WriteDistributor` (src/write_distributor.py): every 50 steps a random mask of coordinates is drawn; drawn
coordinates get the Kahan write, undrawn ones are skipped and their request is either dropped or banked in
the bf16 tab. `distributor_seed` seeds the mask draws.

Also provided: `ObservedTorchAdamW`, a wrapper around torch.optim.AdamW(fused=False) exposing the
same `step(hook=...)` interface (intent recomputed from the post-step bf16 moments as in
exp_01m1tdvatwe1rrek9g34p75vje), used as the null-control condition.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from write_distributor import KENO_WRITERS, WriteDistributor, make_distributor

BASE_WRITERS = ("rne", "kahan", "selective", "sr", "fp32_master")
WRITERS = BASE_WRITERS + KENO_WRITERS
CONDITIONS = WRITERS + ("torch_adamw",)

Hook = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]


def compute_intent(m_bf16: torch.Tensor, v_bf16: torch.Tensor, p_ref32: torch.Tensor,
                   lr: float, wd: float, eps: float, bc1: float, bc2: float) -> torch.Tensor:
    """fp32 AdamW intended update from stored bf16 moments and an fp32 reference parameter."""
    denom = v_bf16.float().sqrt_().div_(math.sqrt(bc2)).add_(eps)
    i = m_bf16.float().div_(denom).mul_(-(lr / bc1))
    if wd != 0.0:
        i.add_(p_ref32, alpha=-(lr * wd))
    return i


def bf16_rne(t32: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even bfloat16 of an fp32 tensor (torch's default cast)."""
    return t32.to(torch.bfloat16)


def bf16_sr(t32: torch.Tensor, gen: Optional[torch.Generator]) -> torch.Tensor:
    """Stochastic rounding of fp32 to bfloat16: add a uniform 16-bit integer to the low half of the
    fp32 bit pattern and truncate. P(round up) equals the fractional position between the two
    neighbouring bf16 values, so the rounding is unbiased for every finite input (sign-magnitude
    representation makes the same trick valid for negative values and subnormals)."""
    t32 = t32.contiguous()
    bits = t32.view(torch.int32)
    r = torch.randint(0, 1 << 16, t32.shape, device=t32.device, dtype=torch.int32, generator=gen)
    bits = (bits + r) & -65536  # -65536 == ~0xFFFF as int32
    return bits.view(torch.float32).to(torch.bfloat16)


class CompensatedAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2, writer: str = "rne", sr_seed: int = 0,
                 distributor_seed: int = 0, sticky_interval: int = 50):
        if writer not in WRITERS:
            raise ValueError(f"writer must be one of {WRITERS}, got {writer!r}")
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.writer = writer
        self.sr_seed = int(sr_seed)
        self._gens: dict = {}
        self.distributor: Optional[WriteDistributor] = None
        self._global_step = 0
        if writer in KENO_WRITERS:
            self.distributor = make_distributor(writer, seed=distributor_seed, sticky_interval=sticky_interval)
        for group in self.param_groups:
            for p in group["params"]:
                if p.dtype != torch.bfloat16:
                    raise TypeError(f"CompensatedAdamW expects bfloat16 parameters, got {p.dtype}")

    def _gen_for(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._gens:
            g = torch.Generator(device=device)
            g.manual_seed(self.sr_seed)
            self._gens[key] = g
        return self._gens[key]

    def _init_state(self, p: torch.Tensor) -> dict:
        st = self.state[p]
        st["step"] = 0
        st["exp_avg"] = torch.zeros_like(p)
        st["exp_avg_sq"] = torch.zeros_like(p)
        if self.writer in ("kahan", "selective"):
            st["comp"] = torch.zeros_like(p)  # bfloat16 compensation buffer
        elif self.writer == "fp32_master":
            st["master"] = p.detach().to(torch.float32).clone()
        elif self.distributor is not None:
            st.update(self.distributor.init_state(p))
        return st

    @torch.no_grad()
    def step(self, closure=None, hook: Optional[Hook] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._global_step += 1
        if self.distributor is not None:
            dev = self.param_groups[0]["params"][0].device
            self.distributor.begin_step(self._global_step, dev)
        for group in self.param_groups:
            lr = float(group["lr"])
            b1, b2 = (float(b) for b in group["betas"])
            eps = float(group["eps"])
            wd = float(group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st = self._init_state(p)
                st["step"] += 1
                t = st["step"]
                bc1 = 1.0 - b1 ** t
                bc2 = 1.0 - b2 ** t
                g = p.grad.float()
                m, v = st["exp_avg"], st["exp_avg_sq"]
                # same op order as torch's foreach path: one bf16 rounding after each op
                m.copy_(m.float().lerp_(g, 1.0 - b1))
                v.copy_(v.float().mul_(b2))
                v.copy_(v.float().addcmul_(g, g, value=1.0 - b2))
                p_pre = p.float()
                p_ref = st["master"] if self.writer == "fp32_master" else p_pre
                i = compute_intent(m, v, p_ref, lr, wd, eps, bc1, bc2)
                self._write(p, p_pre, i, st)
                if hook is not None:
                    hook(p, p_pre, i, p.float())
        return loss

    def _write(self, p: torch.Tensor, p_pre: torch.Tensor, i: torch.Tensor, st: dict) -> None:
        w = self.writer
        if w == "rne":
            p.copy_(bf16_rne(p_pre + i))
        elif w == "kahan":
            c = st["comp"]
            t = p_pre + i + c.float()
            p.copy_(bf16_rne(t))
            c.copy_(t.sub_(p.float()))
        elif w == "selective":
            c = st["comp"]
            t = p_pre + i + c.float()
            p_new = bf16_rne(t)
            moved = p_new != p
            p.copy_(p_new)
            resid = t.sub_(p_new.float())
            resid.masked_fill_(moved, 0.0)
            c.copy_(resid)
        elif w == "sr":
            p.copy_(bf16_sr(p_pre + i, self._gen_for(p.device)))
        elif w == "fp32_master":
            master = st["master"]
            master.add_(i)
            p.copy_(bf16_rne(master))
        elif self.distributor is not None:
            self.distributor.write(p, p_pre, i, st)
        else:  # pragma: no cover
            raise ValueError(w)

    def buffer_bytes(self) -> int:
        """Bytes of writer-specific extra state (compensation buffers or masters)."""
        n = 0
        for st in self.state.values():
            for k in ("comp", "master", "mask"):
                if k in st:
                    n += st[k].numel() * st[k].element_size()
        return n


class ObservedTorchAdamW:
    """torch.optim.AdamW(fused=False) with the same step(hook=...) interface. The intent is recomputed
    after the step from the stored bf16 moments and the pre-step fp32 snapshot, exactly as the
    observer of exp_01m1tdvatwe1rrek9g34p75vje did."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2):
        params = list(params)
        self.opt = torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                     fused=False)
        self.params = params
        self._pre = {p: torch.empty_like(p, dtype=torch.float32) for p in params}

    @property
    def param_groups(self):
        return self.opt.param_groups

    @property
    def state(self):
        return self.opt.state

    def zero_grad(self, set_to_none: bool = True):
        self.opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None, hook: Optional[Hook] = None):
        for p in self.params:
            if p.grad is not None:
                self._pre[p].copy_(p)
        loss = self.opt.step(closure)
        if hook is not None:
            g = self.opt.param_groups[0]
            lr, wd, eps = float(g["lr"]), float(g["weight_decay"]), float(g["eps"])
            b1, b2 = (float(b) for b in g["betas"])
            for p in self.params:
                if p.grad is None:
                    continue
                st = self.opt.state[p]
                t = int(float(st["step"]))
                bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
                pre = self._pre[p]
                i = compute_intent(st["exp_avg"], st["exp_avg_sq"], pre, lr, wd, eps, bc1, bc2)
                hook(p, pre, i, p.float())
        return loss

    def buffer_bytes(self) -> int:
        return 0


def make_optimizer(condition: str, params, lr: float, sr_seed: int = 0, distributor_seed: int = 0, **kw):
    if condition == "torch_adamw":
        return ObservedTorchAdamW(params, lr=lr, **kw)
    return CompensatedAdamW(params, lr=lr, writer=condition, sr_seed=sr_seed, distributor_seed=distributor_seed, **kw)
