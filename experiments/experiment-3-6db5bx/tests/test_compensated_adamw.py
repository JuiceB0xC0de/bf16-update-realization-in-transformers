"""Unit tests for CompensatedAdamW (CPU, random bf16 tensors). Run before any training:
    PYTHONPATH=src apy -m pytest -q tests
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from compensated_adamw import (CompensatedAdamW, ObservedTorchAdamW, WRITERS, bf16_rne, bf16_sr,  # noqa: E402
                               compute_intent)
from realization import Acc, bf16_spacing, cumulative_summary  # noqa: E402

LR, WD, EPS, BETAS = 2e-4, 1e-2, 1e-8, (0.9, 0.999)


def _params(n=200_000, scale=0.02, seed=0, shapes=((400, 250), (50_000,), (100, 100, 10))):
    g = torch.Generator().manual_seed(seed)
    ps = []
    for shp in shapes:
        p = (torch.randn(*shp, generator=g) * scale).to(torch.bfloat16)
        ps.append(torch.nn.Parameter(p))
    return ps


def _grads(ps, seed):
    g = torch.Generator().manual_seed(seed)
    for p in ps:
        p.grad = (torch.randn(p.shape, generator=g) * 1e-3).to(torch.bfloat16)


# --------------------------------------------------------------------------- rne vs torch.optim.AdamW
def test_rne_matches_torch_adamw_write():
    """Same grads, same pre-step weights each step: the rne single write agrees with torch's write on
    >= 99.7% of coordinates at every one of 20 steps (the agreement level measured in the previous
    run on the real model), and the stored bf16 moments match torch bit for bit.

    Agreement depends on the ratio x = |intent| / bf16 spacing: torch's bf16 denominator chain
    perturbs the intent by up to about 2^-8 relatively, so disagreements are writes that land within
    that distance of a rounding boundary. Weight scale 0.1 with lr 2e-4 gives median x about 0.4 at
    step 1 and about 0.1 afterwards, the regime of the real model (about 93% of coordinates unmoved
    per step). At scale 0.02 (median x 3.3 at step 1) agreement falls to 98.8%."""
    a = _params(seed=1, scale=0.1)
    b = [torch.nn.Parameter(p.detach().clone()) for p in a]
    opt_a = CompensatedAdamW(a, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, writer="rne")
    opt_b = torch.optim.AdamW(b, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, fused=False, foreach=False)
    min_agree, min_m, min_v = 1.0, 1.0, 1.0
    for step in range(1, 21):
        _grads(a, 100 + step)
        for pa, pb in zip(a, b):
            pb.grad = pa.grad.clone()
        opt_a.step()
        opt_b.step()
        n = agree = mm = vv = 0
        for pa, pb in zip(a, b):
            n += pa.numel()
            agree += int((pa.detach() == pb.detach()).sum())
            mm += int((opt_a.state[pa]["exp_avg"] == opt_b.state[pb]["exp_avg"]).sum())
            vv += int((opt_a.state[pa]["exp_avg_sq"] == opt_b.state[pb]["exp_avg_sq"]).sum())
            pb.data.copy_(pa.data)  # shared trajectory: compare the write, not accumulated drift
        min_agree = min(min_agree, agree / n)
        min_m = min(min_m, mm / n)
        min_v = min(min_v, vv / n)
    print(f"min exact write agreement {min_agree:.5f}, moments m {min_m:.5f} v {min_v:.5f}")
    assert min_agree >= 0.997, min_agree
    assert min_m == 1.0 and min_v == 1.0, (min_m, min_v)


def test_observed_torch_adamw_hook_reports_actual_write():
    ps = _params(seed=2, scale=0.1)
    opt = ObservedTorchAdamW(ps, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    acc = Acc()
    seen = []

    def hook(p, pre, i, post):
        seen.append(p)
        assert torch.equal(post, p.float())
        acc.add(i, post - pre, pre)

    for step in range(1, 4):
        _grads(ps, 200 + step)
        opt.step(hook=hook)
    assert len(seen) == 3 * len(ps)
    s = acc.summary()
    assert s["exact_write_agreement"] >= 0.997, s
    assert s["length_ratio"] < 1.0 and s["unmoved_fraction"] > 0.5, s  # bf16 write loses part of the intent


# --------------------------------------------------------------------------- kahan
def test_kahan_buffer_equals_fp32_residual_to_bf16_precision():
    ps = _params(seed=3)
    opt = CompensatedAdamW(ps, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, writer="kahan")
    for step in range(1, 6):
        _grads(ps, 300 + step)
        caps = {}

        def hook(p, pre, i, post, caps=caps):
            caps[p] = (pre, i)

        comps_before = {p: opt.state[p]["comp"].float().clone() if opt.state.get(p) else torch.zeros_like(p, dtype=torch.float32)
                        for p in ps}
        opt.step(hook=hook)
        for p in ps:
            pre, i = caps[p]
            t = pre + i + comps_before[p]
            resid = t - p.float()
            c = opt.state[p]["comp"].float()
            err = (c - resid).abs()
            tol = resid.abs() * 2.0 ** -8 + 1e-30
            assert bool((err <= tol).all()), float((err / tol).max())


def test_kahan_delivers_sub_spacing_intents_and_rne_does_not():
    """Constant intent of a quarter spacing per step: rne never moves; kahan moves by ~N/4 spacings."""
    torch.manual_seed(0)
    p0 = (torch.rand(50_000) * 0.02 + 0.01).to(torch.bfloat16)
    u = bf16_spacing(p0.float())
    n_steps = 40
    intent = 0.25 * u
    p_rne = p0.clone()
    p_k = p0.clone()
    c = torch.zeros_like(p0)
    for _ in range(n_steps):
        p_rne = bf16_rne(p_rne.float() + intent)
        t = p_k.float() + intent + c.float()
        p_k = bf16_rne(t)
        c = (t - p_k.float()).to(torch.bfloat16)
    I = n_steps * intent
    r_rne = p_rne.float() - p0.float()
    r_k = p_k.float() - p0.float()
    assert torch.equal(r_rne, torch.zeros_like(r_rne))
    ratio_k = math.sqrt(float((r_k.double() ** 2).sum()) / float((I.double() ** 2).sum()))
    assert 0.98 <= ratio_k <= 1.02, ratio_k


# --------------------------------------------------------------------------- selective
def test_selective_resets_buffer_only_on_movers():
    ps = _params(seed=4)
    opt = CompensatedAdamW(ps, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, writer="selective")
    n_moved = n_stalled = 0
    for step in range(1, 4):
        _grads(ps, 400 + step)
        caps = {}

        def hook(p, pre, i, post, caps=caps):
            caps[p] = (pre, i, post)

        comps_before = {p: opt.state[p]["comp"].float().clone() if opt.state.get(p) else torch.zeros_like(p, dtype=torch.float32)
                        for p in ps}
        opt.step(hook=hook)
        for p in ps:
            pre, i, post = caps[p]
            moved = post != pre
            c = opt.state[p]["comp"].float()
            assert bool((c[moved] == 0).all())
            t = pre + i + comps_before[p]
            resid = (t - post)[~moved]
            err = (c[~moved] - resid).abs()
            assert bool((err <= resid.abs() * 2.0 ** -8 + 1e-30).all())
            n_moved += int(moved.sum())
            n_stalled += int((~moved).sum())
    assert n_moved > 0 and n_stalled > 0, (n_moved, n_stalled)


# --------------------------------------------------------------------------- stochastic rounding
def test_sr_unbiased_within_one_percent():
    torch.manual_seed(0)
    p = (torch.rand(1_000_000) * 0.02 + 0.01).to(torch.bfloat16).float()
    u = bf16_spacing(p)
    for frac in (0.3, 0.5, 0.05):
        i = frac * u
        gen = torch.Generator().manual_seed(7)
        r = bf16_sr(p + i, gen).float() - p
        ratio = float(r.double().sum() / i.double().sum())
        assert abs(ratio - 1.0) <= 0.01, (frac, ratio)
        up = float((r > 0).float().mean())
        assert abs(up - frac) <= 0.01, (frac, up)
    # negatives and exact lattice points
    q = -p
    r = bf16_sr(q - 0.3 * u, torch.Generator().manual_seed(1)).float() - q
    assert abs(float(r.double().sum() / (-0.3 * u).double().sum()) - 1.0) <= 0.01
    assert torch.equal(bf16_sr(p, torch.Generator().manual_seed(2)).float(), p)


def test_sr_is_seeded_and_deterministic():
    ps1 = _params(seed=5)
    ps2 = [torch.nn.Parameter(p.detach().clone()) for p in ps1]
    o1 = CompensatedAdamW(ps1, lr=LR, writer="sr", sr_seed=11)
    o2 = CompensatedAdamW(ps2, lr=LR, writer="sr", sr_seed=11)
    for step in range(3):
        _grads(ps1, 500 + step)
        for a, b in zip(ps1, ps2):
            b.grad = a.grad.clone()
        o1.step()
        o2.step()
    assert all(torch.equal(a.detach(), b.detach()) for a, b in zip(ps1, ps2))


# --------------------------------------------------------------------------- fp32 master
def test_fp32_master_delivers_exactly_on_master():
    ps = _params(seed=6)
    opt = CompensatedAdamW(ps, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, writer="fp32_master")
    p_init = {p: p.detach().float().clone() for p in ps}
    I = {p: torch.zeros_like(p, dtype=torch.float32) for p in ps}
    for step in range(1, 21):
        _grads(ps, 600 + step)

        def hook(p, pre, i, post):
            I[p].add_(i)
            assert torch.equal(post, bf16_rne(opt.state[p]["master"]).float())

        opt.step(hook=hook)
    acc = Acc()
    for p in ps:
        acc.add(I[p], opt.state[p]["master"] - p_init[p])
    s = cumulative_summary(acc)
    assert abs(s["cumulative_length_ratio"] - 1.0) <= 1e-5, s
    assert abs(s["cumulative_overshoot"] - 1.0) <= 1e-5, s
    assert s["cumulative_cosine"] >= 1 - 1e-6, s


# --------------------------------------------------------------------------- shared machinery
@pytest.mark.parametrize("writer", WRITERS)
def test_hook_sees_consistent_pre_post_and_intent(writer):
    ps = _params(seed=7)
    opt = CompensatedAdamW(ps, lr=LR, betas=BETAS, eps=EPS, weight_decay=WD, writer=writer, sr_seed=3)
    calls = []

    def hook(p, pre, i, post):
        assert pre.dtype == i.dtype == post.dtype == torch.float32
        assert torch.equal(post, p.float())
        st = opt.state[p]
        t = st["step"]
        ref = st["master"] if writer == "fp32_master" else pre
        i_ref = compute_intent(st["exp_avg"], st["exp_avg_sq"], ref, LR, WD, EPS, 1 - 0.9 ** t, 1 - 0.999 ** t)
        if writer == "fp32_master":
            # master already advanced by i; recompute with the pre-add master
            i_ref = compute_intent(st["exp_avg"], st["exp_avg_sq"], ref - i, LR, WD, EPS, 1 - 0.9 ** t, 1 - 0.999 ** t)
        assert torch.allclose(i, i_ref, rtol=1e-5, atol=1e-12)
        if writer == "rne":
            assert torch.equal(post, bf16_rne(pre + i).float())
        calls.append(1)

    for step in range(2):
        _grads(ps, 700 + step)
        opt.step(hook=hook)
    assert len(calls) == 2 * len(ps)
    per_coord = {"kahan": 2, "selective": 2, "fp32_master": 4}.get(writer, 3 if writer.startswith("keno_") else 0)
    expected = per_coord * sum(p.numel() for p in ps)  # keno: bf16 tab + byte mask
    assert opt.buffer_bytes() == expected
