"""Unit tests for ThresholdDistributor and the thr_drop writers (CPU). Run before any training:
    PYTHONPATH=src apy -m pytest -q tests
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from write_distributor import (THR_WRITERS, KENO_WRITERS, ThresholdDistributor, WriteDistributor, bf16_rne,  # noqa: E402
                               bf16_spacing, parse_keno_writer, parse_thr_writer, threshold_mask)
from compensated_adamw import CompensatedAdamW, WRITERS, make_optimizer  # noqa: E402
import realization  # noqa: E402

CPU = torch.device("cpu")


def _weights(n=400_000, scale=0.02, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, generator=g) * scale).to(torch.bfloat16)


def _request(shape, scale, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * scale


def _run(dist, p, requests, hook=None):
    st = dist.init_state(p)
    for step, i in enumerate(requests, start=1):
        dist.begin_step(step, CPU)
        pre = p.float()
        dist.write(p, pre, i, st)
        if hook is not None:
            hook(step, pre, i, p.float(), st)
    dist.finish()
    return p, st


# --------------------------------------------------------------------------- selection rule
def test_spacing_matches_realization_module():
    p = _weights().float()
    assert torch.equal(bf16_spacing(p), realization.bf16_spacing(p))
    q = torch.cat([_weights(scale=s, seed=k).float() for k, s in enumerate((1e-6, 1e-3, 1.0, 300.0))])
    assert torch.equal(bf16_spacing(q), realization.bf16_spacing(q))
    assert torch.equal(bf16_spacing(-q), realization.bf16_spacing(-q))
    # spot values: spacing of 1.0 is 2^-7, of 0.75 is 2^-8, of 2^-10 is 2^-17
    x = torch.tensor([1.0, 0.75, 2.0 ** -10, -1.5, 0.0])
    assert torch.equal(bf16_spacing(x), torch.tensor([2.0 ** -7, 2.0 ** -8, 2.0 ** -17, 2.0 ** -7, 0.0]))


def test_threshold_mask_exact_on_synthetic_tensor():
    p = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0])            # spacings 2^-7, 2^-7, 2^-7, 2^-8, 2^-8, 0, 0
    i = torch.tensor([2.0 ** -8, 2.0 ** -7, -3 * 2.0 ** -8, 2.0 ** -8, 0.49 * 2.0 ** -8, 1e-30, 0.0])
    m = threshold_mask(p, i, 1.0)
    assert m.tolist() == [False, True, True, True, False, True, False]
    m = threshold_mask(p, i, 0.5)
    assert m.tolist() == [True, True, True, True, False, True, False]
    m = threshold_mask(p, i, 1.5)
    assert m.tolist() == [False, False, True, False, False, True, False]


def test_theta_half_selects_exactly_where_rne_would_move():
    """theta = 1/2 is the format's own rule: the coordinates rne moves (up to the tie/power-of-two boundary
    cases, which are measure-zero for continuous requests) are those with |i| >= spacing/2 ... except that a
    request crossing a binade boundary downward can move at |i| slightly below half the spacing above. Check
    agreement on random data is above 99.9%."""
    p = _weights(n=200_000).float()
    i = _request(p.shape, 1e-4, 9)
    moved = (bf16_rne(p + i).float() != p)
    sel = threshold_mask(p, i, 0.5)
    agree = float((moved == sel).float().mean())
    assert agree > 0.999, agree


# --------------------------------------------------------------------------- the write
def test_selected_exact_unselected_bit_unchanged():
    dist = ThresholdDistributor(theta=1.0)
    p = _weights()
    p0 = p.clone()
    i = _request(p.shape, 1e-4, 21)
    st = dist.init_state(p)
    dist.begin_step(1, CPU)
    pre = p.float()
    mask = threshold_mask(pre, i, 1.0)
    dist.write(p, pre, i, st)
    # unselected: bit-unchanged, tab untouched (zero)
    assert torch.equal(p[~mask], p0[~mask])
    assert torch.all(st["comp"][~mask] == 0)
    # selected: p + c == pre + i to bf16 tab precision (n = round(x + c) exact write with residual kept)
    t = pre[mask] + i[mask]
    assert torch.equal(p[mask], t.to(torch.bfloat16))
    assert torch.allclose(p[mask].float() + st["comp"][mask].float(), t, atol=2 ** -16 * 0.02, rtol=1e-6)
    assert 0.0 < float(mask.float().mean()) < 1.0


def test_tab_carries_residual_across_steps_for_selected_coordinates():
    """A coordinate selected at every step under a constant request accumulates the exact sum (Kahan)."""
    dist = ThresholdDistributor(theta=0.25)
    p = torch.full((10_000,), 1.0, dtype=torch.bfloat16)
    i = torch.full((10_000,), 0.3 * 2.0 ** -7)      # 0.3 spacings: selected at theta 0.25, dropped by rne
    p, st = _run(dist, p, [i] * 20)
    total = 20 * 0.3 * 2.0 ** -7
    delivered = p.float() - 1.0
    assert torch.allclose(delivered + st["comp"].float(), torch.full_like(delivered, total), atol=2 ** -11)  # bf16 tab precision
    assert torch.all(delivered > 0)
    # the same request under rne never moves the weight
    q = torch.full((10_000,), 1.0, dtype=torch.bfloat16)
    for _ in range(20):
        q = bf16_rne(q.float() + i)
    assert torch.all(q == 1.0)


def test_delivery_is_monotone_in_theta_and_theta_zero_limit_is_kahan():
    p0 = _weights(n=200_000, seed=4)
    reqs = [_request(p0.shape, 3e-5, 500 + k) for k in range(15)]
    I = sum(reqs)
    ratios = []
    for theta in (1e-9, 0.5, 1.0, 2.0, 4.0):
        p = p0.clone()
        p, st = _run(ThresholdDistributor(theta), p, reqs)
        R = p.float() - p0.float()
        ratios.append(float(R.norm() / I.norm()))
    assert all(a >= b for a, b in zip(ratios, ratios[1:])), ratios
    # theta -> 0 selects everything: identical to the kahan writer's trajectory
    p = p0.clone()
    p, st = _run(ThresholdDistributor(1e-9), p, reqs)
    q = p0.clone()
    c = torch.zeros_like(q)
    for i in reqs:
        t = q.float() + i + c.float()
        q = bf16_rne(t)
        c = (t - q.float()).to(torch.bfloat16)
    assert torch.equal(p, q)
    assert torch.equal(st["comp"], c)


def test_step_fraction_logged_per_step():
    dist = ThresholdDistributor(theta=1.0)
    p = _weights(n=50_000)
    reqs = [_request(p.shape, 1e-4 * (1 + k), 700 + k) for k in range(6)]
    fr = []
    _run(dist, p, reqs, hook=lambda step, pre, i, post, st: fr.append(float(threshold_mask(pre, i, 1.0).double().mean())))
    s = dist.summary()
    assert s["n_steps"] == 6 and len(s["step_fraction"]) == 6
    assert all(abs(a - b) < 1e-9 for a, b in zip(s["step_fraction"], fr))
    # larger requests -> larger selected fraction
    assert s["step_fraction"][-1] > s["step_fraction"][0]
    assert 0 <= s["fraction_realized_min"] <= s["fraction_realized_mean"] <= s["fraction_realized_max"] <= 1


def test_deterministic_no_randomness():
    p0 = _weights(n=100_000, seed=8)
    reqs = [_request(p0.shape, 1e-4, 900 + k) for k in range(5)]
    outs = []
    for _ in range(2):
        p = p0.clone()
        p, st = _run(ThresholdDistributor(0.8), p, reqs)
        outs.append((p.clone(), st["comp"].clone()))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


# --------------------------------------------------------------------------- names and optimizer plumbing
def test_writer_names():
    assert THR_WRITERS == ("thr_drop_15", "thr_drop_25", "thr_drop_50")
    assert parse_thr_writer("thr_drop_25") == "25"
    with pytest.raises(ValueError):
        parse_thr_writer("thr_tab_25")
    assert "keno_drop_15" in KENO_WRITERS and parse_keno_writer("keno_drop_15") == ("drop", 0.15)
    # the previous run's writer order (mask seeds) is preserved
    assert KENO_WRITERS[:4] == ("keno_drop_25", "keno_drop_50", "keno_drop_75", "keno_drop_range")
    assert all(w in WRITERS for w in THR_WRITERS)


def test_optimizer_requires_theta_and_runs():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 64, bias=False).to(torch.bfloat16)
    with pytest.raises(ValueError):
        CompensatedAdamW(lin.parameters(), lr=1e-3, writer="thr_drop_25")
    opt = make_optimizer("thr_drop_25", lin.parameters(), lr=1e-2, theta=1.0)
    seen = []
    for _ in range(3):
        x = torch.randn(8, 64).to(torch.bfloat16)
        loss = lin(x).float().pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step(hook=lambda p, pre, i, post: seen.append((pre, i, post)))
    s = opt.distributor.summary()
    assert s["n_steps"] == 3 and s["theta"] == 1.0 and s["level"] == "25"
    pre, i, post = seen[-1]
    moved = post != pre
    # every moved coordinate was selected; unselected coordinates unchanged
    sel = threshold_mask(pre, i, 1.0)
    assert torch.all(sel[moved])
    assert opt.buffer_bytes() == lin.weight.numel() * 2


def test_keno_15_fraction():
    dist = WriteDistributor("drop", 0.15, sticky_interval=10, seed=3)
    p = _weights()
    _run(dist, p, [_request(p.shape, 1e-4, 100 + k) for k in range(20)])
    s = dist.summary()
    assert abs(s["fraction_realized_mean"] - 0.15) < 0.005
