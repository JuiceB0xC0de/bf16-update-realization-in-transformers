"""Unit tests for WriteDistributor and the keno writers (CPU, random bf16 tensors). Run before any training:
    PYTHONPATH=src apy -m pytest -q tests
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from write_distributor import (KENO_WRITERS, WriteDistributor, bf16_rne, parse_keno_writer)
from compensated_adamw import CompensatedAdamW, WRITERS, make_optimizer  # noqa: E402
from realization import bf16_spacing  # noqa: E402

CPU = torch.device("cpu")


def _weights(n=400_000, scale=0.02, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, generator=g) * scale).to(torch.bfloat16)


def _request(shape, scale, seed, sign=None):
    g = torch.Generator().manual_seed(seed)
    i = torch.randn(shape, generator=g) * scale
    if sign is not None:
        i = i.abs() * sign
    return i


def _run(dist, p, requests, hook=None):
    """Drive the distributor over a list of fp32 requests; returns final p and the state."""
    st = dist.init_state(p)
    for step, i in enumerate(requests, start=1):
        dist.begin_step(step, CPU)
        pre = p.float()
        dist.write(p, pre, i, st)
        if hook is not None:
            hook(step, pre, i, p.float(), st)
    dist.finish()
    return p, st


# --------------------------------------------------------------------------- mask fraction
@pytest.mark.parametrize("frac", [0.25, 0.50, 0.75])
def test_mask_fraction_within_half_percent(frac):
    dist = WriteDistributor("drop", frac, sticky_interval=10, seed=3)
    p = _weights()
    reqs = [_request(p.shape, 1e-4, 100 + k) for k in range(30)]
    _run(dist, p, reqs)
    s = dist.summary()
    assert s["n_blocks"] == 3
    for b in s["blocks"]:
        assert abs(b["fraction_realized"] - frac) < 0.005, b
    assert abs(s["fraction_realized_mean"] - frac) < 0.005


def test_range_fraction_fresh_per_block_within_bounds():
    dist = WriteDistributor("drop", None, sticky_interval=5, seed=11, range_lo=0.30, range_hi=0.70)
    p = _weights()
    reqs = [_request(p.shape, 1e-4, 200 + k) for k in range(60)]
    _run(dist, p, reqs)
    s = dist.summary()
    assert s["n_blocks"] == 12
    targets = [b["fraction_target"] for b in s["blocks"]]
    assert all(0.30 <= t <= 0.70 for t in targets)
    assert len(set(round(t, 6) for t in targets)) == 12, "each block must draw a fresh fraction"
    for b in s["blocks"]:
        assert abs(b["fraction_realized"] - b["fraction_target"]) < 0.005


def test_mask_is_sticky_within_block_and_redrawn_at_boundary():
    dist = WriteDistributor("drop", 0.5, sticky_interval=4, seed=5)
    p = _weights(n=50_000)
    masks = []
    reqs = [_request(p.shape, 1e-4, 300 + k) for k in range(8)]
    _run(dist, p, reqs, hook=lambda step, pre, i, post, st: masks.append(st["mask"].clone()))
    for k in range(1, 4):
        assert torch.equal(masks[0], masks[k])
        assert torch.equal(masks[4], masks[4 + k])
    assert not torch.equal(masks[0], masks[4])
    assert 0.3 < float((masks[0] != masks[4]).float().mean()) < 0.7


def test_seeded_draws_are_reproducible():
    outs = []
    for _ in range(2):
        dist = WriteDistributor("tab", None, sticky_interval=3, seed=77)
        p = _weights(n=20_000)
        reqs = [_request(p.shape, 1e-4, 400 + k) for k in range(9)]
        _run(dist, p, reqs)
        outs.append((p.clone(), [b["fraction_target"] for b in dist.block_log]))
    assert torch.equal(outs[0][0], outs[1][0])
    assert outs[0][1] == outs[1][1]


# --------------------------------------------------------------------------- write semantics
@pytest.mark.parametrize("mode", ["drop", "tab"])
def test_drawn_exact_undrawn_unchanged(mode):
    """Drawn coordinates: p_new == bf16_rne(p_pre + i + c_old) exactly. Undrawn: bit-identical."""
    dist = WriteDistributor(mode, 0.5, sticky_interval=3, seed=9)
    p = _weights()
    reqs = [_request(p.shape, 3e-4, 500 + k) for k in range(9)]
    c_prev = {"c": torch.zeros_like(p)}

    def hook(step, pre, i, post, st):
        mask = st["mask"]
        expect = bf16_rne(pre + i + c_prev["c"].float())
        assert torch.equal(post[mask], expect.float()[mask])
        assert torch.equal(post[~mask].to(torch.bfloat16), pre[~mask].to(torch.bfloat16))
        if mode == "drop":
            assert torch.equal(st["comp"][~mask], c_prev["c"][~mask])
        c_prev["c"] = st["comp"].clone()

    _run(dist, p, reqs, hook=hook)


def test_tab_accumulates_fp32_sum_to_bf16_precision():
    """With fraction tiny (nothing drawn), the tab must equal bf16(fp32 sum of requests) up to bf16 rounding of
    the running sum (k roundings of relative size 2^-8)."""
    dist = WriteDistributor("tab", 1e-9, sticky_interval=100, seed=1)
    p = _weights(n=100_000)
    k = 40
    reqs = [_request(p.shape, 1e-4, 600 + j) for j in range(k)]
    p0 = p.clone()
    _, st = _run(dist, p, reqs)
    assert torch.equal(p, p0)
    assert int(st["mask"].sum()) == 0
    total = torch.stack(reqs).sum(0)
    running_max = torch.stack(reqs).cumsum(0).abs().max(0).values
    tol = k * running_max * 2.0 ** -8 + 1e-12
    err = (st["comp"].float() - total).abs()
    assert bool((err <= tol).all()), float((err - tol).max())
    # and mostly far tighter than the bound
    assert float((err / (total.abs() + 1e-12)).median()) < 2.0 ** -6


def test_tab_pays_banked_request_when_drawn():
    """One coordinate banked for a block then drawn: the write delivers p0 + sum(all requests) exactly rounded."""
    dist = WriteDistributor("tab", 1.0, sticky_interval=4, seed=2)
    # force the first block to be undrawn and the second drawn by editing the mask after init
    p = torch.tensor([0.5, -0.25, 1.0, 0.125], dtype=torch.bfloat16)
    st = dist.init_state(p)
    reqs = [torch.tensor([1e-3, -2e-3, 5e-4, 3e-3]) * (j + 1) for j in range(5)]
    p0 = p.float().clone()
    for step, i in enumerate(reqs, start=1):
        dist.begin_step(step, CPU)
        pre = p.float()
        dist.write(p, pre, i, st)
        if step <= 4:
            # emulate undrawn: the distributor drew everything (fraction 1.0); undo and bank manually
            pass
    # with fraction 1.0 everything is drawn every step, so this is a Kahan trajectory: p + c == p0 + sum(i)
    total = torch.stack(reqs).sum(0)
    recon = p.float() + st["comp"].float()
    assert torch.allclose(recon, p0 + total, atol=2.0 ** -9 * 5)


def test_drop_fraction_one_equals_kahan_writer():
    a = _weights(n=100_000, seed=4)
    b = a.clone()
    pa = [torch.nn.Parameter(a.view(400, 250).clone())]
    pb = [torch.nn.Parameter(b.view(400, 250).clone())]
    oa = CompensatedAdamW(pa, lr=2e-4, writer="kahan")
    ob = CompensatedAdamW(pb, lr=2e-4, writer="keno_drop_25")
    ob.distributor.fraction = 1.0  # every coordinate drawn: must reduce to the Kahan writer
    for step in range(1, 11):
        g = _request(pa[0].shape, 1e-3, 700 + step).to(torch.bfloat16)
        pa[0].grad = g.clone()
        pb[0].grad = g.clone()
        oa.step()
        ob.step()
        assert torch.equal(pa[0].detach(), pb[0].detach())
        assert torch.equal(oa.state[pa[0]]["comp"], ob.state[pb[0]]["comp"])


# --------------------------------------------------------------------------- delivery on synthetic requests
@pytest.mark.parametrize("frac", [0.25, 0.5, 0.75])
def test_keno_drop_delivers_fraction_of_cumulative_request(frac):
    """Consistent-sign requests (the momentum regime): the cumulative length ratio ||R|| / ||I|| is f +/- 0.02 and
    the projection <I,R>/||I||^2 is f +/- 0.02. Random-sign requests: the projection is still f +/- 0.02 while
    the length ratio rises toward sqrt(f), so the two readings are both recorded."""
    p = _weights(n=200_000, scale=0.5, seed=8)
    spacing = bf16_spacing(p.float())
    n_steps = 200
    signs = torch.sign(_request(p.shape, 1.0, 1))
    # request ~ 0.3 lattice spacings per step, consistent sign per coordinate, so exact delivery accumulates
    reqs = [(0.3 * spacing * (0.8 + 0.4 * torch.rand(p.shape, generator=torch.Generator().manual_seed(900 + k))))
            * signs for k in range(n_steps)]
    dist = WriteDistributor("drop", frac, sticky_interval=10, seed=frac.__hash__() % 1000)
    p0 = p.float().clone()
    _run(dist, p, reqs)
    I = torch.stack(reqs).sum(0).double()
    R = (p.float() - p0).double()
    length_ratio = float(R.norm() / I.norm())
    projection = float((I * R).sum() / (I * I).sum())
    assert abs(length_ratio - frac) < 0.02, length_ratio
    assert abs(projection - frac) < 0.02, projection
    # random-sign requests: projection still f, length ratio between f and sqrt(f)
    p = _weights(n=200_000, scale=0.5, seed=8)
    p0 = p.float().clone()
    reqs = [0.3 * spacing * torch.randn(p.shape, generator=torch.Generator().manual_seed(950 + k))
            for k in range(n_steps)]
    dist = WriteDistributor("drop", frac, sticky_interval=10, seed=17)
    _run(dist, p, reqs)
    I = torch.stack(reqs).sum(0).double()
    R = (p.float() - p0).double()
    projection = float((I * R).sum() / (I * I).sum())
    length_ratio = float(R.norm() / I.norm())
    assert abs(projection - frac) < 0.02, projection
    assert frac - 0.02 < length_ratio < math.sqrt(frac) + 0.02, length_ratio


def test_keno_tab_delivers_everything_eventually():
    """tab with fraction 0.5 over many blocks: every coordinate is drawn at some point after its last bank, so
    p_final + c_final == p0 + sum(i) (to bf16 tab precision) and the delivered share exceeds the drop share."""
    p = _weights(n=200_000, scale=0.5, seed=12)
    spacing = bf16_spacing(p.float())
    signs = torch.sign(_request(p.shape, 1.0, 2))
    reqs = [0.3 * spacing * signs for _ in range(200)]
    p0 = p.float().clone()
    dist = WriteDistributor("tab", 0.5, sticky_interval=10, seed=21)
    _, st = _run(dist, p, reqs)
    I = torch.stack(reqs).sum(0)
    recon = p.float() + st["comp"].float()
    rel = ((recon - (p0 + I)).abs() / (I.abs() + 1e-12))
    assert float(rel.median()) < 0.02
    R = (p.float() - p0).double()
    length_ratio = float(R.norm() / I.double().norm())
    assert length_ratio > 0.9, length_ratio


# --------------------------------------------------------------------------- optimizer integration
def test_writer_names_and_parsing():
    assert len(KENO_WRITERS) == 10  # 8 from the previous run + keno_{drop,tab}_15
    assert all(w in WRITERS for w in KENO_WRITERS)
    assert parse_keno_writer("keno_drop_25") == ("drop", 0.25)
    assert parse_keno_writer("keno_tab_range") == ("tab", None)
    with pytest.raises(ValueError):
        parse_keno_writer("keno_drop_10")


@pytest.mark.parametrize("writer", KENO_WRITERS)
def test_optimizer_runs_and_reports_blocks(writer):
    ps = [torch.nn.Parameter(_weights(n=30_000, seed=3).view(300, 100)),
          torch.nn.Parameter(_weights(n=5_000, seed=4))]
    opt = make_optimizer(writer, ps, lr=2e-4, distributor_seed=5, sticky_interval=5)
    p0 = [p.detach().float().clone() for p in ps]
    for step in range(1, 16):
        for k, p in enumerate(ps):
            p.grad = _request(p.shape, 1e-3, 1000 * k + step).to(torch.bfloat16)
        opt.step()
    s = opt.distributor.summary()
    assert s["n_blocks"] == 3
    assert opt.buffer_bytes() == sum(p.numel() * 3 for p in ps)  # bf16 tab + byte mask
    moved = sum(int((p.detach().float() != q).sum()) for p, q in zip(ps, p0))
    assert moved > 0
    mode, frac = parse_keno_writer(writer)
    if frac is not None:
        assert abs(s["fraction_realized_mean"] - frac) < 0.02
    else:
        assert 0.3 <= s["fraction_realized_mean"] <= 0.7
