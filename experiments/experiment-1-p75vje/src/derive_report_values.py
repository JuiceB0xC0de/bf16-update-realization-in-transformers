"""Derive the handful of pooled / summary numbers the report cites that are not already in
results/analysis/*.json, and write them to results/analysis/derived.json.

Everything here is a re-aggregation of analyze.py outputs (no new measurement).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def hist_summary(counts, log2_min):
    """Fraction of intended updates below half a bf16 spacing (x < 0.5, i.e. log2 x < -1) and the
    median log2 x, from a log2-binned histogram whose bin i covers [log2_min+i, log2_min+i+1)."""
    c = np.asarray(counts, dtype=float)
    lowers = log2_min + np.arange(len(c))
    total = c.sum()
    below_half = c[lowers + 1 <= -1].sum() / total  # bins whose upper edge is <= -1
    cum = np.cumsum(c) / total
    i = int(np.searchsorted(cum, 0.5))
    # linear interpolation inside the median bin
    prev = cum[i - 1] if i > 0 else 0.0
    frac = (0.5 - prev) / (cum[i] - prev) if cum[i] > prev else 0.0
    return {"fraction_below_half_spacing": float(below_half), "median_log2_x": float(lowers[i] + frac), "n": int(total)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/analysis")
    args = ap.parse_args()
    A = args.analysis
    per_order = json.load(open(os.path.join(A, "per_order.json")))
    series = json.load(open(os.path.join(A, "step_series.json")))
    xh = json.load(open(os.path.join(A, "x_hist.json")))
    verdicts = json.load(open(os.path.join(A, "verdicts.json")))

    orders = sorted(series.keys())
    rho = np.concatenate([np.array(series[o]["overshoot"]) for o in orders])
    lr_ = np.concatenate([np.array(series[o]["length_ratio"]) for o in orders])
    unm = np.concatenate([np.array(series[o]["unmoved_fraction"]) for o in orders])
    cos = np.concatenate([np.array(series[o]["cosine"]) for o in orders])
    gap = np.concatenate([np.abs(np.array(series[o]["overshoot"]) - np.array(series[o]["predicted_overshoot"])) for o in orders])
    exact = np.concatenate([np.array(series[o]["exact_write_agreement"]) for o in orders])
    multi = np.concatenate([np.array(series[o]["multi_spacing_share_of_movers"]) for o in orders])
    loss_first = {o: float(series[o]["loss"][0]) for o in orders}
    loss_last50 = {o: float(np.mean(series[o]["loss"][-50:])) for o in orders}

    pooled = {
        "n_steps": int(len(rho)),
        "median_overshoot": float(np.median(rho)),
        "p05_overshoot": float(np.percentile(rho, 5)),
        "p95_overshoot": float(np.percentile(rho, 95)),
        "min_overshoot": float(rho.min()),
        "max_overshoot": float(rho.max()),
        "median_overshoot_after_step10": float(np.median(np.concatenate([np.array(series[o]["overshoot"][10:]) for o in orders]))),
        "median_length_ratio": float(np.median(lr_)),
        "max_length_ratio": float(lr_.max()),
        "median_unmoved_fraction": float(np.median(unm)),
        "min_unmoved_fraction": float(unm.min()),
        "median_cosine": float(np.median(cos)),
        "median_abs_online_predicted_gap": float(np.median(gap)),
        "max_abs_online_predicted_gap": float(gap.max()),
        "median_exact_write_agreement": float(np.median(exact)),
        "min_exact_write_agreement": float(exact.min()),
        "median_multi_spacing_share_of_movers": float(np.median(multi)),
        "mamba_median_clip1": verdicts["mamba_reference_medians"]["clip1"],
        "gap_to_mamba_median": float(np.median(rho) - verdicts["mamba_reference_medians"]["clip1"]),
        "loss_first_step": loss_first,
        "loss_last50_mean": loss_last50,
    }

    # family medians pooled over both orders (median of the per-step family rho over 600 steps)
    families = {}
    for f in ("attention", "mlp", "embedding", "norm"):
        vals = np.concatenate([np.array(series[o]["family_overshoot"][f], dtype=float) for o in orders])
        unmf = np.concatenate([np.array(series[o]["family_unmoved"][f], dtype=float) for o in orders])
        lrf = np.concatenate([np.array(series[o]["family_length_ratio"][f], dtype=float) for o in orders])
        n_coords = per_order[orders[0]]["families"][f]["n_coords"]
        families[f] = {
            "n_coords": n_coords,
            "share_of_parameters": float(n_coords / sum(per_order[orders[0]]["families"][g]["n_coords"] for g in ("attention", "mlp", "embedding", "norm"))),
            "median_overshoot": float(np.nanmedian(vals)),
            "p05_overshoot": float(np.nanpercentile(vals, 5)),
            "p95_overshoot": float(np.nanpercentile(vals, 95)),
            "median_unmoved": float(np.nanmedian(unmf)),
            "median_length_ratio": float(np.nanmedian(lrf)),
            "n_steps_defined": int(np.sum(~np.isnan(vals))),
            "moving_coords_per_step_median": float(np.nanmedian((1 - unmf) * n_coords)),
        }

    # x-histogram summaries at step 100 (whole model, per order) and the Mamba bundle
    xs = {}
    for k, fam in xh["transformer"].items():
        if k.endswith("/step100"):
            xs[k] = {f: hist_summary(fam[f], xh["log2_min"]) for f in fam}
    xs["mamba_step100_bundle"] = hist_summary(xh["mamba_step100_bundle"], xh["log2_min"])

    shadow = verdicts["q3_magnitude_dependence"]
    shadow_summary = {
        o: {"rho_at_1_16": shadow["per_order"][o]["median_overshoot"][0],
            "rho_at_1_2": shadow["per_order"][o]["median_overshoot"][3],
            "rho_at_1": shadow["per_order"][o]["median_overshoot"][4],
            "rho_at_16": shadow["per_order"][o]["median_overshoot"][-1],
            "unmoved_at_1_16": shadow["per_order"][o]["median_unmoved"][0],
            "unmoved_at_16": shadow["per_order"][o]["median_unmoved"][-1],
            **{k: shadow["per_order"][o][k] for k in ("span", "slope_per_log2_factor", "plateau_end_factor", "plateau_span", "max_adjacent_rise", "monotone_within_tol")}}
        for o in orders}
    shadow_summary["mamba_reference_bundle"] = {
        "rho_at_1_16": shadow["mamba_reference_bundle_shadow"][0],
        "rho_at_1_2": shadow["mamba_reference_bundle_shadow"][3],
        "rho_at_1": shadow["mamba_reference_bundle_shadow"][4],
        "rho_at_16": shadow["mamba_reference_bundle_shadow"][-1],
        **shadow["mamba_reference_bundle_stats"]}

    out = {"pooled_both_orders": pooled, "families_pooled": families, "x_hist_step100": xs, "shadow_summary": shadow_summary,
           "verdicts": {k: verdicts[k]["verdict"] for k in ("q1_signature_present", "q2_write_model_transfers", "q3_magnitude_dependence")}}
    path = os.path.join(A, "derived.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
