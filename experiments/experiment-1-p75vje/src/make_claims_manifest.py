"""Build claims_manifest.json: every quantitative claim in report_summary.json / the report, with
the source file and JSON key path it is read from. Values are read from the files, not typed."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REL = "experiments/experiment-1-p75vje"


def get(obj, path):
    parts = path if isinstance(path, list) else path.split("/")
    for p in parts:
        if isinstance(obj, list):
            obj = obj[int(p)]
        else:
            obj = obj[p]
    return obj


def main():
    files = {
        "verdicts": "results/analysis/verdicts.json",
        "per_order": "results/analysis/per_order.json",
        "derived": "results/analysis/derived.json",
        "detail": "results/analysis/detail_checks.json",
        "ref": "results/reference_check.json",
    }
    data = {k: json.load(open(os.path.join(ROOT, v))) for k, v in files.items()}
    claims = []

    def add(claim_id, text, src, path, kind="quantitative"):
        val = get(data[src], path)
        claims.append({"id": claim_id, "claim": text, "value": val, "source_file": f"{REL}/{files[src]}", "key": path if isinstance(path, str) else path, "kind": kind})

    # verdicts
    add("verdict_q1", "Q1 verdict: signature present", "verdicts", "q1_signature_present/verdict", "categorical")
    add("verdict_q2", "Q2 verdict: write model transfers", "verdicts", "q2_write_model_transfers/verdict", "categorical")
    add("verdict_q3", "Q3 verdict: magnitude dependence", "verdicts", "q3_magnitude_dependence/verdict", "categorical")
    add("q3_tolerance", "Q3 adjacent-rise tolerance", "verdicts", "q3_magnitude_dependence/rise_tolerance")

    # Q1 per order
    for o in ("order0", "order1"):
        add(f"{o}_median_rho", f"{o} median whole-model rho over 300 steps", "per_order", f"{o}/median_overshoot")
        add(f"{o}_min_rho", f"{o} minimum rho (step 1)", "per_order", f"{o}/min_overshoot")
        add(f"{o}_max_rho", f"{o} maximum rho", "per_order", f"{o}/max_overshoot")
        add(f"{o}_p05_rho", f"{o} 5th percentile rho", "per_order", f"{o}/p05_overshoot")
        add(f"{o}_p95_rho", f"{o} 95th percentile rho", "per_order", f"{o}/p95_overshoot")
        add(f"{o}_steps_len_below_one", f"{o} steps with length ratio below one", "per_order", f"{o}/steps_length_ratio_below_one")
        add(f"{o}_max_len", f"{o} maximum length ratio", "per_order", f"{o}/max_length_ratio")
        add(f"{o}_median_len", f"{o} median length ratio", "per_order", f"{o}/median_length_ratio")
        add(f"{o}_min_unmoved", f"{o} minimum unmoved fraction", "per_order", f"{o}/min_unmoved_fraction")
        add(f"{o}_median_unmoved", f"{o} median unmoved fraction", "per_order", f"{o}/median_unmoved_fraction")
        add(f"{o}_median_cos", f"{o} median cosine", "per_order", f"{o}/median_cosine")
        add(f"{o}_clip_fired", f"{o} steps on which clipping fired", "per_order", f"{o}/clip_fired_steps")
        add(f"{o}_n_truncated", f"{o} truncated rows", "per_order", f"{o}/n_truncated")
        add(f"{o}_max_len_tokens", f"{o} longest example in tokens", "per_order", f"{o}/n_tokens_max")
        add(f"{o}_median_tokens", f"{o} median tokens per step", "per_order", f"{o}/n_tokens_median")
        add(f"{o}_foreach", f"{o} torch resolved foreach path", "per_order", f"{o}/resolved_foreach", "categorical")
        add(f"{o}_shadow_median", f"{o} median shadow rho at the nine factors", "per_order", f"{o}/shadow_median_overshoot")
        add(f"{o}_shadow_unmoved", f"{o} median shadow unmoved fraction at the nine factors", "per_order", f"{o}/shadow_median_unmoved")
        for k in ("span", "slope_per_log2_factor", "max_adjacent_rise", "plateau_end_factor", "plateau_span", "monotone_within_tol", "strictly_decreasing"):
            add(f"{o}_shadow_{k}", f"{o} shadow {k}", "per_order", f"{o}/shadow_stats/{k}")
        for f in ("attention", "mlp", "embedding", "norm"):
            add(f"{o}_{f}_n_coords", f"{o} {f} coordinates", "per_order", f"{o}/families/{f}/n_coords")
            add(f"{o}_{f}_median_rho", f"{o} {f} median rho", "per_order", f"{o}/families/{f}/median_overshoot")
            add(f"{o}_{f}_median_unmoved", f"{o} {f} median unmoved", "per_order", f"{o}/families/{f}/median_unmoved")
        add(f"{o}_n_detail", f"{o} detail coordinates per bundle", "per_order", f"{o}/n_detail_coordinates")
        add(f"{o}_online_max_gap", f"{o} max |predicted - stored| online", "per_order", f"{o}/max_abs_predicted_gap")
        add(f"{o}_online_median_gap", f"{o} median |predicted - stored| online", "per_order", f"{o}/median_abs_predicted_gap")
        add(f"{o}_online_min_exact", f"{o} min exact write agreement online", "per_order", f"{o}/min_exact_write_agreement")
        add(f"{o}_loss_first10", f"{o} mean loss first 10 steps", "per_order", f"{o}/loss_first10_mean")
        add(f"{o}_loss_last50", f"{o} mean loss last 50 steps", "per_order", f"{o}/loss_last50_mean")

    # pooled
    P = "pooled_both_orders"
    for k in ("n_steps", "median_overshoot", "p05_overshoot", "p95_overshoot", "min_overshoot", "max_overshoot", "median_length_ratio",
              "max_length_ratio", "median_unmoved_fraction", "min_unmoved_fraction", "median_cosine", "median_abs_online_predicted_gap",
              "max_abs_online_predicted_gap", "median_exact_write_agreement", "min_exact_write_agreement", "median_multi_spacing_share_of_movers",
              "gap_to_mamba_median"):
        add(f"pooled_{k}", f"pooled over both orders: {k}", "derived", f"{P}/{k}")
    add("pooled_loss_first_order0", "order 0 loss at step 1", "derived", f"{P}/loss_first_step/order0")
    add("pooled_loss_first_order1", "order 1 loss at step 1", "derived", f"{P}/loss_first_step/order1")
    add("mamba_median_clip1", "Mamba-3 reference median rho (clip 1)", "verdicts", "mamba_reference_medians/clip1")
    add("mamba_median_clip8", "Mamba-3 reference median rho (clip 8)", "verdicts", "mamba_reference_medians/clip8")
    for f in ("attention", "mlp", "embedding", "norm"):
        for k in ("n_coords", "share_of_parameters", "median_overshoot", "p05_overshoot", "p95_overshoot", "median_unmoved", "median_length_ratio", "moving_coords_per_step_median"):
            add(f"fam_{f}_{k}", f"{f} pooled {k}", "derived", f"families_pooled/{f}/{k}")

    # Q2 captured steps
    add("q2_min_exact", "minimum exact write agreement over the twelve captured steps", "verdicts", "q2_write_model_transfers/min_exact_write_agreement")
    add("q2_max_gap", "maximum |predicted - stored rho| over the twelve captured steps", "verdicts", "q2_write_model_transfers/max_abs_gap")
    add("q2_median_gap", "median |predicted - stored rho| over the twelve captured steps", "verdicts", "q2_write_model_transfers/median_abs_gap")
    rows = data["verdicts"]["q2_write_model_transfers"]["captured_steps"]
    for i, r in enumerate(rows):
        for k in ("stored", "predicted", "gap", "exact_write_agreement", "round_agreement", "n"):
            add(f"q2_{r['order']}_step{r['step']}_{k}", f"{r['order']} step {r['step']} detail {k}", "verdicts", f"q2_write_model_transfers/captured_steps/{i}/{k}")

    # Q3
    add("q3_mamba_bundle_shadow", "Mamba bundle shadow rho at the nine factors", "verdicts", "q3_magnitude_dependence/mamba_reference_bundle_shadow")
    for k in ("span", "slope_per_log2_factor", "plateau_end_factor", "strictly_decreasing"):
        add(f"q3_mamba_{k}", f"Mamba bundle shadow {k}", "verdicts", f"q3_magnitude_dependence/mamba_reference_bundle_stats/{k}")
    add("q3_mamba_reviewed_small", "Mamba reviewed sweep rho at smallest factor", "verdicts", "q3_magnitude_dependence/mamba_reviewed_sweep/small_factor")
    add("q3_mamba_reviewed_large", "Mamba reviewed sweep rho at largest factor", "verdicts", "q3_magnitude_dependence/mamba_reviewed_sweep/large_factor")

    # x histogram
    for o in ("order0", "order1"):
        add(f"xhist_{o}_whole_below_half", f"{o} step 100 fraction of intended updates below half a spacing", "derived", ["x_hist_step100", f"{o}/step100", "whole_model", "fraction_below_half_spacing"])
        add(f"xhist_{o}_whole_median_log2x", f"{o} step 100 median log2 x", "derived", ["x_hist_step100", f"{o}/step100", "whole_model", "median_log2_x"])
        for f in ("attention", "mlp", "embedding", "norm"):
            add(f"xhist_{o}_{f}_below_half", f"{o} step 100 {f} fraction below half a spacing", "derived", ["x_hist_step100", f"{o}/step100", f, "fraction_below_half_spacing"])
    add("xhist_mamba_below_half", "Mamba bundle fraction of intended updates below half a spacing", "derived", "x_hist_step100/mamba_step100_bundle/fraction_below_half_spacing")
    add("xhist_mamba_median_log2x", "Mamba bundle median log2 x", "derived", "x_hist_step100/mamba_step100_bundle/median_log2_x")

    # detail step 100 order 0 groups
    groups = data["detail"]["order0"][3]["groups"]
    assert data["detail"]["order0"][3]["step"] == 100
    for g in groups:
        for k in ("n", "overshoot", "length_ratio", "unmoved_fraction", "exact_write_agreement"):
            add(f"detail_o0_s100_{g.replace(' ', '_').replace('(', '').replace(')', '')}_{k}", f"order 0 step 100 detail group '{g}' {k}", "detail", f"order0/3/groups/{g}/{k}")
    add("detail_o0_s100_bc2", "order 0 step 100 bias_correction2", "detail", "order0/3/settings/bias_correction2")

    # reference check
    ref = data["ref"]
    for k in ("overshoot", "predicted_overshoot", "exact_write_agreement", "round_agreement", "length_ratio", "cosine", "unmoved_fraction", "multi_spacing_share_of_movers", "n"):
        if k in ref["pooled"]:
            add(f"ref_{k}", f"Mamba step-100 bundle reference check {k}", "ref", f"pooled/{k}")
    if "reviewed" in ref:
        for k, v in ref["reviewed"].items():
            if not isinstance(v, (dict, list)):
                add(f"ref_reviewed_{k}", f"reviewed Mamba value {k}", "ref", f"reviewed/{k}")

    out = {"experiment": "exp_01m1tdvatwe1rrek9g34p75vje", "n_claims": len(claims), "claims": claims}
    json.dump(out, open(os.path.join(ROOT, "claims_manifest.json"), "w"), indent=1)
    print(f"wrote {len(claims)} claims")


if __name__ == "__main__":
    main()
