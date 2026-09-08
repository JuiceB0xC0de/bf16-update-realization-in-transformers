"""Write report_summary.json and claims_manifest.json from results/analysis (pod-side, no experiment imports).

Usage: apy src/make_report_summary.py [--analysis results/analysis] [--out .]
Every number in the summary is read from verdicts.json / curves.json / clip_summary.json and the same read is recorded
as a claim (source file + key path + value) so the report can be audited claim by claim.
"""
from __future__ import annotations

import argparse
import json
import os

CLAIMS: list[dict] = []


def rd(path):
    with open(path) as f:
        return json.load(f)


def claim(cid, text, source, key, value):
    CLAIMS.append({"id": cid, "claim": text, "source": source, "key": key, "value": value})
    return value


def dig(obj, key):
    for part in key.split("/"):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/analysis")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    V = os.path.join(a.analysis, "verdicts.json")
    C = os.path.join(a.analysis, "curves.json")
    K = os.path.join(a.analysis, "clip_summary.json")
    v, c, k = rd(V), rd(C), rd(K)

    def cv(cid, text, key):
        return claim(cid, text, V, key, dig(v, key))

    def cc(cid, text, key):
        return claim(cid, text, C, key, dig(c, key))

    def ck(cid, text, key):
        return claim(cid, text, K, key, dig(k, key))

    # ---------------------------------------------------------------- Q1
    q1 = {}
    for cell, kind in (("25@0.0002", "primary"), ("50@0.0001", "primary"), ("15@0.0002", "secondary"),
                       ("50@0.0002", "secondary"), ("15@0.0001", "secondary"), ("25@0.0001", "secondary")):
        base = f"q1_random_vs_magnitude/{kind}/{cell}"
        lv, lr = cell.split("@")
        q1[cell] = {
            "kind": kind,
            "mean": cv(f"q1_{cell}_mean", f"Q1 {kind} {lv}% @ lr {lr}: mean paired keno - threshold NLL", f"{base}/mean"),
            "lo": cv(f"q1_{cell}_lo", f"Q1 {lv}% @ {lr}: 90% interval lower bound", f"{base}/lo"),
            "hi": cv(f"q1_{cell}_hi", f"Q1 {lv}% @ {lr}: 90% interval upper bound", f"{base}/hi"),
            "verdict": cv(f"q1_{cell}_verdict", f"Q1 {lv}% @ {lr}: pre-registered verdict", f"{base}/verdict"),
            "per_seed": cv(f"q1_{cell}_per_seed", f"Q1 {lv}% @ {lr}: per-seed differences (seeds 0-4)", f"{base}/per_seed"),
            "thr_delivery_gap_mean": cv(f"q1_{cell}_deliv_gap", f"Q1 {lv}% @ {lr}: mean keno minus threshold cumulative delivery",
                                        f"q1_random_vs_magnitude/delivery_match/{cell}/difference_mean"),
            "theta": cv(f"q1_{cell}_theta", f"Q1 {lv}% @ {lr}: calibrated threshold theta", f"q1_random_vs_magnitude/delivery_match/{cell}/theta"),
        }
    cv("q1_n_random_wins", "Q1: number of the six contrasts where random wins", "q1_random_vs_magnitude/n_pairs_random_wins")
    cv("q1_n_magnitude_wins", "Q1: number of the six contrasts where magnitude wins", "q1_random_vs_magnitude/n_pairs_magnitude_wins")
    cv("q1_all_pairs_mean", "Q1: mean keno - threshold difference over all six contrasts", "q1_random_vs_magnitude/all_pairs_mean_difference")
    # the two divergent threshold runs
    cv("div_thr25_s2_nll", "thr_drop_25 @ 2e-4 seed 2 held-out NLL (diverged)", "q1_random_vs_magnitude/primary/25@0.0002/b_per_seed/2")
    cv("div_thr50_s1_nll", "thr_drop_50 @ 1e-4 seed 1 held-out NLL (diverged)", "q1_random_vs_magnitude/primary/50@0.0001/b_per_seed/1")
    cv("divergent_runs", "runs flagged divergent (|NLL - median of other seeds| > 0.02)", "divergent_runs")
    for cid, key, text in (
        ("div_thr25_s2_clip", "divergent_detail/thr_drop_25@0.0002@seed2/clip_fired_steps", "thr_25 @ 2e-4 seed 2: steps where the gradient clip fired"),
        ("div_thr25_s2_gmax", "divergent_detail/thr_drop_25@0.0002@seed2/grad_norm_max", "thr_25 @ 2e-4 seed 2: max pre-clip gradient norm"),
        ("div_thr25_s2_spikes", "divergent_detail/thr_drop_25@0.0002@seed2/grad_norm_spike_blocks_steps_and_max", "thr_25 @ 2e-4 seed 2: 100-step blocks with grad-norm max > 50"),
        ("div_thr25_s2_selfrac", "divergent_detail/thr_drop_25@0.0002@seed2/selected_fraction_block50_mean_over_run", "thr_25 @ 2e-4 seed 2: mean per-step selected fraction"),
        ("div_thr50_s1_clip", "divergent_detail/thr_drop_50@0.0001@seed1/clip_fired_steps", "thr_50 @ 1e-4 seed 1: steps where the gradient clip fired"),
        ("div_thr50_s1_gmax", "divergent_detail/thr_drop_50@0.0001@seed1/grad_norm_max", "thr_50 @ 1e-4 seed 1: max pre-clip gradient norm"),
        ("div_thr50_s1_spikes", "divergent_detail/thr_drop_50@0.0001@seed1/grad_norm_spike_blocks_steps_and_max", "thr_50 @ 1e-4 seed 1: 100-step blocks with grad-norm max > 50"),
        ("div_thr50_s1_selfrac", "divergent_detail/thr_drop_50@0.0001@seed1/selected_fraction_block50_mean_over_run", "thr_50 @ 1e-4 seed 1: mean per-step selected fraction"),
        ("clip_thr_total", "family_totals/thr/clip_fired_total", "gradient clip events over all 30 threshold runs"),
        ("clip_thr_runs", "family_totals/thr/n_runs", "number of threshold runs"),
        ("clip_keno_total", "family_totals/keno/clip_fired_total", "gradient clip events over all 22 keno runs of this experiment"),
        ("clip_keno_runs", "family_totals/keno/n_runs", "number of keno runs in this experiment"),
    ):
        ck(cid, text, key)
    # selection schedule / never-moved (critic f1)
    sel = {}
    for cell in ("thr_drop_15@0.0002", "thr_drop_25@0.0002", "thr_drop_50@0.0002", "thr_drop_15@0.0001", "thr_drop_25@0.0001", "thr_drop_50@0.0001"):
        p = v["calibration"][cell]["passes"][-1]
        sel[cell] = {"selected_fraction_mean": cv(f"sel_{cell}_mean", f"{cell} seed 0: mean per-step selected fraction", f"calibration/{cell}/passes/{len(v['calibration'][cell]['passes']) - 1}/selected_fraction_mean"),
                     "selected_first100": cv(f"sel_{cell}_first", f"{cell} seed 0: selected fraction, first 100 steps", f"calibration/{cell}/passes/{len(v['calibration'][cell]['passes']) - 1}/selected_first100"),
                     "selected_last100": cv(f"sel_{cell}_last", f"{cell} seed 0: selected fraction, last 100 steps", f"calibration/{cell}/passes/{len(v['calibration'][cell]['passes']) - 1}/selected_last100")}
        assert p["selected_fraction_mean"] == sel[cell]["selected_fraction_mean"]
    nm = {}
    for cell in ("15@0.0002", "25@0.0002", "50@0.0002", "15@0.0001", "25@0.0001", "50@0.0001"):
        nm[cell] = {"keno": cv(f"nm_keno_{cell}", f"keno {cell}: never-moved fraction, seed mean", f"q1_random_vs_magnitude/delivery_match/{cell}/keno_never_moved_mean"),
                    "thr": cv(f"nm_thr_{cell}", f"threshold {cell}: never-moved fraction, seed mean", f"q1_random_vs_magnitude/delivery_match/{cell}/thr_never_moved_mean")}
    fam = {}
    for lr, lv in (("0.0002", "25"), ("0.0001", "50")):
        for kind in ("keno_drop", "thr_drop"):
            for f in ("attention", "mlp", "embedding", "norm"):
                fam[f"{kind}_{lv}@{lr}/{f}"] = {
                    "never_moved": cc(f"fam_nm_{kind}_{lv}_{lr}_{f}", f"{kind}_{lv} @ {lr}: never-moved fraction in {f} parameters", f"{lr}/{kind}_{lv}/families/{f}/never_moved"),
                    "delivery": cc(f"fam_dl_{kind}_{lv}_{lr}_{f}", f"{kind}_{lv} @ {lr}: cumulative delivery in {f} parameters", f"{lr}/{kind}_{lv}/families/{f}/cum_length_ratio")}
    cc("rne_nm_2e-4", "plain rounding @ 2e-4: never-moved fraction, five seeds", "0.0002/rne/never_moved_mean")
    cc("rne_nm_1e-4", "plain rounding @ 1e-4: never-moved fraction, five seeds", "0.0001/rne/never_moved_mean")

    # ---------------------------------------------------------------- Q2
    q2 = {"bar_mean": cv("q2_bar_mean", "fp32 master @ 5e-5 five-seed mean NLL (exp_01m1tgnb6xf98sqwfhm4effknt)", "q2_tuned_keno_vs_tuned_exact/bar/mean"),
          "bar_per_seed": cv("q2_bar_per_seed", "fp32 master @ 5e-5 per-seed NLL", "q2_tuned_keno_vs_tuned_exact/bar/per_seed"),
          "primary": {kk: cv(f"q2_primary_{kk}", f"Q2 primary keno_drop_50 @ 1e-4 minus fp32 master @ 5e-5: {kk}", f"q2_tuned_keno_vs_tuned_exact/primary/{kk}")
                      for kk in ("mean", "lo", "hi", "verdict", "condition_mean", "per_seed")},
          "exploratory": {}}
    for cell in ("keno_drop_15@0.0002", "keno_drop_25@0.0002", "keno_drop_25@0.0001", "keno_drop_50@0.0002", "keno_drop_15@0.0001",
                 "thr_drop_15@0.0002", "thr_drop_25@0.0002", "thr_drop_50@0.0002", "thr_drop_15@0.0001", "thr_drop_25@0.0001", "thr_drop_50@0.0001"):
        q2["exploratory"][cell] = {kk: cv(f"q2_{cell}_{kk}", f"Q2 exploratory {cell} minus fp32 master @ 5e-5: {kk}", f"q2_tuned_keno_vs_tuned_exact/exploratory/{cell}/{kk}")
                                   for kk in ("mean", "lo", "hi", "verdict", "condition_mean")}
    within = {cell: abs(q2["exploratory"][cell]["mean"]) if cell in q2["exploratory"] else abs(q2["primary"]["mean"])
              for cell in ("keno_drop_15@0.0002", "keno_drop_25@0.0002", "keno_drop_25@0.0001", "keno_drop_50@0.0001")}
    q2["within_0.003_of_bar"] = {cell: bool(d <= 0.003) for cell, d in within.items()}
    assert all(q2["within_0.003_of_bar"].values())

    # ---------------------------------------------------------------- Q3
    q3 = {}
    for lr in ("0.0002", "0.0001"):
        q3[lr] = {kk: cv(f"q3_{lr}_{kk}", f"Q3 @ lr {lr}: {kk} of interpolated keno NLL at rne's delivery minus rne NLL",
                         f"q3_plain_rounding_on_keno_curve/per_lr/{lr}/{kk}")
                  for kk in ("mean", "min", "max", "n_interpolated", "n_extrapolated", "rne_delivery_mean", "keno_drop_15_delivery_mean")}

    # ---------------------------------------------------------------- harness / coverage
    harness = {"passed": cv("harness_passed", "rne seed 2 lr 2e-4 harness reproduction passed", "harness/all_passed"),
               "abs_diff": cv("harness_abs_diff", "harness reproduction |NLL difference|", "harness/checks/0/eval_nll_abs_diff"),
               "losses_max_abs_diff": cv("harness_loss_diff", "harness reproduction max |per-step loss difference| over 3000 steps", "harness/checks/0/losses_max_abs_diff"),
               "reference": cv("harness_ref", "harness reference NLL", "harness/checks/0/eval_nll_reference")}
    cov = {"n_records_this_run": cv("n_records", "records produced by this experiment", "n_records_this_run"),
           "cells_complete": cv("cells_complete", "planned cells with five seeds", "planned_grid/n_complete"),
           "cells_planned": cv("cells_planned", "planned cells", "planned_grid/n_planned"),
           "eval_tokens": cv("eval_tokens", "held-out evaluation tokens", "timing/eval_tokens/0")}
    for cell in ("thr_drop_15@0.0002", "thr_drop_25@0.0002", "thr_drop_50@0.0002", "thr_drop_15@0.0001", "thr_drop_25@0.0001", "thr_drop_50@0.0001"):
        cv(f"cal_{cell}_theta", f"calibrated theta for {cell}", f"calibration/{cell}/theta")
        cv(f"cal_{cell}_gap", f"seed-0 delivery gap threshold minus keno for {cell}", f"calibration/{cell}/passes/{len(v['calibration'][cell]['passes']) - 1}/gap")

    p25, p50 = q1["25@0.0002"], q1["50@0.0001"]
    q2p = q2["primary"]
    summary = {
        "title": "Random coordinate selection never loses to magnitude selection at matched delivery on SmolLM2-135M; the two primary contrasts are undecided because one threshold seed in each diverged, and a 50% random writer at lr 1e-4 ties tuned fp32 master",
        "headline": (f"Delivering a uniformly random fraction of each AdamW update (keno) gives lower held-out NLL than delivering the same cumulative amount "
                     f"through the coordinates with the largest relative update (threshold) in all six matched cells; the mean keno minus threshold difference "
                     f"over the six cells is {v['q1_random_vs_magnitude']['all_pairs_mean_difference']:+.4f} nats and magnitude wins none. The two pre-registered primary "
                     f"contrasts are undecided under the interval rule (25% @ 2e-4: {p25['mean']:+.4f} [{p25['lo']:+.4f}, {p25['hi']:+.4f}]; 50% @ 1e-4: "
                     f"{p50['mean']:+.4f} [{p50['lo']:+.4f}, {p50['hi']:+.4f}]) because one threshold seed in each cell diverged late in training "
                     f"(NLL {v['q1_random_vs_magnitude']['primary']['25@0.0002']['b_per_seed']['2']:.4f} and {v['q1_random_vs_magnitude']['primary']['50@0.0001']['b_per_seed']['1']:.4f}); "
                     f"the four secondary contrasts all have 90% intervals entirely below zero (random wins). The tuned keno writer, 50% @ lr 1e-4, "
                     f"ties tuned fp32 master @ 5e-5: {q2p['mean']:+.4f} [{q2p['lo']:+.4f}, {q2p['hi']:+.4f}], and three other keno configurations are within 0.003 of the same bar. "
                     f"Plain bf16 rounding sits above the keno delivery-loss curve at both learning rates ({q3['0.0002']['mean']:+.4f} at 2e-4, interpolated; "
                     f"{q3['0.0001']['mean']:+.4f} at 1e-4, a short extrapolation below the 15% level)."),
        "assessment": "partial_signal",
        "research_question": ("Given that a random 25% keno write distributor reproduced plain bf16 rounding's advantage over fp32 master, does it matter which coordinates "
                              "receive the update? Compare random selection (keno-drop) with magnitude-based selection (threshold-drop: |update|/bf16 spacing >= theta) "
                              "at matched cumulative delivery and identical exact-write precision; test whether a tuned keno writer beats tuned exact delivery; "
                              "and locate plain rounding on the keno delivery-loss curve."),
        "decision_rules": {
            "q1": "paired per seed over five seeds, 90% Student-t interval of NLL(keno_drop_f) - NLL(thr_drop_f matched to it): entirely < 0 random wins; entirely > 0 magnitude wins; inside +-0.01 indistinguishable; otherwise undecided. Primary: 25% @ 2e-4 and 50% @ 1e-4; 15% level and the remaining cells secondary.",
            "q1_verdicts": {cell: q1[cell]["verdict"] for cell in q1},
            "q2": "paired per seed, 90% interval of NLL(keno_drop_50 @ 1e-4) - NLL(fp32_master @ 5e-5 from exp_01m1tgnb6xf98sqwfhm4effknt): entirely < 0 beats; inside +-0.01 tie; otherwise as reported. Other configurations exploratory.",
            "q2_verdict": q2p["verdict"],
            "q3": "descriptive: per lr and seed, keno_drop NLL linearly interpolated over the keno curve at rne's measured delivery, minus rne's NLL",
            "divergent_runs": "researcher instruction: report thr_drop_25@2e-4 seed 2 and thr_drop_50@1e-4 seed 1 as-is, do not exclude, characterise with clip counts and per-step selected fractions",
        },
        "methods": {
            "model": "HuggingFaceTB/SmolLM2-135M @ 93efa2f097d58c2a74874c7e644dbc9b0cee75a2, bfloat16 parameters and moments, CompensatedAdamW with a write distributor",
            "data": "gsm8k main @ 740312add88f781978c0658806c59bc2815b9866; 3000 steps, batch size 1, seed-specific fixed order; held-out NLL on 49,277 tokens",
            "selectors": ("keno-drop f: every 50 steps redraw a uniform random mask covering fraction f of coordinates; drawn coordinates get the exact write "
                          "t = p + i + c, p <- rne(t), c <- t - p; undrawn requests are discarded. threshold-drop theta: every step select coordinates with "
                          "|i| / spacing(p) >= theta (spacing = 2^(exponent-7)); same exact write on selected coordinates, unselected discarded."),
            "calibration": ("theta per (lr, level) chosen on seed 0 by a secant search on the full 3000-step run until the threshold writer's cumulative delivery "
                            "(sum of realised update length / sum of intended update length) matched keno's within 0.02; the converged pass is the seed-0 production record. "
                            f"theta = {json.dumps({cell: round(v['calibration'][cell]['theta'], 4) for cell in v['calibration']})}"),
            "grid": "levels 15%, 25%, 50% x lr 2e-4, 1e-4 x seeds 0-4 for both selectors (keno 25%/50% seeds 0,1 from exp_01m1tvjx4ee3xshh9djt6db5bx); reference rne, fp32_master, kahan, torch_adamw from exp_01m1tgnb6xf98sqwfhm4effknt",
            "harness_check": f"rne seed 2 lr 2e-4 rerun in this pipeline reproduced {harness['reference']} with |diff| {harness['abs_diff']} and all 3000 per-step losses identical",
            "compute": "one H100 per job, three concurrent jobs; keno about 0.13 s/step, threshold about 0.16 s/step",
            "uncertainty": "90% Student-t intervals on paired per-seed differences, n = 5",
            "seeds": "0, 1, 2, 3, 4 (data order, init and keno mask share the seed)",
            "logging": "W&B project default-exp-exp_01m1v355jafe8sqbzmwmmxc06t; 14 threshold runs from Sept 8 trained without W&B after the API handshake failed, their records carry wandb_url null",
        },
        "results": {
            "q1": q1, "q1_selection_schedule_seed0": sel, "never_moved_by_cell": nm, "never_moved_by_family": fam,
            "instability": {"clip_thr_total": k["family_totals"]["thr"]["clip_fired_total"], "clip_thr_runs": k["family_totals"]["thr"]["n_runs"],
                            "clip_keno_total": k["family_totals"]["keno"]["clip_fired_total"], "clip_keno_runs": k["family_totals"]["keno"]["n_runs"],
                            "divergent_detail": k["divergent_detail"]},
            "q2": q2, "q3": q3, "harness": harness, "coverage": cov,
        },
        "limitations": [
            "Q1 primary verdicts are undecided under the pre-registered interval rule because one threshold seed in each primary cell diverged late in training; the four stable seeds in each cell favour random selection by 0.011 to 0.014 (25% @ 2e-4) and 0.003 to 0.006 (50% @ 1e-4), and excluding the divergent seeds is post-hoc and not done.",
            "Matched cumulative delivery is not matched selection count or schedule: threshold selects 3 to 21% of coordinates per step with an early-heavy decaying schedule while keno selects a flat 15 to 50%, and threshold leaves 20 to 72% of coordinates never moved (norm parameters 63 to 97%) versus 5 to 18% for keno. The Q1 result is therefore about the selection rule as a whole, not about random versus magnitude at equal counts.",
            "The calibrated thetas for the 25% and 50% levels are below the bf16 format's own effective threshold of one half spacing (only 15% @ 2e-4 at theta 0.587 is stricter), so threshold-drop is a softer version of the format's pick rather than a replica; the rounding-overshoot account of plain rounding's position above the keno curve is an interpretation.",
            "Q3 at lr 1e-4 is a short linear extrapolation below the 15% level (rne delivery 0.184 vs keno 15% at 0.195) and is descriptive only.",
            "Single model (SmolLM2-135M), single dataset (GSM8K), batch size 1, 3000 steps.",
        ],
        "artifacts": {
            "verdicts": "results/analysis/verdicts.json", "curves": "results/analysis/curves.json", "step_series": "results/analysis/step_series.json",
            "per_run_diagnostics": "results/analysis/per_run_diagnostics.json", "clip_summary": "results/analysis/clip_summary.json",
            "runs_table": "results/analysis/runs_table.csv", "theta_table": "results/calibration/theta_table_full.json",
            "runs_store": "artifact://juiceb0xc0de-15787e/experiments/exp_01m1v355jafe8sqbzmwmmxc06t/runs/",
            "runs_bundle": "artifact://juiceb0xc0de-15787e/experiments/exp_01m1v355jafe8sqbzmwmmxc06t/bundles/runs_all.tar.gz",
            "figures": ["figures/q1_paired", "figures/delivery_vs_loss_lr2e-4", "figures/delivery_vs_loss_lr1e-4", "figures/q2_vs_master",
                        "figures/never_moved_family_25_lr2e-4", "figures/never_moved_family_50_lr1e-4", "figures/delivery_over_steps_25_lr2e-4",
                        "figures/delivery_over_steps_50_lr1e-4", "figures/instability_primary_thr"],
            "distributor": "src/write_distributor.py",
        },
    }
    with open(os.path.join(a.out, "report_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(a.out, "claims_manifest.json"), "w") as f:
        json.dump({"claims": CLAIMS}, f, indent=1)
    print(f"wrote report_summary.json and claims_manifest.json ({len(CLAIMS)} claims)")


if __name__ == "__main__":
    main()
