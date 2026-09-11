"""Analysis of the observer run: verdicts for the three questions plus compact plot-ready tables.

Reads --run-dir (manifest.json, calibration.json, order*/steps.jsonl, order*/detail/*.npz,
order*/run_info.json) and the reference check (results/reference_check.json), writes small JSON
files under --out. Runs on CPU (torch + numpy); the detail bundles are re-analysed with the same
realization.analyze_bundle that passed the Mamba reference check.

Verdict rules (pre-registered in the plan):
  Q1 present: median whole-model bf16 rho over 300 steps > 1.05, length ratio < 1 on every step,
              unmoved fraction > 50% (every step); absent: median within 0.02 of 1 or below 1.
  Q2 supports: exact write agreement >= 99% and |predicted - stored| <= 0.02 on every captured
              step; refutes: agreement < 95% or gap > 0.05 on any captured step.
  Q3 supports: shadow rho falls monotonically with the factor and spans > 0.05 across 1/16..16;
              refutes: flat or rising.
  Q3 operationalisation (declared before the 300-step results were read; the smoke run and the
  Mamba reference bundle both show a plateau at factors below 1 followed by a fall):
    "falls monotonically" = the median shadow rho is non-increasing across adjacent factors
    within a tolerance of SHADOW_RISE_TOL = 0.005 (no adjacent rise larger than 0.005) AND the
    least-squares slope of median rho against log2(factor) is negative. "Spans > 0.05" =
    median rho at factor 1/16 minus median rho at factor 16 > 0.05. "Flat or rising" = span
    <= 0.05 or the fitted slope is >= 0. Anything else is indeterminate. The plateau width
    (largest factor before the first adjacent drop larger than SHADOW_RISE_TOL) is reported
    descriptively so the report can say where the fall begins instead of claiming the whole
    curve moves with magnitude.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from realization import analyze_bundle, SHADOW_FACTORS  # noqa: E402

SHADOW_RISE_TOL = 0.005  # declared Q3 tolerance (see module docstring)


def shadow_stats(sh_med, factors):
    """Q3 statistics for a median shadow curve (one value per factor)."""
    sh_med = np.asarray(sh_med, dtype=float)
    diffs = np.diff(sh_med)
    lf = np.log2(np.asarray(factors, dtype=float))
    slope = float(np.polyfit(lf, sh_med, 1)[0])
    drops = np.where(diffs < -SHADOW_RISE_TOL)[0]
    plateau_end_factor = float(factors[drops[0]]) if len(drops) else float(factors[-1])
    return {
        "monotone_within_tol": bool(np.all(diffs <= SHADOW_RISE_TOL)),
        "strictly_decreasing": bool(np.all(diffs < 0)),
        "max_adjacent_rise": float(max(diffs.max(), 0.0)),
        "n_rises": int(np.sum(diffs > 0)),
        "slope_per_log2_factor": slope,
        "span": float(sh_med[0] - sh_med[-1]),
        "plateau_end_factor": plateau_end_factor,
        "plateau_span": float(sh_med[0] - sh_med[list(factors).index(plateau_end_factor)]),
    }

FAMILIES = ("attention", "mlp", "embedding", "norm")
MAMBA_MEDIAN_CLIP1 = 1.334
MAMBA_MEDIAN_CLIP8 = 1.335


def med(xs):
    xs = [x for x in xs if x == x]  # drop nan
    return float(np.median(xs)) if xs else float("nan")


def load_steps(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def region_group(base: str) -> str:
    if base.startswith("embed/seen"):
        return "embedding (seen rows)"
    if base.startswith("embed/unseen"):
        return "embedding (unseen rows)"
    if base.startswith("norms/"):
        return "norm"
    proj = base.split("/")[1]
    return "attention" if proj in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"


def analyze_detail(path):
    z = np.load(path, allow_pickle=False)
    g = {k: float(z[f"param/{k}"][0]) for k in
         ("group_lr", "group_weight_decay", "group_eps", "bias_correction1", "bias_correction2")}
    arrays = {k: z[k] for k in z.files if not k.startswith("param/")}
    res = analyze_bundle(arrays, lr=g["group_lr"], wd=g["group_weight_decay"], eps=g["group_eps"],
                         bc1=g["bias_correction1"], bc2=g["bias_correction2"], exclude_regions=())
    # regroup per-tensor into families by pooling sums is not available from summaries; re-run per group
    groups = {}
    for grp in sorted({region_group(b) for b in res["per_tensor"]}):
        sub = analyze_bundle(arrays, lr=g["group_lr"], wd=g["group_weight_decay"], eps=g["group_eps"],
                             bc1=g["bias_correction1"], bc2=g["bias_correction2"], exclude_regions=(),
                             include=lambda b, grp=grp: region_group(b) == grp)
        groups[grp] = {k: v for k, v in sub["pooled"].items() if k not in ("x_hist_log2",)}
    pooled = res["pooled"]
    return {"step": int(z["param/step"][0]), "settings": g, "pooled": pooled, "groups": groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ref", default=os.path.join(os.path.dirname(__file__), "..", "results",
                                                  "reference_check.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = json.load(open(os.path.join(args.run_dir, "manifest.json")))
    calib = json.load(open(os.path.join(args.run_dir, "calibration.json")))
    ref = json.load(open(args.ref))
    orders = sorted(int(d[len("order"):]) for d in os.listdir(args.run_dir) if d.startswith("order"))
    print("orders:", orders, "clip bound:", manifest.get("clip_bound"))

    series = {}
    per_order = {}
    detail_all = {}
    for seed in orders:
        odir = os.path.join(args.run_dir, f"order{seed}")
        steps = load_steps(os.path.join(odir, "steps.jsonl"))
        info = json.load(open(os.path.join(odir, "run_info.json")))
        n = len(steps)
        W = [s["whole_model_bf16"] for s in steps]
        ser = {
            "step": [s["step"] for s in steps],
            "loss": [s["loss"] for s in steps],
            "grad_norm_preclip": [s["grad_norm_preclip"] for s in steps],
            "clip_fired": [int(s["clip_fired"]) for s in steps],
            "n_tokens": [s["n_tokens"] for s in steps],
            "overshoot": [w["overshoot"] for w in W],
            "predicted_overshoot": [w["predicted_overshoot"] for w in W],
            "length_ratio": [w["length_ratio"] for w in W],
            "cosine": [w["cosine"] for w in W],
            "unmoved_fraction": [w["unmoved_fraction"] for w in W],
            "multi_spacing_share_of_movers": [w["multi_spacing_share_of_movers"] for w in W],
            "exact_write_agreement": [w["exact_write_agreement"] for w in W],
            "round_agreement": [w["round_agreement"] for w in W],
            "intent_norm": [w["intent_norm"] for w in W],
            "realized_norm": [w["realized_norm"] for w in W],
            "family_overshoot": {f: [s["families"][f]["overshoot"] for s in steps] for f in FAMILIES},
            "family_unmoved": {f: [s["families"][f]["unmoved_fraction"] for s in steps] for f in FAMILIES},
            "family_length_ratio": {f: [s["families"][f]["length_ratio"] for s in steps] for f in FAMILIES},
            "shadow_overshoot": [[sh["overshoot"] for sh in w["shadow"]] for w in W],
            "shadow_unmoved": [[sh["unmoved_fraction"] for sh in w["shadow"]] for w in W],
        }
        series[f"order{seed}"] = ser

        rho = np.array(ser["overshoot"])
        lr_ = np.array(ser["length_ratio"])
        unm = np.array(ser["unmoved_fraction"])
        sh = np.array(ser["shadow_overshoot"])  # steps x factors
        sh_med = np.median(sh, axis=0)
        diffs = np.diff(sh_med)
        fam = {}
        for f in FAMILIES:
            fo = np.array(ser["family_overshoot"][f], dtype=float)
            fam[f] = {
                "n_coords": steps[0]["families"][f]["n"],
                "median_overshoot": med(fo.tolist()),
                "n_steps_defined": int(np.sum(~np.isnan(fo))),
                "median_unmoved": med(ser["family_unmoved"][f]),
                "median_length_ratio": med(ser["family_length_ratio"][f]),
                "median_multi_spacing": med([s["families"][f]["multi_spacing_share_of_movers"] for s in steps]),
                "median_exact_write_agreement": med([s["families"][f]["exact_write_agreement"] for s in steps]),
                "step100": {k: steps[99]["families"][f][k] for k in
                            ("overshoot", "length_ratio", "cosine", "unmoved_fraction",
                             "multi_spacing_share_of_movers", "exact_write_agreement")} if n >= 100 else None,
            }
        per_order[f"order{seed}"] = {
            "seed": seed, "n_steps": n,
            "median_overshoot": float(np.median(rho)), "min_overshoot": float(rho.min()),
            "max_overshoot": float(rho.max()), "p05_overshoot": float(np.percentile(rho, 5)),
            "p95_overshoot": float(np.percentile(rho, 95)),
            "median_after_step10": float(np.median(rho[10:])) if n > 10 else float("nan"),
            "steps_length_ratio_below_one": int(np.sum(lr_ < 1)),
            "max_length_ratio": float(lr_.max()), "median_length_ratio": float(np.median(lr_)),
            "median_cosine": med(ser["cosine"]),
            "min_unmoved_fraction": float(unm.min()), "median_unmoved_fraction": float(np.median(unm)),
            "median_multi_spacing_share": med(ser["multi_spacing_share_of_movers"]),
            "median_exact_write_agreement": med(ser["exact_write_agreement"]),
            "min_exact_write_agreement": float(np.min(ser["exact_write_agreement"])),
            "max_abs_predicted_gap": float(np.max(np.abs(np.array(ser["predicted_overshoot"]) - rho))),
            "median_abs_predicted_gap": float(np.median(np.abs(np.array(ser["predicted_overshoot"]) - rho))),
            "clip_fired_steps": int(sum(ser["clip_fired"])),
            "loss_first10_mean": float(np.mean(ser["loss"][:10])),
            "loss_last50_mean": float(np.mean(ser["loss"][-50:])),
            "n_tokens_median": float(np.median(ser["n_tokens"])), "n_tokens_max": int(max(ser["n_tokens"])),
            "n_truncated": info["n_truncated"], "max_full_len": info["max_full_len"],
            "resolved_foreach": info["optimizer"]["resolved_foreach"],
            "optimizer_defaults": info["optimizer"]["defaults"],
            "shadow_median_overshoot": sh_med.tolist(),
            "shadow_median_unmoved": np.median(np.array(ser["shadow_unmoved"]), axis=0).tolist(),
            "shadow_stats": shadow_stats(sh_med, SHADOW_FACTORS),
            "shadow_monotone_decreasing": bool(np.all(diffs < 0)),
            "shadow_n_rises": int(np.sum(diffs > 0)),
            "shadow_span": float(sh_med[0] - sh_med[-1]),
            "shadow_monotone_per_step_fraction": float(np.mean(np.all(np.diff(sh, axis=1) < 0, axis=1))),
            "shadow_monotone_within_tol_per_step_fraction": float(np.mean(np.all(np.diff(sh, axis=1) <= SHADOW_RISE_TOL, axis=1))),
            "families": fam,
            "n_detail_coordinates": info["n_detail_coordinates"],
        }
        # detail bundles
        det = []
        for path in sorted(glob.glob(os.path.join(odir, "detail", "step_*.npz"))):
            d = analyze_detail(path)
            st = d["step"]
            online = steps[st - 1]["whole_model_bf16"]
            d["whole_model_online"] = {k: online[k] for k in
                                       ("overshoot", "predicted_overshoot", "exact_write_agreement",
                                        "round_agreement", "length_ratio", "cosine", "unmoved_fraction")}
            d["pooled_compact"] = {k: v for k, v in d["pooled"].items() if k not in ("x_hist_log2",)}
            d["x_hist_log2"] = d["pooled"]["x_hist_log2"]
            del d["pooled"]
            det.append(d)
            print(f"order{seed} step {st}: detail rho {d['pooled_compact']['overshoot']:.5f} "
                  f"pred {d['pooled_compact']['predicted_overshoot']:.5f} "
                  f"exact {d['pooled_compact']['exact_write_agreement']:.5f} n={d['pooled_compact']['n']}")
        detail_all[f"order{seed}"] = det

    # ---------------------------------------------------------------- verdicts
    all_rho = np.concatenate([np.array(series[k]["overshoot"]) for k in series])
    q1 = {
        "rule": "present if median whole-model bf16 rho > 1.05, length ratio < 1 on every step, unmoved > 50% on every step",
        "per_order": {k: {"median_overshoot": v["median_overshoot"],
                          "all_steps_length_ratio_below_one": v["steps_length_ratio_below_one"] == v["n_steps"],
                          "min_unmoved_fraction": v["min_unmoved_fraction"]} for k, v in per_order.items()},
        "pooled_median_overshoot": float(np.median(all_rho)),
    }
    q1_ok = all(v["median_overshoot"] > 1.05 and v["steps_length_ratio_below_one"] == v["n_steps"]
                and v["min_unmoved_fraction"] > 0.5 for v in per_order.values())
    q1_absent = all(abs(v["median_overshoot"] - 1) <= 0.02 or v["median_overshoot"] < 1 for v in per_order.values())
    q1["verdict"] = "present" if q1_ok else ("absent" if q1_absent else "indeterminate")

    q2_rows = []
    for k, det in detail_all.items():
        for d in det:
            p = d["pooled_compact"]
            q2_rows.append({"order": k, "step": d["step"], "n": p["n"], "stored": p["overshoot"],
                            "predicted": p["predicted_overshoot"], "gap": p["predicted_minus_stored"],
                            "exact_write_agreement": p["exact_write_agreement"],
                            "round_agreement": p["round_agreement"],
                            "whole_model_online_stored": d["whole_model_online"]["overshoot"],
                            "whole_model_online_predicted": d["whole_model_online"]["predicted_overshoot"],
                            "whole_model_online_exact": d["whole_model_online"]["exact_write_agreement"]})
    q2_support = all(r["exact_write_agreement"] >= 0.99 and abs(r["gap"]) <= 0.02 for r in q2_rows) and q2_rows
    q2_refute = any(r["exact_write_agreement"] < 0.95 or abs(r["gap"]) > 0.05 for r in q2_rows)
    q2 = {"rule": "supports if exact write agreement >= 99% and |predicted - stored| <= 0.02 on every captured step; refutes if agreement < 95% or gap > 0.05 on any",
          "captured_steps": q2_rows,
          "min_exact_write_agreement": min(r["exact_write_agreement"] for r in q2_rows) if q2_rows else None,
          "max_abs_gap": max(abs(r["gap"]) for r in q2_rows) if q2_rows else None,
          "median_abs_gap": float(np.median([abs(r["gap"]) for r in q2_rows])) if q2_rows else None,
          "verdict": "supports" if q2_support else ("refutes" if q2_refute else "indeterminate")}

    ref_shadow = [s["overshoot"] for s in ref["pooled"]["shadow"]]
    q3 = {"rule": "supports if the median shadow rho falls monotonically across factors 1/16..16 and spans > 0.05; refutes if flat or rising",
          "operationalisation": ("monotonic = no adjacent rise of the median curve larger than SHADOW_RISE_TOL and a negative "
                                 "least-squares slope of median rho against log2(factor); span = rho(1/16) - rho(16) > 0.05; "
                                 "flat or rising = span <= 0.05 or slope >= 0; declared before the 300-step results were read"),
          "rise_tolerance": SHADOW_RISE_TOL,
          "factors": SHADOW_FACTORS,
          "per_order": {k: {"median_overshoot": v["shadow_median_overshoot"], "median_unmoved": v["shadow_median_unmoved"],
                            **v["shadow_stats"],
                            "per_step_strictly_monotone_fraction": v["shadow_monotone_per_step_fraction"],
                            "per_step_monotone_within_tol_fraction": v["shadow_monotone_within_tol_per_step_fraction"]}
                       for k, v in per_order.items()},
          "mamba_reference_bundle_shadow": ref_shadow,
          "mamba_reference_bundle_stats": shadow_stats(ref_shadow, SHADOW_FACTORS),
          "mamba_reviewed_sweep": {"small_factor": 1.3323, "large_factor": 1.1661, "span_factor": 256}}
    q3_support = all(v["shadow_stats"]["monotone_within_tol"] and v["shadow_stats"]["slope_per_log2_factor"] < 0
                     and v["shadow_stats"]["span"] > 0.05 for v in per_order.values())
    q3_refute = any(v["shadow_stats"]["span"] <= 0.05 or v["shadow_stats"]["slope_per_log2_factor"] >= 0
                    for v in per_order.values())
    q3["verdict"] = "supports" if q3_support else ("refutes" if q3_refute else "indeterminate")

    verdicts = {"q1_signature_present": q1, "q2_write_model_transfers": q2, "q3_magnitude_dependence": q3,
                "mamba_reference_medians": {"clip1": MAMBA_MEDIAN_CLIP1, "clip8": MAMBA_MEDIAN_CLIP8,
                                            "step100_bundle_stored": ref["pooled"]["overshoot"]},
                "manifest": manifest, "calibration": {k: v for k, v in calib.items() if k != "rows"}}
    shadow_series = {k: {"step": v["step"], "shadow_overshoot": v.pop("shadow_overshoot"),
                         "shadow_unmoved": v.pop("shadow_unmoved")} for k, v in series.items()}
    for name, obj in (("verdicts.json", verdicts), ("per_order.json", per_order),
                      ("detail_checks.json", detail_all), ("step_series.json", series),
                      ("shadow_series.json", shadow_series)):
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(obj, f)
        print(f"wrote {name} ({os.path.getsize(os.path.join(args.out, name))} bytes)")
    # x histogram per family at step 100 (whole model, online) + Mamba reference
    hist = {"log2_min": ref["pooled"]["x_hist_log2"]["log2_min"],
            "mamba_step100_bundle": ref["pooled"]["x_hist_log2"]["counts"], "transformer": {}}
    for k in series:
        steps = load_steps(os.path.join(args.run_dir, k, "steps.jsonl"))
        for st in (1, 10, 100, 300):
            if len(steps) >= st:
                hist["transformer"][f"{k}/step{st}"] = {
                    f: steps[st - 1]["families"][f]["x_hist_log2"]["counts"] for f in FAMILIES
                    if "x_hist_log2" in steps[st - 1]["families"][f]}
                hist["transformer"][f"{k}/step{st}"]["whole_model"] = steps[st - 1]["whole_model_bf16"]["x_hist_log2"]["counts"]
    with open(os.path.join(args.out, "x_hist.json"), "w") as f:
        json.dump(hist, f)
    print("VERDICTS:", {k: v["verdict"] for k, v in verdicts.items() if isinstance(v, dict) and "verdict" in v})


if __name__ == "__main__":
    main()
