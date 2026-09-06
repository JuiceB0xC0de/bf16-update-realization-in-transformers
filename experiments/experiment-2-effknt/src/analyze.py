"""Analysis of the writer x learning-rate sweep: paired per-seed differences with Student-t 90% intervals,
the three pre-registered verdicts, the null-control noise floor, and plot-ready delivery / step-series tables.

Usage: python src/analyze.py --runs-dir <dir with seed{N}/runs/*.json> --out <analysis dir>

Verdict rules (declared in the accepted plan; operationalised here before the production results were read):
  Q1 delivery   supports if, at every lr, Kahan's seed-mean cumulative length ratio >= 0.98 and |rho - 1| <= 0.02
                and the baseline (rne) stays below 0.9 length ratio; refutes if at every lr |len(kahan) - len(rne)| < 0.05;
                otherwise partial.
  Q2 kahan vs   "reproduces master" if the 90% interval of kahan - fp32_master lies inside [-0.01, +0.01] at every lr;
     master     "differs from master" if any interval lies entirely outside [-0.01, +0.01];
                otherwise "not decided at this power".
  Q3 lr         supports if the 90% interval of rne - fp32_master is entirely below 0 at 2e-4 and entirely above 0 at 5e-5;
     reversal   refutes if all three intervals exclude 0 with the same sign; otherwise "not decided at this power",
                with the point estimates reported.
Null floor: max over lr of |mean(rne - torch_adamw)|, with its interval.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats

CONDS = ["torch_adamw", "rne", "kahan", "fp32_master", "selective", "sr"]
FAMS = ("attention", "mlp", "embedding", "norm")
LEVEL = 0.90


def load_runs(runs_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "seed*", "runs", "*.json"))):
        with open(path) as f:
            r = json.load(f)
        r["_path"] = path
        runs.append(r)
    side = {}
    for sd in sorted(glob.glob(os.path.join(runs_dir, "seed*"))):
        seed = int(os.path.basename(sd)[4:])
        side[seed] = {}
        for name in ("untrained_eval", "equivalence", "seed_manifest"):
            p = os.path.join(sd, name + ".json")
            if os.path.exists(p):
                with open(p) as f:
                    side[seed][name] = json.load(f)
    return runs, side


def lr_key(lr: float) -> str:
    return f"{lr:g}"


def paired(a: dict, b: dict) -> dict:
    """a, b: seed -> value. Returns paired difference a - b over common seeds with a Student-t interval."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    if n >= 2:
        sd = float(d.std(ddof=1))
        half = float(stats.t.ppf(0.5 + LEVEL / 2, n - 1) * sd / math.sqrt(n))
        lo, hi = mean - half, mean + half
        t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf") * np.sign(mean) if mean else 0.0
        p = float(2 * stats.t.sf(abs(t), n - 1)) if np.isfinite(t) else 0.0
    else:
        sd, lo, hi, t, p = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    return {"n": n, "seeds": seeds, "mean": mean, "sd": sd, "ci_lo": lo, "ci_hi": hi, "level": LEVEL,
            "t": float(t), "p_two_sided": p, "per_seed": d.tolist(),
            "excludes_zero": bool(n >= 2 and (lo > 0 or hi < 0))}


def summarize_vals(vals):
    v = np.array(vals, dtype=np.float64)
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
            "min": float(v.min()), "max": float(v.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs, side = load_runs(args.runs_dir)
    assert runs, "no runs found"
    lrs = sorted({r["lr"] for r in runs}, reverse=True)
    seeds = sorted({r["seed"] for r in runs})
    conds = [c for c in CONDS if any(r["condition"] == c for r in runs)]
    print(f"{len(runs)} runs, lrs {lrs}, seeds {seeds}, conditions {conds}")

    # ------------------------------------------------------------------ run table
    nll = defaultdict(dict)  # (cond, lr) -> seed -> nll
    table_rows = []
    for r in runs:
        w = r["final_delivery"]["whole"]
        nll[(r["condition"], lr_key(r["lr"]))][r["seed"]] = r["eval"]["nll_per_token"]
        row = {"seed": r["seed"], "condition": r["condition"], "lr": r["lr"], "eval_nll": r["eval"]["nll_per_token"],
               "eval_tokens": r["eval"]["n_predicted_tokens"], "loss_last200_mean": r["loss_last200_mean"],
               "cum_length_ratio": w["cumulative_length_ratio"], "cum_overshoot": w["cumulative_overshoot"],
               "cum_cosine": w["cumulative_cosine"], "never_moved": w["never_moved_fraction"],
               "clip_fired_steps": r["clip_fired_steps"], "s_per_step": r["s_per_step"],
               "extra_state_bytes": r["extra_state_bytes"], "peak_gpu_mem_gb": r["peak_gpu_mem_gb"]}
        for f in FAMS:
            row[f"never_moved_{f}"] = r["final_delivery"]["families"][f]["never_moved_fraction"]
            row[f"cum_length_ratio_{f}"] = r["final_delivery"]["families"][f]["cumulative_length_ratio"]
        ms = r["final_delivery"].get("master_side")
        row["master_side_length_ratio"] = ms["cumulative_length_ratio"] if ms else None
        table_rows.append(row)
    table_rows.sort(key=lambda x: (-x["lr"], CONDS.index(x["condition"]), x["seed"]))
    with open(os.path.join(args.out, "runs_table.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        wri.writeheader()
        wri.writerows(table_rows)

    # ------------------------------------------------------------------ per-condition NLL summaries
    nll_summary = {lr_key(lr): {c: summarize_vals(list(nll[(c, lr_key(lr))].values())) for c in conds
                                 if nll[(c, lr_key(lr))]} for lr in lrs}
    untrained = {s: side[s]["untrained_eval"]["nll_per_token"] for s in side if "untrained_eval" in side[s]}

    # ------------------------------------------------------------------ paired differences
    pairs = {}
    for lr in lrs:
        k = lr_key(lr)
        pairs[k] = {}
        for a in conds:
            for b in conds:
                if a != b and nll[(a, k)] and nll[(b, k)]:
                    pairs[k][f"{a}-{b}"] = paired(nll[(a, k)], nll[(b, k)])

    def pd(lr, a, b):
        return pairs[lr_key(lr)].get(f"{a}-{b}")

    # ------------------------------------------------------------------ delivery table
    delivery = {}
    for lr in lrs:
        k = lr_key(lr)
        delivery[k] = {}
        for c in conds:
            rs = [r for r in runs if r["condition"] == c and lr_key(r["lr"]) == k]
            if not rs:
                continue
            ent = {}
            for m in ("cumulative_length_ratio", "cumulative_overshoot", "cumulative_cosine", "never_moved_fraction"):
                ent[m] = summarize_vals([r["final_delivery"]["whole"][m] for r in rs])
            ent["families"] = {f: {m: summarize_vals([r["final_delivery"]["families"][f][m] for r in rs])
                                   for m in ("cumulative_length_ratio", "never_moved_fraction")} for f in FAMS}
            if c == "fp32_master":
                ent["master_side_length_ratio"] = summarize_vals(
                    [r["final_delivery"]["master_side"]["cumulative_length_ratio"] for r in rs
                     if r["final_delivery"].get("master_side")])
            ent["s_per_step"] = summarize_vals([r["s_per_step"] for r in rs])
            ent["clip_fired_total"] = int(sum(r["clip_fired_steps"] for r in rs))
            ent["extra_state_bytes"] = rs[0]["extra_state_bytes"]
            delivery[k][c] = ent

    # ------------------------------------------------------------------ step series (seed means)
    series = {}
    for lr in lrs:
        k = lr_key(lr)
        series[k] = {}
        for c in conds:
            rs = [r for r in runs if r["condition"] == c and lr_key(r["lr"]) == k]
            if not rs:
                continue
            steps = [pt["step"] for pt in rs[0]["series"]]
            def col(getter):
                arr = np.array([[getter(pt) for pt in r["series"]] for r in rs], dtype=np.float64)
                return [round(float(x), 6) for x in arr.mean(0)]
            series[k][c] = {
                "steps": steps,
                "cumulative_length_ratio": col(lambda pt: pt["cumulative"]["whole"]["cumulative_length_ratio"]),
                "cumulative_overshoot": col(lambda pt: pt["cumulative"]["whole"]["cumulative_overshoot"]),
                "never_moved_fraction": col(lambda pt: pt["cumulative"]["whole"]["never_moved_fraction"]),
                "step_unmoved_fraction": col(lambda pt: pt["per_step"]["whole"]["unmoved_fraction"]),
                "step_overshoot": col(lambda pt: pt["per_step"]["whole"]["overshoot"]),
            }
            # loss curve: 100-step block means, averaged over seeds
            L = np.array([r["losses"] for r in rs], dtype=np.float64)
            nb = L.shape[1] // 100
            blocks = L[:, : nb * 100].reshape(len(rs), nb, 100).mean(2).mean(0)
            series[k][c]["loss_block100_steps"] = [(i + 1) * 100 for i in range(nb)]
            series[k][c]["loss_block100_mean"] = [round(float(x), 5) for x in blocks]

    # ------------------------------------------------------------------ verdicts
    verdicts = {"n_runs": len(runs), "seeds": seeds, "lrs": lrs, "conditions": conds,
                "untrained_nll_per_seed": untrained,
                "equivalence_per_seed": {s: {k2: v for k2, v in side[s]["equivalence"].items() if k2 != "rows"}
                                         for s in side if "equivalence" in side[s]}}
    # Q1
    q1 = {"rule": "supports if at every lr kahan seed-mean cumulative length ratio >= 0.98 and |rho-1| <= 0.02 and rne "
                  "length ratio < 0.9; refutes if at every lr |len(kahan)-len(rne)| < 0.05; else partial", "per_lr": {}}
    sup = ref = True
    for lr in lrs:
        k = lr_key(lr)
        kh, rn, fm = delivery[k].get("kahan"), delivery[k].get("rne"), delivery[k].get("fp32_master")
        if not (kh and rn):
            continue
        ent = {"kahan_length_ratio": kh["cumulative_length_ratio"]["mean"],
               "kahan_overshoot": kh["cumulative_overshoot"]["mean"],
               "kahan_never_moved": kh["never_moved_fraction"]["mean"],
               "rne_length_ratio": rn["cumulative_length_ratio"]["mean"],
               "rne_overshoot": rn["cumulative_overshoot"]["mean"],
               "rne_never_moved": rn["never_moved_fraction"]["mean"],
               "fp32_master_bf16_copy_length_ratio": fm["cumulative_length_ratio"]["mean"] if fm else None,
               "fp32_master_bf16_copy_never_moved": fm["never_moved_fraction"]["mean"] if fm else None}
        ent["kahan_minus_master_length_ratio"] = (ent["kahan_length_ratio"] - ent["fp32_master_bf16_copy_length_ratio"]
                                                  if fm else None)
        ent["supports_here"] = (ent["kahan_length_ratio"] >= 0.98 and abs(ent["kahan_overshoot"] - 1) <= 0.02
                                and ent["rne_length_ratio"] < 0.9)
        ent["indistinguishable_here"] = abs(ent["kahan_length_ratio"] - ent["rne_length_ratio"]) < 0.05
        sup &= ent["supports_here"]
        ref &= ent["indistinguishable_here"]
        q1["per_lr"][k] = ent
    q1["verdict"] = "supports" if sup else ("refutes" if ref else "partial")
    verdicts["q1_delivery"] = q1
    # Q2
    q2 = {"rule": "reproduces master if 90% CI of kahan - fp32_master inside [-0.01, 0.01] at every lr; differs if any CI "
                  "entirely outside; else not decided at this power", "per_lr": {}}
    inside_all, outside_any = True, False
    for lr in lrs:
        d = pd(lr, "kahan", "fp32_master")
        if d is None:
            continue
        inside = d["ci_lo"] >= -0.01 and d["ci_hi"] <= 0.01
        outside = d["ci_lo"] > 0.01 or d["ci_hi"] < -0.01
        q2["per_lr"][lr_key(lr)] = {"mean": d["mean"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "inside_band": inside,
                                   "outside_band": outside, "per_seed": d["per_seed"]}
        inside_all &= inside
        outside_any |= outside
    q2["verdict"] = ("kahan reproduces master weights" if inside_all else
                     "kahan differs from master" if outside_any else "not decided at this power")
    verdicts["q2_kahan_vs_master"] = q2
    # Q3
    q3 = {"rule": "supports if 90% CI of rne - fp32_master < 0 at 2e-4 and > 0 at 5e-5; refutes if all three CIs exclude 0 "
                  "with the same sign; else not decided at this power", "per_lr": {}}
    signs = []
    for lr in lrs:
        d = pd(lr, "rne", "fp32_master")
        if d is None:
            continue
        s = 1 if d["ci_lo"] > 0 else (-1 if d["ci_hi"] < 0 else 0)
        signs.append(s)
        q3["per_lr"][lr_key(lr)] = {"mean": d["mean"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "ci_sign": s,
                                   "per_seed": d["per_seed"]}
    hi_lr, lo_lr = lr_key(max(lrs)), lr_key(min(lrs))
    if q3["per_lr"].get(hi_lr, {}).get("ci_sign") == -1 and q3["per_lr"].get(lo_lr, {}).get("ci_sign") == 1:
        q3["verdict"] = "supports"
    elif signs and all(s == signs[0] for s in signs) and signs[0] != 0:
        q3["verdict"] = "refutes"
    else:
        q3["verdict"] = "not decided at this power"
    q3["point_estimate_signs"] = {k: int(np.sign(v["mean"])) for k, v in q3["per_lr"].items()}
    q3["point_estimates_flip_sign"] = len({v for v in q3["point_estimate_signs"].values()}) > 1
    verdicts["q3_lr_reversal"] = q3
    # null floor
    null = {lr_key(lr): pd(lr, "rne", "torch_adamw") for lr in lrs if pd(lr, "rne", "torch_adamw")}
    verdicts["null_control_rne_minus_torch"] = {
        "per_lr": {k: {"mean": v["mean"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"], "per_seed": v["per_seed"]}
                   for k, v in null.items()},
        "max_abs_mean": max(abs(v["mean"]) for v in null.values()) if null else None,
        "max_abs_ci_bound": max(max(abs(v["ci_lo"]), abs(v["ci_hi"])) for v in null.values()) if null else None,
        "reference_C": 0.000396}
    # every writer vs master and vs rne, compact
    verdicts["vs_master"] = {lr_key(lr): {c: {"mean": pd(lr, c, "fp32_master")["mean"],
                                              "ci_lo": pd(lr, c, "fp32_master")["ci_lo"],
                                              "ci_hi": pd(lr, c, "fp32_master")["ci_hi"]}
                                          for c in conds if c != "fp32_master" and pd(lr, c, "fp32_master")}
                             for lr in lrs}
    verdicts["vs_rne"] = {lr_key(lr): {c: {"mean": pd(lr, c, "rne")["mean"], "ci_lo": pd(lr, c, "rne")["ci_lo"],
                                           "ci_hi": pd(lr, c, "rne")["ci_hi"]}
                                       for c in conds if c != "rne" and pd(lr, c, "rne")}
                          for lr in lrs}
    verdicts["nll_summary"] = nll_summary
    verdicts["timing"] = {"s_per_step_mean": float(np.mean([r["s_per_step"] for r in runs])),
                          "s_per_step_max": float(np.max([r["s_per_step"] for r in runs])),
                          "clip_fired_total": int(sum(r["clip_fired_steps"] for r in runs)),
                          "eval_tokens": sorted({r["eval"]["n_predicted_tokens"] for r in runs})}

    for name, obj in (("verdicts", verdicts), ("paired_differences", pairs), ("delivery_table", delivery),
                      ("step_series", series)):
        with open(os.path.join(args.out, name + ".json"), "w") as f:
            json.dump(obj, f, indent=None if name == "step_series" else 1)
    print(json.dumps({"q1": q1["verdict"], "q2": q2["verdict"], "q3": q3["verdict"],
                      "null_max_abs_mean": verdicts["null_control_rne_minus_torch"]["max_abs_mean"]}, indent=1))
    for lr in lrs:
        k = lr_key(lr)
        print(f"lr {k}: " + ", ".join(f"{c} {nll_summary[k][c]['mean']:.4f}" for c in conds if c in nll_summary[k]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
