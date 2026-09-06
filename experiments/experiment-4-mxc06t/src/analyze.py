"""Analysis: random (keno_drop) versus magnitude-based (thr_drop) selection at matched delivery and write precision,
tuned keno against tuned exact delivery, and plain rounding's position on the keno curve.

Usage:
  python src/analyze.py --runs-dir <root with */seed<S>/runs/*.json> --exp3-table results/reference/exp3_keno_runs_table.csv
                        --exp2-table results/reference/exp2_runs_table.csv --out results/analysis

Sources of each held-out NLL (nats/token, 256 GSM8K test questions, 49,277 tokens):
  this run      keno_drop_15 (seeds 0-4), keno_drop_25/50 (seeds 2-4), thr_drop_15/25/50 (seeds 0-4), harness rne seed 2
  exp3 table    keno_drop_25/50 seeds 0,1 (and 75/range for the curve, seeds 0,1), harness rne seeds 0,1
  exp2 table    rne all lrs seeds 0-4, fp32_master at 5e-5 seeds 0-4, kahan / fp32_master at 2e-4, 1e-4

Pre-registered rules (accepted plan), 90% Student-t intervals on paired per-seed differences over five seeds:
  Q1 primary   keno_drop_25 - thr_drop_25 at lr 2e-4;  keno_drop_50 - thr_drop_50 at lr 1e-4
               interval entirely < 0: random wins; entirely > 0: magnitude wins; inside +-0.01: indistinguishable;
               otherwise undecided. Other (level, lr) pairs secondary, same rule.
  Q2 primary   keno_drop_50 at lr 1e-4 - fp32_master at lr 5e-5 (paired per seed)
               interval entirely < 0: beats; inside +-0.01: tie; otherwise as reported. Other keno configs exploratory.
  Q3           per lr and seed, keno-drop loss interpolated (over the keno_drop_15/25/50 curve, plus 75 where present) at plain
               rounding's measured delivery, minus plain rounding's loss. Descriptive (mixes selection and precision).
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

T90_DF4 = 2.131846786  # Student t, 0.95 quantile, 4 degrees of freedom
BAND = 0.01
FAMS = ("attention", "mlp", "embedding", "norm")
LEVELS = ("15", "25", "50")
SEEDS = [0, 1, 2, 3, 4]
Q1_PRIMARY = (("0.0002", "25"), ("0.0001", "50"))
Q2_PRIMARY = ("keno_drop_50", "0.0001")


def lr_key(lr: float) -> str:
    return f"{lr:g}"


def fnum(v):
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def load_table(path):
    """runs_table.csv -> {(cond, lrk): {seed: row(dict of floats)}}"""
    out = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            out[(row["condition"], lr_key(float(row["lr"])))][int(row["seed"])] = {k: fnum(v) for k, v in row.items() if k != "condition"}
    return out


def load_runs(root):
    runs, calib, harness = [], {}, []
    for path in sorted(glob.glob(os.path.join(root, "**", "seed*", "runs", "*.json"), recursive=True)):
        with open(path) as f:
            r = json.load(f)
        r["_path"] = path
        runs.append(r)
        if "harness_check" in r:
            harness.append({"condition": r["condition"], "lr": r["lr"], "seed": r["seed"], **r["harness_check"]})
    for path in sorted(glob.glob(os.path.join(root, "**", "calibration", "theta_calibration_*.json"), recursive=True)):
        with open(path) as f:
            c = json.load(f)
        calib[f"{c['writer']}@{lr_key(c['lr'])}"] = {k: v for k, v in c.items() if k != "passes"} | {
            "passes": [{k: v for k, v in p.items() if k != "record"} for p in c["passes"]]}
    return runs, calib, harness


def tinterval(d):
    d = np.asarray(d, dtype=float)
    n = len(d)
    m = float(d.mean())
    if n < 2:
        return {"n": n, "mean": m, "sd": None, "lo": None, "hi": None, "per_seed": d.tolist()}
    sd = float(d.std(ddof=1))
    h = T90_DF4 * sd / math.sqrt(n) if n == 5 else float("nan")
    return {"n": n, "mean": m, "sd": sd, "se": sd / math.sqrt(n), "lo": m - h, "hi": m + h, "level": 0.90,
            "t_quantile": T90_DF4 if n == 5 else None, "per_seed": d.tolist()}


def paired(a: dict, b: dict, seeds=SEEDS):
    common = [s for s in seeds if s in a and s in b]
    res = tinterval([a[s] - b[s] for s in common])
    res["seeds"] = common
    res["a_per_seed"] = {s: a[s] for s in common}
    res["b_per_seed"] = {s: b[s] for s in common}
    return res


def q1_verdict(iv):
    if iv["n"] < 5 or iv["lo"] is None:
        return "not measured at five seeds"
    if iv["hi"] < 0:
        return "random wins"
    if iv["lo"] > 0:
        return "magnitude wins"
    if iv["lo"] >= -BAND and iv["hi"] <= BAND:
        return "indistinguishable"
    return "undecided"


def q2_verdict(iv):
    if iv["n"] < 5 or iv["lo"] is None:
        return "not measured at five seeds"
    if iv["hi"] < 0:
        return "beats"
    if iv["lo"] >= -BAND and iv["hi"] <= BAND:
        return "tie"
    if iv["lo"] > 0:
        return "worse"
    return "undecided"


def smean(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--exp3-table", required=True)
    ap.add_argument("--exp2-table", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs, calib, harness = load_runs(args.runs_dir)
    assert runs, "no runs found"
    exp3 = load_table(args.exp3_table)
    exp2 = load_table(args.exp2_table)
    print(f"{len(runs)} run records, {len(calib)} calibrations, {len(harness)} harness checks")

    # ------------------------------------------------------------------ this run's table
    nll = defaultdict(dict)       # (cond, lrk) -> seed -> nll     (all sources merged below)
    deliv = defaultdict(dict)     # (cond, lrk) -> seed -> cum_length_ratio
    never = defaultdict(dict)
    src = defaultdict(dict)       # (cond, lrk) -> seed -> source label
    fam = defaultdict(dict)       # (cond, lrk) -> seed -> {fam: {never_moved, cum_length_ratio}}
    rows = []
    for r in runs:
        k = (r["condition"], lr_key(r["lr"]))
        s = r["seed"]
        if s in nll[k]:
            print(f"WARNING duplicate record for {k} seed {s}: {r['_path']} (keeping first)")
            continue
        w = r["final_delivery"]["whole"]
        nll[k][s] = r["eval"]["nll_per_token"]
        deliv[k][s] = w["cumulative_length_ratio"]
        never[k][s] = w["never_moved_fraction"]
        src[k][s] = "this run"
        fam[k][s] = {f: {"never_moved": r["final_delivery"]["families"][f]["never_moved_fraction"],
                         "cum_length_ratio": r["final_delivery"]["families"][f]["cumulative_length_ratio"]} for f in FAMS}
        ds = r.get("distributor") or {}
        row = {"seed": s, "condition": r["condition"], "lr": r["lr"], "theta": r.get("theta"), "eval_nll": nll[k][s],
               "eval_tokens": r["eval"]["n_predicted_tokens"], "loss_last200_mean": r["loss_last200_mean"],
               "cum_length_ratio": w["cumulative_length_ratio"], "cum_overshoot": w["cumulative_overshoot"],
               "cum_cosine": w["cumulative_cosine"], "never_moved": w["never_moved_fraction"],
               "cum_projection": (w["cumulative_length_ratio"] ** 2 / w["cumulative_overshoot"]) if w["cumulative_overshoot"] else None,
               "clip_fired_steps": r["clip_fired_steps"], "s_per_step": r["s_per_step"], "peak_gpu_mem_gb": r["peak_gpu_mem_gb"],
               "extra_state_bytes": r["extra_state_bytes"],
               "selected_fraction_mean": ds.get("fraction_realized_mean"),
               "selected_first100": ds.get("fraction_realized_first100_mean"), "selected_last100": ds.get("fraction_realized_last100_mean"),
               "mask_n_blocks": ds.get("n_blocks"), "mask_max_abs_target_gap": ds.get("max_abs_target_gap"),
               "harness_passed": r.get("harness_check", {}).get("passed"), "harness_abs_diff": r.get("harness_check", {}).get("eval_nll_abs_diff"),
               "calibration_pass": (r.get("calibration") or {}).get("pass"), "record": os.path.relpath(r["_path"], args.runs_dir)}
        for f in FAMS:
            row[f"never_moved_{f}"] = fam[k][s][f]["never_moved"]
            row[f"cum_length_ratio_{f}"] = fam[k][s][f]["cum_length_ratio"]
        rows.append(row)
    rows.sort(key=lambda x: (-x["lr"], x["condition"], x["seed"]))
    with open(os.path.join(args.out, "runs_table.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)

    # ------------------------------------------------------------------ merge reused results
    def merge(table, label, conds):
        for (cond, lrk), by_seed in table.items():
            if cond not in conds:
                continue
            for s, row in by_seed.items():
                if s in nll[(cond, lrk)]:
                    continue
                nll[(cond, lrk)][s] = row["eval_nll"]
                deliv[(cond, lrk)][s] = row["cum_length_ratio"]
                never[(cond, lrk)][s] = row["never_moved"]
                src[(cond, lrk)][s] = label
                if all(f"never_moved_{f}" in row for f in FAMS):
                    fam[(cond, lrk)][s] = {f: {"never_moved": row[f"never_moved_{f}"], "cum_length_ratio": row[f"cum_length_ratio_{f}"]} for f in FAMS}
    # exp3: keno seeds 0,1 (its harness rne is identical to exp2's rne, skip)
    merge(exp3, "exp_01m1tvjx4ee3xshh9djt6db5bx", {"keno_drop_25", "keno_drop_50", "keno_drop_75", "keno_drop_range"})
    merge(exp2, "exp_01m1tgnb6xf98sqwfhm4effknt", {"rne", "fp32_master", "kahan", "torch_adamw"})
    # this run's harness rne (seed 2) must not replace exp2's rne in the contrasts: keep exp2 as the rne source
    for lrk in ("0.0002", "0.0001", "5e-05"):
        for s, row in exp2.get(("rne", lrk), {}).items():
            nll[("rne", lrk)][s] = row["eval_nll"]
            deliv[("rne", lrk)][s] = row["cum_length_ratio"]
            never[("rne", lrk)][s] = row["never_moved"]
            src[("rne", lrk)][s] = "exp_01m1tgnb6xf98sqwfhm4effknt"

    # ------------------------------------------------------------------ harness
    harness_summary = {"checks": harness, "all_passed": bool(harness) and all(h["passed"] for h in harness),
                       "max_eval_nll_abs_diff": max((h.get("eval_nll_abs_diff", float("nan")) for h in harness), default=None)}

    # ------------------------------------------------------------------ Q1
    q1 = {"rule": "paired per seed (5 seeds), 90% Student-t interval of NLL(keno_drop_f) - NLL(thr_drop_f): entirely < 0 random "
                  "wins; entirely > 0 magnitude wins; inside +-0.01 indistinguishable; else undecided",
          "primary": {}, "secondary": {}, "delivery_match": {}}
    for lrk in ("0.0002", "0.0001"):
        for lv in LEVELS:
            kd, th = f"keno_drop_{lv}", f"thr_drop_{lv}"
            iv = paired(nll[(kd, lrk)], nll[(th, lrk)])
            iv["verdict"] = q1_verdict(iv)
            iv["sources"] = {"keno": dict(sorted(src[(kd, lrk)].items())), "thr": dict(sorted(src[(th, lrk)].items()))}
            dm = paired(deliv[(kd, lrk)], deliv[(th, lrk)])
            match = {"keno_delivery_mean": smean(deliv[(kd, lrk)].values()), "thr_delivery_mean": smean(deliv[(th, lrk)].values()),
                     "difference_mean": dm["mean"], "difference_per_seed": dm["per_seed"],
                     "within_0.02_every_seed": bool(dm["per_seed"]) and all(abs(x) <= 0.02 for x in dm["per_seed"]),
                     "within_0.02_seed_mean": dm["mean"] is not None and abs(dm["mean"]) <= 0.02,
                     "keno_never_moved_mean": smean(never[(kd, lrk)].values()), "thr_never_moved_mean": smean(never[(th, lrk)].values()),
                     "theta": (calib.get(f"{th}@{lrk}") or {}).get("theta")}
            key = f"{lv}@{lrk}"
            (q1["primary"] if (lrk, lv) in Q1_PRIMARY else q1["secondary"])[key] = iv
            q1["delivery_match"][key] = match
    # pooled reading across the six pairs (descriptive)
    allp = [iv for d in (q1["primary"], q1["secondary"]) for iv in d.values() if iv["n"] == 5]
    q1["all_pairs_mean_difference"] = smean([iv["mean"] for iv in allp])
    q1["n_pairs_random_wins"] = sum(iv["verdict"] == "random wins" for iv in allp)
    q1["n_pairs_magnitude_wins"] = sum(iv["verdict"] == "magnitude wins" for iv in allp)

    # ------------------------------------------------------------------ Q2
    master = nll[("fp32_master", "5e-05")]
    q2 = {"rule": "paired per seed (5 seeds), 90% interval of NLL(keno config) - NLL(fp32_master @ 5e-5): entirely < 0 beats; "
                  "inside +-0.01 tie; entirely > 0 worse; else undecided",
          "bar": {"condition": "fp32_master", "lr": 5e-5, "per_seed": master, "mean": smean(master.values())},
          "primary": None, "exploratory": {}}
    for lrk in ("0.0002", "0.0001"):
        for cond in [f"keno_drop_{lv}" for lv in LEVELS] + [f"thr_drop_{lv}" for lv in LEVELS]:
            if not nll[(cond, lrk)]:
                continue
            iv = paired(nll[(cond, lrk)], master)
            iv["verdict"] = q2_verdict(iv)
            iv["condition_mean"] = smean(nll[(cond, lrk)].values())
            if (cond, lrk) == Q2_PRIMARY:
                q2["primary"] = {"condition": cond, "lr": lrk, **iv}
            else:
                q2["exploratory"][f"{cond}@{lrk}"] = iv
    best = min(((c, l), smean(v.values())) for (c, l), v in nll.items() if c.startswith(("keno_drop", "thr_drop")) and len(v) == 5)
    q2["best_five_seed_config"] = {"condition": best[0][0], "lr": best[0][1], "mean": best[1]}

    # ------------------------------------------------------------------ Q3
    q3 = {"rule": "per lr and seed: keno_drop loss linearly interpolated over the keno curve (15/25/50, plus 75 where present) at rne's "
                  "measured delivery for that seed, minus rne's loss; descriptive", "per_lr": {}}
    for lrk in ("0.0002", "0.0001"):
        per_seed = {}
        for s in SEEDS:
            pts = []
            for lv in ("15", "25", "50", "75"):
                k = (f"keno_drop_{lv}", lrk)
                if s in nll[k] and s in deliv[k]:
                    pts.append((deliv[k][s], nll[k][s], lv))
            if s not in nll[("rne", lrk)] or len(pts) < 2:
                continue
            pts.sort()
            x0 = deliv[("rne", lrk)][s]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if xs[0] <= x0 <= xs[-1]:
                y = float(np.interp(x0, xs, ys))
                status = "interpolated"
            else:
                # linear extrapolation from the two nearest points
                (xa, ya), (xb, yb) = ((xs[0], ys[0]), (xs[1], ys[1])) if x0 < xs[0] else ((xs[-2], ys[-2]), (xs[-1], ys[-1]))
                y = ya + (yb - ya) * (x0 - xa) / (xb - xa)
                status = "extrapolated"
            per_seed[s] = {"rne_delivery": x0, "rne_nll": nll[("rne", lrk)][s], "keno_nll_at_rne_delivery": y,
                           "keno_minus_rne": y - nll[("rne", lrk)][s], "status": status,
                           "curve": [{"level": p[2], "delivery": p[0], "nll": p[1]} for p in pts]}
        d = [v["keno_minus_rne"] for v in per_seed.values()]
        q3["per_lr"][lrk] = {"per_seed": per_seed, "mean": smean(d), "min": min(d) if d else None, "max": max(d) if d else None,
                             "n_interpolated": sum(v["status"] == "interpolated" for v in per_seed.values()),
                             "n_extrapolated": sum(v["status"] == "extrapolated" for v in per_seed.values()),
                             "rne_delivery_mean": smean([v["rne_delivery"] for v in per_seed.values()]),
                             "keno_drop_15_delivery_mean": smean(deliv[("keno_drop_15", lrk)].values())}

    # ------------------------------------------------------------------ curves: delivery vs loss, seed means + per seed
    curves = {}
    for lrk in ("0.0002", "0.0001"):
        curves[lrk] = {}
        conds = [f"keno_drop_{lv}" for lv in ("15", "25", "50", "75", "range")] + [f"thr_drop_{lv}" for lv in LEVELS] + \
                ["rne", "kahan", "fp32_master"]
        for c in conds:
            k = (c, lrk)
            if not nll[k]:
                continue
            seeds = sorted(nll[k])
            ent = {"n_seeds": len(seeds), "seeds": seeds, "nll_mean": smean(nll[k].values()), "nll_per_seed": nll[k],
                   "delivery_mean": smean(deliv[k].values()), "delivery_per_seed": deliv[k],
                   "never_moved_mean": smean(never[k].values()), "sources": sorted(set(src[k].values()))}
            if len(seeds) >= 2:
                ent["nll_sd"] = float(np.std([nll[k][s] for s in seeds], ddof=1))
            if fam[k]:
                ent["families"] = {f: {"never_moved": smean([fam[k][s][f]["never_moved"] for s in fam[k]]),
                                       "cum_length_ratio": smean([fam[k][s][f]["cum_length_ratio"] for s in fam[k]])} for f in FAMS}
            if c.startswith("thr_"):
                ent["theta"] = (calib.get(f"{c}@{lrk}") or {}).get("theta")
                rs = [r for r in runs if r["condition"] == c and lr_key(r["lr"]) == lrk and r.get("distributor")]
                ent["selected_fraction_mean"] = smean([r["distributor"]["fraction_realized_mean"] for r in rs])
                ent["selected_first100"] = smean([r["distributor"]["fraction_realized_first100_mean"] for r in rs])
                ent["selected_last100"] = smean([r["distributor"]["fraction_realized_last100_mean"] for r in rs])
            rs = [r for r in runs if r["condition"] == c and lr_key(r["lr"]) == lrk]
            if rs:
                ent["clip_fired_total_this_run"] = int(sum(r["clip_fired_steps"] for r in rs))
                ent["s_per_step_this_run"] = smean([r["s_per_step"] for r in rs])
            curves[lrk][c] = ent
        # fp32 master at 5e-5 for reference
    curves["5e-05"] = {c: {"nll_mean": smean(nll[(c, "5e-05")].values()), "nll_per_seed": nll[(c, "5e-05")],
                           "delivery_mean": smean(deliv[(c, "5e-05")].values())} for c in ("rne", "fp32_master", "kahan") if nll[(c, "5e-05")]}

    # ------------------------------------------------------------------ step series (seed means over this run's records)
    series = {}
    for lrk in ("0.0002", "0.0001"):
        series[lrk] = {}
        for c in sorted({r["condition"] for r in runs}):
            rs = [r for r in runs if r["condition"] == c and lr_key(r["lr"]) == lrk]
            if not rs:
                continue
            steps = [pt["step"] for pt in rs[0]["series"]]

            def col(getter):
                arr = np.array([[getter(pt) for pt in r["series"]] for r in rs], dtype=np.float64)
                return [round(float(x), 6) for x in arr.mean(0)]
            ent = {"n_seeds": len(rs), "steps": steps,
                   "cumulative_length_ratio": col(lambda pt: pt["cumulative"]["whole"]["cumulative_length_ratio"]),
                   "never_moved_fraction": col(lambda pt: pt["cumulative"]["whole"]["never_moved_fraction"]),
                   "step_length_ratio": col(lambda pt: pt["per_step"]["whole"]["length_ratio"])}
            L = np.array([r["losses"] for r in rs], dtype=np.float64)
            nb = L.shape[1] // 100
            ent["loss_block100_steps"] = [(i + 1) * 100 for i in range(nb)]
            ent["loss_block100_mean"] = [round(float(x), 5) for x in L[:, : nb * 100].reshape(len(rs), nb, 100).mean(2).mean(0)]
            if c.startswith("thr_") and rs[0].get("distributor"):
                sf = np.array([r["distributor"]["step_fraction"] for r in rs], dtype=np.float64).mean(0)
                nb2 = len(sf) // 50
                ent["selected_fraction_block50_steps"] = [(i + 1) * 50 for i in range(nb2)]
                ent["selected_fraction_block50_mean"] = [round(float(x), 6) for x in sf[: nb2 * 50].reshape(nb2, 50).mean(1)]
            series[lrk][c] = ent

    verdicts = {"n_records_this_run": len(runs), "harness": harness_summary, "calibration": calib,
                "q1_random_vs_magnitude": q1, "q2_tuned_keno_vs_tuned_exact": q2, "q3_plain_rounding_on_keno_curve": q3,
                "timing": {"s_per_step_by_family": {c: smean([r["s_per_step"] for r in runs if r["condition"].startswith(c)])
                                                    for c in ("keno_", "thr_", "rne")},
                           "clip_fired_total": int(sum(r["clip_fired_steps"] for r in runs)),
                           "eval_tokens": sorted({r["eval"]["n_predicted_tokens"] for r in runs})},
                "coverage": {f"{c}@{l}": {"n_seeds": len(v), "sources": sorted(set(src[(c, l)].values()))}
                             for (c, l), v in sorted(nll.items()) if v}}
    for name, obj in (("verdicts", verdicts), ("curves", curves), ("step_series", series)):
        with open(os.path.join(args.out, name + ".json"), "w") as f:
            json.dump(obj, f, indent=None if name == "step_series" else 1)
    print(json.dumps({"harness_all_passed": harness_summary["all_passed"],
                      "q1_primary": {k: (v["verdict"], round(v["mean"], 4), [round(v["lo"], 4), round(v["hi"], 4)] if v["lo"] is not None else None)
                                     for k, v in q1["primary"].items()},
                      "q1_secondary": {k: (v["verdict"], round(v["mean"], 4)) for k, v in q1["secondary"].items()},
                      "delivery_match": {k: (round(v["keno_delivery_mean"], 4) if v["keno_delivery_mean"] else None,
                                             round(v["thr_delivery_mean"], 4) if v["thr_delivery_mean"] else None, v["within_0.02_seed_mean"])
                                         for k, v in q1["delivery_match"].items()},
                      "q2_primary": (q2["primary"]["verdict"], round(q2["primary"]["mean"], 4),
                                     [round(q2["primary"]["lo"], 4), round(q2["primary"]["hi"], 4)] if q2["primary"] and q2["primary"]["lo"] is not None else None)
                      if q2["primary"] else None,
                      "q3": {k: (v["mean"], v["n_interpolated"], v["n_extrapolated"]) for k, v in q3["per_lr"].items()}}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
