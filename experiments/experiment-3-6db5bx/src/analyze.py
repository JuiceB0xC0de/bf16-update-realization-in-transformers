"""Analysis of the keno writer sweep: paired per-seed differences against the previous run's plain-rounding (rne)
and fp32-master results, the three pre-registered verdicts, the delivery-vs-loss table and step series.

Usage: python src/analyze.py --runs-dir <dir with seed{S}_lr{lr}/runs/*.json> --prev-table results/reference/prev_runs_table.csv
                             --out results/analysis

Verdict rules (accepted plan, operationalised before the production results were read):
  Q1 how much, or which   d = mean over seeds {0,1} of NLL(keno_drop_25) - NLL(rne) at lr 2e-4.
                          |d| <= 0.01: "amount delivered is the knob; the format's selection is incidental"
                          d > +0.02:   "the format's magnitude-based pick is doing real work"
                          d < -0.02:   "random selection beats the format's pick"
                          otherwise:   "undecided at two seeds"
  Q2 banking              for each fraction f in {25, 50, 75, range} and each lr:
                          |mean(keno_tab_f - fp32_master)| <= 0.01 -> "delayed delivery behaves as delivery"
                          |mean(keno_tab_f - keno_drop_f)| <= 0.01 -> "requests not written promptly might as well be discarded"
                          (both can hold, neither can hold; reported as found)
  Q3 beats tuned exact    any writer whose seed-mean NLL (seeds 0,1) is below 1.150 (fp32_master at lr 5e-5 on the
     delivery             previous grid) is flagged; reported as found.
No interval is claimed at n = 2; point estimate and both per-seed values are reported for every difference.
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

KENO = ["keno_drop_25", "keno_drop_50", "keno_drop_75", "keno_drop_range",
        "keno_tab_25", "keno_tab_50", "keno_tab_75", "keno_tab_range"]
FRACS = ["25", "50", "75", "range"]
REF_CONDS = ["rne", "fp32_master", "kahan", "torch_adamw"]
FAMS = ("attention", "mlp", "embedding", "norm")
Q3_BAR = 1.150
BAND_Q1 = 0.01
BAND_Q1_STRONG = 0.02
BAND_Q2 = 0.01


def lr_key(lr: float) -> str:
    return f"{lr:g}"


def load_runs(runs_dir):
    runs, side = [], {}
    for shard in sorted(glob.glob(os.path.join(runs_dir, "seed*_lr*"))):
        if not os.path.isdir(shard):
            continue
        name = os.path.basename(shard)
        side[name] = {}
        for extra in ("harness_check", "untrained_eval", "seed_manifest", "summary"):
            p = os.path.join(shard, extra + ".json")
            if os.path.exists(p):
                with open(p) as f:
                    side[name][extra] = json.load(f)
        for path in sorted(glob.glob(os.path.join(shard, "runs", "*.json"))):
            with open(path) as f:
                r = json.load(f)
            r["_path"] = path
            r["_shard"] = name
            runs.append(r)
    return runs, side


def load_prev_table(path):
    """Previous run's runs_table.csv -> {(cond, lr_key): {seed: row}}"""
    out = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            out[(row["condition"], lr_key(float(row["lr"])))][int(row["seed"])] = {
                k: (float(v) if v not in ("", None) else None) for k, v in row.items()
                if k not in ("condition",)}
    return out


def paired(a: dict, b: dict) -> dict:
    """a, b: seed -> value. Paired difference a - b over common seeds; point estimate and per-seed values only."""
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    return {"n": len(d), "seeds": seeds, "mean": float(np.mean(d)) if d else float("nan"),
            "per_seed": d, "min": min(d) if d else None, "max": max(d) if d else None,
            "same_sign": bool(d and all(np.sign(x) == np.sign(d[0]) for x in d))}


def smean(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else None


def delivery_row(w: dict) -> dict:
    lr_ = w["cumulative_length_ratio"]
    rho = w["cumulative_overshoot"]
    return {"cum_length_ratio": lr_, "cum_overshoot": rho, "cum_cosine": w["cumulative_cosine"],
            "never_moved": w["never_moved_fraction"],
            # projection of the realised change on the request: <I,R>/||I||^2 = len^2 / rho
            "cum_projection": (lr_ ** 2 / rho) if rho else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--prev-table", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs, side = load_runs(args.runs_dir)
    assert runs, "no runs found"
    prev = load_prev_table(args.prev_table)
    seeds = sorted({r["seed"] for r in runs})
    lrs = sorted({r["lr"] for r in runs}, reverse=True)
    print(f"{len(runs)} runs, seeds {seeds}, lrs {[lr_key(x) for x in lrs]}")

    # ------------------------------------------------------------------ run table (this experiment's runs)
    nll = defaultdict(dict)       # (cond, lrk) -> seed -> nll
    deliv = defaultdict(dict)     # (cond, lrk) -> seed -> delivery row
    rows = []
    for r in runs:
        k = (r["condition"], lr_key(r["lr"]))
        nll[k][r["seed"]] = r["eval"]["nll_per_token"]
        d = delivery_row(r["final_delivery"]["whole"])
        deliv[k][r["seed"]] = d
        row = {"seed": r["seed"], "condition": r["condition"], "lr": r["lr"], "eval_nll": r["eval"]["nll_per_token"],
               "eval_tokens": r["eval"]["n_predicted_tokens"], "loss_last200_mean": r["loss_last200_mean"], **d,
               "clip_fired_steps": r["clip_fired_steps"], "s_per_step": r["s_per_step"],
               "extra_state_bytes": r["extra_state_bytes"], "peak_gpu_mem_gb": r["peak_gpu_mem_gb"],
               "role": "harness" if "harness_check" in r else "keno", "shard": r["_shard"]}
        for f in FAMS:
            row[f"never_moved_{f}"] = r["final_delivery"]["families"][f]["never_moved_fraction"]
            row[f"cum_length_ratio_{f}"] = r["final_delivery"]["families"][f]["cumulative_length_ratio"]
        ds = r.get("distributor")
        row["mask_n_blocks"] = ds["n_blocks"] if ds else None
        row["mask_fraction_target_mean"] = ds["fraction_target_mean"] if ds else None
        row["mask_fraction_realized_mean"] = ds["fraction_realized_mean"] if ds else None
        row["mask_max_abs_target_gap"] = ds["max_abs_target_gap"] if ds else None
        hc = r.get("harness_check")
        row["harness_ref_nll"] = hc.get("eval_nll_reference") if hc else None
        row["harness_abs_diff"] = hc.get("eval_nll_abs_diff") if hc else None
        row["harness_passed"] = hc.get("passed") if hc else None
        rows.append(row)
    rows.sort(key=lambda x: (-x["lr"], (KENO + REF_CONDS).index(x["condition"]) if x["condition"] in KENO + REF_CONDS else 99, x["seed"]))
    with open(os.path.join(args.out, "runs_table.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)

    # ------------------------------------------------------------------ reference points from the previous run
    ref_nll = defaultdict(dict)     # (cond, lrk) -> seed -> nll   (previous run, seeds 0..4)
    ref_deliv = defaultdict(dict)
    for (cond, lrk), by_seed in prev.items():
        for s, row in by_seed.items():
            ref_nll[(cond, lrk)][s] = row["eval_nll"]
            ref_deliv[(cond, lrk)][s] = {"cum_length_ratio": row["cum_length_ratio"], "cum_overshoot": row["cum_overshoot"],
                                         "cum_cosine": row["cum_cosine"], "never_moved": row["never_moved"],
                                         "cum_projection": row["cum_length_ratio"] ** 2 / row["cum_overshoot"]}

    def ref_sub(cond, lrk, use_seeds):
        return {s: v for s, v in ref_nll[(cond, lrk)].items() if s in use_seeds}

    # ------------------------------------------------------------------ harness
    harness = {}
    for shard, sd in side.items():
        if "harness_check" in sd:
            harness[shard] = sd["harness_check"]
    harness_all_passed = bool(harness) and all(h.get("all_passed", False) for h in harness.values())
    harness_max_diff = max((c.get("eval_nll_abs_diff", float("nan")) for h in harness.values() for c in h["checks"]),
                           default=None)

    # ------------------------------------------------------------------ paired differences
    pairs = {}
    for lr in lrs:
        lrk = lr_key(lr)
        P = pairs[lrk] = {}
        for w in KENO:
            if not nll[(w, lrk)]:
                continue
            for ref in ("rne", "fp32_master", "kahan"):
                if ref_nll[(ref, lrk)]:
                    P[f"{w}-{ref}"] = paired(nll[(w, lrk)], ref_sub(ref, lrk, seeds))
        for fr in FRACS:
            t, d = f"keno_tab_{fr}", f"keno_drop_{fr}"
            if nll[(t, lrk)] and nll[(d, lrk)]:
                P[f"{t}-{d}"] = paired(nll[(t, lrk)], nll[(d, lrk)])
        # this run's harness rne vs the previous run's rne (should be ~0)
        if nll[("rne", lrk)]:
            P["rne_rerun-rne_prev"] = paired(nll[("rne", lrk)], ref_sub("rne", lrk, seeds))

    def pd(lr, a, b):
        return pairs[lr_key(lr)].get(f"{a}-{b}")

    # ------------------------------------------------------------------ verdicts
    verdicts = {"n_runs": len(runs), "seeds": seeds, "lrs": lrs, "q3_bar": Q3_BAR,
                "harness": {"all_passed": harness_all_passed, "max_eval_nll_abs_diff": harness_max_diff,
                            "per_shard": harness}}
    # Q1
    d1 = pd(2e-4, "keno_drop_25", "rne")
    q1 = {"rule": "mean over seeds of NLL(keno_drop_25) - NLL(rne) at lr 2e-4: |d|<=0.01 amount is the knob; d>+0.02 the "
                  "format's pick matters; d<-0.02 random selection wins; else undecided at two seeds",
          "difference": d1}
    if d1 is None or d1["n"] == 0:
        q1["verdict"] = "not measured"
    elif abs(d1["mean"]) <= BAND_Q1:
        q1["verdict"] = "amount delivered is the knob; the format's selection is incidental"
    elif d1["mean"] > BAND_Q1_STRONG:
        q1["verdict"] = "the format's magnitude-based pick is doing real work"
    elif d1["mean"] < -BAND_Q1_STRONG:
        q1["verdict"] = "random selection beats the format's pick"
    else:
        q1["verdict"] = "undecided at two seeds"
    if d1 and d1["n"]:
        kd = deliv[("keno_drop_25", "0.0002")]
        rn = {s: v for s, v in ref_deliv[("rne", "0.0002")].items() if s in seeds}
        q1["delivery_match"] = {
            "keno_drop_25_length_ratio_mean": smean([v["cum_length_ratio"] for v in kd.values()]),
            "rne_length_ratio_mean": smean([v["cum_length_ratio"] for v in rn.values()]),
            "keno_drop_25_projection_mean": smean([v["cum_projection"] for v in kd.values()]),
            "rne_projection_mean": smean([v["cum_projection"] for v in rn.values()]),
            "keno_drop_25_never_moved_mean": smean([v["never_moved"] for v in kd.values()]),
            "rne_never_moved_mean": smean([v["never_moved"] for v in rn.values()])}
        # the same contrast at lr 1e-4, reported for context (not part of the rule)
        q1["context_lr1e-4"] = pd(1e-4, "keno_drop_25", "rne")
        # Matched-delivery reading (added after review, before results were read): the plan expected keno_drop_25 to
        # deliver ~0.25 by length ratio, but for a mask independent of the request the length ratio is
        # sqrt(f^2 + f(1-f)x) > f, so keno_drop_25 will sit above rne on the delivery axis. Read the keno-drop curve
        # (drop 25/50/75/range, plus kahan = keno_drop at f=1 from the previous run) at rne's measured delivery.
        q1["matched_delivery"] = {}
        for lr in lrs:
            lrk = lr_key(lr)
            pts = []
            for w in ("keno_drop_25", "keno_drop_50", "keno_drop_75", "keno_drop_range"):
                if nll[(w, lrk)]:
                    pts.append((smean([v["cum_length_ratio"] for v in deliv[(w, lrk)].values()]),
                                smean([v["cum_projection"] for v in deliv[(w, lrk)].values()]),
                                smean(nll[(w, lrk)].values()), w))
            kh = {s: v for s, v in ref_deliv[("kahan", lrk)].items() if s in seeds}
            if kh and ref_nll[("kahan", lrk)]:
                pts.append((smean([v["cum_length_ratio"] for v in kh.values()]), smean([v["cum_projection"] for v in kh.values()]),
                            smean([v for s, v in ref_nll[("kahan", lrk)].items() if s in seeds]), "kahan (keno_drop at f=1)"))
            rn = {s: v for s, v in ref_deliv[("rne", lrk)].items() if s in seeds}
            rne_len = smean([v["cum_length_ratio"] for v in rn.values()])
            rne_proj = smean([v["cum_projection"] for v in rn.values()])
            rne_nll = smean([v for s, v in ref_nll[("rne", lrk)].items() if s in seeds])

            def interp(xcol, x0):
                P = sorted((p[xcol], p[2], p[3]) for p in pts)
                xs = [p[0] for p in P]
                if x0 < xs[0]:
                    return {"status": "below measured range", "nearest": P[0][2], "nearest_x": xs[0], "nearest_nll": P[0][1],
                            "extrapolated_nll": P[0][1] + (P[1][1] - P[0][1]) * (x0 - xs[0]) / (xs[1] - xs[0]) if len(P) > 1 else None}
                if x0 > xs[-1]:
                    return {"status": "above measured range", "nearest": P[-1][2], "nearest_x": xs[-1], "nearest_nll": P[-1][1]}
                y = float(np.interp(x0, xs, [p[1] for p in P]))
                return {"status": "interpolated", "nll": y}
            q1["matched_delivery"][lrk] = {
                "keno_drop_curve": [{"writer": p[3], "cum_length_ratio": p[0], "cum_projection": p[1], "nll_mean": p[2]} for p in pts],
                "rne": {"cum_length_ratio": rne_len, "cum_projection": rne_proj, "nll_mean": rne_nll},
                "by_length_ratio": interp(0, rne_len), "by_projection": interp(1, rne_proj)}
            for key in ("by_length_ratio", "by_projection"):
                m = q1["matched_delivery"][lrk][key]
                y = m.get("nll") if m["status"] == "interpolated" else m.get("extrapolated_nll")
                m["keno_drop_minus_rne_at_matched_delivery"] = (y - rne_nll) if y is not None else None
        q1["confounds"] = [
            "delivery mismatch: keno_drop_25's measured length ratio and projection differ from rne's; see matched_delivery",
            "write precision: drawn coordinates receive the exact Kahan write, rne quantises every write, so the keno_drop "
            "curve runs from undertrained toward kahan (f=1), not toward rne; a keno_drop - rne difference mixes selection, "
            "precision and delivery amount and is not attributable to selection alone"]
    verdicts["q1_how_much_or_which"] = q1
    # Q2
    q2 = {"rule": "per fraction and lr: |mean(tab - fp32_master)|<=0.01 -> delayed delivery behaves as delivery; "
                  "|mean(tab - drop)|<=0.01 -> banked requests might as well be discarded", "per_lr": {}}
    for lr in lrs:
        lrk = lr_key(lr)
        if not any(k.startswith("keno_tab_") for k in pairs[lrk]):
            continue  # no keno runs at this lr (harness-only lr)
        q2["per_lr"][lrk] = {}
        for fr in FRACS:
            tm = pd(lr, f"keno_tab_{fr}", "fp32_master")
            td = pairs[lrk].get(f"keno_tab_{fr}-keno_drop_{fr}")
            ent = {"tab_minus_master": tm, "tab_minus_drop": td}
            ent["behaves_as_delivery"] = bool(tm and tm["n"] and abs(tm["mean"]) <= BAND_Q2)
            ent["banking_indistinguishable_from_drop"] = bool(td and td["n"] and abs(td["mean"]) <= BAND_Q2)
            if ent["behaves_as_delivery"] and not ent["banking_indistinguishable_from_drop"]:
                ent["reading"] = "delayed delivery behaves as delivery"
            elif ent["banking_indistinguishable_from_drop"] and not ent["behaves_as_delivery"]:
                ent["reading"] = "requests not written promptly might as well be discarded"
            elif ent["behaves_as_delivery"] and ent["banking_indistinguishable_from_drop"]:
                ent["reading"] = "tab within 0.01 of both master and drop (master and drop themselves within 0.02)"
            else:
                ent["reading"] = "tab differs from both master and drop by more than 0.01"
            q2["per_lr"][lrk][fr] = ent
    verdicts["q2_banking"] = q2
    # Q3
    q3 = {"rule": f"flag any writer with seed-mean held-out NLL below {Q3_BAR} (fp32_master at lr 5e-5, previous grid)",
          "bar": Q3_BAR, "flagged": [], "seed_means": {}}
    master_5e5_prev = ref_nll[("fp32_master", "5e-05")]
    q3["reference_fp32_master_5e-5"] = {"seed_mean_all_prev_seeds": smean(master_5e5_prev.values()),
                                        "seed_mean_seeds_0_1": smean([v for s, v in master_5e5_prev.items() if s in seeds]),
                                        "per_seed": master_5e5_prev}
    for lr in lrs:
        lrk = lr_key(lr)
        for w in KENO:
            if nll[(w, lrk)]:
                m = smean(nll[(w, lrk)].values())
                q3["seed_means"][f"{w}@{lrk}"] = {"mean": m, "per_seed": nll[(w, lrk)], "below_bar": m < Q3_BAR}
                if m < Q3_BAR:
                    q3["flagged"].append({"writer": w, "lr": lr, "seed_mean_nll": m, "per_seed": nll[(w, lrk)]})
    q3["best_keno"] = min(q3["seed_means"].items(), key=lambda kv: kv[1]["mean"])[0] if q3["seed_means"] else None
    q3["verdict"] = (f"{len(q3['flagged'])} writer(s) below {Q3_BAR}" if q3["flagged"]
                     else f"no keno writer below {Q3_BAR} (best {q3['best_keno']})")
    verdicts["q3_beats_tuned_exact_delivery"] = q3

    # ------------------------------------------------------------------ delivery vs loss table
    dvl = {}
    for lr in lrs:
        lrk = lr_key(lr)
        dvl[lrk] = {}
        for w in KENO:
            if not nll[(w, lrk)]:
                continue
            dv = deliv[(w, lrk)]
            ent = {"source": "this run", "n_seeds": len(dv), "nll_mean": smean(nll[(w, lrk)].values()),
                   "nll_per_seed": nll[(w, lrk)]}
            for m in ("cum_length_ratio", "cum_projection", "cum_overshoot", "cum_cosine", "never_moved"):
                ent[m] = smean([v[m] for v in dv.values()])
                ent[m + "_per_seed"] = {s: v[m] for s, v in dv.items()}
            rs = [r for r in runs if r["condition"] == w and lr_key(r["lr"]) == lrk]
            ent["families"] = {f: {"never_moved": smean([r["final_delivery"]["families"][f]["never_moved_fraction"] for r in rs]),
                                   "cum_length_ratio": smean([r["final_delivery"]["families"][f]["cumulative_length_ratio"] for r in rs])}
                               for f in FAMS}
            ent["mask_fraction_realized_mean"] = smean([r["distributor"]["fraction_realized_mean"] for r in rs if r.get("distributor")])
            ent["mask_fraction_target_mean"] = smean([r["distributor"]["fraction_target_mean"] for r in rs if r.get("distributor")])
            ent["clip_fired_total"] = int(sum(r["clip_fired_steps"] for r in rs))
            ent["s_per_step"] = smean([r["s_per_step"] for r in rs])
            ent["extra_state_bytes"] = rs[0]["extra_state_bytes"]
            dvl[lrk][w] = ent
        for ref in REF_CONDS:
            if not ref_nll[(ref, lrk)]:
                continue
            sub = {s: v for s, v in ref_nll[(ref, lrk)].items() if s in seeds}
            dsub = {s: v for s, v in ref_deliv[(ref, lrk)].items() if s in seeds}
            ent = {"source": "previous run (seeds 0,1)", "n_seeds": len(sub), "nll_mean": smean(sub.values()), "nll_per_seed": sub,
                   "nll_mean_all_prev_seeds": smean(ref_nll[(ref, lrk)].values()), "n_prev_seeds": len(ref_nll[(ref, lrk)])}
            for m in ("cum_length_ratio", "cum_projection", "cum_overshoot", "cum_cosine", "never_moved"):
                ent[m] = smean([v[m] for v in dsub.values()])
            prow = prev[(ref, lrk)]
            ent["families"] = {f: {"never_moved": smean([prow[s][f"never_moved_{f}"] for s in sub]),
                                   "cum_length_ratio": smean([prow[s][f"cum_length_ratio_{f}"] for s in sub])} for f in FAMS}
            ent["extra_state_bytes"] = prow[next(iter(sub))]["extra_state_bytes"] if sub else None
            dvl[lrk][ref] = ent
    # 5e-5 reference row for the Q3 bar
    lrk = "5e-05"
    dvl[lrk] = {}
    for ref in ("rne", "fp32_master", "kahan"):
        if ref_nll[(ref, lrk)]:
            sub = {s: v for s, v in ref_nll[(ref, lrk)].items() if s in seeds}
            dsub = {s: v for s, v in ref_deliv[(ref, lrk)].items() if s in seeds}
            dvl[lrk][ref] = {"source": "previous run (seeds 0,1)", "n_seeds": len(sub), "nll_mean": smean(sub.values()),
                             "nll_per_seed": sub, "nll_mean_all_prev_seeds": smean(ref_nll[(ref, lrk)].values()),
                             **{m: smean([v[m] for v in dsub.values()]) for m in
                                ("cum_length_ratio", "cum_projection", "cum_overshoot", "cum_cosine", "never_moved")}}

    # ------------------------------------------------------------------ step series (seed means, keno + harness rne)
    series = {}
    for lr in lrs:
        lrk = lr_key(lr)
        series[lrk] = {}
        for c in KENO + ["rne"]:
            rs = [r for r in runs if r["condition"] == c and lr_key(r["lr"]) == lrk]
            if not rs:
                continue
            steps = [pt["step"] for pt in rs[0]["series"]]

            def col(getter):
                arr = np.array([[getter(pt) for pt in r["series"]] for r in rs], dtype=np.float64)
                return [round(float(x), 6) for x in arr.mean(0)]
            series[lrk][c] = {
                "steps": steps,
                "cumulative_length_ratio": col(lambda pt: pt["cumulative"]["whole"]["cumulative_length_ratio"]),
                "cumulative_overshoot": col(lambda pt: pt["cumulative"]["whole"]["cumulative_overshoot"]),
                "never_moved_fraction": col(lambda pt: pt["cumulative"]["whole"]["never_moved_fraction"]),
            }
            L = np.array([r["losses"] for r in rs], dtype=np.float64)
            nb = L.shape[1] // 100
            blocks = L[:, : nb * 100].reshape(len(rs), nb, 100).mean(2).mean(0)
            series[lrk][c]["loss_block100_steps"] = [(i + 1) * 100 for i in range(nb)]
            series[lrk][c]["loss_block100_mean"] = [round(float(x), 5) for x in blocks]
            if c != "rne":
                blk = [b for r in rs for b in r["distributor"]["blocks"]]
                series[lrk][c]["block_fraction_target"] = [round(b["fraction_target"], 4) for b in rs[0]["distributor"]["blocks"]]
                series[lrk][c]["block_fraction_realized"] = [round(b["fraction_realized"], 5) for b in rs[0]["distributor"]["blocks"]]
                series[lrk][c]["max_abs_target_gap_all_seeds"] = max(abs(b["fraction_realized"] - b["fraction_target"]) for b in blk)

    verdicts["timing"] = {"s_per_step_mean": float(np.mean([r["s_per_step"] for r in runs])),
                          "s_per_step_max": float(np.max([r["s_per_step"] for r in runs])),
                          "clip_fired_total": int(sum(r["clip_fired_steps"] for r in runs)),
                          "eval_tokens": sorted({r["eval"]["n_predicted_tokens"] for r in runs}),
                          "untrained_nll_per_shard": {k: v["untrained_eval"]["nll_per_token"] for k, v in side.items()
                                                      if "untrained_eval" in v}}
    verdicts["mask_realisation"] = {
        "max_abs_target_gap_over_all_keno_runs": max(r["distributor"]["max_abs_target_gap"] for r in runs if r.get("distributor")),
        "n_blocks_per_run": sorted({r["distributor"]["n_blocks"] for r in runs if r.get("distributor")})}

    for name, obj in (("verdicts", verdicts), ("paired_differences", pairs), ("delivery_vs_loss", dvl),
                      ("step_series", series)):
        with open(os.path.join(args.out, name + ".json"), "w") as f:
            json.dump(obj, f, indent=None if name == "step_series" else 1)
    print(json.dumps({"harness_all_passed": harness_all_passed, "q1": q1["verdict"],
                      "q1_d": d1["mean"] if d1 else None,
                      "q2": {lrk: {fr: e["reading"] for fr, e in v.items()} for lrk, v in q2["per_lr"].items()},
                      "q3": q3["verdict"]}, indent=1))
    for lr in lrs:
        lrk = lr_key(lr)
        print(f"lr {lrk}: " + ", ".join(f"{c} {e['nll_mean']:.4f}/{e['cum_length_ratio']:.3f}" for c, e in dvl[lrk].items()
                                        if e.get("nll_mean") is not None))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
