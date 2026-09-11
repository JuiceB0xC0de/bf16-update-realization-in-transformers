"""Write plot-ready data.json for each figure bundle from the analysis outputs, then render them.

Usage: uv run python src/prepare_figure_data.py --analysis results/analysis
Each figures/<name>/plot.py rebuilds its figure from the co-located data.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
FIG = os.path.join(EXP, "figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default=os.path.join(EXP, "results", "analysis"))
    ap.add_argument("--hist-step", type=int, default=100)
    args = ap.parse_args()
    A = args.analysis
    colors = json.load(open(os.path.join(FIG, "entity_colors.json")))
    series = json.load(open(os.path.join(A, "step_series.json")))
    verdicts = json.load(open(os.path.join(A, "verdicts.json")))
    per_order = json.load(open(os.path.join(A, "per_order.json")))
    hist = json.load(open(os.path.join(A, "x_hist.json")))
    orders = sorted(series)
    label = {k: k.replace("order", "order ") for k in orders}

    # 1. overshoot per step
    rows = [{"step": st, "order": label[k], "overshoot": rho}
            for k in orders for st, rho in zip(series[k]["step"], series[k]["overshoot"])]
    json.dump({"rows": rows, "mamba_median": verdicts["mamba_reference_medians"]["clip1"],
               "colors": {label[k]: colors[label[k]] for k in orders}},
              open(os.path.join(FIG, "overshoot_per_step", "data.json"), "w"))

    # 2. predicted vs stored on the captured steps (detail bundles)
    q2 = verdicts["q2_write_model_transfers"]["captured_steps"]
    rows = [{"order": label[r["order"]], "step": r["step"], "stored": r["stored"], "predicted": r["predicted"],
             "gap": r["gap"], "exact_write_agreement": r["exact_write_agreement"], "n": r["n"]} for r in q2]
    json.dump({"rows": rows, "colors": {**{label[k]: colors[label[k]] for k in orders},
                                        "diagonal": colors["Mamba-3 187M reference"]}},
              open(os.path.join(FIG, "predicted_vs_stored", "data.json"), "w"))

    # 3. shadow curve
    q3 = verdicts["q3_magnitude_dependence"]
    ser = [{"name": label[k], "overshoot": q3["per_order"][k]["median_overshoot"]} for k in orders]
    ser.append({"name": "Mamba-3 187M reference", "overshoot": q3["mamba_reference_bundle_shadow"]})
    json.dump({"factors": q3["factors"], "series": ser,
               "colors": {s["name"]: colors[s["name"]] for s in ser}},
              open(os.path.join(FIG, "shadow_curve", "data.json"), "w"))

    # 4. x histogram at hist-step, order 0, per family + Mamba bundle
    key = f"{orders[0]}/step{args.hist_step}"
    fams = hist["transformer"][key]
    lo = hist["log2_min"]
    nb = len(hist["mamba_step100_bundle"])
    bins = list(range(lo, lo + nb))
    ser = []
    for f in ("attention", "mlp", "embedding", "norm"):
        c = fams.get(f)
        if c and sum(c):
            tot = sum(c)
            ser.append({"name": f, "share": [v / tot for v in c], "n": tot})
    mt = sum(hist["mamba_step100_bundle"])
    ser.append({"name": "Mamba-3 187M reference", "share": [v / mt for v in hist["mamba_step100_bundle"]], "n": mt})
    nz = [b for s in ser for b, v in zip(bins, s["share"]) if v > 1e-4]
    json.dump({"log2_bin_lower": bins, "series": ser, "step": args.hist_step, "order": label[orders[0]],
               "x_range": [min(nz) - 0.5, max(nz) + 1.5],
               "colors": {s["name"]: colors[s["name"]] for s in ser}},
              open(os.path.join(FIG, "x_histogram", "data.json"), "w"))

    # 5. family overshoot per step (order 0), skipping families that never moved
    k0 = orders[0]
    rows = []
    for f, vals in series[k0]["family_overshoot"].items():
        if all(v != v for v in vals):
            continue
        rows += [{"step": st, "family": f, "overshoot": v} for st, v in zip(series[k0]["step"], vals) if v == v]
    json.dump({"rows": rows, "order": label[k0], "colors": {f: colors[f] for f in ("attention", "mlp", "embedding", "norm")}},
              open(os.path.join(FIG, "family_overshoot_per_step", "data.json"), "w"))

    for name in ("overshoot_per_step", "predicted_vs_stored", "shadow_curve", "x_histogram", "family_overshoot_per_step"):
        print(f"--- {name}")
        subprocess.run([sys.executable, os.path.join(FIG, name, "plot.py")], check=True, cwd=EXP)


if __name__ == "__main__":
    main()
