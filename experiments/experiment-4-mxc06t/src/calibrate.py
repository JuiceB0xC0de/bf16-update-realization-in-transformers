"""Calibrate the threshold selector's theta so its cumulative delivery matches a keno_drop level.

Two modes.

probe      short passes (default 300 steps, seed 0) of thr_drop at a list of thetas and of keno_drop levels, at one
           lr; writes a JSON table of theta -> cumulative length ratio. Used interactively to get a starting theta and
           the local slope d(delivery)/d(log theta) for every target.

calibrate  full-horizon (3000 steps, seed 0) secant iteration on log(theta) for ONE (lr, level) target: the target
           delivery is either given (--target, the keno_drop level's seed-0 value from the previous run) or measured
           first by running the keno_drop level itself (--run-keno, which also yields that seed-0 production record).
           Every pass is a complete production-schema run record (train_sweep.train_run, tagged calN); the pass that
           lands within --tol of the target is copied to seed0/runs/<thr>_lr<lr>.json as the seed-0 production run,
           and theta_calibration_<thr>_lr<lr>.json records every pass. Exit 5 if no pass lands within tol.

Why full horizon: the delivery of a magnitude-based selector declines over the run (requests shrink as training
settles, so fewer pass the threshold), unlike the random selector's near-constant fraction, so a 300-step match does
not transfer to 3000 steps (the previous run's rne delivery fell 0.438 -> 0.264 between steps 300 and 3000 while
keno_drop_25 fell 0.365 -> 0.309).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_sweep import (add_common_args, banner, evaluate, load_data, load_model, log, lr_key, seed_examples,  # noqa: E402
                         train_run, wandb_or_none, write_seed_manifest)
from write_distributor import THR_WRITERS  # noqa: E402


def delivery_of(rec) -> float:
    return rec["final_delivery"]["whole"]["cumulative_length_ratio"]


def next_theta_secant(passes, target, slope_guess):
    """passes: list of (theta, delivery). Delivery decreases with theta. Returns next theta (log-space secant,
    regula falsi when the target is bracketed, slope-guess step when only one pass exists)."""
    pts = sorted(passes, key=lambda x: x[0])
    if len(pts) == 1:
        th, d = pts[0]
        # log theta_1 = log theta_0 + (d - target) / |slope|, slope in delivery per log-theta unit
        return th * math.exp((d - target) / max(abs(slope_guess), 1e-3))
    # bracket: largest theta with d > target and smallest theta with d < target
    above = [p for p in pts if p[1] > target]
    below = [p for p in pts if p[1] < target]
    if above and below:
        (ta, da), (tb, db) = max(above, key=lambda x: x[0]), min(below, key=lambda x: x[0])
    else:
        # not bracketed: secant through the two passes with delivery closest to the target
        (ta, da), (tb, db) = sorted(pts, key=lambda x: abs(x[1] - target))[:2]
    la, lb = math.log(ta), math.log(tb)
    if db == da:
        return math.exp((la + lb) / 2)
    l_new = la + (target - da) * (lb - la) / (db - da)
    # keep the step inside a sane range
    lo, hi = math.log(min(p[0] for p in pts)) - 1.5, math.log(max(p[0] for p in pts)) + 1.5
    return math.exp(min(max(l_new, lo), hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "calibrate"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    # probe
    ap.add_argument("--thetas", type=float, nargs="*", default=[])
    ap.add_argument("--keno", nargs="*", default=[], help="keno_drop levels to run in probe mode, e.g. keno_drop_15")
    # calibrate
    ap.add_argument("--writer", default=None, help="thr_drop_15 | thr_drop_25 | thr_drop_50")
    ap.add_argument("--target", type=float, default=None, help="target cumulative length ratio")
    ap.add_argument("--run-keno", default=None, help="measure the target by running this keno writer first (also a production run)")
    ap.add_argument("--theta0", type=float, default=None)
    ap.add_argument("--slope", type=float, default=0.15, help="|d delivery / d log theta| guess for the second pass")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--max-passes", type=int, default=5)
    add_common_args(ap)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.manual_seed(args.torch_seed)
    np.random.seed(args.torch_seed)
    banner(args)
    tokenizer, ds, eval_rows, eval_examples = load_data(args)
    order, examples = seed_examples(tokenizer, ds, args.seed, args)
    out = os.path.join(args.out, f"seed{args.seed}")
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    os.makedirs(os.path.join(out, "calibration"), exist_ok=True)
    model = load_model(device)
    pristine = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    wandb_mod = wandb_or_none(args)
    lrk = lr_key(args.lr)

    if args.mode == "probe":
        rows = []
        plan = [(k, None) for k in args.keno] + [("thr_probe", th) for th in args.thetas]
        total = len(plan) * args.steps
        for j, (cond, th) in enumerate(plan):
            model.load_state_dict(pristine)
            if th is None:
                rec = train_run(model, cond, args.lr, args.seed, examples, eval_examples, device, args, wandb_mod,
                                j * args.steps, total, j, tag=f"probe{args.steps}")
                rows.append({"condition": cond, "theta": None, "steps": args.steps, "cum_length_ratio": delivery_of(rec),
                             "eval_nll": rec["eval"]["nll_per_token"], "never_moved": rec["final_delivery"]["whole"]["never_moved_fraction"]})
            else:
                rec = train_run(model, "thr_drop_25", args.lr, args.seed, examples, eval_examples, device, args, wandb_mod,
                                j * args.steps, total, j, theta=th, tag=f"probe{args.steps}")
                ds_ = rec["distributor"]
                rows.append({"condition": "thr_drop", "theta": th, "steps": args.steps, "cum_length_ratio": delivery_of(rec),
                             "eval_nll": rec["eval"]["nll_per_token"], "never_moved": rec["final_delivery"]["whole"]["never_moved_fraction"],
                             "selected_fraction_mean": ds_["fraction_realized_mean"],
                             "selected_first100": ds_["fraction_realized_first100_mean"], "selected_last100": ds_["fraction_realized_last100_mean"],
                             "s_per_step": rec["s_per_step"]})
            log("PROBE " + json.dumps(rows[-1]))
            path = os.path.join(out, "calibration", f"probe_lr{lrk}_steps{args.steps}.json")
            with open(path, "w") as f:
                json.dump({"lr": args.lr, "seed": args.seed, "steps": args.steps, "rows": rows}, f, indent=1)
        log(f"probe done -> {path}")
        return

    # ------------------------------------------------------------------ calibrate
    assert args.writer in THR_WRITERS, args.writer
    level = args.writer.split("_")[-1]
    total = (args.max_passes + (1 if args.run_keno else 0)) * args.steps
    j = 0
    target = args.target
    target_source = {"given": args.target}
    if args.run_keno:
        keno_path = os.path.join(out, "runs", f"{args.run_keno}_lr{lrk}.json")
        if os.path.exists(keno_path):
            rec = json.load(open(keno_path))
            log(f"reusing existing {keno_path}")
        else:
            log(f"phase: target run {args.run_keno} lr {lrk} seed {args.seed}")
            model.load_state_dict(pristine)
            rec = train_run(model, args.run_keno, args.lr, args.seed, examples, eval_examples, device, args, wandb_mod,
                            0, total, 0)
            with open(keno_path + ".tmp", "w") as f:
                json.dump(rec, f)
            os.replace(keno_path + ".tmp", keno_path)
        target = delivery_of(rec)
        target_source = {"run": keno_path, "eval_nll": rec["eval"]["nll_per_token"]}
        j = 1
    assert target is not None
    log(f"calibrating {args.writer} at lr {lrk}: target delivery {target:.4f} (tol {args.tol}), theta0 {args.theta0}")
    write_seed_manifest(os.path.join(out, "seed_manifest.json"), tokenizer, args.seed, order, examples, eval_rows,
                        eval_examples, args, [args.writer] + ([args.run_keno] if args.run_keno else []), [args.lr])
    passes = []
    theta = float(args.theta0)
    result = {"writer": args.writer, "level": level, "lr": args.lr, "seed": args.seed, "target": target,
              "target_source": target_source, "tol": args.tol, "passes": [], "converged": False}
    cal_json = os.path.join(out, "calibration", f"theta_calibration_{args.writer}_lr{lrk}.json")
    for k in range(args.max_passes):
        log(f"phase: calibration pass {k + 1}/{args.max_passes} theta={theta:.5f}")
        model.load_state_dict(pristine)
        t0 = time.time()
        rec = train_run(model, args.writer, args.lr, args.seed, examples, eval_examples, device, args, wandb_mod,
                        (j + k) * args.steps, total, j + k, theta=theta, tag=f"cal{k + 1}")
        d = delivery_of(rec)
        rec_path = os.path.join(out, "calibration", f"{args.writer}_lr{lrk}_pass{k + 1}_theta{theta:.5f}.json")
        with open(rec_path, "w") as f:
            json.dump(rec, f)
        passes.append((theta, d))
        entry = {"pass": k + 1, "theta": theta, "cum_length_ratio": d, "gap": d - target, "eval_nll": rec["eval"]["nll_per_token"],
                 "selected_fraction_mean": rec["distributor"]["fraction_realized_mean"],
                 "selected_first100": rec["distributor"]["fraction_realized_first100_mean"],
                 "selected_last100": rec["distributor"]["fraction_realized_last100_mean"],
                 "never_moved": rec["final_delivery"]["whole"]["never_moved_fraction"], "record": rec_path,
                 "minutes": (time.time() - t0) / 60}
        result["passes"].append(entry)
        log("CAL " + json.dumps(entry))
        if abs(d - target) <= args.tol:
            result.update({"converged": True, "theta": theta, "delivery": d, "n_passes": k + 1})
            final = os.path.join(out, "runs", f"{args.writer}_lr{lrk}.json")
            rec["calibration"] = {"target": target, "tol": args.tol, "pass": k + 1, "theta": theta}
            with open(final + ".tmp", "w") as f:
                json.dump(rec, f)
            os.replace(final + ".tmp", final)
            result["production_record"] = final
            with open(cal_json, "w") as f:
                json.dump(result, f, indent=1)
            log(f"CONVERGED {args.writer} lr {lrk}: theta={theta:.5f} delivery {d:.4f} (target {target:.4f}) after {k + 1} passes -> {final}")
            return
        with open(cal_json, "w") as f:
            json.dump(result, f, indent=1)
        theta = next_theta_secant(passes, target, args.slope)
    best = min(result["passes"], key=lambda e: abs(e["gap"]))
    result.update({"converged": False, "best_pass": best})
    with open(cal_json, "w") as f:
        json.dump(result, f, indent=1)
    log(f"NOT CONVERGED after {args.max_passes} passes; best theta {best['theta']:.5f} gap {best['gap']:+.4f}")
    sys.exit(5)


if __name__ == "__main__":
    main()
