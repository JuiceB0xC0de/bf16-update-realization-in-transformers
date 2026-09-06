"""Schema-faithful toy shards to exercise src/analyze.py on the pod. Values are random; not results."""
import json, os, sys, numpy as np
root = sys.argv[1]
rng = np.random.default_rng(0)
FAMS = ("attention", "mlp", "embedding", "norm")
def rec(cond, lr, seed, theta=None, harness=False):
    steps = list(range(25, 3001, 25)); steps = [1] + steps
    d = {"15": 0.22, "25": 0.30, "50": 0.55}[cond.split("_")[-1]] if cond != "rne" else 0.26
    d += rng.normal(0, 0.005)
    r = {"condition": cond, "lr": lr, "seed": seed, "steps": 3000, "theta": theta, "clip_fired_steps": int(rng.integers(0, 3)),
         "extra_state_bytes": 269030016, "losses": list(rng.normal(1.2, 0.3, 3000)), "loss_last200_mean": 1.15,
         "s_per_step": 0.15, "peak_gpu_mem_gb": 3.9,
         "eval": {"nll_per_token": 1.16 + rng.normal(0, 0.01) - (0.01 if cond.startswith("keno") else 0), "n_predicted_tokens": 49277},
         "final_delivery": {"whole": {"cumulative_length_ratio": d, "cumulative_overshoot": d * 1.2, "cumulative_cosine": 0.8, "never_moved_fraction": 0.1},
                            "families": {f: {"cumulative_length_ratio": d, "never_moved_fraction": 0.1 + 0.05 * i} for i, f in enumerate(FAMS)}},
         "series": [{"step": s, "loss": 1.2, "cumulative": {"whole": {"cumulative_length_ratio": d + 0.1 * (1 - s / 3000), "never_moved_fraction": 0.1}},
                     "per_step": {"whole": {"length_ratio": d}}} for s in steps]}
    if cond.startswith("thr"):
        r["distributor"] = {"mode": "threshold_drop", "theta": theta, "fraction_realized_mean": 0.05, "fraction_realized_first100_mean": 0.08,
                            "fraction_realized_last100_mean": 0.04, "step_fraction": list(rng.uniform(0.03, 0.08, 3000))}
    elif cond.startswith("keno"):
        r["distributor"] = {"n_blocks": 60, "fraction_target_mean": 0.25, "fraction_realized_mean": 0.25, "max_abs_target_gap": 1e-4}
    if harness:
        r["harness_check"] = {"passed": True, "eval_nll_abs_diff": 0.0, "eval_nll_reference": r["eval"]["nll_per_token"]}
    return r
def put(job, r):
    p = os.path.join(root, job, f"seed{r['seed']}", "runs"); os.makedirs(p, exist_ok=True)
    json.dump(r, open(os.path.join(p, f"{r['condition']}_lr{r['lr']:g}.json"), "w"))
lrs = [2e-4, 1e-4]
put("keno_A", rec("rne", 2e-4, 2, harness=True))
for lr in lrs:
    for s in (1, 2, 3, 4): put("keno_A", rec("keno_drop_15", lr, s))
    for s in (2, 3, 4):
        for lv in ("25", "50"): put("keno_B", rec(f"keno_drop_{lv}", lr, s))
    for lv in ("15", "25", "50"):
        th = 0.4
        job = f"cal_{lv}_lr{lr:g}"
        if lv == "15": put(job, rec("keno_drop_15", lr, 0))
        r = rec(f"thr_drop_{lv}", lr, 0, theta=th); r["calibration"] = {"pass": 2, "theta": th}; put(job, r)
        os.makedirs(os.path.join(root, job, "seed0", "calibration"), exist_ok=True)
        json.dump({"writer": f"thr_drop_{lv}", "level": lv, "lr": lr, "seed": 0, "target": 0.3, "converged": True, "theta": th, "delivery": 0.31,
                   "n_passes": 2, "passes": [{"pass": 1, "theta": 0.5, "cum_length_ratio": 0.25, "gap": -0.05, "record": "x"}, {"pass": 2, "theta": th, "cum_length_ratio": 0.31, "gap": 0.01, "record": "y"}]},
                  open(os.path.join(root, job, "seed0", "calibration", f"theta_calibration_thr_drop_{lv}_lr{lr:g}.json"), "w"))
        for s in (1, 2, 3, 4): put(f"thr_{s}", rec(f"thr_drop_{lv}", lr, s, theta=th))
print("toy runs written")
