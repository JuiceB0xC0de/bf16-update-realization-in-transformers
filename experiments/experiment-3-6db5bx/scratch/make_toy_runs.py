"""Schema-faithful toy shards for exercising src/analyze.py on the pod (numbers are made up)."""
import json
import os
import random
import sys

out = sys.argv[1]
KENO = ["keno_drop_25", "keno_drop_50", "keno_drop_75", "keno_drop_range",
        "keno_tab_25", "keno_tab_50", "keno_tab_75", "keno_tab_range"]
FAMS = ("attention", "mlp", "embedding", "norm")
FRAC = {"25": 0.25, "50": 0.5, "75": 0.75, "range": 0.5}
rng = random.Random(0)


def cum(lr_, rho, never):
    return {"n": 134515008, "cumulative_length_ratio": lr_, "cumulative_overshoot": rho, "cumulative_cosine": 0.5,
            "never_moved_fraction": never, "intent_norm": 1.0, "realized_norm": lr_}


def rec(cond, lr, seed, steps=3000, harness=None):
    f = FRAC[cond.split("_")[2]] if cond.startswith("keno") else 0.26
    lr_ = f if cond.startswith("keno_drop") else (0.95 if cond.startswith("keno_tab") else 0.26)
    nll = 1.17 + rng.uniform(-0.02, 0.02) + (0.05 if cond.startswith("keno_tab") else 0)
    series = []
    for s in [1] + list(range(25, steps + 1, 25)):
        series.append({"step": s, "loss": 1.3, "per_step": {"whole": {}, "families": {}},
                       "cumulative": {"whole": cum(lr_ * min(1, s / 500 + 0.3), 0.9, 0.7), "families": {}}})
    dist = None
    if cond.startswith("keno"):
        blocks = [{"block": b, "first_step": 50 * b + 1, "fraction_target": f if "range" not in cond else rng.uniform(0.3, 0.7),
                   "n_drawn": 1, "n": 4} for b in range(steps // 50)]
        for b in blocks:
            b["fraction_realized"] = b["fraction_target"] + rng.uniform(-1e-4, 1e-4)
        dist = {"mode": cond.split("_")[1], "fraction": None if "range" in cond else f, "sticky_interval": 50,
                "n_blocks": len(blocks), "fraction_target_mean": sum(b["fraction_target"] for b in blocks) / len(blocks),
                "fraction_realized_mean": sum(b["fraction_realized"] for b in blocks) / len(blocks),
                "fraction_realized_min": 0.2, "fraction_realized_max": 0.8, "max_abs_target_gap": 1e-4, "blocks": blocks}
    r = {"condition": cond, "lr": lr, "seed": seed, "steps": steps, "hyperparams": {}, "distributor": dist,
         "clip": 32.0, "clip_fired_steps": rng.randint(0, 3), "extra_state_bytes": 0,
         "losses": [1.5 + rng.uniform(-0.3, 0.3) for _ in range(steps)], "grad_norms_preclip": [], "series": series,
         "final_delivery": {"whole": cum(lr_, 0.9, 0.6), "families": {fa: cum(lr_, 0.9, 0.6) for fa in FAMS}},
         "eval": {"nll_per_token": nll, "n_predicted_tokens": 49277, "n_examples": 256}, "loss_last200_mean": 1.2,
         "train_time_s": 330.0, "s_per_step": 0.11, "eval_time_s": 20.0, "peak_gpu_mem_gb": 3.9}
    if harness is not None:
        r["harness_check"] = harness
    return r


for seed in (0, 1):
    for lr, lrs in ((2e-4, "2e-4"), (1e-4, "1e-4")):
        shard = os.path.join(out, f"seed{seed}_lr{lrs}")
        os.makedirs(os.path.join(shard, "runs"), exist_ok=True)
        hc = {"reference": "x", "eval_nll_reference": 1.165, "eval_nll_observed": 1.1652, "eval_nll_abs_diff": 0.0002,
              "passed": True, "n_losses_compared": 3000, "losses_max_abs_diff": 0.0}
        json.dump({"seed": seed, "tol": 0.001, "checks": [{"condition": "rne", "lr": lr, **hc}], "all_passed": True},
                  open(os.path.join(shard, "harness_check.json"), "w"))
        json.dump({"nll_per_token": 1.94}, open(os.path.join(shard, "untrained_eval.json"), "w"))
        json.dump(rec("rne", lr, seed, harness=hc), open(os.path.join(shard, "runs", f"rne_lr{lr:g}.json"), "w"))
        for c in KENO:
            json.dump(rec(c, lr, seed), open(os.path.join(shard, "runs", f"{c}_lr{lr:g}.json"), "w"))
print("toy shards under", out)
