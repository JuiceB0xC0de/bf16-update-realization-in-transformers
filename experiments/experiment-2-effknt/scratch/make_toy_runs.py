"""Generate a schema-faithful toy sweep (random numbers) to exercise src/analyze.py on the pod.
Values are not scientific. Usage: apy scratch/make_toy_runs.py /tmp/toy_runs"""
import json
import os
import sys

import numpy as np

CONDS = ["torch_adamw", "rne", "kahan", "fp32_master", "selective", "sr"]
LRS = [2e-4, 1e-4, 5e-5]
FAMS = ("attention", "mlp", "embedding", "norm")
out = sys.argv[1]
rng = np.random.default_rng(0)
steps = 3000
stat_steps = [1] + list(range(25, steps + 1, 25))


def cum(cond, frac):
    base = {"torch_adamw": 0.5, "rne": 0.5, "kahan": 1.0, "fp32_master": 1.0, "selective": 0.8, "sr": 1.0}[cond]
    lr = base * (0.6 + 0.4 * frac) + rng.normal(0, 0.005)
    never = {"torch_adamw": 0.4, "rne": 0.4, "kahan": 0.05, "fp32_master": 0.05, "selective": 0.1, "sr": 0.02}[cond]
    d = {"n": 1000, "cumulative_length_ratio": lr, "cumulative_overshoot": 1 / max(lr, 0.3) * 0.9 + rng.normal(0, 0.01),
         "cumulative_cosine": min(0.999, 0.5 + 0.5 * lr), "never_moved_fraction": never * (1.2 - 0.2 * frac),
         "intent_norm": 1.0, "realized_norm": lr}
    return {"whole": d, "families": {f: dict(d) for f in FAMS}}


def step_stats():
    d = {"n": 1000, "overshoot": 1.32, "length_ratio": 0.52, "cosine": 0.4, "unmoved_fraction": 0.93,
         "exact_write_agreement": 0.999, "intent_norm": 1.0, "realized_norm": 0.5}
    return {"whole": d, "families": {f: dict(d) for f in FAMS}}


for seed in range(5):
    sd = os.path.join(out, f"seed{seed}", "runs")
    os.makedirs(sd, exist_ok=True)
    json.dump({"nll_per_token": 1.95 + rng.normal(0, 1e-6), "n_predicted_tokens": 45000, "n_examples": 256,
               "mean_example_nll": 1.94}, open(os.path.join(out, f"seed{seed}", "untrained_eval.json"), "w"))
    json.dump({"min_exact_write_agreement": 0.9979, "min_exp_avg_equal": 1.0, "min_exp_avg_sq_equal": 1.0,
               "max_abs_loss_diff": 0.0, "torch_resolved_foreach": True, "passed": True, "n_steps": 20},
              open(os.path.join(out, f"seed{seed}", "equivalence.json"), "w"))
    json.dump({"seed": seed, "steps": steps, "train_row_indices": list(range(steps)),
               "eval": {"n": 256, "n_predicted_tokens": 45000, "row_indices": list(range(256))}},
              open(os.path.join(out, f"seed{seed}", "seed_manifest.json"), "w"))
    for lr in LRS:
        for k, cond in enumerate(CONDS):
            base_nll = 1.20 + 0.1 * np.log10(lr / 2e-4) ** 2 + 0.03 * seed
            eff = {"torch_adamw": 0.0, "rne": 0.0003, "kahan": 0.08, "fp32_master": 0.08, "selective": 0.03, "sr": 0.05}[cond]
            eff *= (1 if lr > 1.5e-4 else -1) if cond != "torch_adamw" else 1
            losses = (1.9 * np.exp(-np.arange(steps) / 800) + 1.1 + rng.normal(0, 0.3, steps)).tolist()
            rec = {"condition": cond, "lr": lr, "seed": seed, "steps": steps,
                   "hyperparams": {"lr": lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
                   "clip": 32.0, "clip_fired_steps": 0, "sr_seed": None, "extra_state_bytes": 0,
                   "losses": losses, "grad_norms_preclip": (rng.random(steps) * 10).tolist(),
                   "series": [{"step": s, "loss": losses[s - 1], "per_step": step_stats(), "cumulative": cum(cond, s / steps)}
                              for s in stat_steps],
                   "final_delivery": cum(cond, 1.0), "eval": {"nll_per_token": base_nll + eff + rng.normal(0, 0.002),
                                                             "n_predicted_tokens": 45000, "n_examples": 256,
                                                             "mean_example_nll": 1.2},
                   "loss_first50_mean": 1.9, "loss_last200_mean": 1.1, "train_time_s": 330.0, "s_per_step": 0.11,
                   "eval_time_s": 4.0, "peak_gpu_mem_gb": 3.8, "wandb_url": None}
            if cond == "fp32_master":
                rec["final_delivery"]["master_side"] = {"cumulative_length_ratio": 1.0000001, "cumulative_overshoot": 1.0,
                                                        "cumulative_cosine": 1.0, "never_moved_fraction": 0.0}
            json.dump(rec, open(os.path.join(sd, f"{cond}_lr{lr:g}.json"), "w"))
print("toy runs written to", out)
