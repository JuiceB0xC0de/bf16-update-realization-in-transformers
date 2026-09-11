"""One seed of the writer x learning-rate sweep: SmolLM2-135M (bf16) fine-tuned on GSM8K, batch 1.

For one data-order seed, every (condition, lr) pair trains on the identical 3,000-row stream from the
pristine checkpoint, so all contrasts are paired within seed. Conditions are the five
CompensatedAdamW writers plus plain torch.optim.AdamW (null control).

Per run this records
  - loss and pre-clip grad norm at every step,
  - every --stats-every steps: pooled per-step intended-vs-realized statistics (whole model and per
    family) and the cumulative delivery so far (R = p - p_init against I = sum of intents),
  - final cumulative delivery (whole model and per family) and the held-out NLL per token on the
    fixed 256-question GSM8K test probe (all tokens scored).
The untrained model's held-out NLL is evaluated once per job. A 20-step equivalence run of the rne
writer against torch.optim.AdamW on the real model (identical grads, shared trajectory) runs first
and aborts the job if the exact write agreement falls below 0.997.

Outputs (under --out, one directory per seed):
  seed_manifest.json, untrained_eval.json, equivalence.json, runs/<condition>_lr<lr>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from compensated_adamw import CONDITIONS, CompensatedAdamW, make_optimizer  # noqa: E402
from realization import Acc, cumulative_summary  # noqa: E402

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REV = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DATA_ID = "openai/gsm8k"
DATA_REV = "740312add88f781978c0658806c59bc2815b9866"
N_TRAIN_ROWS = 7473
N_TEST_ROWS = 1319
EXPECTED_PARAMS = 134_515_008
FAMILIES = ("attention", "mlp", "embedding", "norm")
DEFAULT_CONDITIONS = ("torch_adamw", "rne", "kahan", "fp32_master", "selective", "sr")
EVAL_PERM_SEED = 1234


def family_of(name: str) -> str:
    if "embed_tokens" in name:
        return "embedding"
    if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    if any(k in name for k in ("gate_proj", "up_proj", "down_proj")):
        return "mlp"
    if "norm" in name:
        return "norm"
    raise ValueError(f"unclassified parameter {name}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- data
def build_examples(tokenizer, ds, order, max_len: int):
    eos = tokenizer.eos_token_id
    assert eos is not None
    out = []
    for idx in [int(x) for x in order]:
        row = ds[idx]
        text = row["question"] + "\n" + row["answer"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [eos]
        full = len(ids)
        ids = ids[:max_len]
        out.append({"row": idx, "ids": ids, "full_len": full, "truncated": full > max_len})
    return out


# ----------------------------------------------------------------------------- model
def load_model(device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model.to(device)
    model.train()
    n = sum(p.numel() for p in model.parameters())
    assert n == EXPECTED_PARAMS, n
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "expected tied embedding"
    assert model.config.num_hidden_layers == 30
    return model


@torch.no_grad()
def evaluate(model, examples, device) -> dict:
    """Held-out NLL per predicted token, all tokens scored (sum of token NLLs / number of predicted tokens)."""
    model.eval()
    nll_sum = 0.0
    n_tok = 0
    per_example = []
    for ex in examples:
        ids = torch.tensor([ex["ids"]], device=device)
        out = model(input_ids=ids, labels=ids)
        n = len(ex["ids"]) - 1
        s = float(out.loss.float()) * n
        nll_sum += s
        n_tok += n
        per_example.append(s / n)
    model.train()
    return {"nll_per_token": nll_sum / n_tok, "n_predicted_tokens": n_tok, "n_examples": len(examples),
            "mean_example_nll": float(np.mean(per_example))}


# ----------------------------------------------------------------------------- one run
class Delivery:
    """Cumulative intent accumulator and per-step statistics for one run."""

    def __init__(self, named):
        self.named = named
        self.fam = {id(p): family_of(n) for n, p in named}
        self.p_init = {id(p): p.detach().float().clone() for _, p in named}
        self.I = {id(p): torch.zeros_like(p, dtype=torch.float32) for _, p in named}
        self.step_accs = None

    def start_step_stats(self):
        self.step_accs = {f: Acc() for f in FAMILIES}

    def hook(self, p, pre, i, post):
        self.I[id(p)].add_(i)
        if self.step_accs is not None:
            self.step_accs[self.fam[id(p)]].add(i, post - pre, pre)

    def finish_step_stats(self) -> dict:
        whole = Acc()
        for a in self.step_accs.values():
            whole.merge(a)
        out = {"whole": whole.summary(), "families": {f: a.summary() for f, a in self.step_accs.items()}}
        self.step_accs = None
        return out

    @torch.no_grad()
    def cumulative(self) -> dict:
        accs = {f: Acc() for f in FAMILIES}
        for _, p in self.named:
            r = p.float() - self.p_init[id(p)]
            accs[self.fam[id(p)]].add(self.I[id(p)], r)
        whole = Acc()
        for a in accs.values():
            whole.merge(a)
        return {"whole": cumulative_summary(whole), "families": {f: cumulative_summary(a) for f, a in accs.items()}}


def train_run(model, condition, lr, seed, examples, eval_examples, device, args, wandb_mod, job_step_base,
              job_total_steps, run_index):
    from silico.slurm_telemetry import report_progress, register_wandb_url, register_key_metrics
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    torch.manual_seed(args.torch_seed)
    sr_seed = 100_000 * seed + 1_000 * run_index + 7
    opt = make_optimizer(condition, params, lr, sr_seed=sr_seed)
    g = opt.param_groups[0]
    hp = {"lr": float(g["lr"]), "betas": [float(b) for b in g["betas"]], "eps": float(g["eps"]),
          "weight_decay": float(g["weight_decay"])}
    deliv = Delivery(named)
    run_name = f"{condition}-lr{lr:g}-seed{seed}"
    wb = None
    if wandb_mod is not None:
        wb = wandb_mod.init(project=f"default-exp-{os.environ.get('SILICO_EXPERIMENT_ID', 'local')}", name=run_name,
                            config={"condition": condition, "lr": lr, "seed": seed, "max_steps": args.steps,
                                    "model": MODEL_ID, "model_revision": MODEL_REV, "dataset": DATA_ID,
                                    "dataset_revision": DATA_REV, "clip": args.clip, **hp})
        register_wandb_url()
        register_key_metrics(["train/loss", "eval/nll_heldout", "delivery/cumulative_length_ratio",
                              "delivery/never_moved_fraction"])
    losses, gnorms, series = [], [], []
    clip_fired = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_steps = args.steps
    for step in range(1, n_steps + 1):
        ex = examples[step - 1]
        ids = torch.tensor([ex["ids"]], device=device)
        loss = model(input_ids=ids, labels=ids).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        pre_norm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        fired = pre_norm > args.clip
        clip_fired += int(fired)
        want_stats = step == 1 or step % args.stats_every == 0 or step == n_steps
        if want_stats:
            deliv.start_step_stats()
        opt.step(hook=deliv.hook)
        lv = float(loss.detach())
        losses.append(lv)
        gnorms.append(pre_norm)
        if wb is not None:
            wb.log({"train/loss": lv, "train/grad_norm_preclip": pre_norm}, step=step)
        if want_stats:
            st = deliv.finish_step_stats()
            cum = deliv.cumulative()
            series.append({"step": step, "loss": lv, "per_step": st, "cumulative": cum})
            w, c = st["whole"], cum["whole"]
            if wb is not None:
                wb.log({"delivery/cumulative_length_ratio": c["cumulative_length_ratio"],
                        "delivery/cumulative_overshoot": c["cumulative_overshoot"],
                        "delivery/cumulative_cosine": c["cumulative_cosine"],
                        "delivery/never_moved_fraction": c["never_moved_fraction"],
                        "step_stats/overshoot": w["overshoot"], "step_stats/length_ratio": w["length_ratio"],
                        "step_stats/unmoved_fraction": w["unmoved_fraction"],
                        "step_stats/exact_write_agreement": w["exact_write_agreement"]}, step=step)
            elapsed = time.time() - t0
            log(f"{run_name} step {step}/{n_steps} loss={lv:.4f} gnorm={pre_norm:.2f} "
                f"step[rho={w['overshoot']:.3f} len={w['length_ratio']:.3f} unmoved={w['unmoved_fraction']:.3f} "
                f"exact={w['exact_write_agreement']:.4f}] cum[len={c['cumulative_length_ratio']:.4f} "
                f"rho={c['cumulative_overshoot']:.4f} cos={c['cumulative_cosine']:.4f} "
                f"never={c['never_moved_fraction']:.4f}] ({elapsed / step:.3f}s/step)")
        report_progress(step=job_step_base + step, total_steps=job_total_steps, phase=run_name)
    torch.cuda.synchronize()
    train_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    t1 = time.time()
    ev = evaluate(model, eval_examples, device)
    eval_time = time.time() - t1
    final = deliv.cumulative()
    if condition == "fp32_master":
        # master-side delivery check (should be ~1): I against master - p_init
        acc = Acc()
        for _, p in named:
            acc.add(deliv.I[id(p)], opt.state[p]["master"] - deliv.p_init[id(p)])
        final["master_side"] = cumulative_summary(acc)
    rec = {
        "condition": condition, "lr": lr, "seed": seed, "steps": n_steps, "hyperparams": hp,
        "clip": args.clip, "clip_fired_steps": clip_fired, "sr_seed": sr_seed if condition == "sr" else None,
        "extra_state_bytes": opt.buffer_bytes(),
        "losses": losses, "grad_norms_preclip": gnorms, "series": series,
        "final_delivery": final, "eval": ev,
        "loss_first50_mean": float(np.mean(losses[:50])), "loss_last200_mean": float(np.mean(losses[-200:])),
        "train_time_s": train_time, "s_per_step": train_time / n_steps, "eval_time_s": eval_time,
        "peak_gpu_mem_gb": peak_mem, "wandb_url": (wb.url if wb is not None else None),
    }
    c = final["whole"]
    log(f"{run_name} DONE: eval nll {ev['nll_per_token']:.5f} ({ev['n_predicted_tokens']} tok), "
        f"cum len {c['cumulative_length_ratio']:.4f} rho {c['cumulative_overshoot']:.4f} "
        f"cos {c['cumulative_cosine']:.4f} never {c['never_moved_fraction']:.4f}, "
        f"{train_time / n_steps:.3f}s/step, eval {eval_time:.1f}s, peak {peak_mem:.2f} GB, clip fired {clip_fired}")
    if wb is not None:
        wb.log({"eval/nll_heldout": ev["nll_per_token"]}, step=n_steps)
        wb.summary.update({"eval/nll_heldout": ev["nll_per_token"], **{f"final/{k}": v for k, v in c.items()}})
        wb.finish()
    del opt, deliv
    torch.cuda.empty_cache()
    return rec


# ----------------------------------------------------------------------------- equivalence run
def equivalence_run(model_a, model_b, examples, device, n_steps, lr, clip) -> dict:
    """rne (model A) vs torch.optim.AdamW (model B) on the real model with identical grads and a shared
    parameter trajectory (B's params are resynced to A after each comparison; moments evolve independently)."""
    pa = [p for p in model_a.parameters() if p.requires_grad]
    pb = [p for p in model_b.parameters() if p.requires_grad]
    opt_a = CompensatedAdamW(pa, lr=lr, writer="rne")
    opt_b = torch.optim.AdamW(pb, lr=lr, fused=False)
    from torch.optim.optimizer import _default_to_fused_or_foreach
    _, foreach = _default_to_fused_or_foreach(pb, differentiable=False, use_fused=False)
    rows = []
    for step in range(1, n_steps + 1):
        ids = torch.tensor([examples[step - 1]["ids"]], device=device)
        with torch.no_grad():
            loss_b = float(model_b(input_ids=ids, labels=ids).loss)
        loss_a = model_a(input_ids=ids, labels=ids).loss
        opt_a.zero_grad(set_to_none=True)
        opt_b.zero_grad(set_to_none=True)
        loss_a.backward()
        with torch.no_grad():
            for a, b in zip(pa, pb):
                b.grad = a.grad.clone()
        torch.nn.utils.clip_grad_norm_(pa, clip)
        torch.nn.utils.clip_grad_norm_(pb, clip)
        opt_a.step()
        opt_b.step()
        with torch.no_grad():
            n = agree = m_eq = v_eq = 0
            for a, b in zip(pa, pb):
                n += a.numel()
                agree += int((a == b).sum())
                m_eq += int((opt_a.state[a]["exp_avg"] == opt_b.state[b]["exp_avg"]).sum())
                v_eq += int((opt_a.state[a]["exp_avg_sq"] == opt_b.state[b]["exp_avg_sq"]).sum())
                b.copy_(a)
        rows.append({"step": step, "loss_rne": float(loss_a.detach()), "loss_torch": loss_b,
                     "exact_write_agreement": agree / n, "exp_avg_equal": m_eq / n, "exp_avg_sq_equal": v_eq / n})
        log(f"equivalence step {step}: agreement {agree / n:.5f} m_eq {m_eq / n:.5f} v_eq {v_eq / n:.5f} "
            f"loss rne {float(loss_a):.5f} torch {loss_b:.5f}")
    out = {"n_steps": n_steps, "lr": lr, "torch_resolved_foreach": bool(foreach), "rows": rows,
           "min_exact_write_agreement": min(r["exact_write_agreement"] for r in rows),
           "min_exp_avg_equal": min(r["exp_avg_equal"] for r in rows),
           "min_exp_avg_sq_equal": min(r["exp_avg_sq_equal"] for r in rows),
           "max_abs_loss_diff": max(abs(r["loss_rne"] - r["loss_torch"]) for r in rows),
           "threshold": 0.997}
    out["passed"] = out["min_exact_write_agreement"] >= out["threshold"]
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="seed output directory")
    ap.add_argument("--seed", type=int, required=True, help="data-order seed")
    ap.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    ap.add_argument("--lrs", type=float, nargs="+", default=[2e-4, 1e-4, 5e-5])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--stats-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--clip", type=float, default=32.0)
    ap.add_argument("--equivalence-steps", type=int, default=20)
    ap.add_argument("--torch-seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    for c in args.conditions:
        assert c in CONDITIONS, c

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    torch.manual_seed(args.torch_seed)
    np.random.seed(args.torch_seed)

    import transformers
    import datasets
    log(f"torch {torch.__version__} cuda {torch.version.cuda} gpu {torch.cuda.get_device_name(0)} "
        f"transformers {transformers.__version__} datasets {datasets.__version__}")
    log(f"config: {vars(args)}")

    from transformers import AutoTokenizer
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    ds = load_dataset(DATA_ID, "main", split="train", revision=DATA_REV)
    ds_test = load_dataset(DATA_ID, "main", split="test", revision=DATA_REV)
    assert len(ds) == N_TRAIN_ROWS and len(ds_test) == N_TEST_ROWS, (len(ds), len(ds_test))

    order = np.random.default_rng(args.seed).permutation(N_TRAIN_ROWS)[: args.steps]
    examples = build_examples(tokenizer, ds, order, args.max_len)
    eval_rows = np.random.default_rng(EVAL_PERM_SEED).permutation(N_TEST_ROWS)[: args.eval_n]
    eval_examples = build_examples(tokenizer, ds_test, eval_rows, args.max_len)
    lens = [len(e["ids"]) for e in examples]
    assert len(set(order.tolist())) == len(order)
    manifest = {
        "model_id": MODEL_ID, "model_revision": MODEL_REV, "dataset": DATA_ID, "dataset_revision": DATA_REV,
        "seed": args.seed, "steps": args.steps, "batch_size": 1, "lrs": args.lrs, "conditions": args.conditions,
        "clip": args.clip, "max_len": args.max_len, "stats_every": args.stats_every, "torch_seed": args.torch_seed,
        "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "format": "question + '\\n' + answer + eos; every token scored; add_special_tokens=False",
        "train_row_indices": [int(x) for x in order], "n_tokens_per_step": lens,
        "n_truncated": int(sum(e["truncated"] for e in examples)),
        "eval": {"split": "test", "perm_seed": EVAL_PERM_SEED, "n": args.eval_n,
                 "row_indices": [int(x) for x in eval_rows],
                 "n_predicted_tokens": int(sum(len(e["ids"]) - 1 for e in eval_examples)),
                 "n_truncated": int(sum(e["truncated"] for e in eval_examples))},
        "first_train_example": tokenizer.decode(examples[0]["ids"]),
        "first_eval_example": tokenizer.decode(eval_examples[0]["ids"]),
    }
    log(f"seed {args.seed}: {len(examples)} train rows, tokens/step min {min(lens)} median {int(np.median(lens))} "
        f"max {max(lens)}, truncated {manifest['n_truncated']}; eval {len(eval_examples)} rows, "
        f"{manifest['eval']['n_predicted_tokens']} predicted tokens, truncated {manifest['eval']['n_truncated']}")
    log("first train example:\n" + manifest["first_train_example"][:500])
    with open(os.path.join(args.out, "seed_manifest.json"), "w") as f:
        json.dump(manifest, f)

    model = load_model(device)
    pristine = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

    untrained_path = os.path.join(args.out, "untrained_eval.json")
    ev0 = evaluate(model, eval_examples, device)
    log(f"untrained model held-out NLL per token: {ev0['nll_per_token']:.5f}")
    with open(untrained_path, "w") as f:
        json.dump(ev0, f, indent=1)

    if args.equivalence_steps > 0:
        log(f"phase: equivalence run, {args.equivalence_steps} steps at lr {args.lrs[0]:g}")
        model_b = load_model(device)
        model_b.load_state_dict(pristine)
        eq = equivalence_run(model, model_b, examples, device, args.equivalence_steps, args.lrs[0], args.clip)
        del model_b
        torch.cuda.empty_cache()
        with open(os.path.join(args.out, "equivalence.json"), "w") as f:
            json.dump(eq, f, indent=1)
        log(f"equivalence: min exact write agreement {eq['min_exact_write_agreement']:.5f} "
            f"(m eq {eq['min_exp_avg_equal']:.5f}, v eq {eq['min_exp_avg_sq_equal']:.5f}), "
            f"max |loss diff| {eq['max_abs_loss_diff']:.5f}, foreach={eq['torch_resolved_foreach']}, "
            f"passed={eq['passed']}")
        if not eq["passed"]:
            log("equivalence check FAILED; stopping before any training run")
            sys.exit(3)

    wandb_mod = None
    if not args.no_wandb and os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb_mod = wandb
    elif not args.no_wandb:
        log("WANDB_API_KEY not set; logging to stdout and files only")

    plan = [(lr, c) for lr in args.lrs for c in args.conditions]
    total = len(plan) * args.steps
    log(f"phase: {len(plan)} runs x {args.steps} steps")
    summary = []
    for k, (lr, cond) in enumerate(plan):
        path = os.path.join(args.out, "runs", f"{cond}_lr{lr:g}.json")
        if os.path.exists(path):
            log(f"skip existing {path}")
            continue
        model.load_state_dict(pristine)
        rec = train_run(model, cond, lr, args.seed, examples, eval_examples, device, args, wandb_mod,
                        job_step_base=k * args.steps, job_total_steps=total, run_index=k)
        with open(path + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(path + ".tmp", path)
        summary.append({"condition": cond, "lr": lr, "eval_nll": rec["eval"]["nll_per_token"],
                        **rec["final_delivery"]["whole"], "s_per_step": rec["s_per_step"]})
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump({"seed": args.seed, "untrained_nll": ev0["nll_per_token"], "runs": summary}, f, indent=1)
    log(f"seed {args.seed} complete: {len(summary)} runs written under {args.out}")


if __name__ == "__main__":
    main()
