"""One shard of the selector sweep: SmolLM2-135M (bf16) fine-tuned on GSM8K, batch 1.

Adapted from exp_01m1tvjx4ee3xshh9djt6db5bx (train_sweep.py there; itself from exp_01m1tgnb6xf98sqwfhm4effknt):
same model, data streams, probe, clip bound, delivery statistics and per-run record schema. New here: a shard is
a list of (seed, lr, condition) tasks (several seeds per job), the threshold writers `thr_drop_{15,25,50}` take
their calibrated theta from --theta-table (or a per-task override), and the keno_drop_15 level. The harness
check (rerun of plain rounding `rne` compared with the stored previous-run record, abort beyond --harness-tol)
is unchanged.

For one data-order seed, every (condition, lr) pair trains on the identical 3,000-row stream from the
pristine checkpoint, so all contrasts are paired within seed.

Per run this records
  - loss and pre-clip grad norm at every step,
  - every --stats-every steps: pooled per-step intended-vs-realized statistics (whole model and per
    family) and the cumulative delivery so far (R = p - p_init against I = sum of intents),
  - final cumulative delivery (whole model and per family) and the held-out NLL per token on the
    fixed 256-question GSM8K test probe (all tokens scored).
The untrained model's held-out NLL is evaluated once per job. A 20-step equivalence run of the rne
writer against torch.optim.AdamW on the real model (identical grads, shared trajectory) runs first
and aborts the job if the exact write agreement falls below 0.997.

Outputs (under --out/seed<S>/):
  seed_manifest.json, untrained_eval.json, harness_check.json, summary.json, runs/<condition>_lr<lr>.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from compensated_adamw import CONDITIONS, CompensatedAdamW, make_optimizer  # noqa: E402
from write_distributor import THR_WRITERS, parse_thr_writer  # noqa: E402
from realization import Acc, cumulative_summary  # noqa: E402

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REV = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DATA_ID = "openai/gsm8k"
DATA_REV = "740312add88f781978c0658806c59bc2815b9866"
N_TRAIN_ROWS = 7473
N_TEST_ROWS = 1319
EXPECTED_PARAMS = 134_515_008
FAMILIES = ("attention", "mlp", "embedding", "norm")
DEFAULT_CONDITIONS = ("keno_drop_15", "keno_drop_25", "keno_drop_50", "thr_drop_15", "thr_drop_25", "thr_drop_50")
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


def lr_key(lr: float) -> str:
    return f"{lr:g}"


def train_run(model, condition, lr, seed, examples, eval_examples, device, args, wandb_mod, job_step_base,
              job_total_steps, run_index, theta=None, tag=None):
    from silico.slurm_telemetry import report_progress, register_wandb_url, register_key_metrics
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    torch.manual_seed(args.torch_seed)
    sr_seed = 100_000 * seed + 1_000 * run_index + 7
    # mask draws are seeded per (data seed, writer) so the same writer sees the same masks at every lr
    distributor_seed = 100_000 * seed + 1_000 * CONDITIONS.index(condition) + 7
    extra = {}
    if condition.startswith("keno_"):
        extra["sticky_interval"] = args.sticky_interval
    if condition in THR_WRITERS:
        assert theta is not None, f"{condition} needs theta"
        extra["theta"] = float(theta)
    opt = make_optimizer(condition, params, lr, sr_seed=sr_seed, distributor_seed=distributor_seed, **extra)
    dist = getattr(opt, "distributor", None)
    g = opt.param_groups[0]
    hp = {"lr": float(g["lr"]), "betas": [float(b) for b in g["betas"]], "eps": float(g["eps"]),
          "weight_decay": float(g["weight_decay"])}
    deliv = Delivery(named)
    run_name = f"{condition}-lr{lr:g}-seed{seed}" + (f"-{tag}" if tag else "")
    wb = None
    if wandb_mod is not None:
        wb = wandb_mod.init(project=f"default-exp-{os.environ.get('SILICO_EXPERIMENT_ID', 'local')}", name=run_name,
                            config={"condition": condition, "lr": lr, "seed": seed, "max_steps": args.steps,
                                    "model": MODEL_ID, "model_revision": MODEL_REV, "dataset": DATA_ID,
                                    "dataset_revision": DATA_REV, "clip": args.clip,
                                    "sticky_interval": args.sticky_interval if condition.startswith("keno_") else None,
                                    "theta": theta, "tag": tag, **hp})
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
            if dist is not None and wb is not None:
                wb.log({"selector/fraction": dist.block_fraction}, step=step)
            mask_txt = (f" sel[block={dist.block} f={dist.block_fraction:.3f}]" if dist is not None else "")
            log(f"{run_name} step {step}/{n_steps} loss={lv:.4f} gnorm={pre_norm:.2f}{mask_txt} "
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
    dist_summary = dist.summary() if dist is not None else None
    rec = {
        "condition": condition, "lr": lr, "seed": seed, "steps": n_steps, "hyperparams": hp, "theta": theta, "tag": tag,
        "distributor": dist_summary, "distributor_seed": distributor_seed if dist is not None else None,
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
    if dist_summary is not None and "n_blocks" in dist_summary:
        log(f"{run_name} mask: {dist_summary['n_blocks']} blocks, target mean {dist_summary['fraction_target_mean']:.4f}, "
            f"realised mean {dist_summary['fraction_realized_mean']:.4f} "
            f"[{dist_summary['fraction_realized_min']:.4f}, {dist_summary['fraction_realized_max']:.4f}], "
            f"max |realised - target| {dist_summary['max_abs_target_gap']:.5f}")
    elif dist_summary is not None:
        log(f"{run_name} threshold theta={dist_summary['theta']}: selected fraction mean "
            f"{dist_summary['fraction_realized_mean']:.4f} [{dist_summary['fraction_realized_min']:.4f}, "
            f"{dist_summary['fraction_realized_max']:.4f}], first100 {dist_summary['fraction_realized_first100_mean']:.4f} "
            f"last100 {dist_summary['fraction_realized_last100_mean']:.4f}")
    if wb is not None:
        wb.log({"eval/nll_heldout": ev["nll_per_token"]}, step=n_steps)
        wb.summary.update({"eval/nll_heldout": ev["nll_per_token"], **{f"final/{k}": v for k, v in c.items()}})
        wb.finish()
    del opt, deliv
    torch.cuda.empty_cache()
    return rec


# ----------------------------------------------------------------------------- harness check
def find_reference(ref_dirs, condition, lr, seed):
    """Reference run JSON for (condition, lr, seed): <dir>/{cond}_lr{lr}.json, <dir>/seed{seed}/runs/{cond}_lr{lr}.json,
    or <dir>/{cond}_lr{lr}_seed{seed}*.json (a loss-prefix file)."""
    name = f"{condition}_lr{lr:g}"
    for d in ref_dirs:
        for cand in (os.path.join(d, f"seed{seed}", "runs", name + ".json"), os.path.join(d, name + ".json")):
            if os.path.exists(cand):
                return cand
        hits = sorted(glob.glob(os.path.join(d, f"{name}_seed{seed}*.json")))
        if hits:
            return hits[0]
    return None


def harness_compare(rec: dict, ref_path: str, tol: float) -> dict:
    with open(ref_path) as f:
        ref = json.load(f)
    assert ref.get("condition") == rec["condition"] and abs(float(ref.get("lr")) - rec["lr"]) < 1e-12
    assert int(ref.get("seed")) == rec["seed"], (ref.get("seed"), rec["seed"])
    n = min(len(ref["losses"]), len(rec["losses"]))
    a, b = np.array(ref["losses"][:n]), np.array(rec["losses"][:n])
    out = {"reference": ref_path, "n_losses_compared": n, "losses_max_abs_diff": float(np.abs(a - b).max()),
           "losses_exact_equal_fraction": float(np.mean(a == b)), "tol": tol}
    if "eval" in ref and n == rec["steps"] == ref.get("steps"):
        out["eval_nll_reference"] = ref["eval"]["nll_per_token"]
        out["eval_nll_observed"] = rec["eval"]["nll_per_token"]
        out["eval_nll_abs_diff"] = abs(out["eval_nll_observed"] - out["eval_nll_reference"])
        out["passed"] = out["eval_nll_abs_diff"] <= tol
    else:
        out["passed"] = out["losses_max_abs_diff"] <= tol  # prefix-only reference (smoke)
    return out


# ----------------------------------------------------------------------------- setup shared with calibrate.py
def load_data(args):
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    ds = load_dataset(DATA_ID, "main", split="train", revision=DATA_REV)
    ds_test = load_dataset(DATA_ID, "main", split="test", revision=DATA_REV)
    assert len(ds) == N_TRAIN_ROWS and len(ds_test) == N_TEST_ROWS, (len(ds), len(ds_test))
    eval_rows = np.random.default_rng(EVAL_PERM_SEED).permutation(N_TEST_ROWS)[: args.eval_n]
    eval_examples = build_examples(tokenizer, ds_test, eval_rows, args.max_len)
    return tokenizer, ds, eval_rows, eval_examples


def seed_examples(tokenizer, ds, seed, args):
    order = np.random.default_rng(seed).permutation(N_TRAIN_ROWS)[: args.steps]
    assert len(set(order.tolist())) == len(order)
    return order, build_examples(tokenizer, ds, order, args.max_len)


def write_seed_manifest(path, tokenizer, seed, order, examples, eval_rows, eval_examples, args, conditions, lrs):
    import transformers
    lens = [len(e["ids"]) for e in examples]
    manifest = {
        "model_id": MODEL_ID, "model_revision": MODEL_REV, "dataset": DATA_ID, "dataset_revision": DATA_REV,
        "seed": seed, "steps": args.steps, "batch_size": 1, "lrs": lrs, "conditions": conditions,
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
    log(f"seed {seed}: {len(examples)} train rows, tokens/step min {min(lens)} median {int(np.median(lens))} "
        f"max {max(lens)}, truncated {manifest['n_truncated']}; eval {len(eval_examples)} rows, "
        f"{manifest['eval']['n_predicted_tokens']} predicted tokens, truncated {manifest['eval']['n_truncated']}")
    with open(path, "w") as f:
        json.dump(manifest, f)
    return manifest


def add_common_args(ap):
    ap.add_argument("--sticky-interval", type=int, default=50)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--stats-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--clip", type=float, default=32.0)
    ap.add_argument("--torch-seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")


def theta_for(theta_table, condition, lr):
    if condition not in THR_WRITERS:
        return None
    assert theta_table is not None, "threshold writers need --theta-table"
    return float(theta_table[lr_key(lr)][parse_thr_writer(condition)])


def wandb_or_none(args):
    if not args.no_wandb and os.environ.get("WANDB_API_KEY"):
        import wandb
        return wandb
    if not args.no_wandb:
        log("WANDB_API_KEY not set; logging to stdout and files only")
    return None


def banner(args):
    import transformers
    import datasets
    log(f"torch {torch.__version__} cuda {torch.version.cuda} gpu {torch.cuda.get_device_name(0)} "
        f"transformers {transformers.__version__} datasets {datasets.__version__}")
    log(f"config: {vars(args)}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="shard output root; runs land under <out>/seed<S>/runs/")
    ap.add_argument("--tasks", required=True,
                    help="JSON list of {seed, lr, condition, harness?: bool, theta?: float} or a path to one")
    ap.add_argument("--theta-table", default=None, help="JSON {lr_key: {level: theta}} for thr_drop writers")
    ap.add_argument("--reference-dirs", nargs="*", default=[], help="directories holding reference run JSONs")
    ap.add_argument("--harness-tol", type=float, default=0.001)
    add_common_args(ap)
    args = ap.parse_args()
    tasks = json.load(open(args.tasks)) if os.path.exists(args.tasks) else json.loads(args.tasks)
    for t in tasks:
        assert t["condition"] in CONDITIONS, t
        t["lr"] = float(t["lr"])
        t["seed"] = int(t["seed"])
        t.setdefault("harness", False)
    # harness tasks first (they abort the shard on mismatch), then by seed
    tasks.sort(key=lambda t: (not t["harness"], t["seed"]))
    theta_table = json.load(open(args.theta_table)) if args.theta_table else None
    for t in tasks:
        if t["condition"] in THR_WRITERS:
            t["theta"] = float(t["theta"]) if t.get("theta") is not None else theta_for(theta_table, t["condition"], t["lr"])

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.manual_seed(args.torch_seed)
    np.random.seed(args.torch_seed)
    banner(args)
    log(f"{len(tasks)} tasks: " + ", ".join(f"s{t['seed']}/{t['condition']}@{lr_key(t['lr'])}"
                                             + (f"(th={t['theta']:.4g})" if t.get('theta') is not None else "")
                                             + ("[harness]" if t["harness"] else "") for t in tasks))
    tokenizer, ds, eval_rows, eval_examples = load_data(args)
    model = load_model(device)
    pristine = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    ev0 = evaluate(model, eval_examples, device)
    log(f"untrained model held-out NLL per token: {ev0['nll_per_token']:.5f}")
    wandb_mod = wandb_or_none(args)

    total = len(tasks) * args.steps
    seeds_done = {}
    summaries = {}
    for k, t in enumerate(tasks):
        seed, lr, cond = t["seed"], t["lr"], t["condition"]
        out = os.path.join(args.out, f"seed{seed}")
        os.makedirs(os.path.join(out, "runs"), exist_ok=True)
        if seed not in seeds_done:
            order, examples = seed_examples(tokenizer, ds, seed, args)
            conds = sorted({x["condition"] for x in tasks if x["seed"] == seed})
            lrs = sorted({x["lr"] for x in tasks if x["seed"] == seed}, reverse=True)
            write_seed_manifest(os.path.join(out, "seed_manifest.json"), tokenizer, seed, order, examples, eval_rows,
                                eval_examples, args, conds, lrs)
            with open(os.path.join(out, "untrained_eval.json"), "w") as f:
                json.dump(ev0, f, indent=1)
            seeds_done[seed] = examples
            summaries.setdefault(seed, [])
        examples = seeds_done[seed]
        path = os.path.join(out, "runs", f"{cond}_lr{lr:g}.json")
        if os.path.exists(path):
            log(f"skip existing {path}")
            continue
        log(f"phase: task {k + 1}/{len(tasks)} seed {seed} {cond} lr {lr:g}" + ("" if t.get("theta") is None else f" theta {t['theta']}"))
        model.load_state_dict(pristine)
        rec = train_run(model, cond, lr, seed, examples, eval_examples, device, args, wandb_mod,
                        job_step_base=k * args.steps, job_total_steps=total, run_index=k, theta=t.get("theta"))
        if t["harness"]:
            ref = find_reference(args.reference_dirs, cond, lr, seed)
            if ref is None:
                log(f"harness check: no reference found for {cond} lr {lr:g} seed {seed} in {args.reference_dirs}")
                cmp = {"reference": None, "passed": False}
            else:
                cmp = harness_compare(rec, ref, args.harness_tol)
                log(f"harness check {cond} lr {lr:g} seed {seed}: " + json.dumps(cmp))
            rec["harness_check"] = cmp
            hpath = os.path.join(out, "harness_check.json")
            prev = json.load(open(hpath)) if os.path.exists(hpath) else {"seed": seed, "tol": args.harness_tol, "checks": []}
            prev["checks"].append({"condition": cond, "lr": lr, **cmp})
            prev["all_passed"] = all(h["passed"] for h in prev["checks"])
            with open(hpath, "w") as f:
                json.dump(prev, f, indent=1)
        with open(path + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(path + ".tmp", path)
        if t["harness"] and not rec["harness_check"]["passed"]:
            log("HARNESS CHECK FAILED; stopping before the sweep runs")
            sys.exit(4)
        summaries[seed].append({"condition": cond, "lr": lr, "theta": t.get("theta"), "eval_nll": rec["eval"]["nll_per_token"],
                                **rec["final_delivery"]["whole"], "s_per_step": rec["s_per_step"]})
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump({"seed": seed, "untrained_nll": ev0["nll_per_token"], "runs": summaries[seed]}, f, indent=1)
    log(f"shard complete: {sum(len(v) for v in summaries.values())} runs written under {args.out}")


if __name__ == "__main__":
    main()
