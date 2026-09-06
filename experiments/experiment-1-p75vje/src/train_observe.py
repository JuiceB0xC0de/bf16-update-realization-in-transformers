"""Observer-wrapped AdamW fine-tuning of SmolLM2-135M (bf16) on GSM8K.

Pipeline (plan steps 2-4):
  1. clip calibration: 20 unclipped steps from the pristine checkpoint on order 0; the clip
     bound is 2 x p95(pre-clip grad norm) rounded up to a power of two.
  2. for each data order (seed 0, 1): 300 steps of torch.optim.AdamW(fused=False) from the
     pristine checkpoint, parameters and moments in bfloat16. Every step the observer
     snapshots pre-step weights in fp32, runs the step, and records pooled intended-vs-
     realized statistics for the whole model and per family, plus the nine-factor shadow curve.
     At the detail steps it stores pre-step weight, both post-step moments and the stored
     change for a fixed slice of about two million coordinates, in the Mamba bundle layout.

Outputs (under --out):
  manifest.json, calibration.json, order{seed}/steps.jsonl, order{seed}/detail/step_XXXXXX.npz,
  order{seed}/examples.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from realization import Acc, intent  # noqa: E402

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REV = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DATA_ID = "openai/gsm8k"
DATA_REV = "740312add88f781978c0658806c59bc2815b9866"
N_TRAIN_ROWS = 7473
EXPECTED_PARAMS = 134_515_008

FAMILIES = ("attention", "mlp", "embedding", "norm")
DETAIL_LAYERS = (0, 7, 15, 22, 29)
DETAIL_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


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
def build_examples(tokenizer, ds, order: np.ndarray, max_len: int):
    eos = tokenizer.eos_token_id
    assert eos is not None
    out = []
    for idx in order.tolist():
        row = ds[int(idx)]
        text = row["question"] + "\n" + row["answer"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [eos]
        full = len(ids)
        ids = ids[:max_len]
        out.append({"row": int(idx), "ids": ids, "full_len": full, "truncated": full > max_len})
    return out


def token_sets(examples, vocab: int, seed: int, n_rows: int):
    counts = np.zeros(vocab, dtype=np.int64)
    for ex in examples:
        np.add.at(counts, np.asarray(ex["ids"]), 1)
    seen = np.flatnonzero(counts)
    unseen = np.flatnonzero(counts == 0)
    order = np.lexsort((seen, -counts[seen]))  # most frequent first, ties by id
    seen_rows = np.sort(seen[order][:n_rows])
    rng = np.random.default_rng(seed + 1000)
    unseen_rows = np.sort(rng.choice(unseen, size=min(n_rows, len(unseen)), replace=False))
    return seen_rows, unseen_rows, int(len(seen)), int(len(unseen))


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


def resolved_foreach(params) -> bool:
    from torch.optim.optimizer import _default_to_fused_or_foreach
    fused, foreach = _default_to_fused_or_foreach(list(params), differentiable=False, use_fused=False)
    return bool(foreach)


# ----------------------------------------------------------------------------- observer
class Observer:
    def __init__(self, model, optimizer):
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.opt = optimizer
        self.pre = {n: torch.empty_like(p, dtype=torch.float32) for n, p in self.named}
        self.fam = {n: family_of(n) for n, _ in self.named}
        g = optimizer.param_groups[0]
        self.lr, self.wd, self.eps = float(g["lr"]), float(g["weight_decay"]), float(g["eps"])
        self.b1, self.b2 = (float(b) for b in g["betas"])

    def snapshot(self):
        with torch.no_grad():
            for n, p in self.named:
                self.pre[n].copy_(p)

    def bias_corrections(self, t: int):
        return 1.0 - self.b1 ** t, 1.0 - self.b2 ** t

    @torch.no_grad()
    def measure(self, step: int, detail_spec=None):
        st = self.opt.state[self.named[0][1]]
        t = int(float(st["step"]))
        assert t == step, (t, step)
        bc1, bc2 = self.bias_corrections(t)
        accs = {f: Acc() for f in FAMILIES}
        detail = {}
        for n, p in self.named:
            s = self.opt.state[p]
            m, v = s["exp_avg"], s["exp_avg_sq"]
            assert m.dtype == torch.bfloat16 and v.dtype == torch.bfloat16, (n, m.dtype, v.dtype)
            pre = self.pre[n]
            i = intent(m, v, pre, self.lr, self.wd, self.eps, bc1, bc2)
            r = p.float() - pre
            accs[self.fam[n]].add(i, r, pre, shadow=True, hist=True)
            if detail_spec is not None:
                for key, sel in detail_spec.get(n, []):
                    detail[f"{key}/param_pre"] = pre[sel].reshape(-1).cpu().numpy()
                    detail[f"{key}/exp_avg_post"] = m[sel].float().reshape(-1).cpu().numpy()
                    detail[f"{key}/exp_avg_sq_post"] = v[sel].float().reshape(-1).cpu().numpy()
                    detail[f"{key}/delta"] = r[sel].reshape(-1).cpu().numpy()
        whole = Acc()
        for a in accs.values():
            whole.merge(a)
        rec = {"bc1": bc1, "bc2": bc2, "whole_model_bf16": whole.summary(),
               "families": {f: a.summary() for f, a in accs.items()}}
        if detail_spec is not None:
            detail["param/group_lr"] = np.array([self.lr], dtype=np.float64)
            detail["param/group_weight_decay"] = np.array([self.wd], dtype=np.float64)
            detail["param/group_eps"] = np.array([self.eps], dtype=np.float64)
            detail["param/bias_correction1"] = np.array([bc1], dtype=np.float64)
            detail["param/bias_correction2"] = np.array([bc2], dtype=np.float64)
            detail["param/step"] = np.array([step], dtype=np.int64)
        return rec, detail


def build_detail_spec(model, seen_rows, unseen_rows, proj_rows: int):
    """name -> list of (npz key prefix, index selector)."""
    spec = {}
    for L in DETAIL_LAYERS:
        for proj in DETAIL_PROJS:
            sub = "self_attn" if proj in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            name = f"model.layers.{L}.{sub}.{proj}.weight"
            spec.setdefault(name, []).append((f"layer_{L:02d}/{proj}/rows0-{proj_rows}", slice(0, proj_rows)))
    emb = "model.embed_tokens.weight"
    dev = next(model.parameters()).device
    spec[emb] = [("embed/seen_rows", torch.as_tensor(seen_rows, device=dev)),
                 ("embed/unseen_rows", torch.as_tensor(unseen_rows, device=dev))]
    for n, _ in model.named_parameters():
        if "norm" in n:
            spec[n] = [(f"norms/{n}", slice(None))]
    n_coords = 0
    for n, p in model.named_parameters():
        for key, sel in spec.get(n, []):
            n_coords += p[sel].numel()
    return spec, n_coords


# ----------------------------------------------------------------------------- training
def run_steps(model, optimizer, examples, device, n_steps, clip, observer=None, detail_steps=(),
              detail_dir=None, detail_spec=None, log_fn=None, wb=None, phase="train", jsonl=None):
    from silico.slurm_telemetry import report_progress
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        ex = examples[step - 1]
        ids = torch.tensor([ex["ids"]], device=device)
        out = model(input_ids=ids, labels=ids)
        loss = out.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        pre_norm = torch.nn.utils.clip_grad_norm_(params, clip)
        pre_norm = float(pre_norm)
        coef = min(1.0, clip / (pre_norm + 1e-6)) if math.isfinite(clip) else 1.0
        fired = math.isfinite(clip) and pre_norm > clip
        if observer is not None:
            observer.snapshot()
        optimizer.step()
        rec = {"step": step, "row": ex["row"], "n_tokens": len(ex["ids"]), "loss": float(loss.detach()),
               "grad_norm_preclip": pre_norm, "clip_coef": coef, "clip_fired": bool(fired),
               "lr": float(optimizer.param_groups[0]["lr"])}
        if observer is not None:
            want_detail = step in detail_steps
            m, detail = observer.measure(step, detail_spec if want_detail else None)
            rec.update(m)
            if want_detail:
                os.makedirs(detail_dir, exist_ok=True)
                path = os.path.join(detail_dir, f"step_{step:06d}.npz")
                np.savez(path + ".tmp.npz", **detail)
                os.replace(path + ".tmp.npz", path)
                rec["detail_file"] = os.path.basename(path)
        rows.append(rec)
        if jsonl is not None:
            jsonl.write(json.dumps(rec) + "\n")
            jsonl.flush()
        if wb is not None and observer is not None:
            w = rec["whole_model_bf16"]
            wb.log({"train/loss": rec["loss"], "train/grad_norm_preclip": pre_norm,
                    "train/clip_fired": int(fired), "train/n_tokens": rec["n_tokens"],
                    "overshoot/bf16_whole_model": w["overshoot"],
                    "overshoot/predicted_whole_model": w["predicted_overshoot"],
                    "realized/length_ratio": w["length_ratio"], "realized/cosine": w["cosine"],
                    "realized/unmoved_fraction": w["unmoved_fraction"],
                    "realized/exact_write_agreement": w["exact_write_agreement"],
                    **{f"overshoot/{f}": rec["families"][f]["overshoot"] for f in FAMILIES}},
                   step=step)
        report_progress(step=step, total_steps=n_steps, phase=phase)
        if step <= 10 or step % 25 == 0 or step == n_steps:
            extra = ""
            if observer is not None:
                w = rec["whole_model_bf16"]
                extra = (f" rho={w['overshoot']:.4f} pred={w['predicted_overshoot']:.4f} "
                         f"len={w['length_ratio']:.4f} cos={w['cosine']:.4f} "
                         f"unmoved={w['unmoved_fraction']:.4f} exact={w['exact_write_agreement']:.5f}")
            log(f"{phase} step {step}/{n_steps} loss={rec['loss']:.4f} gnorm={pre_norm:.3f} "
                f"clip={'Y' if fired else 'n'} tok={rec['n_tokens']}{extra} "
                f"({(time.time() - t0) / step:.2f}s/step)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--orders", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--calib-steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--detail-steps", default="1,10,50,100,200,300")
    ap.add_argument("--proj-rows", type=int, default=32)
    ap.add_argument("--embed-rows", type=int, default=1024, help="seen and unseen rows each")
    ap.add_argument("--clip", type=float, default=None, help="override the calibrated clip bound")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    device = torch.device("cuda:0")
    os.makedirs(args.out, exist_ok=True)
    detail_steps = tuple(int(s) for s in args.detail_steps.split(",") if s)

    log(f"torch {torch.__version__}  cuda {torch.version.cuda}  gpu {torch.cuda.get_device_name(0)}")
    import transformers
    import datasets
    log(f"transformers {transformers.__version__}  datasets {datasets.__version__}")
    log(f"config: {vars(args)}")

    from transformers import AutoTokenizer
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    ds = load_dataset(DATA_ID, "main", split="train", revision=DATA_REV)
    assert len(ds) == N_TRAIN_ROWS, len(ds)
    log(f"dataset rows {len(ds)}; eos id {tokenizer.eos_token_id} ({tokenizer.eos_token!r}); "
        f"vocab {len(tokenizer)}")

    manifest = {
        "model_id": MODEL_ID, "model_revision": MODEL_REV, "dataset": DATA_ID,
        "dataset_revision": DATA_REV, "dataset_split": "train", "n_dataset_rows": len(ds),
        "steps": args.steps, "orders": args.orders, "lr": args.lr, "max_len": args.max_len,
        "batch_size": 1, "detail_steps": list(detail_steps), "proj_rows": args.proj_rows,
        "embed_rows_each": args.embed_rows, "detail_layers": list(DETAIL_LAYERS),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "seed": args.seed,
        "format": "question + '\\n' + answer + eos; every token scored; add_special_tokens=False",
    }

    def make_optimizer(model):
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=False)
        return opt

    # ---------------------------------------------------------------- clip calibration
    calib_path = os.path.join(args.out, "calibration.json")
    if args.clip is not None:
        clip = args.clip
        calib = {"override": clip}
        log(f"clip bound override {clip}")
    else:
        log("phase: clip calibration (unclipped, pristine checkpoint, order 0)")
        model = load_model(device)
        opt = make_optimizer(model)
        order0 = np.random.default_rng(0).permutation(N_TRAIN_ROWS)[: args.calib_steps]
        ex0 = build_examples(tokenizer, ds, order0, args.max_len)
        rows = run_steps(model, opt, ex0, device, args.calib_steps, float("inf"), phase="calibration")
        norms = np.array([r["grad_norm_preclip"] for r in rows])
        p95 = float(np.percentile(norms, 95))
        clip = float(2.0 ** math.ceil(math.log2(2.0 * p95)))
        calib = {"n_steps": args.calib_steps, "grad_norms": norms.tolist(), "p95": p95,
                 "rule": "2 * p95 rounded up to a power of two", "clip_bound": clip,
                 "min": float(norms.min()), "max": float(norms.max()),
                 "losses": [r["loss"] for r in rows], "rows": [r["row"] for r in rows]}
        log(f"calibration norms: min {norms.min():.3f} median {np.median(norms):.3f} p95 {p95:.3f} "
            f"max {norms.max():.3f} -> clip bound {clip}")
        del opt, model
        torch.cuda.empty_cache()
    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=1)
    manifest["clip_bound"] = clip

    # ---------------------------------------------------------------- main runs
    from silico.slurm_telemetry import register_wandb_url, register_key_metrics, bridge_wandb
    wandb = None
    if not args.no_wandb and os.environ.get("WANDB_API_KEY"):
        import wandb as _wandb
        wandb = _wandb
    elif not args.no_wandb:
        log("WANDB_API_KEY not set; logging to stdout and files only")

    for seed in args.orders:
        log(f"phase: main run, order seed {seed}")
        odir = os.path.join(args.out, f"order{seed}")
        os.makedirs(odir, exist_ok=True)
        order = np.random.default_rng(seed).permutation(N_TRAIN_ROWS)[: args.steps]
        examples = build_examples(tokenizer, ds, order, args.max_len)
        model = load_model(device)
        opt = make_optimizer(model)
        foreach = resolved_foreach(model.parameters())
        seen_rows, unseen_rows, n_seen, n_unseen = token_sets(examples, model.config.vocab_size, seed,
                                                             args.embed_rows)
        spec, n_detail = build_detail_spec(model, seen_rows, unseen_rows, args.proj_rows)
        lens = [len(e["ids"]) for e in examples]
        run_info = {
            "seed": seed, "row_indices": [int(x) for x in order], "n_tokens_per_step": lens,
            "n_truncated": int(sum(e["truncated"] for e in examples)),
            "max_full_len": int(max(e["full_len"] for e in examples)),
            "optimizer": {"class": "torch.optim.AdamW", "defaults": {k: (list(v) if isinstance(v, tuple) else v)
                                                                      for k, v in opt.defaults.items()
                                                                      if k in ("lr", "betas", "eps", "weight_decay",
                                                                               "amsgrad", "fused", "foreach",
                                                                               "maximize", "capturable")},
                          "resolved_foreach": foreach, "param_dtype": "bfloat16", "moment_dtype": "bfloat16"},
            "clip_bound": clip, "n_seen_tokens": n_seen, "n_unseen_tokens": n_unseen,
            "embed_seen_rows": seen_rows.tolist(), "embed_unseen_rows": unseen_rows.tolist(),
            "n_detail_coordinates": int(n_detail),
            "first_example_text": tokenizer.decode(examples[0]["ids"]),
            "first_example_ids_head": examples[0]["ids"][:24],
        }
        log(f"order {seed}: {args.steps} rows, tokens/step min {min(lens)} median {int(np.median(lens))} "
            f"max {max(lens)}, truncated {run_info['n_truncated']}, detail coords {n_detail}, "
            f"foreach={foreach}, seen tokens {n_seen}")
        log("first example:\n" + run_info["first_example_text"][:600])
        with open(os.path.join(odir, "run_info.json"), "w") as f:
            json.dump(run_info, f)

        wb = None
        if wandb is not None:
            wb = wandb.init(project=f"default-exp-{os.environ.get('SILICO_EXPERIMENT_ID', 'local')}",
                            name=f"smollm2-135m-adamw-bf16-order{seed}-lr{args.lr:g}",
                            config={**manifest, "order_seed": seed, "max_steps": args.steps,
                                    "resolved_foreach": foreach})
            register_wandb_url()
            register_key_metrics(["train/loss", "overshoot/bf16_whole_model", "realized/length_ratio",
                                  "realized/unmoved_fraction"])
            bridge_wandb()
        observer = Observer(model, opt)
        with open(os.path.join(odir, "steps.jsonl"), "w") as jsonl:
            rows = run_steps(model, opt, examples, device, args.steps, clip, observer=observer,
                             detail_steps=detail_steps, detail_dir=os.path.join(odir, "detail"),
                             detail_spec=spec, wb=wb, phase=f"train-order{seed}", jsonl=jsonl)
        rho = np.array([r["whole_model_bf16"]["overshoot"] for r in rows])
        summ = {"seed": seed, "median_overshoot": float(np.median(rho)), "min_overshoot": float(rho.min()),
                "max_overshoot": float(rho.max()),
                "median_length_ratio": float(np.median([r["whole_model_bf16"]["length_ratio"] for r in rows])),
                "median_unmoved": float(np.median([r["whole_model_bf16"]["unmoved_fraction"] for r in rows])),
                "clip_fired_steps": int(sum(r["clip_fired"] for r in rows)),
                "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        log(f"order {seed} done: {summ}")
        with open(os.path.join(odir, "summary.json"), "w") as f:
            json.dump(summ, f, indent=1)
        if wb is not None:
            wb.summary.update(summ)
            wb.finish()
        del observer, opt, model
        torch.cuda.empty_cache()

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log(f"all runs complete; outputs under {args.out}")


if __name__ == "__main__":
    main()
