"""Build a toy run directory (same file layout as train_observe.py) on CPU to exercise analyze.py."""
import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import train_observe as T
from test_observer_cpu import Toy  # noqa
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
json.dump({"clip_bound": 4.0, "steps": 12}, open(os.path.join(out, "manifest.json"), "w"))
json.dump({"p95": 1.0, "clip_bound": 4.0, "grad_norms": [1.0]}, open(os.path.join(out, "calibration.json"), "w"))
for seed in (0, 1):
    torch.manual_seed(seed)
    m = Toy().to(torch.bfloat16)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, fused=False)
    obs = T.Observer(m, opt)
    spec = {"model.layers.0.self_attn.q_proj.weight": [("layer_00/q_proj/rows0-8", slice(0, 8))],
            "model.embed_tokens.weight": [("embed/seen_rows", torch.arange(0, 8)), ("embed/unseen_rows", torch.arange(8, 16))],
            "model.norm.weight": [("norms/model.norm.weight", slice(None))]}
    odir = os.path.join(out, f"order{seed}"); os.makedirs(os.path.join(odir, "detail"), exist_ok=True)
    json.dump({"n_truncated": 0, "max_full_len": 16, "optimizer": {"resolved_foreach": False, "defaults": {}}, "n_detail_coordinates": 1}, open(os.path.join(odir, "run_info.json"), "w"))
    with open(os.path.join(odir, "steps.jsonl"), "w") as f:
        for step in range(1, 13):
            ids = torch.randint(0, 64, (1, 16))
            loss = torch.nn.functional.cross_entropy(m(ids)[0].float(), ids[0])
            opt.zero_grad(); loss.backward(); obs.snapshot(); opt.step()
            want = step in (1, 10)
            rec, detail = obs.measure(step, spec if want else None)
            rec.update({"step": step, "row": 0, "n_tokens": 16, "loss": float(loss), "grad_norm_preclip": 1.0, "clip_coef": 1.0, "clip_fired": False, "lr": 2e-4})
            if want:
                np.savez(os.path.join(odir, "detail", f"step_{step:06d}.npz"), **detail)
            f.write(json.dumps(rec) + "\n")
print("ok")
