"""Reference check: the analysis must reproduce the reviewed Mamba step-100 numbers.

Downloads the public Camera-3 step-100 bundle, runs realization.analyze_bundle over the
bfloat16 in-projection slices (D and dt_bias are float32 and excluded), and compares with
the reviewed values. Exit status 1 if the check fails (rho gap > 0.01 or exact-write agreement < 99%).
The reviewed 99.89% agreement is the exact write-model agreement (r_hat == r, 1,056 mismatches);
the percoord.py rounding-in-ULP-units definition is reported alongside as round_agreement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from realization import analyze_bundle  # noqa: E402

URL = ("https://huggingface.co/datasets/juiceb0xc0de/mmb3-slt-6xcam-results/"
       "resolve/d30ed00178a04810ea197a7f621e543c1aa9f091/observed-runs/20260810T091311Z/"
       "e1_s000100/camera-3/step_000100_optimizer_delivery.npz")

REVIEWED = {  # from the reviewed synthesis report exp_01m1sty5nefxts8x13s1b4dv5a
    "n": 958_464,
    "overshoot": 1.299,
    "predicted_overshoot": 1.298,
    "exact_write_agreement": 0.9989,  # 1,056 mismatches of 958,464
    "length_ratio": 0.6084,
    "cosine": 0.4684,
    "multi_spacing_share_of_movers": 0.3588,
    "unmoved_fraction": 0.8774,
}
TOL = {"n": 0, "overshoot": 0.01, "predicted_overshoot": 0.01, "exact_write_agreement": 0.001,
       "length_ratio": 0.001, "cosine": 0.001, "multi_spacing_share_of_movers": 0.001,
       "unmoved_fraction": 0.001}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("REF_BUNDLE_CACHE", "/tmp/c3_step100.npz"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results",
                                                  "reference_check.json"))
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        req = urllib.request.Request(URL)
        tok = os.environ.get("HF_TOKEN")
        if tok:
            req.add_header("Authorization", "Bearer " + tok)
        with urllib.request.urlopen(req, timeout=600) as r, open(args.cache + ".part", "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(args.cache + ".part", args.cache)
    print(f"bundle: {args.cache} ({os.path.getsize(args.cache)} bytes)")

    z = np.load(args.cache, allow_pickle=False)
    g = {k: float(z[f"param/{k}"][0]) for k in
         ("group_lr", "group_weight_decay", "group_eps", "bias_correction1", "bias_correction2")}
    print("group settings:", g)
    arrays = {k: z[k] for k in z.files if k.startswith("layer_")}
    res = analyze_bundle(arrays, lr=g["group_lr"], wd=g["group_weight_decay"], eps=g["group_eps"],
                         bc1=g["bias_correction1"], bc2=g["bias_correction2"])
    pooled = res["pooled"]
    print(json.dumps({k: v for k, v in pooled.items() if k not in ("shadow", "x_hist_log2")}, indent=1))
    print("shadow:", [(s["factor"], round(s["overshoot"], 4)) for s in pooled["shadow"]])

    checks = {}
    ok = True
    for k, ref in REVIEWED.items():
        got = pooled[k]
        gap = abs(got - ref)
        passed = gap <= TOL[k] + 1e-12
        checks[k] = {"reviewed": ref, "observed": got, "gap": gap, "tolerance": TOL[k], "pass": passed}
        ok &= passed
        print(f"  {k:<32s} reviewed {ref:<12} observed {got:.6g}  gap {gap:.3g}  {'PASS' if passed else 'FAIL'}")
    print("REFERENCE CHECK:", "PASS" if ok else "FAIL")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"bundle_url": URL, "group_settings": g, "pooled": pooled,
                   "n_tensors": len(res["per_tensor"]), "checks": checks, "pass": ok,
                   "torch_version": torch.__version__}, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
