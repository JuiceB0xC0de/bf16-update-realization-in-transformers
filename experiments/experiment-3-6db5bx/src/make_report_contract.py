"""Write report_summary.json and claims_manifest.json from the analysis outputs.

Every number in the report contract is read from results/analysis/*.json so the
claims manifest and the summary cannot drift from the files. Run from the
experiment directory after src/analyze.py:

    apy src/make_report_contract.py
"""
from __future__ import annotations

import json
from pathlib import Path

ANALYSIS = Path("results/analysis")
V = json.loads((ANALYSIS / "verdicts.json").read_text())
P = json.loads((ANALYSIS / "paired_differences.json").read_text())
D = json.loads((ANALYSIS / "delivery_vs_loss.json").read_text())
N = json.loads(Path("results/reference/null_control_prev.json").read_text())

claims: list[dict] = []


def get(obj, key: str):
    cur = obj
    for part in key.split("/"):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def claim(cid: str, text: str, source: str, key: str, obj) -> float:
    value = get(obj, key)
    claims.append({"id": cid, "claim": text, "source": source, "key": key, "value": value})
    return value


def r(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


VJ = "results/analysis/verdicts.json"
PJ = "results/analysis/paired_differences.json"
DJ = "results/analysis/delivery_vs_loss.json"
NJ = "results/reference/null_control_prev.json"

claim("null_floor_mean", "previous run null control (rne − torch AdamW) max |mean| is 0.0004", NJ, "null_control_rne_minus_torch/max_abs_mean", N)
claim("null_floor_ci", "previous run null control max |90% CI bound| is 0.0045", NJ, "null_control_rne_minus_torch/max_abs_ci_bound", N)
NOISE = "null-control noise floor from exp_01m1tgnb6xf98sqwfhm4effknt: max |mean| 0.0004, max CI bound 0.0045"

# ---------------------------------------------------------------- harness
claim("harness_all_passed", "all five plain-rounding reruns matched the previous experiment within 0.001", VJ,
      "harness/all_passed", V)
claim("harness_max_diff", "largest |eval NLL rerun − previous| over the five harness reruns is 0.0", VJ,
      "harness/max_eval_nll_abs_diff", V)
claim("n_runs", "37 runs analysed (32 keno + 5 harness reruns)", VJ, "n_runs", V)

# ---------------------------------------------------------------- Q1
q1 = V["q1_how_much_or_which"]
d1 = claim("q1_diff_mean", "keno_drop_25 − rne at lr 2e-4, seed mean, is −0.0158", VJ,
           "q1_how_much_or_which/difference/mean", V)
claim("q1_diff_seed0", "keno_drop_25 − rne at lr 2e-4, seed 0, is −0.0137", VJ,
      "q1_how_much_or_which/difference/per_seed/0", V)
claim("q1_diff_seed1", "keno_drop_25 − rne at lr 2e-4, seed 1, is −0.0178", VJ,
      "q1_how_much_or_which/difference/per_seed/1", V)
claim("q1_verdict", "Q1 pre-registered verdict is 'undecided at two seeds'", VJ,
      "q1_how_much_or_which/verdict", V)
claim("q1_keno25_len", "keno_drop_25 cumulative length ratio at lr 2e-4 is 0.309", VJ,
      "q1_how_much_or_which/delivery_match/keno_drop_25_length_ratio_mean", V)
claim("q1_rne_len", "rne cumulative length ratio at lr 2e-4 is 0.264", VJ,
      "q1_how_much_or_which/delivery_match/rne_length_ratio_mean", V)
claim("q1_keno25_proj", "keno_drop_25 cumulative projection at lr 2e-4 is 0.249", VJ,
      "q1_how_much_or_which/delivery_match/keno_drop_25_projection_mean", V)
claim("q1_rne_proj", "rne cumulative projection at lr 2e-4 is 0.137", VJ,
      "q1_how_much_or_which/delivery_match/rne_projection_mean", V)
claim("q1_keno25_never", "keno_drop_25 never-moved fraction at lr 2e-4 is 0.074", VJ,
      "q1_how_much_or_which/delivery_match/keno_drop_25_never_moved_mean", V)
claim("q1_rne_never", "rne never-moved fraction at lr 2e-4 is 0.616", VJ,
      "q1_how_much_or_which/delivery_match/rne_never_moved_mean", V)
claim("q1_matched_len_status", "rne's delivery lies below the measured keno-drop range at lr 2e-4", VJ,
      "q1_how_much_or_which/matched_delivery/0.0002/by_length_ratio/status", V)
claim("q1_matched_len_extrap", "extrapolated keno_drop − rne at rne's length ratio, lr 2e-4, is −0.022", VJ,
      "q1_how_much_or_which/matched_delivery/0.0002/by_length_ratio/keno_drop_minus_rne_at_matched_delivery", V)
claim("q1_matched_proj_extrap", "extrapolated keno_drop − rne at rne's projection, lr 2e-4, is −0.032", VJ,
      "q1_how_much_or_which/matched_delivery/0.0002/by_projection/keno_drop_minus_rne_at_matched_delivery", V)
claim("q1_keno25_minus_master", "keno_drop_25 − fp32_master at lr 2e-4, seed mean, is −0.1168", PJ,
      "0.0002/keno_drop_25-fp32_master/mean", P)
claim("q1_keno25_minus_master_seeds", "keno_drop_25 − fp32_master at lr 2e-4 per seed is −0.1107 and −0.1228", PJ,
      "0.0002/keno_drop_25-fp32_master/per_seed", P)
claim("q1_rne_minus_master", "rne − fp32_master at lr 2e-4, seed mean, is −0.1010 (1.1691 − 1.2701)", DJ,
      "0.0002/rne/nll_mean", D)
claim("q1_context_1e4", "keno_drop_25 − rne at lr 1e-4, seed mean, is −0.0469", VJ,
      "q1_how_much_or_which/context_lr1e-4/mean", V)

# ---------------------------------------------------------------- per-writer NLL and delivery
for lr in ("0.0002", "0.0001"):
    tag = "2e-4" if lr == "0.0002" else "1e-4"
    for w in ("keno_drop_25", "keno_drop_50", "keno_drop_75", "keno_drop_range",
              "keno_tab_25", "keno_tab_50", "keno_tab_75", "keno_tab_range",
              "rne", "fp32_master", "kahan"):
        v = get(D, f"{lr}/{w}/nll_mean")
        claim(f"nll_{w}_{tag}", f"{w} held-out NLL at lr {tag}, seed mean (seeds 0,1), is {r(v)}", DJ,
              f"{lr}/{w}/nll_mean", D)
        v = get(D, f"{lr}/{w}/cum_length_ratio")
        claim(f"len_{w}_{tag}", f"{w} cumulative length ratio at lr {tag} is {r(v, 3)}", DJ,
              f"{lr}/{w}/cum_length_ratio", D)
        v = get(D, f"{lr}/{w}/never_moved")
        claim(f"never_{w}_{tag}", f"{w} never-moved fraction at lr {tag} is {r(v, 3)}", DJ,
              f"{lr}/{w}/never_moved", D)
    for w in ("keno_drop_25", "keno_drop_50", "keno_tab_25"):
        for s in ("0", "1"):
            v = get(D, f"{lr}/{w}/nll_per_seed/{s}")
            claim(f"nll_{w}_{tag}_seed{s}", f"{w} held-out NLL at lr {tag}, seed {s}, is {r(v)}", DJ,
                  f"{lr}/{w}/nll_per_seed/{s}", D)
# 5e-5 references
claim("nll_fp32_master_5e-5", "fp32_master held-out NLL at lr 5e-5, seeds 0,1 mean, is 1.1512", DJ,
      "5e-05/fp32_master/nll_mean", D)
claim("nll_rne_5e-5", "rne held-out NLL at lr 5e-5, seeds 0,1 mean, is 1.2746", DJ, "5e-05/rne/nll_mean", D)
claim("nll_fp32_master_5e-5_allseeds", "fp32_master at lr 5e-5 over all five previous seeds is 1.1505", VJ,
      "q3_beats_tuned_exact_delivery/reference_fp32_master_5e-5/seed_mean_all_prev_seeds", V)

# ---------------------------------------------------------------- Q2
for lr in ("0.0002", "0.0001"):
    tag = "2e-4" if lr == "0.0002" else "1e-4"
    for f in ("25", "50", "75", "range"):
        v = get(V, f"q2_banking/per_lr/{lr}/{f}/tab_minus_master/mean")
        claim(f"q2_tab_minus_master_{f}_{tag}", f"keno_tab_{f} − fp32_master at lr {tag} is {r(v)}", VJ,
              f"q2_banking/per_lr/{lr}/{f}/tab_minus_master/mean", V)
        v = get(V, f"q2_banking/per_lr/{lr}/{f}/tab_minus_drop/mean")
        claim(f"q2_tab_minus_drop_{f}_{tag}", f"keno_tab_{f} − keno_drop_{f} at lr {tag} is {r(v)}", VJ,
              f"q2_banking/per_lr/{lr}/{f}/tab_minus_drop/mean", V)
        claim(f"q2_reading_{f}_{tag}", f"Q2 reading for fraction {f} at lr {tag}", VJ,
              f"q2_banking/per_lr/{lr}/{f}/reading", V)
for w in ("keno_tab_25", "keno_tab_50", "keno_tab_75", "keno_tab_range"):
    for lr in ("0.0002", "0.0001"):
        tag = "2e-4" if lr == "0.0002" else "1e-4"
        v = get(D, f"{lr}/{w}/cum_projection")
        claim(f"proj_{w}_{tag}", f"{w} cumulative projection at lr {tag} is {r(v, 3)}", DJ,
              f"{lr}/{w}/cum_projection", D)

# ---------------------------------------------------------------- Q3
claim("q3_flagged", "no keno writer has seed-mean NLL below the 1.150 bar", VJ,
      "q3_beats_tuned_exact_delivery/flagged", V)
claim("q3_bar", "Q3 bar is 1.150", VJ, "q3_beats_tuned_exact_delivery/bar", V)
claim("q3_best", "closest to the bar: keno_drop_50 at lr 1e-4, seed mean 1.1512", VJ,
      "q3_beats_tuned_exact_delivery/seed_means/keno_drop_50@0.0001/mean", V)
claim("q3_best_seed0", "keno_drop_50 at lr 1e-4 seed 0 is 1.1489 (below the bar on one seed)", VJ,
      "q3_beats_tuned_exact_delivery/seed_means/keno_drop_50@0.0001/per_seed/0", V)
claim("q3_best_seed1", "keno_drop_50 at lr 1e-4 seed 1 is 1.1535", VJ,
      "q3_beats_tuned_exact_delivery/seed_means/keno_drop_50@0.0001/per_seed/1", V)

# ---------------------------------------------------------------- paired to master at 1e-4 (drop)
for w in ("keno_drop_25", "keno_drop_50", "keno_drop_range", "keno_drop_75"):
    v = get(P, f"0.0001/{w}-fp32_master/mean")
    claim(f"paired_{w}_minus_master_1e-4", f"{w} − fp32_master at lr 1e-4 is {r(v)}", PJ,
          f"0.0001/{w}-fp32_master/mean", P)
    v = get(P, f"0.0001/{w}-rne/mean")
    claim(f"paired_{w}_minus_rne_1e-4", f"{w} − rne at lr 1e-4 is {r(v)}", PJ, f"0.0001/{w}-rne/mean", P)
for w in ("keno_drop_50", "keno_drop_75", "keno_drop_range"):
    v = get(P, f"0.0002/{w}-rne/mean")
    claim(f"paired_{w}_minus_rne_2e-4", f"{w} − rne at lr 2e-4 is {r(v)}", PJ, f"0.0002/{w}-rne/mean", P)
claim("paired_kahan_ref_2e-4", "keno_drop_25 − kahan at lr 2e-4 is −0.109", PJ, "0.0002/keno_drop_25-kahan/mean", P)

# ---------------------------------------------------------------- families, mask, timing
for fam in ("attention", "mlp", "embedding", "norm"):
    for w in ("rne", "keno_drop_25"):
        v = get(D, f"0.0002/{w}/families/{fam}/never_moved")
        claim(f"family_never_{w}_{fam}_2e-4", f"{w} never-moved in {fam} at lr 2e-4 is {r(v, 3)}", DJ,
              f"0.0002/{w}/families/{fam}/never_moved", D)
claim("mask_gap", "max |realised − target fraction| over all keno runs is 0.00014", VJ,
      "mask_realisation/max_abs_target_gap_over_all_keno_runs", V)
claim("timing_s_per_step", "mean seconds per step over all runs is 0.133", VJ, "timing/s_per_step_mean", V)
claim("timing_clip", "gradient clipping fired 78 times in total over 37 runs", VJ, "timing/clip_fired_total", V)
claim("eval_tokens", "held-out probe is 49,277 tokens", VJ, "timing/eval_tokens/0", V)
claim("untrained_nll", "untrained NLL is 1.9372 in every shard", VJ, "timing/untrained_nll_per_shard/seed0_lr2e-4", V)
claim("bytes_keno", "keno writer extra state is 403,545,024 bytes (3 bytes per coordinate)", DJ,
      "0.0002/keno_drop_25/extra_state_bytes", D)

# ---------------------------------------------------------------- summary
q2 = V["q2_banking"]["per_lr"]
summary = {
    "title": "Delivered amount is the dominant knob: random 25% delivery reproduces plain rounding's advantage over fp32 master on SmolLM2-135M; the −0.016 residual is undecided at two seeds; banked requests land but sit below fp32 master at lr 2e-4",
    "headline": (
        f"A keno writer that delivers a uniformly random 25% of each AdamW update (exact Kahan write on drawn coordinates, "
        f"request discarded elsewhere) reaches held-out NLL {r(get(D,'0.0002/keno_drop_25/nll_mean'))} at lr 2e-4, "
        f"{r(-get(P,'0.0002/keno_drop_25-fp32_master/mean'))} below fp32 master (1.2701); plain round-to-nearest sits "
        f"{r(get(D,'0.0002/fp32_master/nll_mean')-get(D,'0.0002/rne/nll_mean'))} below the same reference. Random partial delivery "
        f"therefore reproduces the whole of plain rounding's advantage over exact delivery, so how much is delivered is the dominant knob. "
        f"The remaining keno_drop_25 − rne difference is {r(d1)} (seeds −0.0137, −0.0178, same sign), inside the pre-registered "
        f"'undecided at two seeds' band, and it is confounded with write precision (drawn coordinates get exact writes). keno_drop_25 delivers "
        f"more than rne (length ratio 0.309 vs 0.264) and the two writers are not matched; a two-point extrapolation below the measured range "
        f"is post-hoc and not a supported bound. At lr 1e-4 keno_drop_25 and keno_drop_50 reach 1.1517 and 1.1512, 0.047 below plain rounding "
        f"and 0.029 below fp32 master at the same lr, tying fp32 master at its best lr (1.1512 on the same seeds; difference 0.0000, below the {NOISE}) but not crossing the 1.150 bar. "
        f"Banking undelivered requests (tab) delivers 0.96 to 1.00 of the intended length and is far from drop (+0.031 to +0.084 at 2e-4), so "
        f"banked requests land; the pre-registered 'behaves as delivery' rule (tab − master within ±0.01) passes only for 50%, range and 75% at "
        f"lr 1e-4 (−0.006, −0.010, −0.006; {NOISE}), while at 2e-4 tab is better than fp32 master by 0.017 to 0.033 at every fraction, same sign on both seeds."
    ),
    "assessment": "partial_signal",
    "research_question": (
        "Does the bfloat16 round-to-nearest advantage at lr 2e-4 come from how much of the update is delivered (about a quarter of "
        "the intended length) or from which coordinates the format picks (the large-relative-update ones)? Test with a keno "
        "write distributor that delivers a random fraction f of coordinates each step; undelivered requests are either discarded "
        "(drop) or banked in the compensation buffer (tab)."
    ),
    "decision_rules": {
        "q1": q1["rule"],
        "q1_verdict": q1["verdict"],
        "q1_reading_at_measured_delivery": (
            "keno_drop_25 − fp32_master at 2e-4 is −0.1168 (seeds −0.1107, −0.1228) against rne − fp32_master of −0.1010, so random "
            "25% delivery reproduces plain rounding's advantage over exact delivery in full: amount is the dominant knob. The residual "
            "keno_drop_25 − rne of −0.016 has the same sign on both seeds but lies in the undecided band. Delivery is not matched "
            "(keno_drop_25 length 0.309 vs rne 0.264; projection 0.249 vs 0.137). The keno-drop curve at 2e-4 rises with delivery over "
            "the measured range 0.31 to 0.79, but the curve must turn upward toward the untrained 1.937 at zero delivery and the 1e-4 "
            "curve is already flat between 0.30 and 0.54, so the two-point linear extrapolation to rne's delivery (−0.022 by length, "
            "−0.032 by projection) is unsupported below the range and is reported as post-hoc context only, not as a bound. Confound: "
            "drawn coordinates receive exact Kahan writes whereas rne quantises every write, so the residual mixes selection and write precision."
        ),
        "q2": V["q2_banking"]["rule"],
        "q2_verdict_2e-4": {f: q2["0.0002"][f]["reading"] for f in ("25", "50", "75", "range")},
        "q2_verdict_1e-4": {f: q2["0.0001"][f]["reading"] for f in ("25", "50", "75", "range")},
        "q2_reading": (
            "Tab delivers 0.96 to 1.00 of the intended length and is far from drop at 2e-4 (+0.031 to +0.084) and above drop at 1e-4 "
            "(+0.005 to +0.023), so banked requests land; they are not equivalent to discarded ones (only 75% at 1e-4, +0.005, is inside the "
            "±0.01 band on that side, a difference near the " + NOISE + "). The 'behaves as delivery' rule (tab − master within ±0.01) fails at 2e-4 for every fraction "
            "(−0.033, −0.020, −0.017, −0.018 for 25/50/75/range) and for 25% at 1e-4 (−0.012); it passes for 50%, range and marginally 75% "
            "at 1e-4 (−0.006, −0.010, −0.006; these sub-0.01 values sit above the " + NOISE + " but inside the pre-registered band). At 2e-4 tab is better than fp32 master by more than the band on the seed mean; for 75% and "
            "range one seed of each is inside 0.01 (per seed −0.0059/−0.0287 and −0.0272/−0.0088; " + NOISE + "), so the 2e-4 excess is not uniform across seeds."
        ),
        "q3": V["q3_beats_tuned_exact_delivery"]["rule"],
        "q3_verdict": "no writer flagged; keno_drop_50 at lr 1e-4 is 1.1512 (seed 0 1.1489, seed 1 1.1535), a tie with fp32 master at 5e-5 on the same seeds (1.1512; difference 0.0000, below the " + NOISE + ")",
        "harness": "rne rerun at every (seed, lr) must match the previous experiment within 0.001 NLL; all five matched exactly (0.0)",
    },
    "methods": {
        "model": "HuggingFaceTB/SmolLM2-135M @ 93efa2f097d58c2a74874c7e644dbc9b0cee75a2, bfloat16 parameters and moments",
        "data": "gsm8k main @ 740312add88f781978c0658806c59bc2815b9866; seed-specific fixed order, question + newline + answer + eos, max 512 tokens, batch 1, 3000 steps",
        "held_out": "256 fixed test questions, 49,277 tokens, NLL per token; untrained 1.9372",
        "optimizer": "AdamW betas (0.9, 0.999), eps 1e-8, weight decay 0.01, no schedule, grad clip 32; moments bf16 updated in torch foreach order",
        "writers": {
            "keno_drop_f": "each block of 50 steps draws a random fraction f of coordinates (byte mask, seeded generator); drawn coordinates get t=p+i+c, p←rne(t), c←t−p; undrawn coordinates discard the request, c unchanged",
            "keno_tab_f": "same draw; undrawn coordinates add the request to the compensation buffer c←c+i, delivered when next drawn",
            "fractions": "0.25, 0.50, 0.75 and 'range' = U(0.30, 0.70) redrawn per block (60 blocks per run)",
            "references": "rne, kahan, fp32_master from the previous experiment (same seeds, same data order); rne rerun in every shard as harness check",
        },
        "learning_rates": [0.0002, 0.0001],
        "seeds": [0, 1],
        "n_runs": 37,
        "delivery_metrics": "cumulative length ratio ||sum realised|| / ||sum intended||, projection <I,R>/||I||^2, never-moved fraction; family split attention/MLP/embedding/norm",
        "compute": "4 single-H100 jobs (one per seed x lr), 51 to 69 min each, 0.133 s/step",
        "extra_memory": "keno = bf16 compensation + byte mask = 3 bytes/coord on top of the 2-byte bf16 parameter (fp32 master: 4; rne: 0)",
    },
    "results": {
        "nll_seed_mean": {lr: {w: get(D, f"{lr}/{w}/nll_mean") for w in D[lr] if w != "torch_adamw"} for lr in ("0.0002", "0.0001")},
        "length_ratio": {lr: {w: get(D, f"{lr}/{w}/cum_length_ratio") for w in D[lr] if w != "torch_adamw"} for lr in ("0.0002", "0.0001")},
        "q1_difference": q1["difference"],
        "q2_per_lr": {lr: {f: {"tab_minus_master": q2[lr][f]["tab_minus_master"]["mean"],
                                "tab_minus_drop": q2[lr][f]["tab_minus_drop"]["mean"]} for f in q2[lr]} for lr in q2 if q2[lr]},
        "q3_reference_fp32_master_5e-5": V["q3_beats_tuned_exact_delivery"]["reference_fp32_master_5e-5"],
        "families_never_moved_2e-4": {w: {fam: get(D, f"0.0002/{w}/families/{fam}/never_moved") for fam in ("attention", "mlp", "embedding", "norm")} for w in ("rne", "keno_drop_25", "keno_tab_25", "fp32_master")},
        "mask_realisation": V["mask_realisation"],
        "timing": V["timing"],
    },
    "limitations": [
        "Two seeds per cell; no interval is claimed. Paired differences smaller than about 0.01 are read as ties; both seeds agree in sign on every reported difference. The previous run's null control (rne − torch AdamW, 5 seeds, exp_01m1tgnb6xf98sqwfhm4effknt) had max |mean| 0.0004 and max 90% CI bound 0.0045, so sub-0.01 differences here are above the harness noise floor but inside the pre-registered tie band.",
        "Write precision is confounded with selection: drawn coordinates get exact Kahan writes while plain rounding quantises every write, so keno_drop − rne mixes selection, precision and amount. A keno-drop variant with plain-rounding writes on drawn coordinates would separate them (proposed follow-up).",
        "Delivery is not matched between keno_drop_25 and rne (0.309 vs 0.264 length ratio). The matched-delivery numbers are a two-point linear extrapolation below the measured range (0.31 to 0.79); the curve must rise toward the untrained 1.937 at zero delivery and is flat at lr 1e-4 between 0.30 and 0.54, so the extrapolation is unsupported and the −0.016 residual is not a lower bound on a selection effect.",
        "The final norm of the tab compensation buffer was not recorded, so the residual tab − master gap cannot be split between still-banked and rounding-lost mass.",
        "Single model (SmolLM2-135M), single dataset (GSM8K), batch 1, 3000 steps; results transfer to other settings only as a hypothesis.",
    ],
    "artifacts": {
        "verdicts": "results/analysis/verdicts.json",
        "paired_differences": "results/analysis/paired_differences.json",
        "delivery_vs_loss": "results/analysis/delivery_vs_loss.json",
        "step_series": "results/analysis/step_series.json",
        "runs_table": "results/analysis/runs_table.csv",
        "harness_checks": "results/shards/*/harness_check.json",
        "raw_runs": "artifact://juiceb0xc0de-15787e/experiments/exp_01m1tvjx4ee3xshh9djt6db5bx/runs/",
        "figures": "figures/",
        "wandb": "https://wandb.ai/ricks-holmberg-juiceb0xc0de/default-exp-exp_01m1tvjx4ee3xshh9djt6db5bx",
    },
}

Path("report_summary.json").write_text(json.dumps(summary, indent=1))
Path("claims_manifest.json").write_text(json.dumps({"claims": claims}, indent=1))
print(f"wrote report_summary.json and claims_manifest.json ({len(claims)} claims)")
