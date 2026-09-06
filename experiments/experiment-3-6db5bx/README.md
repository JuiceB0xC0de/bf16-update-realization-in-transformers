# A keno write distributor: does the bfloat16 advantage come from how much is delivered or which weights get it?

## What this is
`WriteDistributor` (`src/write_distributor.py`) sits between AdamW's fp32 request and a bfloat16 weight. Every 50 steps it draws a random mask over every coordinate (a fixed fraction 25/50/75%, or a fresh U(0.30, 0.70) draw per block); drawn coordinates get the exact Kahan write (request + tab, one rounding, residual kept in a bf16 tab), undrawn coordinates are skipped and their request is either discarded (`drop`) or banked in the tab (`tab`). The eight writers `keno_{drop,tab}_{25,50,75,range}` plug into the previous experiment's `CompensatedAdamW`. This experiment fine-tunes `HuggingFaceTB/SmolLM2-135M` (bf16) on `openai/gsm8k` for 3,000 steps at batch size 1 under every keno writer at lr 2e-4 and 1e-4 for data-order seeds 0 and 1 (32 runs), reruns plain rounding (`rne`) as a harness check (5 runs), and compares against the previous run's plain rounding, Kahan and fp32-master results on the same data streams.

## Layout
- `src/write_distributor.py`: the distributor (new). `src/compensated_adamw.py`: the optimizer from the previous run, extended with the keno writers. `src/realization.py`: delivery statistics (unchanged).
- `src/train_sweep.py`: one (seed, lr) shard per job: harness rerun(s) first with abort on mismatch, then the eight keno writers; JSON per run.
- `src/analyze.py`: paired per-seed differences against the previous run, the three verdicts, matched-delivery reading, delivery-vs-loss table, step series.
- `src/prepare_figures.py`: figure bundles under `figures/`.
- `tests/`: 45 unit tests (16 for the distributor).
- `results/reference/`: the previous run's `runs_table.csv` and the 100-step loss prefix used by the smoke. `results/smoke/`: smoke summary. `results/analysis/`: analysis outputs.
- `scratch/make_toy_runs.py`: schema-faithful toy shards for exercising the analysis on the pod.

## How to reproduce
1. Unit tests (CPU): `PYTHONPATH=src apy -m pytest -q tests`
2. One shard (1x H100, job-core image, about 1 hour): `uv run python src/train_sweep.py --out "$SILICO_EXPERIMENT_ARTIFACTS_DIR/runs/seed0_lr2e-4" --seed 0 --lrs 2e-4 --steps 3000 --stats-every 25 --eval-n 256 --sticky-interval 50 --harness rne:2e-4 --reference-dirs <dir with the previous run's seed{S}/runs/rne_lr*.json>` with `HF_TOKEN` and `WANDB_API_KEY` as job settings and `HF_HOME` under `scratch/`. Shards: (0, 2e-4), (0, 1e-4; harness also rne:5e-5), (1, 2e-4), (1, 1e-4).
3. Analysis (CPU): `apy src/analyze.py --runs-dir <dir with seed*_lr*/> --prev-table results/reference/prev_runs_table.csv --out results/analysis`
4. Figures (pod): `uv run --no-sync python src/prepare_figures.py --analysis results/analysis`

Seeds: torch seed 42; data orders `numpy.random.default_rng(seed).permutation(7473)[:3000]`; probe rows `default_rng(1234).permutation(1319)[:256]` of the test split; mask draws seeded `100000*seed + 1000*CONDITIONS.index(writer) + 7`.

## Outputs

All numbers below are seed means over seeds 0 and 1 (held-out NLL, nats/token, 256 GSM8K test questions, 49,277 tokens).

| writer | lr 2e-4 | lr 1e-4 |
|---|---|---|
| plain rounding (rne, previous run) | 1.1691 | 1.1986 |
| keno_drop_25 | 1.1533 | 1.1517 |
| keno_drop_50 | 1.1879 | 1.1512 |
| keno_drop_75 | 1.2214 | 1.1699 |
| keno_drop_range | 1.1883 | 1.1544 |
| keno_tab_25 | 1.2370 | 1.1686 |
| keno_tab_50 | 1.2497 | 1.1742 |
| keno_tab_75 | 1.2528 | 1.1748 |
| keno_tab_range | 1.2521 | 1.1706 |
| kahan (previous run) | 1.2624 | 1.1807 |
| fp32 master (previous run) | 1.2701 | 1.1804 |

fp32 master at lr 5e-5 (same seeds): 1.1512. Q1 rule: keno_drop_25 − rne at 2e-4 = −0.0158 (−0.0137, −0.0178), "undecided at two seeds". Q3: no writer below 1.150.

Provenance
- Model: `HuggingFaceTB/SmolLM2-135M` @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, bf16 parameters and moments.
- Data: `gsm8k` main @ `740312add88f781978c0658806c59bc2815b9866`, seed-specific fixed order, batch 1, 3000 steps.
- Image: `job-core` (torch 2.11.0+cu130, transformers 5.12.1), one H100 per shard.
- Jobs: 281598836202 (seed 0, lr 2e-4), 439124401464 (seed 0, lr 1e-4 + rne 5e-5 harness), 639692456089 (seed 1, lr 2e-4), 325308821433 (seed 1, lr 1e-4); 51 to 69 min each. Bundle utility job 572562273379.
- Date: 2026-09-06.
- W&B: https://wandb.ai/ricks-holmberg-juiceb0xc0de/default-exp-exp_01m1tvjx4ee3xshh9djt6db5bx (project `default-exp-exp_01m1tvjx4ee3xshh9djt6db5bx`, 37 runs).

Durable files
- Raw per-run records (losses, grad norms, 25-step delivery series, cumulative delivery whole-model and per family, distributor block log, held-out NLL): `artifact://juiceb0xc0de-15787e/experiments/exp_01m1tvjx4ee3xshh9djt6db5bx/runs/<seed>_<lr>/runs/<writer>_lr<lr>.json`, plus `harness_check.json`, `summary.json`, `seed_manifest.json`, `untrained_eval.json` per shard.
- Job logs: `artifact://juiceb0xc0de-15787e/experiments/exp_01m1tvjx4ee3xshh9djt6db5bx/logs/`.
- One-file bundle of everything above: `artifact://juiceb0xc0de-15787e/experiments/exp_01m1tvjx4ee3xshh9djt6db5bx/bundles/runs_all.tar.gz` (4.7 MB).
- Analysis (this worktree): `results/analysis/{verdicts,paired_differences,delivery_vs_loss,step_series}.json`, `runs_table.csv`, `run_headlines.json`; shard summaries and harness checks under `results/shards/`.
- Figure bundles: `figures/*/` (data.json, plot.py, html, alt.txt) with `figures/entity_colors.json`.
- Report contract: `report_summary.json`, `claims_manifest.json` (built by `src/make_report_contract.py`).
- Reusable distributor: `src/write_distributor.py` (WriteDistributor, KENO_WRITERS, make_distributor) with `tests/test_write_distributor.py`; plugs into `src/compensated_adamw.py` via `distributor=`.
