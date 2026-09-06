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
(filled in at completion)
