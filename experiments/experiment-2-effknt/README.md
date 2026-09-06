# Build a compensated bfloat16 AdamW writer and test whether delivering the intended update helps

## What this is
`CompensatedAdamW` is a `torch.optim.Optimizer` that keeps parameters and moments in bfloat16, computes the AdamW intended update in one fp32 path, and writes the weight with one of five interchangeable writers: `rne` (round-to-nearest, the baseline), `kahan` (bf16 compensation buffer), `selective` (Kahan that resets the buffer on moved coordinates), `sr` (stochastic rounding), `fp32_master` (fp32 master copy). Plain `torch.optim.AdamW` is the null control. This experiment fine-tunes `HuggingFaceTB/SmolLM2-135M` (bf16) on `openai/gsm8k` for 3,000 steps at batch size 1 under every writer at three learning rates (2e-4, 1e-4, 5e-5) and five data-order seeds (90 runs), and measures (1) how much of the cumulative intended update each writer delivers, (2) whether Kahan reproduces the fp32-master held-out loss, and (3) whether the baseline-vs-master sign flips with learning rate as it did on Mamba.

## Layout
- `src/compensated_adamw.py`: the optimizer (five writers), `ObservedTorchAdamW` (null control with the same `step(hook=...)` interface), `bf16_sr`, `compute_intent`.
- `src/realization.py`: per-step pooled statistics and cumulative delivery (length ratio, rho, cosine, never-moved), lightened from the previous experiment.
- `src/train_sweep.py`: one seed per job: equivalence check, 18 runs, held-out eval, JSON outputs.
- `src/analyze.py`: paired per-seed differences with Student-t 90% intervals, the three verdicts, null floor, delivery and step-series tables.
- `src/prepare_figures.py`: renders the six figure bundles under `figures/` from `results/analysis/`.
- `tests/test_compensated_adamw.py`: 13 unit tests (write agreement vs torch, Kahan residual, selective reset, SR unbiasedness and determinism, master-side delivery, hook consistency).
- `results/smoke/`: smoke and equivalence numbers; `results/analysis/`: copies of the analysis outputs.
- `scratch/make_toy_runs.py`: schema-faithful toy sweep used to exercise the analysis on the pod.

## How to reproduce
From this directory in the `job-core` image (`uv run python`); the pod-side `apy` interpreter runs the CPU steps.

1. Unit tests (CPU): `PYTHONPATH=src apy -m pytest -q tests`
2. One seed (1x H100, about 2 hours): `uv run python src/train_sweep.py --out "$SILICO_EXPERIMENT_ARTIFACTS_DIR/runs/seed0" --seed 0 --steps 3000 --stats-every 25 --eval-n 256 --equivalence-steps 20` with `HF_TOKEN` and `WANDB_API_KEY` as job settings and `HF_HUB_CACHE` under `scratch/`. Repeat for seeds 1 to 4.
3. Analysis (CPU): `uv run python src/analyze.py --runs-dir <dir containing seed0..seed4> --out results/analysis`
4. Figures (pod): `uv run --no-sync python src/prepare_figures.py --analysis results/analysis`

Seeds: torch seed 42 for every run; data orders `numpy.random.default_rng(seed).permutation(7473)[:3000]` for seeds 0 to 4; held-out probe rows `default_rng(1234).permutation(1319)[:256]` of the test split; SR generator seeded per (seed, run index). Clip bound 32 (calibrated in the previous experiment; never fired there).

## Outputs
Filled in at completion.
