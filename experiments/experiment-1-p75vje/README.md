# Does the BF16 update-realization pattern appear in a small pretrained Transformer under AdamW?

## What this is
Fine-tunes `HuggingFaceTB/SmolLM2-135M` (revision 93efa2f, all parameters bfloat16) on `openai/gsm8k` (main/train, revision 740312a) with plain `torch.optim.AdamW` for 300 steps at batch size 1, lr 2e-4, and records, at every step, how the intended AdamW update compares with the change that actually reaches the bfloat16 weights: pooled overshoot, length ratio, cosine, unmoved fraction, the round-to-nearest write model, and a nine-factor shadow curve. Two fixed data orders (seeds 0 and 1) are the replicate. The same analysis is first checked on the public Mamba step-100 bundle.

## Layout
- `src/realization.py`: the statistics (intent formula, bf16 spacing, pooled overshoot, write model, shadow curve).
- `src/reference_check.py`: reproduces the reviewed Mamba step-100 numbers; `results/reference_check.json` is its output.
- `src/train_observe.py`: the GPU job (clip calibration, two 300-step observer-wrapped runs, detail captures).
- `src/analyze.py`: verdicts and plot-ready tables from the job outputs.
- `src/prepare_figure_data.py`: turns the analysis JSONs into the per-figure `data.json` files and renders every `figures/*/plot.py`.
- `figures/<name>/{plot.py,data.json,<name>.html,alt.txt}`: standalone figure bundles (overshoot per step, predicted vs stored, shadow curve, x histogram, family overshoot); `figures/entity_colors.json` fixes the colour of each order, family and the Mamba reference.
- `results/`: small reviewable outputs (`reference_check.json`, `analysis/*.json` copied from the analysis job).
- `scratch/`: CPU observer test and a toy-run generator used to exercise `analyze.py` and the figures.

## How to reproduce
All commands run from this directory in the `job-core` image (`uv run python`; the pod-side `apy` interpreter also runs the CPU steps).

1. Reference check on the public Mamba step-100 bundle (CPU):
   `uv run python src/reference_check.py --out results/reference_check.json`
2. Production run (1×H100, about 8 minutes, produced by on-demand job 653635765785 on 2026-09-06):
   `uv run python src/train_observe.py --out "$SILICO_EXPERIMENT_ARTIFACTS_DIR/run1" --steps 300 --orders 0 1 --calib-steps 20 --detail-steps 1,10,50,100,200,300`
   with `HF_TOKEN` and `WANDB_API_KEY` forwarded as job settings and `HF_HUB_CACHE`/`WANDB_DIR` pointed at `scratch/` inside the worktree.
3. Analysis (CPU, job 212300581606):
   `uv run python src/analyze.py --run-dir <run1> --out <analysis> --ref results/reference_check.json`
4. Figures (CPU): `uv run python src/prepare_figure_data.py --analysis results/analysis --hist-step 100`

Seeds: torch/numpy seed 42 for everything except the data orders, which are `numpy.random.default_rng(seed).permutation(7473)[:300]` for seeds 0 and 1. The gradient-clip bound (32) was calibrated as 2·p95 of 20 unclipped grad norms rounded up to a power of two and never fired.

## Outputs
- Run outputs (durable): `artifact://juiceb0xc0de-15787e/experiments/exp_01m1tdvatwe1rrek9g34p75vje/run1/` — `calibration.json`, `manifest.json`, `order{0,1}/steps.jsonl` (300 per-step records incl. the nine-factor shadow and log2 histograms), `order{0,1}/run_info.json`, `order{0,1}/summary.json`, `order{0,1}/detail/step_{000001,000010,000050,000100,000200,000300}.npz` (fp32 pre-step weights, bf16 post-step weights, moments and hyperparameters for 2,013,504 coordinates: rows 0:32 of the seven projections in layers 0, 7, 15, 22, 29; 1024 seen and 1024 unseen embedding rows; every norm weight). Job log: `.../run1_stdout.log`.
- Analysis outputs (durable): `artifact://juiceb0xc0de-15787e/experiments/exp_01m1tdvatwe1rrek9g34p75vje/analysis/` — `verdicts.json`, `per_order.json`, `detail_checks.json`, `step_series.json`, `shadow_series.json`, `x_hist.json`; small copies in `results/analysis/`.
- Model: `HuggingFaceTB/SmolLM2-135M` @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Data: `openai/gsm8k` main/train @ `740312add88f781978c0658806c59bc2815b9866`. Mamba reference: public replay bundle revision d30ed00178a04810ea197a7f621e543c1aa9f091, step 100.
- W&B: project `default-exp-exp_01m1tdvatwe1rrek9g34p75vje`, runs `smollm2-135m-adamw-bf16-order{0,1}-lr0.0002`.
- No external LLM or judge was used.
