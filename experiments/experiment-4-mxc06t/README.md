# Random versus magnitude-based selection at matched delivery and write precision

## What this is
Follow-up to the keno write distributor (exp_01m1tvjx4ee3xshh9djt6db5bx). A second selector, `thr_drop_{15,25,50}` (`ThresholdDistributor` in `src/write_distributor.py`), delivers AdamW's request exactly (Kahan write, bf16 tab) to the coordinates the bfloat16 format itself would pick, by request size relative to the weight's bf16 spacing (|i| / spacing(p) >= theta, re-evaluated every step), and drops the rest. theta is calibrated per learning rate and level so the cumulative delivery matches the corresponding random `keno_drop` level. Also adds `keno_drop_15` and seeds 2 to 4. Model `HuggingFaceTB/SmolLM2-135M` (bf16), `openai/gsm8k`, 3,000 steps at batch 1, lr 2e-4 and 1e-4, five data-order seeds.

## Layout
- `src/write_distributor.py`: keno `WriteDistributor` (previous run) + `ThresholdDistributor` (new). `src/compensated_adamw.py`: the optimizer with all writers. `src/realization.py`: delivery statistics.
- `src/train_sweep.py`: one shard = list of (seed, lr, condition) tasks; harness rerun with abort on mismatch. `src/calibrate.py`: theta probes and full-horizon secant calibration.
- `src/analyze.py`: verdicts, paired intervals, delivery-vs-loss curves. `src/prepare_figures.py`: figure bundles.
- `tests/`: 63 unit tests. `results/reference/`: previous runs' tables. `results/calibration/`: probes and calibrated thetas. `results/shard_plans/`: task lists per job. `results/analysis/`: outputs.

## How to reproduce
(filled in on completion)

## Outputs
(filled in on completion)
