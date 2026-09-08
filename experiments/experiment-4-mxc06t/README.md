# Random versus magnitude-based selection at matched delivery and write precision

## What this is
Follow-up to the keno write distributor (exp_01m1tvjx4ee3xshh9djt6db5bx). A second selector, `thr_drop_{15,25,50}` (`ThresholdDistributor` in `src/write_distributor.py`), delivers AdamW's request exactly (Kahan write, bf16 tab) to the coordinates the bfloat16 format itself would pick, by request size relative to the weight's bf16 spacing (|i| / spacing(p) >= theta, re-evaluated every step), and drops the rest. theta is calibrated per learning rate and level so the cumulative delivery matches the corresponding random `keno_drop` level. Also adds `keno_drop_15` and seeds 2 to 4. Model `HuggingFaceTB/SmolLM2-135M` (bf16), `openai/gsm8k`, 3,000 steps at batch 1, lr 2e-4 and 1e-4, five data-order seeds.

## Layout
- `src/write_distributor.py`: keno `WriteDistributor` (previous run) + `ThresholdDistributor` (new). `src/compensated_adamw.py`: the optimizer with all writers. `src/realization.py`: delivery statistics.
- `src/train_sweep.py`: one shard = list of (seed, lr, condition) tasks; harness rerun with abort on mismatch. `src/calibrate.py`: theta probes and full-horizon secant calibration.
- `src/analyze.py`: verdicts, paired intervals, delivery-vs-loss curves. `src/prepare_figures.py`: figure bundles.
- `tests/`: 63 unit tests. `results/reference/`: previous runs' tables. `results/calibration/`: probes and calibrated thetas. `results/shard_plans/`: task lists per job. `results/analysis/`: outputs.

## How to reproduce
All GPU work ran as on-demand `job-core` jobs, one H100 each, three concurrent, from the worktree root with
`cd experiments/experiment-4-mxc06t && export HF_HOME=$PWD/scratch/hf_home && uv run python src/<script>`.
Outputs go under `$SILICO_EXPERIMENT_ARTIFACTS_DIR/runs/<shard>/seed<S>/runs/<condition>_lr<lr>.json`.

1. Calibrate theta per (lr, level) on seed 0 (six jobs, one per cell; the converged pass is the seed-0 production record):
   `src/calibrate.py calibrate --lr 2e-4 --seed 0 --writer thr_drop_15 --run-keno keno_drop_15 --theta0 0.7 --out $SILICO_EXPERIMENT_ARTIFACTS_DIR/runs/cal_15_lr2e-4`
   (25/50 levels pass `--target` from `results/reference/exp3_keno_runs_table.csv` instead of `--run-keno`). Result: `results/calibration/theta_table.json`.
2. Production shards: `src/train_sweep.py --tasks results/shard_plans/<plan>.json --theta-table results/calibration/theta_table.json --reference-dirs <exp2 runs> --out $SILICO_EXPERIMENT_ARTIFACTS_DIR/runs/<plan>`
   for plans keno_A, keno_B, thr_primary_s12, thr_primary_s34, thr_secondary_s1, thr_rest_A, thr_rest_B, thr_rest_C (thr_secondary_s2..s4 were superseded by thr_rest_*). keno_A includes the harness rerun (rne seed 2 lr 2e-4, must match 1.163403438942883 within 0.001; it matched exactly).
3. Analysis (CPU job, whole `runs/` folder staged as input): `src/analyze.py --runs-dir <runs> --exp3-table results/reference/exp3_keno_runs_table.csv --exp2-table results/reference/exp2_runs_table.csv --out results/analysis`.
4. Pod-side: `uv run --no-sync python src/prepare_figures.py --analysis results/analysis` (figure bundles), `apy src/make_report_summary.py` (report_summary.json + claims_manifest.json, every number traced to a key in results/analysis).
Tests: `PYTHONPATH=$PWD apy -m pytest -q tests` (63 tests).

## Outputs
- Run records (53 this run, all cells five seeds): `artifact://juiceb0xc0de-15787e/experiments/exp_01m1v355jafe8sqbzmwmmxc06t/runs/` and one tarball `.../bundles/runs_all.tar.gz`. Each record holds per-step losses, grad norms (pre-clip), clip events, 25-step delivery series, cumulative delivery whole-model and per family, never-moved fractions, per-step selected fractions (threshold), held-out NLL on 49,277 tokens, config and W&B URL (null for the 14 threshold runs of Sept 8, which trained without W&B after an API handshake failure).
- Analysis: `results/analysis/{verdicts,curves,step_series,per_run_diagnostics,clip_summary}.json`, `results/analysis/runs_table.csv` (also at `.../analysis/` in the store). Calibration: `results/calibration/theta_table_full.json`.
- Figures: `figures/<name>/` bundles (q1_paired, q2_vs_master, delivery_vs_loss_lr2e-4, delivery_vs_loss_lr1e-4, never_moved_family_*, delivery_over_steps_*, instability_primary_thr).
- Report: `report_summary.json`, `claims_manifest.json` (221 claims); published page in the experiment's Results tab.
- Model `HuggingFaceTB/SmolLM2-135M` @ 93efa2f097d58c2a74874c7e644dbc9b0cee75a2; data `openai/gsm8k` main @ 740312add88f781978c0658806c59bc2815b9866. W&B project `default-exp-exp_01m1v355jafe8sqbzmwmmxc06t`. Dates: calibration and production Sept 6-8 2026, analysis Sept 8 2026.
- Jobs: calibration 473451514148, 362365063042, 441959342487, 443823741902, 101246367718, 925148905685; keno 240403904134, 183558143092; threshold 805546448579, 308044629741, 872194154649, 295709102359, 895734294267, 522950432861 (628952642813, 999181512626, 384387341829 failed at W&B init and produced nothing); analysis 341252512167.

## Results in brief
Q1 (paired keno minus threshold, 90% t, n=5): primary 25% @ 2e-4 -0.0273 [-0.0593, +0.0048] undecided; primary 50% @ 1e-4 -0.0122 [-0.0292, +0.0048] undecided; both because one threshold seed diverged late (1.2405 seed 2; 1.1976 seed 1). Secondary 15%@2e-4, 50%@2e-4, 15%@1e-4, 25%@1e-4: random wins. Magnitude wins 0 of 6.
Q2: keno_drop_50 @ 1e-4 minus fp32_master @ 5e-5 = +0.0009 [-0.0003, +0.0020], tie; keno_15@2e-4, keno_25@2e-4, keno_25@1e-4 within 0.003 of the bar.
Q3: keno interpolated at rne's delivery minus rne = -0.0169 at 2e-4 (interpolated), -0.0347 at 1e-4 (short extrapolation below the 15% level).
