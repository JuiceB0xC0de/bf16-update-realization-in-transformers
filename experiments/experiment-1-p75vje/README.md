# Does the BF16 update-realization pattern appear in a small pretrained Transformer under AdamW?

## What this is
Fine-tunes `HuggingFaceTB/SmolLM2-135M` (revision 93efa2f, all parameters bfloat16) on `openai/gsm8k` (main/train, revision 740312a) with plain `torch.optim.AdamW` for 300 steps at batch size 1, lr 2e-4, and records, at every step, how the intended AdamW update compares with the change that actually reaches the bfloat16 weights: pooled overshoot, length ratio, cosine, unmoved fraction, the round-to-nearest write model, and a nine-factor shadow curve. Two fixed data orders (seeds 0 and 1) are the replicate. The same analysis is first checked on the public Mamba step-100 bundle.

## Layout
- `src/realization.py`: the statistics (intent formula, bf16 spacing, pooled overshoot, write model, shadow curve).
- `src/reference_check.py`: reproduces the reviewed Mamba step-100 numbers; `results/reference_check.json` is its output.
- `src/train_observe.py`: the GPU job (clip calibration, two 300-step observer-wrapped runs, detail captures).
- `src/analyze.py`: verdicts and plot-ready tables from the job outputs.
- `results/`: small reviewable outputs.

## How to reproduce
(filled in as the run completes)

## Outputs
(filled in as the run completes)
