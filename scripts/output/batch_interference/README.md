# Batch-versus-batch interference probe — results

Run 2026-08-12, three arms, ~60 min each. Every arm probed its own finished
`last.pt` (global_step 3400, LR at the 9.0e-6 decay floor) over one frozen epoch
of 34 train batches: 34 optimizer steps and 1,156 measured losses per arm.

Analysis and interpretation live in
`diagnostics/010-batch-interference-capacity-probe.md`. This file is the artifact
index and the headline table.

## What the numbers are

For each arm, one optimizer step is taken on each batch in turn — always from a
bit-identical restored checkpoint — and the loss change it causes on every batch
is recorded as `delta = pre - post`. **Positive means the step helped that
batch.** See `scripts/batch_interference_probe.py` for the method.

| file | contents |
|---|---|
| `<arm>/batch_interference_long.csv` | one row per (step batch, evaluated batch) |
| `<arm>/batch_interference_matrix.csv` | the same deltas as a 34x34 matrix |
| `<arm>/batch_interference_summary.csv` | per step batch: self vs the other 33 |
| `<arm>/batch_interference_meta.json` | checkpoint, step, LR, seeds, restore drift |
| `<arm>-console.log` | full per-step console output |

## Headline

| arm | baseline loss | self delta | other-batch mean | others hurt | steps raising total loss | self-gain cancelled (median) |
|---|---|---|---|---|---|---|
| 01 no-reg | 0.1213 | +7.03e-4 (34/34) | **-6.65e-6** | **50.4%** | **11/34** | **+43.8%** |
| 02 t=1 x0.25 | 0.4924 | +1.12e-3 (34/34) | **+9.95e-5** | 39.8% | **0/34** | -314.3% |
| 03 both | 0.4672 | +8.73e-4 (34/34) | **-1.32e-5** | 45.8% | **12/34** | **+57.4%** |

Every step in every arm improves its own batch — 102 of 102. The arms differ
entirely in what that costs everything else.

## Who absorbs the damage

Mean delta on batch j caused by the other 33 batches' steps, split by how well
fit batch j already was:

| arm | easiest 10 batches | % hurt | hardest 10 batches | % hurt |
|---|---|---|---|---|
| 01 no-reg | **-2.71e-5** | **64.5%** | -3.99e-6 | 51.5% |
| 02 t=1 x0.25 | +1.16e-4 | 34.2% | +6.68e-5 | 41.5% |
| 03 both | **-5.83e-5** | 48.2% | +1.92e-5 | 39.7% |

In the two unregularized arms the already-well-fit batches are the ones that get
degraded. In arm 03 the transfer is explicit: the easiest batches lose (-5.83e-5)
while the hardest ones gain (+1.92e-5).

## Validity

- `restore_max_abs_drift = 0.000e+00` in all three arms: every step began from
  bit-identical weights, so no delta inherited a previous step's state.
- The probe aborts unless a batch evaluated twice on untouched weights returns
  the identical float; all three arms passed.
- Losses measured in fp32 (the step itself keeps the run's bf16), because a
  single step at the 9.0e-6 LR floor moves loss by ~1e-4 and bf16 cannot resolve
  that.
- Baseline loss is NOT comparable between fog-on and fog-off arms, nor between
  t=1-oversampled and baseline arms — the same caveats as
  `tests/output/memorization/README.md`. Arms 02 and 03 sit at nearly the same
  loss scale (0.492 vs 0.467), which is what makes their comparison the clean one.

## Reproduce

```bash
bash scripts/run_batch_interference_probe.sh
```
