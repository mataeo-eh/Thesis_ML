# Analysis Scripts

Run scripts through the Thesis_ML virtual environment from the submodule root.

## Context-window estimator

```powershell
& .\.venv\Scripts\python.exe .\scripts\estimate_context_window.py
```

The script derives the default parquet location from the repository layout, so it does not embed or emit a machine-specific path. It streams parquet metadata plus the two upgrade columns and writes `scripts/output/context_window_estimate.json`.

Each sample is one full replay from one player perspective:

- Input counts all self content and zero-fog enemy content, plus one delimiter per timestep and one terminal `[EOS]` token.
- Output counts the `[BOS]`/perspective-outcome prefix, all enemy content, one delimiter per timestep, and one terminal `[END]` token.
- An entity contributes one token when its instance is present in a row. Each listed cumulative upgrade contributes one token in every row where it is present.
- Padding is excluded. Token statistics cover both perspectives for every replay; timestep statistics count each replay once. Both include minimum, maximum, mean, median, mode, and all tied modes.

Use `--input-dir`, `--pattern`, or `--output` to override defaults. Prefer repository-relative arguments.

## Batch-versus-batch interference probe

```bash
./.venv/Scripts/python.exe scripts/batch_interference_probe.py --config configs/memorization_01_no_regularization.yaml
bash scripts/run_batch_interference_probe.sh   # all three memorization arms, back to back
```

Asks whether the trained model has capacity to spare. It restores a finished
checkpoint, freezes one epoch of training batches together with one corrupted
view of each (fixed `t`, fixed noise, fixed self-conditioning mask), records the
baseline loss on every view, and then — restoring the checkpoint exactly each
time — takes a single real optimizer step on each batch in turn and re-measures
the loss on all of them.

`delta_loss = pre_loss - post_loss`, so **positive means the step helped that
batch and negative means it hurt it**. A step that improves its own batch while
pushing other batches' losses up means those batches are competing for the same
parameters, which is the signature of a capacity-limited model. Uniformly
non-negative deltas mean the opposite: capacity to spare.

Nothing is written back into the probed run — no checkpoint, no EMA update, no
scheduler advance, no metrics. Outputs land in
`scripts/output/batch_interference/<arm>/`, with any previous run's files moved
into a timestamped `_archive-*` subfolder:

| file | contents |
|---|---|
| `batch_interference_long.csv` | one row per (step batch, evaluated batch) |
| `batch_interference_matrix.csv` | the same deltas as a step-by-eval matrix |
| `batch_interference_summary.csv` | per step batch: self delta vs the others |
| `batch_interference_meta.json` | checkpoint, step, LR, seeds, restore drift |

Loss measurement runs in fp32 by default (`--eval-precision config` opts out)
because bf16 cannot resolve the change a single late-schedule step produces; the
optimizer step itself always uses the run's configured precision. `--lr-scale`
amplifies the step for readability at the cost of no longer describing a step
training would actually take, and `--max-batches` bounds a smoke run.
