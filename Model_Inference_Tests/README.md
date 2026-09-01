# Model_Inference_Tests

One entry point for "the training run finished — now show me how the model
behaves on replays it has never seen."

Everything here is **read-only** with respect to the run being tested. No
checkpoint, config, manifest, or source replay is modified. Tests write into
their own run directory under `output/`, with one deliberate exception: the
cross-model leaderboard CSV described below, which lives at the root of
`output/` and accumulates one row per model scored.

## Layout

```
Model_Inference_Tests/
  run_inference_tests.py     the runner — this is what you invoke
  inference_test_api.py      shared plumbing every test builds on
  Test_Scripts/              one module per test, run in filename order
  output/                    results, one directory per run (git-ignored)
    model_comparison_bits_per_token.csv   the cross-model leaderboard, appended
```

## Running

```powershell
& .\.venv\Scripts\python.exe .\Model_Inference_Tests\run_inference_tests.py
```

With no arguments that scores `tests/output/smallTrainingTestV3/checkpoints/best/epoch-0033.pt`
under `configs/smallTrainingTestV3.yaml`. Both are overridable:

```powershell
& .\.venv\Scripts\python.exe .\Model_Inference_Tests\run_inference_tests.py `
    --checkpoint tests\output\smallTrainingTestV3\checkpoints\durable\epoch-0040.pt `
    --config configs\smallTrainingTestV3.yaml
```

Useful flags:

| flag | effect |
|---|---|
| `--list` | print the discovered tests and exit; loads nothing |
| `--only SUBSTRING` | run only matching tests (repeatable) |
| `--skip SUBSTRING` | skip matching tests (repeatable) |
| `--raw` | score the raw final-step weights instead of the EMA weights |
| `--n-replays N` | how many held-out replays to draw windows from (0 = all 23) |
| `--n-windows-per-replay N` | windows per replay (0 = all) |
| `--max-examples N` | global cap, applied by even striding |
| `--fog-rate R` | fixed fog for every example; default `config.eval.fog_rate` |
| `--device cpu` | stay off a GPU that is busy |
| `--option NAME=VALUE` | per-test knob (repeatable), see each test's module docstring |

Every test also honours its own window budget via `--option`, e.g.
`--option build_order_windows=100` or `--option ce_noise_levels=0.25,0.5,1.0`.

## Which split is scored

The **test** split, always. The runner derives it from the config's split rule and
cross-checks it against the run's recorded `metrics/replay_selection.json`; if the
two disagree it refuses to run rather than reporting "held-out" numbers on
replays the model may have trained on. For `smallTrainingTestV3` that is the 23
replays outside the 870 train / 50 dev selection.

Windows come from the run's existing window manifest — the replays are not
re-tokenized, so a test never disagrees with what training saw.

## Output layout

```
output/
  model_comparison_bits_per_token.csv    shared, appended, one row per model scored
  smallTrainingTestV3-epoch-0033__2026-Aug-26_02-41PM/
    SUMMARY.md                 what ran, headline numbers, what each test measures
    summary.json               machine-readable: provenance + every test's metrics
    <test name>/
      console.log              everything that test printed
      ...its artifacts...
```

The directory name is `<run>-<checkpoint>__<date>`. A second run in the same
minute gets a `-2` suffix rather than overwriting the first.

## The one-number model comparison

`model_comparison_leaderboard` is the test to run when the question is "which of
my models is best", not "how does this model behave". It scores held-out
**bits per token** — the unweighted cross entropy per scored canvas position, and
the same quantity the training loop writes into `epoch_metrics.csv` — then
appends one row to the shared CSV:

```
output/model_comparison_bits_per_token.csv
```

That file is the only artifact in this package that outlives a single run.
Every invocation adds a row; nothing is ever replaced. Each row carries the
score, the architecture that produced it (`arch_shape`, `params_millions`, the
toggles), the checkpoint, the `run_datetime` it was measured at, and the full
evaluation condition.

Lower is better. `effective_token_choices` (= `2 ** bits_per_token`) is the same
number read as "how many equally-likely tokens is the model effectively choosing
between" — 1.0 means it already knows, 291 (the vocabulary size) means it is
guessing.

Each row carries that pair twice, side by side:

| column | condition | what it answers |
|---|---|---|
| `bits_per_token` / `effective_token_choices` | the run's training t-distribution | did training progress — comparable to the run's own `epoch_metrics.csv` column |
| `bits_per_token_t_1.00` / `effective_token_choices_t_1.00` | t = 1.0, a canvas that is entirely noise | **how well the model does the job it is actually asked to do at inference** |

The second pair is the one to lead with when comparing understanding across
models. The deployed sampler starts from a fully-noised canvas, so t = 1.0 is the
condition the model really faces; the training t-distribution is dominated by
lightly-corrupted examples whose canvas tokens are already correct, which makes
the headline pair look far better than the model's grasp of the game warrants.
Expect a large gap between the two — that gap is the measurement, not a defect.

t = 1.0 is always scored, even if you override `leaderboard_t_grid` to exclude
it, so those two columns are never blank.

`fraction_of_uniform_prior_removed_t_1.00` says how much of the naive
`log2(291) = 8.18` bits cost the model removes at that condition — the honest
version of "how much has it learned". Both prior-comparison columns carry the
`_t_1.00` suffix because computing them at the headline t-distribution produces a
~99% figure for any half-trained model and separates nothing.

**Re-scoring one checkpoint under one condition gives bit-identical numbers.**
That takes more than a fixed seed: `SC2DiffusionDataset` redraws fog every time a
window is served, so the test builds a *fresh loader for every pass* to keep each
one at serving 0. Without that, a level's score would depend on how many passes
ran before it, and two runs with different `leaderboard_t_grid` values would
disagree on a level they share.

That column is deliberately **not** called perplexity. The arithmetic is
identical, and it plays the same role in a comparison, but perplexity is the
exponentiated per-token negative log-likelihood of a sequence under a model that
factorises that sequence's probability — which a discrete-diffusion denoiser does
not do. Calling it perplexity would invite lining it up against published LM
perplexity numbers, and that comparison is not valid.

Three things to know before ranking two rows:

- **`eval_condition_key` must match.** It hashes the split, fog condition, window
  budget, seed, vocabulary, the whole t-schedule, and `window_selection_key`
  (which windows were scored, not just how many) — everything that changes
  *which positions get scored*. Rows with different keys were not measured on the
  same thing and must not be ranked against each other. Sort by this column
  first, then by `bits_per_token` within it.
- **`bits_per_token` is taken at the run's own configured t-distribution**, so it
  is only strictly comparable between models sharing a `diffusion.schedule`.
  The `_t_1.00` pair and `bits_per_token_uniform_t` (which averages a *hardcoded*
  t = 0.25/0.5/0.75/1.0 grid) are both schedule-independent — reach for those
  when the `t_*` columns of two rows disagree. The remaining
  `bits_per_token_t_*` columns carry the rest of the curve.
- **Nothing here is an LLM's next-token perplexity.** Both columns are
  conditioned on the canvas corruption process, so they compare your models to
  each other and to the `unigram_entropy_baseline` floor, and to nothing outside
  this project.

Re-running the same checkpoint deliberately appends another row rather than
replacing the old one — the date-time is what separates them. When redundant
repeats pile up, sort by `dedupe_key`: it is equal exactly when the model,
architecture, condition, and score are all identical, so the duplicates land
together and the extras can be deleted.

## The tests

| test | uses the model | applies to |
|---|---|---|
| `canvas_reconstruction_figures` | yes (iterative sampler) | any checkpoint |
| `build_order_accuracy` | yes (iterative sampler) | any checkpoint |
| `heldout_canvas_cross_entropy` | yes (one forward pass per batch) | any checkpoint |
| `outcome_position_probe` | yes (one forward pass per batch) | any checkpoint |
| `noise_recovery_sweep` | yes (one forward pass per batch per level) | any checkpoint |
| `unigram_entropy_baseline` | **no** — data only, CPU | any run profile |
| `debut_report_and_timelines` | yes (iterative sampler) | **debut fine-tuned only** — skipped otherwise |
| `model_comparison_leaderboard` | yes (one forward pass per batch per level) | any checkpoint |

`debut_report_and_timelines` declares `REQUIRES_DEBUT_FINETUNE = True`. On a
pre-training checkpoint (`data.debut_mode = false`) the runner skips it and says
why, because debut metrics on pre-training weights produce numbers that look fine
and mean nothing.

Run `--list` for each test's full description and the exact files it writes.

## Cost

The two sampler-bound tests dominate. On an RTX 3070 with the V3 model, one
window through the full iterative sampler is roughly 18 seconds, so
`build_order_accuracy` at its default 40 windows is ~12 minutes. The forward-pass
tests are seconds per batch. `--only` plus a small `--option *_windows=N` is the
way to iterate quickly.

## Adding a test

Drop a `test_NN_<name>.py` into `Test_Scripts/`. It must define:

```python
TEST_NAME = "short_slug"              # also its output subdirectory name
TEST_TITLE = "One-line human title"
TEST_DESCRIPTION = "What question this answers."
TEST_OUTPUTS = ("`file.json` -- what it holds", ...)
USES_MODEL = True                     # False for a data-only baseline
REQUIRES_DEBUT_FINETUNE = False       # True gates it to fine-tuned checkpoints

def run(context: TestContext) -> TestResult: ...
```

Write only inside `context.out_dir`. The single exception in this package is
`model_comparison_leaderboard`, whose whole purpose is a file that accumulates
across runs; do not add a second one without the same justification. Get the
model with
`context.shared.model()`, held-out examples with `context.shared.examples(...)`,
and batches with `context.shared.dataloader(...)` — all memoized, so the
checkpoint is read from disk once per run regardless of how many tests ask.

Prefer wrapping existing package code over reimplementing it. Every test here is
a thin selector-and-writer around `thesis_ml.eval`, `thesis_ml.inference`,
`thesis_ml.viz`, or `thesis_ml.train`; that is what keeps these numbers
comparable to the ones the training pipeline reports.
