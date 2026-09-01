# Running Thesis_ML

This project is intentionally packaged as `git clone` + `uv sync` + one command. No Dockerfile is used.

## Local Setup

```bash
git clone <repo-url>
cd local-play-bootstrap-main/Thesis_ML
uv sync --extra dev
```

The project pins `torch` to PyTorch's official CUDA 13.0 wheel index through
`tool.uv.sources`; `uv sync` therefore installs a CUDA-enabled build rather
than silently selecting the CPU wheel. Verify a local GPU environment with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Run a tiny smoke training pipeline:

```bash
uv run thesis-ml-train --config config/default.yaml --smoke
```

Acquire replay data independently:

```bash
uv run thesis-ml-acquire --config config/default.yaml
```

Run training against configured parquet data:

```bash
uv run thesis-ml-train --config config/default.yaml
```

Run the local proof-of-life profiles:

```bash
uv run thesis-ml-train --config configs/local_overfit.yaml
uv run thesis-ml-train --config configs/local_overfit_v2.yaml
uv run thesis-ml-train --config configs/local_full.yaml
```

The current full-corpus V3 run is launched on Windows with
`tests\smallTrainingTestV3.bat`; its console log, metrics, cache, and checkpoint
families are isolated below `tests\output\smallTrainingTestV3\`. The V3 profile
uses six-row microbatches and seven-batch accumulation (42 windows and roughly
275k valid tokens per optimizer step), with a 6.5 GiB reclaim-first reservation
ceiling for the 8 GiB RTX 3070.

Run the parameter-count/capacity suite with:

```bat
tests\SizeAblationTest.bat
```

It runs 005m, 015m, the current 030m baseline, the deeper 030m control, 060m,
and 120m in that order. Re-running the launcher skips validated finished arms
and resumes the first interrupted arm from its own rolling checkpoint; a failed
or ambiguous arm stops the sequence before any larger model starts. Use
`tests\SizeAblationTest.bat --dry-run` to print the decisions without launching
training. Generated status, metrics, logs, and checkpoints live below
`tests\output\SizeAblationTest\` and remain ignored.

On Windows, equivalent thin launchers write console output and run artifacts to
`tests\output\overfitV2\` and `tests\output\smallTrainingTestV2\`:

```bat
tests\overfit.bat
tests\smallTrainingTestV2.bat
```

Both launchers forward extra CLI arguments, so `--max-steps 1` provides a
bounded launch check. The local profiles require a CUDA-enabled PyTorch build.

For a bounded full-pipeline smoke that still exercises preprocessing, loading,
checkpoint save, and resume, add `--max-steps N`. The first non-smoke launch
builds or incrementally refreshes the config-owned tokenized replay artifacts
and window manifest before training. Production runs also require the configured
feature-statistics artifact; set `pipeline.prepare_feature_statistics: true`
explicitly for the run that should compute it from the selected train replay
artifacts, then return the switch to `false` to reuse and verify that artifact.

## Configuration

All input/output locations are in `config/default.yaml`:

- `storage.data_uri`: processed parquet dataset location.
- `storage.raw_uri`: raw replay staging location.
- `storage.checkpoint_uri`: checkpoint output and resume location.
- `storage.log_uri`: pipeline log output.
- `storage.local_cache_dir`: local staging directory used when a URI is remote.
- `data_source.*`: replay source and extractor wrapper settings.
- `pipeline.auto_acquire`: if true, training invokes acquisition when processed data is absent.
- `data.input_budget_tokens` / `data.canvas_budget_tokens`: hard per-window bounds.
- `data.canvas_recon_fraction`: maximum canvas share used by in-window enemy reconstruction; the remainder is reserved for whole future timesteps.
- `data.tokenized_replay_dir` / `data.window_manifest_path`: preprocessing outputs.
- `data.feature_statistics_path`: deterministic normalization statistics derived
  only from selected training replay artifacts; its identity is bound to
  checkpoints, sampling, diagnostics, and finished exports.
- `pipeline.prepare_feature_statistics`: explicit permission to compute or
  replace the feature-statistics artifact; default `false` fails if it is absent.
- `fog.rate_distribution`: one enemy-content omission rate sampled per example
  for the clamped input; `power: 2` is a scaled `Beta(2,1)` draw favoring high omission.
- Debut fine-tuning uses a separate manifest: input windows tile whole
  timesteps under `input_budget_tokens` only, while each output runs from its
  input-window start to replay end or the canvas budget and may overlap the
  neighboring output horizon.
- `pipeline.replay_subset_size`: seeded training-replay subset (`0` means all).
- `pipeline.train_replay_count`: when positive, use an exact count split with this many train replays, `validation_replay_count` dev replays, and every remainder in test; `0` keeps fraction-based splitting.
- `train.epochs`: used when `train.max_steps` is `0`.
- `train.schedule_horizon_epochs`: `0` fits LR/EMA to the actual run; a positive
  value keeps those schedules on that many epochs while `train.epochs` remains
  the stop limit. The capacity suite uses 50 and stops after epoch 3.
- `train.lr_schedule`: `wsd`, `cosine`, or `linear`; all use the configurable fixed `train.warmup` optimizer-step count, while WSD separately configures its final decay ratio and floor and fills the intervening steps with its stable phase.
- `train.accumulation_steps`: microbatches per optimizer step; epoch-derived horizons count optimizer steps.
- `train.early_stopping_patience_epochs`: consecutive dev-loss epochs below the relative-improvement threshold before stopping (`0` disables).
- `train.early_stopping_min_relative_improvement`: relative improvement required to reset patience.
- `train.max_cuda_reserved_gb`: reclaim-first CUDA allocator reservation ceiling (`0` disables). Reaching it releases unused cached blocks and aborts only if the post-trim reservation remains at or above the limit; the overfit profile uses 7.5 GiB and full-corpus V3 uses 6.5 GiB.
- `model.gradient_checkpointing`: recompute transformer blocks during backward to bound saved-activation memory; enabled for the local overfit profile after measured RTX 3070 spillover.

`s3://bucket/prefix` is supported for data, checkpoints, and logs through the same resolver as local paths. AWS credentials must come from the normal AWS environment/instance profile chain.

Kaggle credentials must come from environment variables named by config:

```bash
export KAGGLE_USERNAME=<username>
export KAGGLE_KEY=<api-key>
```

## AWS Recipe

Data acquisition is CPU-bound and training is GPU-bound, so run them separately.

1. Launch a small EC2 CPU instance for acquisition.
2. Install `uv`, clone the repo, and run `uv sync`.
3. Set `KAGGLE_USERNAME` and `KAGGLE_KEY`, or point config at an alternate source.
4. Set `storage.raw_uri` and `storage.data_uri` to persistent S3 prefixes.
5. Run:

```bash
uv run thesis-ml-acquire --config config/default.yaml
```

For training:

1. Launch an EC2 GPU instance with an AWS Deep Learning AMI so CUDA/drivers are already present.
2. Install `uv`, clone the repo, and run `uv sync`.
3. Point `storage.data_uri`, `storage.checkpoint_uri`, and `storage.log_uri` at S3 prefixes.
4. Run:

```bash
uv run thesis-ml-train --config config/default.yaml
```

Spot instances are safe to use: every `train.checkpoint_interval` optimizer steps the run
overwrites and uploads `resume/last.pt` below `storage.checkpoint_uri`, and on startup it
pulls that file back before falling back to a local checkpoint. A
fresh replacement instance pointed at the same S3 prefix therefore resumes where
the preempted one left off, losing at most one checkpoint interval. Best-dev
epoch checkpoints and durable epoch milestones occupy separately configured
`best/` and `durable/` subdirectories. Set
`train.keep_step_checkpoints: true` to also retain timestamped `step-N.pt`
snapshots (otherwise only the rolling `last.pt` is kept, so disk/S3 stays flat).

### Monitoring a long run

Per-step metrics (optimizer-step mean loss, cumulative `epoch_loss_so_far`,
per-class losses, learning rate, corruption fraction, step wall time, tokens/sec,
CUDA memory telemetry, and periodic held-out validation) are appended to the step
JSONL and uploaded to `storage.log_uri` on the checkpoint cadence. The terminal
progress line prints the current and cumulative epoch loss without repeating the
CUDA telemetry; the final cumulative value exactly matches the epoch CSV
`train_loss`. CUDA attention is restricted to
Flash or memory-efficient SDPA, so an incompatible mask fails instead of
falling back to quadratic-memory math attention. Tail or parse the file to
track a multi-day run and abort early if the loss curves go wrong. A reproducible
train/dev/test split (config `pipeline.split_seed` / `test_fraction` /
`dev_fraction`, split over whole replays to avoid leakage) drives the in-training
validation; the test split is held out for final evaluation.

Local profiles also write `epoch_metrics.csv` with epoch train/dev loss,
train/dev per-class losses, p50/p90/p95 input and future horizon lengths,
train/dev future-token loss bucketed by prediction distance, cumulative token
counts, tokens/sec, and cumulative wall-clock elapsed time.

Each split's headline loss is followed immediately by that split's token-level
metrics: `accuracy` and `macro_f1` for the `ground_truth_preserved` and `noised`
canvas states, then `bits_per_token` and `perplexity`. Read the `noised` figures
as the denoising work proper and the `ground_truth_preserved` ones as whether the
model leaves already-correct tokens alone; the two are reported separately
because a single pooled accuracy would largely reflect how corrupted the sampled
examples happened to be. `macro_f1` gives every token id one vote regardless of
frequency, so it drops when a model buys accuracy by collapsing onto the common
tokens. `bits_per_token` is the plain UNWEIGHTED cross entropy per scored
position, which is why it is not `log2` of the loss column, and `perplexity` is
`2 ** bits_per_token`. Those two are comparable across your own models whenever
the splits and the corruption schedule match, but they are not the same quantity
as a language model's next-token perplexity. The overfit profiles
record their seeded 25 train and three disjoint dev replay IDs in
`replay_selection.json` beside that CSV. `tests\overfit.bat` launches the V2
profile, which uses an isolated output/checkpoint namespace, downweights
`[PAD]` to 0.2, and runs all 200 epochs unless manually stopped.

### Throughput knobs

The real training DataLoader uses `pipeline.num_workers` background loader
processes with `pipeline.prefetch_factor` batches prefetched each, plus pinned
memory on CUDA. Each worker memory-maps only the tokenized replay artifact needed
for its current window; the corpus is never loaded into RAM. Inputs and canvases
are padded only to their batch maxima and carry exact attention/loss masks. The
overfit profile uses four persistent workers with four batches prefetched per
worker. Workers build model features, then omit raw record/metadata object graphs
from training batches before IPC; the custom batch pins its tensors and transfers
them to CUDA non-blockingly. Debut targets scan memory-mapped token ids and
materialize only emitted events, and replay outcome JSON is cached per worker.
Worker persistence is config-owned. `local_full.yaml` keeps the same persistent
four-worker, four-prefetch feed path and releases unused CUDA cache after completed
epochs. Step telemetry distinguishes current allocation, lifetime peak allocation,
reserved allocator memory, inactive split memory, device-wide VRAM use, and the
device-minus-reserved gap. Epoch CSV rows average the last two measurements across
optimizer steps so drift outside PyTorch's caching allocator is visible.
CSV writes retry transient Windows locks. If a viewer or sync process keeps the
configured epoch/interval CSV locked, training copies all readable history plus
the current row into a timestamped `*-continued-*.csv`, logs the redirection, and
uses that continuation for later rows, publishing, and process resumes.
Checkpoints written mid-fit store the live cumulative wall-clock total. A
replacement process restores that value as its baseline, so later epoch and
interval rows measure the whole resumed run rather than restarting elapsed time
from zero.

Production stochastic training is paired by base seed, epoch, and manifest
index. The shuffled example order, fog rate/omissions, diffusion time,
corruption positions/replacements, and self-conditioning decision are therefore
repeatable across restarts and across capacity arms even though their
microbatch sizes differ. This pairs the data/noise realization; it does not
claim bitwise-identical CUDA arithmetic across different model shapes.

### Publish a finished training summary

Raw launcher output under `tests\output\` remains ignored because it includes
large checkpoint families, step-level logs, and caches. After a run creates its
`checkpoints\finished\finished_metadata.json`, prepare a compact tracked evidence
bundle with:

```powershell
& .\.venv\Scripts\python.exe .\scripts\prepare_training_report.py `
  tests\output\smallTrainingTestV3
```

The command accepts only a completed or early-stopped run, validates the stamped
architecture identity, extracts first/best/final epoch facts, renders the loss
curve, and copies only small textual evidence into
`reports\training-runs\<date>-<run-name>\`. It never copies `.pt` or
`.safetensors` weights, the step JSONL, console logs, or caches. Invoke the
repository's `training-run-summary` skill in Codex or Claude with the same run
directory to write the short chair-facing `SUMMARY.md` from that evidence.

`configs/local_full.yaml` runs the full debut/outcome task for eight epochs with
870 train replays, 50 dev replays, and all 23 remaining quickstart replays held
out for test. It writes the normal step/epoch telemetry plus
`finetune_report.json` against the true test split.

### Pre-flight GPU smoke test

Before a long run, confirm the full-size model fits and trains on the target GPU
and measure peak VRAM + per-step time (no dataset required — it fabricates a
correctly-shaped random batch):

```bash
uv run python scripts/gpu_smoke_test.py --batch-size 1 --input-len 2048 --steps 5
```

Pass `--vocab-size` matching your real vocabulary for an accurate parameter
count, and raise `--batch-size` until VRAM headroom runs out to find the largest
micro-batch your GPU supports.

### Checkpoint diagnostics

Render held-out replay diagnostics with the EMA checkpoint weights:

```powershell
.\.venv\Scripts\python.exe -m thesis_ml.viz.diagnostics --checkpoint <last.pt> --replay-dir <features-dir> --out-dir <output-dir>
```

The default output is PNG/SVG figures plus a combined PDF. Add `--csv` to write
one side-by-side comparison CSV per example
(`sequenceindex,modelprediction,groundtruth,correct`) lining the predicted and
ground-truth canvases up position-by-position. Add `--json` to write `canvas_logits.json` with the final-canvas top-10
raw logits and softmax confidence values at every sequence position. These
exports are independent and are not created when their flags are omitted.

## Extractor Wrapper

The acquisition command wraps the separate `SC2-gamestate-extractor` repository. By default it runs the configured command from `data_source.extractor_path`, passing:

```text
--process-replay-directory <raw_uri-or-cache> --output <data_uri-or-cache> --workers <workers>
```

The extractor itself remains the source of truth for replay parsing and parquet production.
