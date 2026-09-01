# tests Contract

## Purpose

- Own the pytest suite, owner-provided extractor fixtures, and the thin Windows launchers that exercise local training profiles.

## Ownership

- `test_*.py` own package regression coverage (config, serialization, windowing, dataset, model, training, sampler, eval, pipeline, fine-tune report, launcher checks). `test_diffusion_integration.py` owns bounded real-model checks spanning corruption/loss/backward, self-conditioning, sampling, and checkpoint transfer between training and inference.
- `test_canvas_unigram_baseline.py` owns the closed-form entropy/weighted-optimum baseline and exact equivalence with production canvas-loss mask selection and normalization.
- `test_training_report.py` owns the finished-run report preparer's validation, metric summarization, architecture matching, and artifact allowlist.
- `fixtures/` owns owner-provided sample extractor parquet (`match_*_game_state.parquet`); it is the ground truth for schema-dependent tests.
- `overfit.bat`, `smallTrainingTestV2.bat`, `smallTrainingTestV3.bat`, and `overfit-fine-tune.BAT` are thin launchers; training behavior stays owned by YAML and the Python entry points.
- `SizeAblationTest.bat` is the thin Windows adapter to `scripts/run_size_ablation.py`. The driver owns ascending arm order and restart/finish bookkeeping only; every training knob stays in `configs/size_ablation_*.yaml`.
- `test_gpu_smoke_script.py` keeps the standalone full-size GPU benchmark fixture aligned with the production EOS/BOS/outcome grammar without requiring a GPU.
- `run_ablation_sweep.sh` is a thin sequential launcher for the representational-toggle ablation sweep (`configs/ablation_0*.yaml`). It owns run ORDER and restart bookkeeping only; every training knob stays in YAML. It is restartable: each arm's own `last.pt` `global_step` is the resume state, so a finished arm is skipped without launching Python and an interrupted arm resumes through `train_pipeline._try_resume`. An unreadable checkpoint is skipped with a warning rather than silently retrained.
- `output/` holds per-run launcher artifacts and console logs (generated; not durable contract material). If an epoch/interval CSV is persistently write-locked by a viewer or sync process, training emits a full-history timestamped `*-continued-*.csv` beside it and continues there.
- `output/` remains ignored even when selected evidence is published. Durable copies are created only below `reports/training-runs/` by `scripts/prepare_training_report.py`.

## Local Contracts

- Run tests through `.venv\Scripts\python.exe -m pytest -q` after confirming the venv exists.
- `overfit.bat` launches `configs/local_overfit_v2.yaml`, mirrors flushed progress to its terminal and `tests/output/overfitV2/console.log`, and archives each finished run's log as `console-<timestamp>.log` because Tee-Object truncates `console.log` at every launch.
- `smallTrainingTestV3.bat` launches the current full-corpus profile and keeps console, metrics, checkpoints, and cache under `tests/output/smallTrainingTestV3/`.
- Launchers forward extra CLI args, so `--max-steps N` gives a bounded launch check. CUDA-required profiles must fail before preprocessing when CUDA is unavailable.
- The size-ablation driver skips an arm only when its finished metadata/config match the live three-epoch arm, treats any rolling checkpoint as resumable without loading its large tensors merely to probe progress, and stops on a failure or ambiguous finished export before attempting a larger model.
- `run_ablation_sweep.sh` passes NO `--max-steps`. Each arm's length is owned by `train.epochs: 100` in `configs/local_overfit_v2.yaml`, which every arm extends, so all arms and the baseline share one 3400-step budget (34 steps/epoch) and each runs its config-derived schedules to completion: the linear LR decay reaches its `0.03 × 3.0e-4 = 9.0e-6` floor at the last step and the EMA window (10% of the run = 340 steps) turns over ~10 times. Because no cap is passed, `train_pipeline` treats each arm as a proper finish and writes its `finished/` raw+EMA export alongside `last.pt`. The driver's `STEPS_PER_EPOCH` / `EXPECTED_EPOCHS` shell constants are used ONLY to turn a checkpoint's `global_step` into a skip / resume / fresh decision, never as a cap, and `tests/test_config.py` pins them to the loaded config so they cannot desync.
- Fixtures are the schema authority for tests; do not hardcode field names that contradict them or `SCHEMA.md`.
- Serialization coverage must include variable-width instance IDs, slash-form fractions, sentinel-null entity presence, valid-zero masks, facing sine/cosine, cloak/buff categorical values, direct model-input parity between rich and optimized paths, and artifact/manifest invalidation when source or vocabulary identity changes.
- GPU/VRAM claims require an environment where CUDA is visible; never infer VRAM from a CPU run.
- Diffusion coverage must exercise both process values without mixing their semantics: uniform corruption/all-valid-except-BOS loss/nonmonotonic full-renoising EB by default, and absorbing corruption/masked inverse-time loss/monotonic EB as the ablation. Tests must prove BOS clamping, position-1 outcome corruption eligibility, the restricted uniform replacement support, exact EB prefix math, categorical RNG reproducibility, adaptive stop conjunction, and checkpoint incompatibility.

## Work Guidance

- Add a focused test module beside the subpackage it covers rather than expanding an unrelated one.
- Keep launcher behavior thin: new training behavior belongs in YAML and Python, not in the `.bat` files.

## Verification

- `.venv\Scripts\python.exe -m pytest -q` is the package-wide check; launcher wiring is covered by `test_windows_launchers.py`.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
