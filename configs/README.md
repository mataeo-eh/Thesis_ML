# configs/

Configuration files (YAML/JSON/TOML) for data pipelines, model hyperparameters,
and experiment settings. Keep configs declarative and version-controlled so every
experiment is traceable to the exact settings that produced it.

- `local_overfit.yaml`: V1 baseline for the seeded 25-replay memorization run, retaining its five-epoch relative-improvement patience and default loss weights.
- `local_overfit_v2.yaml`: clean V2 comparison with isolated checkpoint/log/cache paths, `[PAD]` loss weight 0.2, early stopping disabled, and the full 200-epoch schedule.
- `local_full.yaml`: eight-epoch full-corpus local pipeline-validation profile.

The local profiles use equal 4096 input/canvas budgets with a 0.5 reconstruction fraction and share persisted clean replay artifacts. Manifests carry a semantic/config stamp and are rebuilt when these rules change.
