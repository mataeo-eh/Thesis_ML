# configs/

Configuration files (YAML/JSON/TOML) for data pipelines, model hyperparameters,
and experiment settings. Keep configs declarative and version-controlled so every
experiment is traceable to the exact settings that produced it.

- `local_overfit.yaml`: seeded 25-replay memorization profile with three disjoint held-out dev replays, batch size 10, four persistent workers with four-way prefetch, five-epoch relative-improvement patience, fused-only CUDA attention, activation checkpointing, and a 7 GiB reserved-memory ceiling, capped at 200 epochs.
- `local_full.yaml`: eight-epoch full-corpus local pipeline-validation profile.

Both profiles extend `config/default.yaml`, use equal 4096 input/canvas budgets with a 0.5 reconstruction fraction, and share persisted clean replay artifacts. Manifests carry a semantic/config stamp and are rebuilt when these rules change.
