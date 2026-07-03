# tests Contract

## Purpose

- Own the pytest suite, owner-provided extractor fixtures, and the thin Windows launchers that exercise local training profiles.

## Ownership

- `test_*.py` own package regression coverage (config, serialization, windowing, dataset, model, training, sampler, eval, pipeline, fine-tune report, launcher checks).
- `fixtures/` owns owner-provided sample extractor parquet (`match_*_game_state.parquet`); it is the ground truth for schema-dependent tests.
- `overfit.bat`, `smallTrainingTest.bat`, and `overfit-fine-tune.BAT` are thin launchers; training behavior stays owned by YAML and the Python entry points.
- `output/` holds per-run launcher artifacts and console logs (generated; not durable contract material).

## Local Contracts

- Run tests through `.venv\Scripts\python.exe -m pytest -q` after confirming the venv exists.
- `overfit.bat` launches `configs/local_overfit_v2.yaml` and mirrors flushed progress to its terminal and `tests/output/overfitV2/console.log`.
- Launchers forward extra CLI args, so `--max-steps N` gives a bounded launch check. CUDA-required profiles must fail before preprocessing when CUDA is unavailable.
- Fixtures are the schema authority for tests; do not hardcode field names that contradict them or `SCHEMA.md`.
- GPU/VRAM claims require an environment where CUDA is visible; never infer VRAM from a CPU run.

## Work Guidance

- Add a focused test module beside the subpackage it covers rather than expanding an unrelated one.
- Keep launcher behavior thin: new training behavior belongs in YAML and Python, not in the `.bat` files.

## Verification

- `.venv\Scripts\python.exe -m pytest -q` is the package-wide check; launcher wiring is covered by `test_windows_launchers.py`.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
