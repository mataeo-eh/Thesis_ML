# Analysis Scripts

Run scripts through the Thesis_ML virtual environment from the submodule root.

## Context-window estimator

```powershell
& .\.venv\Scripts\python.exe .\scripts\estimate_context_window.py
```

The script derives the default parquet location from the repository layout, so it does not embed or emit a machine-specific path. It streams parquet metadata plus the two upgrade columns and writes `scripts/output/context_window_estimate.json`.

Each sample is one full replay from one player perspective:

- Input counts all self content and zero-fog enemy content, plus one delimiter per player per timestep.
- Output counts all enemy content, plus one delimiter per timestep and one terminal `[END]` token.
- An entity contributes one token when its instance is present in a row. Each listed cumulative upgrade contributes one token in every row where it is present.
- Padding is excluded. Token statistics cover both perspectives for every replay; timestep statistics count each replay once. Both include minimum, maximum, mean, median, mode, and all tied modes.

Use `--input-dir`, `--pattern`, or `--output` to override defaults. Prefer repository-relative arguments.
