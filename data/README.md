# data/

Datasets for the thesis. Split into:

- `raw/` — original, immutable source data (never edited in place).
- `processed/` — cleaned/feature-engineered data derived from `raw/`.

Actual data files are **git-ignored** (see `.gitignore`) because they may be large
or non-distributable. Only this README and the directory structure are tracked.
Document data provenance and any download/preprocessing steps here.

Token dictionary artifacts derived from extractor output must exclude
engine-created ability pseudo-entities that the extractor marks untracked,
including `kd8charge` and creep tumor variants.
