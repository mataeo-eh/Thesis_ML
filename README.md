# Thesis_ML

Thesis-level ML/AI architecture and framing.

This repository is the foundation for a graduate thesis exploring [working title —
fill in your research question]. It is developed as a standalone, open-source
project and is also consumed as a git submodule inside the `SC2_DBN` monorepo.

## License & attribution

Released under the **Apache License 2.0** (see [`LICENSE`](LICENSE)). This is a
permissive open-source license: anyone may use, modify, and redistribute the work,
**provided they preserve attribution** to the original author. See [`NOTICE`](NOTICE)
for the required attribution notice.

> Copyright 2026 Mataeo John Anderson.

## Repository layout

```
Thesis_ML/
├── src/thesis_ml/   # Core library: models, data pipelines, training loops
├── data/            # Datasets (raw/processed). Large files are git-ignored.
├── notebooks/       # Exploratory analysis & experiment write-ups (Jupyter)
├── experiments/     # Reproducible experiment configs, runs, and results
├── configs/         # Hyperparameter / pipeline configuration files
├── tests/           # Unit and integration tests
├── pyproject.toml   # Package metadata & dependencies
├── LICENSE          # Apache License 2.0 (verbatim)
└── NOTICE           # Required attribution notice
```

## Getting started

```bash
# From the repo root, create/activate a virtual environment, then:
pip install -e .
```

## Thesis framing

> _Use this section to articulate the research question, hypotheses, the proposed
> ML/AI architecture, evaluation methodology, and how each module above maps to a
> chapter or contribution of the thesis._
