# Thesis_ML


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


**Research Question**
- Can self-supervised learning and the proper pre-training objective train a generative model with a rich enough representation of SC2 to perform discriminative tasks.

**Hypothesis**
- Self-supervised learning with the pre-training objective of training a generative model to identify if anything is missing from a game-state snapshot, executed as % corruption via omitted tokens to approximate fog-of-war, predict what is missing, and then predict future game states based on what has already been observed, will train a generative model with a rich enough representation of the feature space that the model will be able to be fine-tuned to perform a discriminative task such as predicting opponent build order/strategy.

**ML Architecture**
- A discrete diffusion/ masked denoising backbone using a bidirectional, non-causal transformer. The reference family being the open Ermon-group work — SEDD / MDLM / LLaDA — with Mercury 2 as an existence proof in the realm of Language.
- Two self-supervised pretraining objectives on perfect-information replay data:
    - inter-timestep (given 1–n, predict n+1) and
    - intra-timestep (given prior timesteps plus a partial token set at n with % corruption tokens randomly omitted, reconstruct the full set at n). (determine if anything is missing, and what is missing)
        - | [input tokens] | [output tokens] |
- Pre-backbone pipeline: embedding lookup (entity token ID → vector, learned during pretraining), then contextual encodings added (timestep membership, map position, unit stats).
- Post-training is Supervised Fine Tuning (SFT) -> Strategy labels becomes additional tokens in the generative output stream
    - | [input tokens] | [strategy-label output tokens] | [normal output tokens] |
    - The model iteratively refines the strategy label alongside the future game-state roll-out.
    - The input collapses to a single type: dropping the dual inter-/intra-timestep input regime that only existed for pretraining and only passing the sequence of gamestates, but each gamestate in the sequence maintains its fog-of-war approximation of missing tokens.

**Evaluation**
- Evaluate the model's discriminative ability by measuring accuracy, recall, precision, and F1score against a held-out test set of replays with ground-truth strategy/build order labels.
