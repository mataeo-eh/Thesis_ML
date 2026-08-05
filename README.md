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
- Uniform-state multinomial discrete diffusion with a dense, full-bidirectional Gemma 4-lineage transformer. The project adopts DiffusionGemma's uniform corruption, expected-embedding self-conditioning, dense GeGLU/sandwich-RMSNorm mechanics, and nonmonotonic entropy-bounded sampler while retaining one full output canvas and a clamped input instead of block-autoregressive KV-cache conditioning.
- Self-supervised pretraining receives a fogged observed-game-state input and jointly reconstructs omitted enemy past/present state plus whole-timestep future continuation on the output canvas.
    - | [clamped input tokens] | [uniformly noised output canvas] |
- Input embeddings combine location-agnostic entity-token identity with allowlisted input-only map position, unit statistics, and allegiance. Sequence position uses Llama 3.1-style frequency-scaled RoPE; absolute time and frame-derived values never enter the model.
- Ground truth begins with a perspective-relative `[WIN]` or `[LOSS]` token followed by the normal canvas body. The sampler applies no position-specific restriction: the model learns the position-zero outcome convention and refines it jointly with the strategy roll-out.
- Absorbing `[MASK]` diffusion remains a configuration-selectable scientific ablation, not the production default.

**Evaluation**
- Evaluate the model's discriminative ability by measuring accuracy, recall, precision, and F1score against a held-out test set of replays with ground-truth strategy/build order labels.
