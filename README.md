# Thesis_ML


## License & attribution

Released under the **Apache License 2.0** (see [`LICENSE`](LICENSE)). This is a
permissive open-source license: anyone may use, modify, and redistribute the work,
**provided they preserve attribution** to the original author. See [`NOTICE`](NOTICE)
for the required attribution notice.

> Copyright 2026 Mataeo John Anderson.

## Repository layout

This is a uv workspace holding three independently installable packages. The split
exists so the two model architectures under comparison can be trained on provably
identical data while living in different deep-learning frameworks.

```
Thesis_ML/
├── packages/
│   ├── thesis-shared/      # Framework-agnostic core — NO torch, NO tensorflow.
│   │                       #   tokenization, vocabulary, windowing, features,
│   │                       #   standardization stats, splits, decode, metrics.
│   │                       #   Both arms import this; it is the controlled constant.
│   ├── thesis-diffusion/   # Arm A — uniform discrete diffusion  (PyTorch)
│   └── thesis-ar/          # Arm B — autoregressive baseline     (TensorFlow)
├── config/          # Canonical default configuration
├── configs/         # Versioned run profiles overriding the default
├── data/            # Datasets (raw/processed). Large files are git-ignored.
├── notebooks/       # Exploratory analysis & experiment write-ups (Jupyter)
├── experiments/     # Reproducible experiment configs, runs, and results
├── tests/           # Unit and integration tests
├── pyproject.toml   # Workspace root: members, tooling config
├── LICENSE          # Apache License 2.0 (verbatim)
└── NOTICE           # Required attribution notice
```

Dependencies flow one way: each arm depends on `thesis-shared`, and neither arm
depends on the other.

## Getting started

The two arms are installed into **separate** virtual environments and are never
installed together — a single Python process loads one CUDA runtime, so PyTorch's
and TensorFlow's bundled CUDA libraries conflict.

```bash
# Arm A (discrete diffusion, PyTorch) + shared core:
uv sync --extra dev

# Arm B (autoregressive, TensorFlow) + shared core, in its own environment:
uv venv .venv-tf
uv pip install -e packages/thesis-shared -e packages/thesis-ar
```

TensorFlow has had no native-Windows GPU support since 2.11, so Arm B is CPU-only
on Windows. That is expected: locally it is used for lightweight verification only,
and all real Arm B training runs on Linux cloud compute (install the `cuda` extra
there for `tensorflow[and-cuda]`).

See [`RUN.md`](RUN.md) for the full run procedure.

## Thesis framing


**Research Question**
- Can self-supervised learning and the proper pre-training objective train a generative model with a rich enough representation of SC2 to perform discriminative tasks.

**Hypothesis**
- Self-supervised learning with the pre-training objective of training a generative model to identify if anything is missing from a game-state snapshot, executed as % corruption via omitted tokens to approximate fog-of-war, predict what is missing, and then predict future game states based on what has already been observed, will train a generative model with a rich enough representation of the feature space that the model will be able to be fine-tuned to perform a discriminative task such as predicting opponent build order/strategy.

**Experimental design — two-arm controlled comparison**
- The experiment compares the discrete-diffusion model (Arm A) against a similarly sized **autoregressive** model (Arm B), matched on non-embedding parameter count and put through the same pre-training and post-training regime.
- Everything outside the architecture is held constant, and that is enforced structurally: both arms import the same `thesis-shared` package for tokenization, windowing, feature construction, standardization statistics, replay splits, and the build-order metrics they are scored on.
- Arm A is implemented in PyTorch; Arm B is implemented in TensorFlow. The arms are compared on shared task metrics, never on each other's loss — diffusion `perplexity` is computed over a sampled corruption distribution and is not the same quantity as autoregressive next-token perplexity.

**ML Architecture — Arm A (discrete diffusion)**
- Uniform-state multinomial discrete diffusion with a dense, full-bidirectional Gemma 4-lineage transformer. The project adopts DiffusionGemma's uniform corruption, expected-embedding self-conditioning, dense GeGLU/sandwich-RMSNorm mechanics, and nonmonotonic entropy-bounded sampler while retaining one full output canvas and a clamped input instead of block-autoregressive KV-cache conditioning.
- Self-supervised pretraining receives an EOS-terminated fogged observed-game-state input and jointly reconstructs omitted enemy past/present state plus whole-timestep future continuation on the output canvas.
    - | [clamped input tokens] [EOS] | [clamped BOS] [uniformly noised output canvas] |
- Input embeddings combine location-agnostic entity-token identity with allowlisted input-only map position, unit statistics, and allegiance. Sequence position uses Llama 3.1-style frequency-scaled RoPE; absolute time and frame-derived values never enter the model.
- Ground truth begins with clamped `[BOS]` at canvas position 0 and a perspective-relative `[WIN]` or `[LOSS]` target at position 1, followed by the normal canvas body. The outcome slot remains eligible for corruption and sampling; uniform noise can replace it only with `[PAD]`, `[DELIMITER]`, or a content token, never by injecting a random outcome or boundary token.
- Absorbing `[MASK]` diffusion remains a configuration-selectable scientific ablation, not the production default.

**Evaluation**
- Evaluate the model's discriminative ability by measuring accuracy, recall, precision, and F1score against a held-out test set of replays with ground-truth strategy/build order labels.
