"""thesis_ar - the autoregressive baseline arm (TensorFlow).

Arm B of the thesis comparison: a decoder-only autoregressive transformer trained
on the same replay corpus, the same splits, and the same token vocabulary as the
diffusion arm, sized to match it in non-embedding parameters.

STATUS: scaffold. No model code is implemented yet.

FRAMEWORK: TensorFlow / Keras, exclusively. This package must never import
`torch`, and must never import from `thesis_diffusion`. Everything it shares with
the diffusion arm comes from `thesis_shared`, which is framework-agnostic.

See `AGENTS.md` in this directory for the binding contract, including the
local-execution rule: no meaningful training runs on the owner's Windows machine
(TensorFlow is CPU-only on native Windows), and all real training happens on
Linux cloud compute.
"""

__version__ = "0.0.1"
