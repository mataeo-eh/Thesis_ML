"""thesis_diffusion - the uniform discrete-diffusion arm (PyTorch).

Arm A of the thesis comparison. Owns the bidirectional dense backbone, canvas
corruption and diffusion objective, the training loop and its metrics, iterative
denoising sampling, checkpoint/export handling, and read-only checkpoint
diagnostics. All learnable machinery here is PyTorch.

Framework-agnostic concerns (tokenization, windowing, features, splits, decode,
build-order metrics) are NOT defined here -- they are imported from
`thesis_shared` so that Arm B consumes byte-identical data and is scored by
identical metrics.

See `AGENTS.md` in this directory for the binding contract.
"""

__version__ = "0.0.1"
