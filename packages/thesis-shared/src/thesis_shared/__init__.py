"""thesis_shared - framework-agnostic core shared by both experiment arms.

This package holds everything that MUST be identical between the discrete-diffusion
arm (`thesis_diffusion`, PyTorch) and the autoregressive arm (`thesis_ar`,
TensorFlow) for the comparison between them to mean anything: configuration
loading, tokenization/serialization, the content vocabulary, replay windowing,
input feature construction and standardization statistics, the train/dev/test
replay split, canvas decoding, timing recovery, build-order extraction, and the
build-order metrics both arms are scored with.

Nothing in this package may import `torch` or `tensorflow`. It depends only on the
scientific-Python stack (numpy/pandas/pyarrow/PyYAML). That restriction is what
lets the same code be installed next to either framework, including in a
TensorFlow-only environment that never has PyTorch present.

See `AGENTS.md` in this directory for the binding contract.
"""

__version__ = "0.0.1"
