# thesis_shared.inference Contract — Shared Decode and Timing

## Purpose

- Own the post-generation steps that are identical regardless of how a sequence was produced: validating and decoding a canvas into entity records, and recovering external absolute game time.

## FRAMEWORK CONTRACT

- No `torch`, no `tensorflow`. Inputs are integer token sequences (numpy or plain Python), not framework tensors.

## Ownership

- `decode.py` owns canvas grammar validation and decoding to entity records.
- `timing.py` owns external absolute-clock recovery from the configured cadence.

## Local Contracts

- Decoding is deliberately agnostic to how the sequence was generated. The diffusion arm arrives here after iterative denoising and the autoregressive arm after left-to-right decoding; both must produce a sequence this module can validate without arm-specific branches. If a branch on "which arm produced this" seems necessary, that is a signal the grammar is being violated upstream.
- Timing recovery uses the same configured cadence as preprocessing. Absolute time is metadata and must never re-enter model inputs.
- Grammar validity is required, not advisory: a sequence that fails validation is a defect to diagnose, not a case to silently skip.

## Work Guidance

- Sampling and decoding *strategies* (nonmonotonic denoising, greedy/beam/top-k) are arm-owned. Only validation and decoding of the finished sequence belong here.

## Verification

- Changes require the decode coverage in `tests/test_eval.py` and `tests/test_sampler.py`.
- Confirm that importing these modules pulls neither `torch` nor `tensorflow` into `sys.modules`.
