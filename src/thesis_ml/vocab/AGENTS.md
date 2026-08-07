# vocab Subpackage Contract

## Purpose

- Own the single shared input/output vocabulary: raw entity-type content tokens plus the reserved special tokens of `SPEC.md` §4.

## Ownership

- `content_vocab.py` owns content-token identity and lookup (`ContentToken`, `ContentVocabulary`, `normalize_content_name`, `load_content_vocabulary`, `build_content_vocabulary`).
- `special_tokens.py` owns the reserved special-token constants (`[MASK]`, `[PAD]`, `[END]`, `[DELIMITER]`, `[WIN]`, `[LOSS]`, and their IDs).

## Local Contracts

- One vocabulary is shared by input and output. Content tokens are raw entity-type tokens and carry no spatial information of any kind.
- `[MASK]` is the absorbing-ablation noise state and is never a content target. Uniform corruption, prior sampling, renoising, and categorical candidate sampling exclude it. `[PAD]` is a real semantic canvas token for surplus positions.
- `[WIN]`/`[LOSS]` are targets in both pretraining and outcome fine-tuning. Ground truth places one at canvas position zero, but the sampler imposes no positional token restriction.
- The vocabulary contains no tokens for coordinates, frame numbers, or absolute times.
- Concrete content-token contents derive from the extractor schema documented in `SCHEMA.md`; do not assume field names ahead of it. Engine-created ability pseudo-entities that the extractor marks untracked, including `kd8charge` and creep tumor variants, must not be content tokens.

## Work Guidance

- Add new special tokens by extending `special_tokens.py` and reserving IDs, not by overloading existing tokens.
- Any vocabulary-ID, content-token, state-space, or maximum-ID change must update `../../../Model_Architecture/MODEL_ARCHITECTURE.md` in the same change, including embedding/head shapes, parameter totals, logits, corruption/sampler state space, memory arithmetic, and caveats; then update the canonical `.mmd` labels and regenerate its SVG/PNG. Use `UPDATE_PROMPT.md` and recompute rather than editing one number in isolation.

## Verification

- Vocabulary behavior is exercised through `tests/test_serialization.py` and dataset/model tests; there is no dedicated vocab test module.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
