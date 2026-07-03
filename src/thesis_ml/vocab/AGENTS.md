# vocab Subpackage Contract

## Purpose

- Own the single shared input/output vocabulary: raw entity-type content tokens plus the reserved special tokens of `SPEC.md` §4.

## Ownership

- `content_vocab.py` owns content-token identity and lookup (`ContentToken`, `ContentVocabulary`, `normalize_content_name`, `load_content_vocabulary`, `build_content_vocabulary`).
- `special_tokens.py` owns the reserved special-token constants (`[MASK]`, `[PAD]`, `[END]`, `[DELIMITER]`, `[WIN]`, `[LOSS]`, and their IDs).

## Local Contracts

- One vocabulary is shared by input and output. Content tokens are raw entity-type tokens and carry no spatial information of any kind.
- `[MASK]` is the absorbing noise state and is never a content target. `[PAD]` is a real content token that surplus canvas positions denoise into.
- `[WIN]`/`[LOSS]` are reserved from day one so embeddings exist, but are used only in outcome fine-tuning (`SPEC.md` §8).
- The vocabulary contains no tokens for coordinates, frame numbers, or absolute times.
- Concrete content-token contents derive from the extractor schema documented in `SCHEMA.md`; do not assume field names ahead of it.

## Work Guidance

- Add new special tokens by extending `special_tokens.py` and reserving IDs, not by overloading existing tokens.

## Verification

- Vocabulary behavior is exercised through `tests/test_serialization.py` and dataset/model tests; there is no dedicated vocab test module.

## Child DOX Index

- No child `AGENTS.md` files currently exist.
