# Sequence formats contract

## Purpose

- Own framework-neutral presentation of serialized replay records for both experimental arms. The first implemented new path is an inspection-only joint pretraining preview; neither training pipeline consumes it yet.

## Ownership

- `bpe.py`: deterministic content-ID BPE for the preview, exact decoding, ranked merge provenance and expanded vocabulary export. It has no framework dependency.

- `types.py`: immutable tokens and whole-window results; owner/timestep fields are inspection metadata, not model features.
- `common.py`: perspective validation, structural tokens, and canonical per-owner record selection.
- `unconditioned_joint.py`: `[SELF] content [ENEMY] content [DELIMITER]` per timestep, including empty player blocks, and a separate four-token outcome-footer builder.
- `preview.py`: one-replay loading and readable exports. `scripts/preview_sequence.py` is its thin CLI.
- `../data/window_policies/unconditioned_joint.py`: contiguous whole-timestep budgeting, one leading BOS/delimiter per window, terminal END only at actual replay end.
- `../vocab/sequence_vocabulary.py`: explicit format selection and ownership-token overlay without shifting legacy IDs.

## Local Contracts

- Preview `tokenizer: atomic|bpe` is selectable; atomic remains the default, including older preview YAMLs. `bpe_min_occurrences: 3` stops fitting when no current adjacent pair occurs at least three times. Counts include overlapping pairs; replacements are left-to-right non-overlapping. Frequency ties use ascending token-ID pairs. There is no vocabulary-size cap.
- All canonical content IDs, including entities and upgrades/research, may merge with each other. Structural IDs (PAD/MASK, BOS/EOS, outcomes, ownership and delimiters) are hard boundaries; owner/timestep changes also break spans. Compound IDs append after the full base vocabulary. Base IDs and structural tokens never change.
- Fit on the complete selected replay once, both owners in one perspective, before re-windowing compressed whole timesteps with the existing policy. The threshold applies at merge selection, not to final compound usage after subsequent merges. A learned merge may be applied even to a rare span during encoding.
- BPE exports under `scripts/output/sequence_preview/bpe/` include the re-windowed first sequence, `before/`, `same_window_bpe/`, `comparison.json`, and `bpe_vocabulary.json` (all base IDs, ordered merges, atomic expansions, and tokenizer identity). Compare identical spans separately from increased window coverage. Full-replay counts include one prefix/footer/END, not repeated window overhead. Verify exact token/owner/timestep decoding across the entire replay.
- Tokenizer identity includes base vocabulary, allowed IDs, threshold and ordered merge rules; a vocabulary-name hash alone cannot identify BPE semantics. This replay-fitted vocabulary is in-sample debugging evidence only. Full fitting/tokenization/windowing/manifests belong on cloud compute; production fitting must use training replays only after replay splitting and freeze the tokenizer for dev/test and both arms. Cloud execution and training adapters are not implemented by this preview.

- New preview defaults are declared in `config/sequence_preview.yaml`; missing/unknown format fails before replay I/O. Existing conditioned training remains in its original entry points. Selecting the conditioned format in the new preview fails explicitly rather than silently producing a joint sequence.
- Reuse `serialize_snapshot`: preserve entity-presence rules, upgrades, ordering, and all recorded timesteps. No fog, feature MLP inputs, time features, or conditioned input region in joint pretraining.
- Emit one `[SELF] self-outcome [ENEMY] enemy-outcome` footer per window, after its last timestep delimiter. All four positions remain scored. The joint vocabulary spells existing ID 5 `[LOSE]`; the conditioned vocabulary retains `[LOSS]`. Swap footer ownership and outcome together when changing perspective.
- Joint inspection windows are unpadded, at most the declared budget; reserve four footer positions plus END at replay end within that budget (minimum budget 10). Never split a timestep. Oversized individual timesteps fail explicitly.
- The first BOS position is clamped and unscored; all other emitted positions are scored. The intended joint diffusion policy allows every vocabulary ID as replacement, including MASK/BOS/EOS. It is recorded in preview metadata but not yet integrated into the production corruption/prior/sampler.
- Metadata features must not be supplied to either arm during unconditional pretraining. Reintroducing them requires an explicit observed-input or generated-feature design. AR teacher-forcing does not make unavailable inference-time metadata legitimate input.
- No inference-time rule forces complementary predictions: training scores both ownership markers and both outcomes so the model can learn the relationship. Recorded label validation is data validation only. The second outcome can use the first under causal teacher forcing; report that distinction when interpreting forecasting accuracy. Outcomes never precede entity targets in a window.
- Never generate manifests, statistics, model weights, or processed replay arrays from the preview. Write only below `scripts/output/` and include no absolute paths in exports.
- Production integration still requires explicit format/checkpoint identities, masks/padding, loss taxonomy, decoder, full-vocabulary prior/corruption/renoising consistency, and both arm adapters. Do not pass the joint overlay or its BPE expansion to the legacy pipeline.

- Text-only display: each merged token is one bracketed, fully expanded atomic-name list, e.g. `[scv scv marine]`. Delimiters introduce one-based replay timesteps (`[DELIMITER 1]`); the final delimiter labels replay/window end after the last timestep instead of inventing another timestep. CSV IDs, metadata and sequence vocabulary remain unchanged by display formatting.

## Verification

- `tests/test_sequence_bpe.py` covers frequency thresholds, overlapping counts, deterministic ties, every structural boundary, lossless metadata-preserving decoding, re-windowing, and config compatibility.

- `tests/test_sequence_preview.py` covers ID preservation, whole timesteps and END budgeting, perspective and outcome semantics, configuration errors, and framework-free imports.
- Run `tests/` from the existing project venv after shared changes.
