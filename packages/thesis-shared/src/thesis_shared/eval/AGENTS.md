# thesis_shared.eval Contract — Shared Scoring

## Purpose

- Own build-order extraction and the precision/recall/F1 metrics **both arms are scored on**. This is the comparison's measuring instrument, so it lives outside either arm by construction.

## FRAMEWORK CONTRACT

- No `torch`, no `tensorflow`. Inputs are decoded token sequences and ground-truth records, not framework tensors.

## Ownership

- `buildorder.py` owns build-order event extraction from ground-truth parquet and from decoded predictions.
- `metrics.py` owns `BuildOrderMetrics`, `compare_build_orders`, and `aggregate_metrics`.

## Local Contracts

- **Both arms are compared on these metrics, not on each other's loss.** The diffusion arm's `bits_per_token`/`perplexity` are computed over a sampled corruption distribution; an autoregressive model's perplexity is a next-token quantity. The two are not the same measurement and must never be placed side by side as if they were.
- A metric change re-scores both arms. Never add a metric in an arm that is meant to compare arms — add it here.
- Extraction must treat a decoded prediction and a ground-truth record through the same code path, so a scoring difference can never come from which side produced the sequence.

## Work Guidance

- Arm-side reporting that merely *calls* these metrics (assembling a report, writing files, wiring a harness) belongs in that arm's `eval/`, not here.

## Verification

- Changes require `tests/test_eval.py`.
- Re-run both arms' reports after any metric change; a shared-metric edit invalidates previously recorded scores for both.
