# Evaluation Harness

Ground-truth build order is extracted directly from the project's parsed parquet rows. This keeps evaluation aligned with the model's own training data rather than with external build-order tools that may timestamp the command issue time instead of map appearance time.

## Oracle Representation

The prediction-side extractor operates on timestep-bucketed entity-type counts, the same representation produced by the model decoder. The ground-truth extractor operates on the parsed parquet rows and reduces them to the same event representation:

- one timestep bucket equals `sampling_interval_s`;
- each timestep is a mapping of `entity_type -> count`;
- positions, exact frames, coordinates, and resource values are ignored because the model cannot emit them.

The build order is the ordered multiset of appearance events:

```text
(entity_type, appearance_bucket)
```

For parsed parquet ground truth:

- each unit/building entity emits one event at the first row where that specific entity instance appears;
- each upgrade emits one event at the first row where that upgrade appears in the player's upgrade list;
- entity positions and exact frames are ignored.

For decoded model predictions, entity instance IDs do not exist, so each positive count increase emits that many new events. For example, counts `marine: 2` then `marine: 5` emit two marine events in the first bucket and three additional marine events in the second bucket.

Events are sorted by `(bucket, entity_type)` for deterministic comparison.

## Truncated Horizons

When an example has a truncated horizon, the final timestep is dropped from both prediction and ground truth before extraction. This follows `SPEC.md` §7: the final timestep may be partial when the canvas fills exactly without `[END]`.

## Metrics

Matching is entity-type exact with configurable bucket tolerance:

```text
abs(predicted_bucket - ground_truth_bucket) <= timing_tolerance_buckets
```

Each predicted event can match at most one ground-truth event, and each ground-truth event can match at most one predicted event.

- `precision = true_positive_matches / predicted_event_count`
- `recall = true_positive_matches / ground_truth_event_count`
- `f1 = 2 * precision * recall / (precision + recall)`
- `accuracy = true_positive_matches / ground_truth_event_count`

Accuracy is therefore the fraction of ground-truth build-order events correctly predicted at the model's resolution. Token cross-entropy is not an evaluation metric and is not reported by the harness.

Default `timing_tolerance_buckets` is defined in `config/default.yaml`.
