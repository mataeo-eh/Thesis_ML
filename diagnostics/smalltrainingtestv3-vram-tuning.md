# smallTrainingTestV3 RTX 3070 VRAM tuning

## Scope

- Date: 2026-08-13.
- Baseline source: commit `21ef21b` with the owner-started `smallTrainingTestV3` outputs.
- Hardware: NVIDIA GeForce RTX 3070, 8,192 MiB, Windows WDDM.
- Data/model path: the real full-corpus V3 manifest and replay artifacts, 4,096 input and canvas budgets, 29,318,720 parameters, BF16 autocast, frozen input K/V, 50% training self-conditioning, block activation checkpointing, ten persistent workers, and four-batch worker prefetch.
- Each arm used the production training pipeline and deterministic V3 data order. Generated checkpoints, metrics, caches, and probe YAML stayed under ignored `tests/output/vramBench/`.

## Accumulation audit

`TrainingLoop.fit` does not retain a sequence of forward graphs. For each microbatch it computes loss, immediately calls `backward()` on `loss / number_of_microbatches`, and retains only detached scalar metrics. It performs one optimizer/scheduler/EMA update after the configured microbatches, then clears gradients with `zero_grad(set_to_none=True)`.

The console logger is intentionally one optimizer step behind GPU launch. Seeing batches 1-10 before the first `step=1` line does not mean step 1 accumulated ten batches: batches 1-5 form step 1, batches 6-10 launch step 2, and only then does the host finalize and print step 1.

## Measurements

The baseline values are the four completed steps from the owner-started run. Probe arms ran three cold-start steps; the selected 6x7 arm then resumed through 20 total steps. Token counts are reconstructed from persisted `tokens_per_second * step_wall_seconds` and exclude batch-shape padding.

| Microbatch x accumulation | Windows/update | Mean valid tokens/update | Warm valid tokens/s | Peak allocated | Max reserved | Max device use |
|---|---:|---:|---:|---:|---:|---:|
| 9 x 5 baseline | 45 | 295,616 | 10,571 | 5.470 GiB | 7.262 GiB | 8.000 GiB |
| 7 x 6 | 42 | 277,164 | 13,480 | 4.344 GiB | 5.641 GiB | 6.891 GiB |
| 6 x 7 selected | 42 | 275,440 | 13,707 | 3.845 GiB | 4.980 GiB | 6.083 GiB |
| 5 x 9 | 45 | 298,109 | 13,805 | 3.293 GiB | 4.061 GiB | 5.164 GiB |

The selected arm's valid-token range was 262,233-284,804 per update. After startup, data waits were approximately 1-2 ms while compute took about 19-20 seconds, so the optimized run remained GPU-compute-bound rather than exposing a loader bottleneck.

## Diagnosis

The 29.3M parameter state is not itself the dominant peak. Model, EMA, Adam moments, and gradients total about 559 MiB in persistent FP32 state; sequence activations and temporary workspaces dominate the live peak.

The baseline's post-step live allocation returned to about 0.50 GiB, proving accumulated forward graphs were not retained. Its 5.47 GiB run peak was below physical capacity, but the allocator reserved 7.262 GiB and non-PyTorch Windows/display use consumed the remaining gap, taking device use to exactly 8.000 GiB. WDDM then used shared system memory. The throughput collapse was therefore the physical-memory knee caused by activation/workspace demand plus allocator reservation, not a malformed accumulation loop.

## Adopted profile

- `pipeline.batch_size: 6`.
- `train.accumulation_steps: 7`.
- Effective update: 42 windows and about 275k valid tokens on the measured slice, close to the requested roughly 45 windows/280k tokens.
- `train.max_cuda_reserved_gb: 6.5` as a reclaim-first ceiling. The healthy path reserves about 5.0 GiB, so the guard does not trim normally; it provides protection if later dynamic shapes or fragmentation grow reservation toward the 8 GiB physical limit.
- Model dimensions, parameter count, loss scaling, BF16 precision, self-conditioning, frozen input K/V, activation checkpointing, optimizer, and token budgets remain unchanged.

The 5x9 arm provides more headroom but needs two extra forward/backward microbatches per update and processes closer to 300k tokens. The 7x6 arm has no sustained throughput advantage over 6x7 and leaves about 1 GiB less device headroom. The selected 6x7 arm is therefore the best measured balance of target update size, throughput, and long-run memory margin.

## Final verification

- Live merged-config construction: batch 6, accumulation 7, 42 windows/update, 6.5 GiB ceiling, 291 vocabulary IDs, and 29,318,720 trainable parameters.
- Final isolated canonical-profile GPU launch, three optimizer steps: 3.799 GiB peak allocation, 4.803 GiB reservation, 5.906 GiB total device use, and 13.8k-14.2k warm valid tokens/s. The ceiling did not trigger.
- Focused config/model/training/parameter-count tests: 112 passed, 1 skipped.
- Full package suite: 250 passed, 1 skipped.
- Architecture diagram regenerated and visually checked; `git diff --check` passed.
