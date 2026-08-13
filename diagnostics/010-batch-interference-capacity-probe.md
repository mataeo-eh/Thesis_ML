# 010 — Batch-versus-batch interference: is the model capacity-limited?

**Date:** 2026-08-12
**Status:** observation complete; recommendations scoped, none implemented
**Artifacts:** `scripts/output/batch_interference/` (per-arm CSVs, console logs,
provenance JSON)
**Method:** `scripts/batch_interference_probe.py`, driven by
`scripts/run_batch_interference_probe.sh`

## Question

`tests/output/memorization/README.md` finding A closed on a puzzle it could not
resolve: turning off weight decay and fog bought only ~9% on the like-for-like
loss classes and left train loss at 0.106, "nowhere near the ~0 that actual
memorization of 10 replays would produce." Regularization was not the cap. The
open question was what is.

One candidate is model capacity. A capacity-limited model cannot hold every
training batch at once, so an optimizer step that improves one batch must
degrade others — the batches compete for the same parameters. A model with room
to spare shows no such tension: a step either helps or is neutral elsewhere.

That is directly measurable on a finished checkpoint, and this probe measures it.

## Method

For each arm, from its own finished `last.pt` (global_step 3400, LR at the
9.0e-6 decay floor):

1. Freeze one epoch of 34 training batches AND one corrupted view of each —
   fixed diffusion timestep `t`, fixed noise draw, fixed self-conditioning row
   mask. Every loss in the probe is measured on these same frozen views, so the
   only thing that ever varies is the weights.
2. Record baseline loss on all 34 views.
3. For each batch i: restore the checkpoint exactly, take ONE real optimizer step
   on batch i (restored Adam moments, the run's own LR, its bf16 precision, its
   gradient clipping), then re-measure loss on ALL 34 batches.
4. Record `delta = pre - post`. **Positive means the step helped that batch.**

34 steps and 1,156 measured losses per arm; ~60 minutes per arm on one RTX 3070.

Nothing is written back into the probed runs: the probe's `TrainingLoop` is
constructed with no metrics paths and no publishers, and never calls `fit`,
`save_checkpoint`, `scheduler.step`, or the EMA update.

### Why the observations are trustworthy

- **Restore is exact.** After the final step the probe restores and re-measures
  the entire baseline; all three arms reported
  `restore_max_abs_drift = 0.000e+00`. Every step began from bit-identical
  weights, so no delta inherited a previous step's state.
- **Measurement is deterministic.** The probe aborts unless a batch evaluated
  twice on untouched weights returns the identical float. All three arms passed.
- **Resolution is adequate.** A single step at the 9.0e-6 LR floor moves loss by
  ~1e-4, which bf16 (~3 decimal digits on a loss of 0.1) would quantize away, so
  measurement runs in fp32 while the step keeps the run's configured precision.
- **Arm 02 is the control.** The same measurement on a model at 4x the loss
  produces cooperation rather than competition (below), so the arm 01 result is
  not an artifact of the probe or of single-step noise.

## Observations

| arm | baseline loss | self delta | other-batch mean | others hurt | steps raising total loss | self-gain cancelled (median) |
|---|---|---|---|---|---|---|
| 01 no-reg | 0.1213 | +7.03e-4 (34/34) | -6.65e-6 | 50.4% | 11/34 | +43.8% |
| 02 t=1 x0.25 | 0.4924 | +1.12e-3 (34/34) | +9.95e-5 | 39.8% | 0/34 | -314.3% |
| 03 both | 0.4672 | +8.73e-4 (34/34) | -1.32e-5 | 45.8% | 12/34 | +57.4% |

**A. Every step improves its own batch, in every arm — 102 of 102.** The arms
differ entirely in what that improvement costs everything else.

**B. Arm 01 is competitive.** Its off-diagonal deltas are a coin flip (50.4%
hurt, 1,122 cells, none exactly zero) with a net-negative mean. The median step
gives back 43.8% of its own gain as collateral damage, and for 11 of 34 batches
a step on that batch RAISES the total loss summed over the whole training set.
The extreme case is batch 23: a step there earns +2.29e-5 for itself and costs
the other 33 batches a combined -3.88e-4.

**C. Arm 02 is cooperative, at 4x the loss.** Not one of its 34 steps raises
total training loss. The median step delivers roughly 4x more benefit to the
other 33 batches combined than to its own (`cancelled` is -314%, i.e. amplified
rather than cancelled). This is what a model with unfit shared structure looks
like: steps generalize instead of competing.

**D. At matched loss scale, removing regularization is what re-introduces the
competition.** Arms 02 and 03 sit at nearly the same baseline (0.492 vs 0.467)
and differ only by weight decay and fog. Arm 02: +9.95e-5 mean on others, 0/34
net-negative steps. Arm 03: -1.32e-5 mean, 12/34 net-negative steps, median 57.4%
of self-gain cancelled. Removing the regularizers did not unlock memorization —
it converted cooperative updates into competitive ones.

**E. The memorized batches are what gets spent.** Mean delta on batch j from the
other 33 batches' steps, split by how well fit batch j already was:

| arm | easiest 10 | % hurt | hardest 10 | % hurt |
|---|---|---|---|---|
| 01 no-reg | -2.71e-5 | 64.5% | -3.99e-6 | 51.5% |
| 02 t=1 x0.25 | +1.16e-4 | 34.2% | +6.68e-5 | 41.5% |
| 03 both | -5.83e-5 | 48.2% | +1.92e-5 | 39.7% |

In arm 01 the best-fit batches absorb 6.8x more damage than the hardest ones and
are hurt far more often. Arm 03 makes the transfer explicit: the easiest batches
lose (-5.83e-5) while the hardest ones gain (+1.92e-5). Arm 02 inverts it
entirely — there the easiest batches are the biggest beneficiaries.

**F. The competition is diffuse, not pairwise.** In arm 01,
`corr(delta[i,j], delta[j,i]) = +0.14` over 561 pairs, with mutually
antagonistic (26.6%) and mutually reinforcing (25.7%) pairs almost equally
common. There is no small set of rival batches to point at; the interference is
spread across the parameter set.

## Interpretation (hypothesis, not established)

The observations are consistent with arm 01 having exhausted the capacity it can
bring to bear on this corpus: at loss 0.106 it can still improve any individual
batch on demand, but only by spending fit it already had elsewhere, and it spends
it preferentially from the batches it had memorized best.

Two things this does NOT establish, and the evidence that would settle them:

- **Parameter count is not implicated by these data.** "Capacity" here is
  whatever bounds the model's ability to hold all 34 batches at once, which
  includes width/depth but also the input bottleneck that
  `tests/output/memorization/README.md` finding B already pointed at. Re-running
  this probe across a `model.d_model` / `model.layers` sweep would separate them:
  if interference falls as the model grows, capacity is the binding constraint.
- **The single-step reading may not describe the training trajectory.** These are
  one-step counterfactuals at a decayed LR. Whether the same competition governs
  a full epoch is a different measurement — repeating the probe at an earlier
  checkpoint (e.g. step 340 and 1700) would show whether competition sets in as
  the run converges or was present throughout.

## Scoped recommendations

None of these are implemented; all are the owner's call.

1. Re-run the probe across a model-scale sweep to test whether interference is
   capacity-bound (above). This is the highest-value follow-up and needs no new
   machinery.
2. Re-run at earlier checkpoints of the existing arms to date the onset of
   competition. The checkpoints for this exist only if `keep_step_checkpoints`
   was set; otherwise this requires a re-run.
3. Treat "remove regularization to enable memorization" as closed. Finding D
   shows it changes the character of the updates rather than the achievable
   floor, which is consistent with finding A of the memorization sweep.

## Reproduce

```bash
bash scripts/run_batch_interference_probe.sh
```
