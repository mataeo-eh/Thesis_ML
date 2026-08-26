# smallTrainingTestV3 Training Summary

> **Chair takeaway:** This full-corpus uniform-diffusion run stopped through configured early stopping after 43 of 50 epochs; development loss improved substantially before reaching its best value at epoch 33, then modestly regressed.

## Run at a glance

| Measure | Result |
|---|---:|
| Stop condition | Early stopping at epoch 43 of 50 |
| Best development loss | 0.189912 (epoch 33) |
| Final development loss | 0.207617 (epoch 43; +0.017705 vs. best) |
| Final training loss | 0.196584 |
| Recorded fit time | At least 230.0 hours (9.58 days) across four timing segments |

## Interpretation

This run trained the current 29.3M-parameter, 12-block uniform-state discrete-diffusion transformer on the configured full-corpus split (870 training and 50 development replays), using BF16, six-row microbatches, seven-step accumulation, and the WSD schedule. Its stamped architecture identity, `dense-multinomial-SC2-v2+frozen_input_kv`, matches the current [model architecture](../../../Model_Architecture/MODEL_ARCHITECTURE.md). The result is therefore interpretable against the repository’s active model definition rather than a retired configuration.

Measured development cross-entropy fell from 0.512085 in epoch 1 to 0.189912 at epoch 33, a reduction of 0.322173 (62.9%). Training loss also fell from 1.071460 to 0.196584 by the final epoch. The [training curve](TRAINING_CURVES.png) shows rapid early improvement followed by a low-loss plateau and ordinary epoch-to-epoch variation. The final development loss was 0.207617, so it regressed by 0.017705 from the best epoch; the early-stopping outcome is consistent with retaining the best development result rather than treating the last epoch as best.

The strict best-development checkpoint is `epoch-0033.pt`. Separately, the completed run exported both final raw and EMA weights; EMA is the default serving choice. Those final exports record the epoch-43 state and must not be conflated with the best-development checkpoint. These are token-reconstruction measurements, not evidence of downstream gameplay or strategy quality.

## Limitation and next step

The evidence preparer excluded three malformed trailing metric rows (CSV lines 45–47), while retaining 43 valid epoch rows; the run also did not record a source commit. In addition, three legacy resume resets divide the timing data into four segments: the 827,882.602 seconds (at least 230.0 hours, 9.58 days) is a recorded-fit-time lower bound, and exact end-to-end duration is unavailable. The compact evidence otherwise supports the reported trajectory, but it contains no held-out gameplay evaluation. Next, evaluate the default EMA model and the epoch-33 best-development checkpoint on the held-out test split with the repository’s sampling and task metrics, reporting the comparison alongside these [run facts](RUN_FACTS.json) and [epoch metrics](evidence/epoch_metrics.csv).
