from dataclasses import replace
import csv
import json
import math
from functools import partial
from pathlib import Path
import time

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from thesis_ml.config import (
    ClassLossWeightsConfig,
    FogConfig,
    ProjectConfig,
    UniformDistributionConfig,
    load_config,
)
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.resumable_sampler import ResumableBatchSampler
from thesis_ml.data.dataset import (
    CLASS_CLAMPED,
    CLASS_DELIMITER,
    CLASS_END,
    CLASS_ENEMY_FOGGED,
    CLASS_ENEMY_FUTURE,
    CLASS_ENEMY_OBSERVED,
    CLASS_PAD,
    CLASS_WINLOSS,
    DEBUT_CLASS_ID_TO_NAME,
    PRETRAIN_CLASS_ID_TO_NAME,
    DatasetExample,
)
from thesis_ml.model.loss import RARE_CLASS_T_BUCKET_NAMES
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.train.corruption import corrupt_batch, inverse_t_weights, sample_uniform_noise
from thesis_ml.train.loop import (
    INTERVAL_REPORTS_PER_EPOCH,
    TrainingLoop,
    _accumulate_token_class_counts,
    _append_csv_row,
    _finalize_bits_per_token,
    _finalize_rare_class_t_bucket,
    _finalize_token_metrics,
    _latest_metrics_csv_path,
    _macro_f1_from_counts,
    auxiliary_confidence_loss,
    interval_boundaries,
    optimizer_steps_per_epoch,
)
from thesis_ml.train.train import _synthetic_input_records, make_synthetic_examples, run_smoke_train
from thesis_ml.vocab.special_tokens import (
    BOS_ID,
    CONTENT_TOKEN_OFFSET,
    DELIMITER_ID,
    END_ID,
    EOS_ID,
    LOSS_ID,
    MASK_ID,
    PAD_ID,
    WIN_ID,
)

# All fixtures in this file (except `_make_debut_synthetic_examples`, used only
# by the one debut-mode test below) use the restored pretraining input and
# seven-class taxonomy. `collate_diffusion_examples` requires an
# explicit `debut_mode` at every call site (Worker 3), so these two bound
# partials are the "which grammar is this batch built from" answer for the
# two fixture families in this file.
_collate_pretrain = partial(collate_diffusion_examples, debut_mode=False)
_collate_debut = partial(collate_diffusion_examples, debut_mode=True)


def test_smoke_train_loss_decreases_and_first_step_per_class_logs(tmp_path: Path) -> None:
    logs = run_smoke_train(max_steps=40, seed=17, checkpoint_dir=tmp_path / "smoke")

    first = logs[0]
    last = logs[-1]
    assert last.loss < first.loss

    # The smoke train is a pretraining run whose synthetic fixture carries one
    # label for every stable pretraining class id.
    examples = make_synthetic_examples(_small_config(tmp_path), count=1)
    expected_classes = {
        PRETRAIN_CLASS_ID_TO_NAME[int(label)]
        for label in examples[0].class_labels.unique()
        if int(label) >= 0
    }
    assert set(first.per_class) == expected_classes
    assert all(value > 0 for value in first.per_class.values())


def test_uniform_corruption_never_noises_input_region() -> None:
    config = _small_config()
    input_token_ids = torch.tensor([[100, 101, 102], [103, 104, 105]])
    target_canvas = torch.tensor([[100, 101, 102, 103], [104, 105, 106, 107]])
    generator = torch.Generator(device="cpu").manual_seed(1)

    for t in (0.0, 0.25, 0.75, 1.0):
        corrupted = corrupt_batch(
            input_token_ids=input_token_ids,
            target_canvas=target_canvas,
            process=config.diffusion.process,
            schedule=config.diffusion.schedule,
            vocab_size=128,
            generator=generator,
            t=t,
        )
        assert torch.equal(corrupted.input_token_ids, input_token_ids)
        assert not (corrupted.input_token_ids == 0).any()
        if t == 0.0:
            assert torch.equal(corrupted.noised_canvas, target_canvas)
        if t == 1.0:
            assert corrupted.corrupted_positions.all()
            assert (corrupted.noised_canvas != 0).all()


def test_t_one_oversampling_hits_configured_fraction_and_zero_disables_it() -> None:
    """Over many draws, `t_one_fraction` of examples get t forced to EXACTLY 1.0.

    With t=None (the real training path) and t_one_fraction=0.1, a fixed
    generator over >= 10,000 examples must produce an exact-t==1.0 fraction in
    [0.07, 0.13] (a generous band around the 0.1 Bernoulli mean). With
    t_one_fraction=0.0 the oversampling is fully disabled: no draw is exactly
    1.0 (torch.rand samples [0, 1), so the uniform path alone can never land
    exactly on 1.0).
    """

    config = _small_config()
    schedule = replace(config.diffusion.schedule, t_one_fraction=0.1)

    batch = 10_000
    target_canvas = torch.zeros((batch, 4), dtype=torch.long)
    input_token_ids = torch.zeros((batch, 0), dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(1234)

    corrupted = corrupt_batch(
        input_token_ids=input_token_ids,
        target_canvas=target_canvas,
        process="uniform",
        schedule=schedule,
        vocab_size=128,
        generator=generator,
        t=None,
    )
    exact_one_fraction = float((corrupted.t == 1.0).float().mean())
    assert 0.07 <= exact_one_fraction <= 0.13

    disabled_schedule = replace(schedule, t_one_fraction=0.0)
    generator = torch.Generator(device="cpu").manual_seed(1234)
    disabled = corrupt_batch(
        input_token_ids=input_token_ids,
        target_canvas=target_canvas,
        process="uniform",
        schedule=disabled_schedule,
        vocab_size=128,
        generator=generator,
        t=None,
    )
    assert int((disabled.t == 1.0).sum()) == 0


def test_per_epoch_reseed_makes_corruption_deterministic_and_epochs_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fit() reseeds the generator to base_seed+epoch at every epoch boundary.

    Three claims (the per-epoch reseed contract from train/loop.py):
      1. Two runs with the same seed produce IDENTICAL per-epoch corruption.
      2. Within one run, epoch 0's and epoch 1's branch draws DIFFER (the reseed is
         per-epoch, not a frozen repeat of one stream).
      3. Reseeding a fresh generator to base_seed + epoch reproduces that
         epoch's corruption stream exactly (this is what makes a mid-training
         resume replay the same draws an uninterrupted run would have made).
    """

    import thesis_ml.train.loop as loop_module

    def run_and_capture(seed: int) -> list[torch.Tensor]:
        """Run 2 epochs x 2 steps and record every corruption branch, in order."""

        captured: list[torch.Tensor] = []
        real_corrupt_batch = corrupt_batch

        def spy_corrupt_batch(**kwargs):
            output = real_corrupt_batch(**kwargs)
            captured.append(output.corrupted_positions.clone())
            return output

        monkeypatch.setattr(loop_module, "corrupt_batch", spy_corrupt_batch)
        config = _small_config(tmp_path)
        torch.manual_seed(seed)
        examples = make_synthetic_examples(config, count=4)
        loader = DataLoader(examples, batch_size=2, shuffle=False, collate_fn=_collate_pretrain)
        model = SC2StrategyDiffusionModel(config, vocab_size=128)
        loop = TrainingLoop(model=model, config=config, seed=seed)
        # fixed_t=None so corruption consumes the loop's own generator (the
        # subject of the reseed); 2 batches/epoch x 2 epochs = 4 steps.
        loop.fit(loader, max_steps=4, epochs=2)
        monkeypatch.setattr(loop_module, "corrupt_batch", real_corrupt_batch)
        return captured

    first_run = run_and_capture(seed=123)
    second_run = run_and_capture(seed=123)
    assert len(first_run) == len(second_run) == 4

    # 1. Same seed -> identical corruption branches across BOTH epochs.
    for first_mask, second_mask in zip(first_run, second_run, strict=True):
        assert torch.equal(first_mask, second_mask)

    # 2. Epoch 0 and epoch 1 draw DIFFERENT corruption branches.
    epoch_zero = torch.cat([first_run[0], first_run[1]], dim=0)
    epoch_one = torch.cat([first_run[2], first_run[3]], dim=0)
    assert not torch.equal(epoch_zero, epoch_one)

    # 3. manual_seed(base_seed + epoch) reproduces that epoch's stream: replay
    #    each epoch's two corruption draws with a fresh generator and get the
    #    same masks fit() recorded. (The batches are identical each epoch --
    #    shuffle=False over fixed synthetic data -- so only the generator
    #    stream distinguishes the epochs.)
    config = _small_config(tmp_path)
    torch.manual_seed(123)
    examples = make_synthetic_examples(config, count=4)
    loader = DataLoader(examples, batch_size=2, shuffle=False, collate_fn=_collate_pretrain)
    batches = list(loader)
    for epoch_index in (0, 1):
        replay_generator = torch.Generator(device="cpu").manual_seed(123 + epoch_index)
        for step_in_epoch, batch in enumerate(batches):
            replayed = corrupt_batch(
                input_token_ids=batch.input_token_ids,
                target_canvas=batch.target_canvas,
                process=config.diffusion.process,
                schedule=config.diffusion.schedule,
                vocab_size=128,
                generator=replay_generator,
                t=None,
            )
            recorded = first_run[epoch_index * 2 + step_in_epoch]
            assert torch.equal(replayed.corrupted_positions, recorded)
            # The shared epoch generator also supplies the independent
            # per-example self-conditioning gate after each corruption draw.
            torch.rand(batch.target_canvas.shape[0], generator=replay_generator)


def test_bos_is_clamped_while_outcome_position_one_is_noised_and_scored(tmp_path: Path) -> None:
    """BOS is a visible anchor; the adjacent outcome remains an ordinary target."""

    config = _small_config(tmp_path)
    loop, batch = _loop_and_batch(config, seed=93)
    result = loop.compute_batch_loss(batch, fixed_t=1.0)
    assert not result.corruption.corrupted_positions[:, 0].any()
    assert not result.scored_mask[:, 0].any()
    assert torch.equal(result.corruption.noised_canvas[:, 0], batch.target_canvas[:, 0])
    assert (batch.target_canvas[:, 0] == BOS_ID).all()
    assert (batch.class_labels[:, 0] == CLASS_CLAMPED).all()
    assert result.corruption.corrupted_positions[:, 1].all()
    assert result.scored_mask[:, 1].all()
    assert batch.class_labels[0, 1].item() == CLASS_WINLOSS
    assert "win-loss" in result.loss_output.per_class

    # Its loss weight is nonzero in both modes; semantic [PAD] is also scored.
    from thesis_ml.model.loss import CanvasCrossEntropyLoss

    pretrain_weights = CanvasCrossEntropyLoss(config).class_weights
    debut_weights = CanvasCrossEntropyLoss(_small_debut_config(tmp_path)).class_weights
    assert pretrain_weights[CLASS_WINLOSS].item() > 0.0
    assert debut_weights[CLASS_WINLOSS].item() > 0.0

    # At intermediate t, position 1 takes the normal iid corruption branch while
    # BOS remains exempt.
    # Bernoulli(t) draw: across many examples it is sometimes selected and sometimes not --
    # never always-exempt (and never always-forced).
    generator = torch.Generator(device="cpu").manual_seed(7)
    target_canvas = torch.tensor([BOS_ID, WIN_ID, 20, 21]).repeat(2_000, 1)
    target_canvas[1::2, 1] = LOSS_ID
    corrupted = corrupt_batch(
        input_token_ids=torch.zeros((2_000, 0), dtype=torch.long),
        target_canvas=target_canvas,
        process=config.diffusion.process,
        schedule=config.diffusion.schedule,
        vocab_size=128,
        generator=generator,
        t=0.5,
    )
    assert not corrupted.corrupted_positions[:, 0].any()
    win_rate = float(corrupted.corrupted_positions[0::2, 1].float().mean())
    loss_rate = float(corrupted.corrupted_positions[1::2, 1].float().mean())
    assert 0.4 <= win_rate <= 0.6
    assert 0.4 <= loss_rate <= 0.6

    # At terminal noise both outcome classes take the corruption branch and
    # necessarily change, because neither outcome ID belongs to the injected
    # noise-state support.
    terminal = corrupt_batch(
        input_token_ids=torch.zeros((2_000, 0), dtype=torch.long),
        target_canvas=target_canvas,
        process=config.diffusion.process,
        schedule=config.diffusion.schedule,
        vocab_size=128,
        generator=torch.Generator(device="cpu").manual_seed(8),
        t=1.0,
    )
    assert terminal.corrupted_positions[:, 1].all()
    assert terminal.changed_positions[:, 1].all()
    assert not terminal.noised_canvas[:, 1].eq(WIN_ID).any()
    assert not terminal.noised_canvas[:, 1].eq(LOSS_ID).any()


def test_uniform_training_scores_every_valid_canvas_position(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    loop, batch = _loop_and_batch(config, seed=9)

    result = loop.compute_batch_loss(batch, fixed_t=0.5)

    assert torch.equal(result.scored_mask, batch.canvas_loss_mask)
    assert result.scored_mask[result.corruption.corrupted_positions.logical_not()].any()
    semantic_pad = batch.class_labels == CLASS_PAD
    assert result.scored_mask[semantic_pad].all()
    assert result.canvas_logits.shape[1] == batch.target_canvas.shape[1]


def test_uniform_replacement_can_equal_target_and_is_seed_reproducible() -> None:
    config = _small_config()
    target = torch.ones((64, 8), dtype=torch.long)
    kwargs = {
        "input_token_ids": torch.empty((64, 0), dtype=torch.long),
        "target_canvas": target,
        "process": "uniform",
        "schedule": config.diffusion.schedule,
        "vocab_size": 9,
        "t": 1.0,
    }
    first = corrupt_batch(
        **kwargs, generator=torch.Generator(device="cpu").manual_seed(44)
    )
    second = corrupt_batch(
        **kwargs, generator=torch.Generator(device="cpu").manual_seed(44)
    )
    assert first.corrupted_positions.all()
    assert (first.noised_canvas == target).any()
    assert first.changed_positions.any()
    assert torch.equal(first.noised_canvas, second.noised_canvas)
    assert torch.equal(first.corrupted_positions, second.corrupted_positions)


def test_power_t_sampling_favors_high_noise_and_keeps_five_percent_exact_terminal() -> None:
    base = _small_config()
    schedule = replace(
        base.diffusion.schedule,
        t_distribution="power",
        t_distribution_power=2.0,
        t_one_fraction=0.05,
    )
    target = torch.tensor([BOS_ID, CONTENT_TOKEN_OFFSET]).repeat(50_000, 1)
    result = corrupt_batch(
        input_token_ids=torch.empty((50_000, 0), dtype=torch.long),
        target_canvas=target,
        process="uniform",
        schedule=schedule,
        vocab_size=128,
        generator=torch.Generator(device="cpu").manual_seed(812),
    )

    exact_terminal = result.t.eq(1.0)
    continuous = result.t[~exact_terminal]
    assert float(exact_terminal.float().mean()) == pytest.approx(0.05, abs=0.004)
    assert float(continuous.mean()) == pytest.approx(2.0 / 3.0, abs=0.01)
    assert float((continuous >= 0.75).float().mean()) == pytest.approx(0.4375, abs=0.015)
    assert float((continuous < 0.25).float().mean()) == pytest.approx(0.0625, abs=0.01)


def test_uniform_noise_support_is_pad_delimiter_and_content_only() -> None:
    vocab_size = CONTENT_TOKEN_OFFSET + 8
    sampled = sample_uniform_noise(
        (20_000,),
        vocab_size=vocab_size,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu").manual_seed(812),
    )
    observed = set(sampled.tolist())
    expected = {PAD_ID, DELIMITER_ID, *range(CONTENT_TOKEN_OFFSET, vocab_size)}
    assert observed == expected
    assert observed.isdisjoint({MASK_ID, END_ID, WIN_ID, LOSS_ID, BOS_ID, EOS_ID})


def test_absorbing_corruption_and_loss_retain_masked_inverse_time_objective(
    tmp_path: Path,
) -> None:
    base = _small_config(tmp_path)
    config = replace(base, diffusion=replace(base.diffusion, process="absorbing"))
    loop, batch = _loop_and_batch(config, seed=15)
    result = loop.compute_batch_loss(batch, fixed_t=0.5)

    assert torch.equal(
        result.scored_mask,
        result.corruption.corrupted_positions & batch.canvas_loss_mask,
    )
    assert (result.corruption.noised_canvas[result.corruption.corrupted_positions] == 0).all()
    assert torch.allclose(
        result.corruption.position_weights,
        inverse_t_weights(result.corruption.t, batch.target_canvas.shape[1]),
    )


def test_self_conditioning_training_uses_no_grad_estimate_then_grad_pass(tmp_path: Path) -> None:
    config = replace(
        _small_config(tmp_path),
        model=replace(_small_config(tmp_path).model, self_conditioning=True),
        train=replace(_small_config(tmp_path).train, self_cond_prob=1.0),
    )
    torch.manual_seed(61)
    examples = make_synthetic_examples(config, count=2)
    batch = next(iter(DataLoader(examples, batch_size=2, shuffle=False, collate_fn=_collate_pretrain)))
    model = CountingDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=61)

    result = loop.compute_batch_loss(batch, fixed_t=1.0)

    assert result.self_conditioning_mask.all()
    assert model.forward_records == [(False, False), (True, True)]
    assert loop.global_step == 0
    assert loop.optimizer.state_dict()["state"] == {}


def test_self_conditioning_off_uses_single_legacy_training_forward(tmp_path: Path) -> None:
    config = replace(
        _small_config(tmp_path),
        model=replace(_small_config(tmp_path).model, self_conditioning=False),
        train=replace(_small_config(tmp_path).train, self_cond_prob=1.0),
    )
    torch.manual_seed(62)
    examples = make_synthetic_examples(config, count=2)
    batch = next(iter(DataLoader(examples, batch_size=2, shuffle=False, collate_fn=_collate_pretrain)))
    model = CountingDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=62)

    result = loop.compute_batch_loss(batch, fixed_t=1.0)

    assert not result.self_conditioning_mask.any()
    assert model.forward_records == [(True, False)]


def test_schedule_weighting_uses_inverse_t_not_flat() -> None:
    weights = inverse_t_weights(torch.tensor([0.25, 0.75]), canvas_len=3)

    assert weights[0, 0].item() == pytest.approx(4.0)
    assert weights[1, 0].item() == pytest.approx(4.0 / 3.0)
    assert not torch.allclose(weights, torch.ones_like(weights))
    assert weights[0, 0] > weights[1, 0]


def test_checkpoint_roundtrip_restores_model_optimizer_and_step(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    loop, batch = _loop_and_batch(config, seed=21)
    loop.fit([batch], max_steps=1, fixed_t=1.0)
    checkpoint = loop.save_checkpoint(tmp_path / "manual.pt")

    restored_model = SC2StrategyDiffusionModel(config, vocab_size=128)
    restored = TrainingLoop(model=restored_model, config=config, seed=21)
    restored.load_checkpoint(checkpoint)

    assert restored.global_step == loop.global_step
    assert restored.total_tokens_ingested == loop.total_tokens_ingested
    assert restored.unique_token_ids_seen == loop.unique_token_ids_seen
    for saved, loaded in zip(loop.model.parameters(), restored.model.parameters(), strict=True):
        assert torch.allclose(saved, loaded)
    for saved, loaded in zip(loop.ema_model.parameters(), restored.ema_model.parameters(), strict=True):
        assert torch.allclose(saved, loaded)
    _assert_optimizer_states_match(loop.optimizer.state_dict(), restored.optimizer.state_dict())

    logs = restored.fit([batch], max_steps=2, fixed_t=1.0)
    assert restored.global_step == 2
    assert logs


def _fit_keeping_step_checkpoints(tmp_path: Path, *, seed: int) -> tuple[TrainingLoop, DataLoader]:
    """Run a short fit() that leaves behind per-step `step-N.pt` snapshots.

    Those snapshots are the faithful stand-in for a preempted run: unlike
    `last.pt` (which fit() rewrites on a clean return) they are written by
    `_maybe_checkpoint` while fit() is still in flight and are never touched
    again. Resuming from one is exactly what happens after a process is killed
    mid-epoch.

    Args:
        tmp_path: pytest temp dir, used for the checkpoint directory.
        seed: training seed for the loop.

    Returns:
        `(loop, dataloader)` -- the finished loop and the loader it trained on,
        so a caller can resume into a fresh loop over the same data.

    Depends on: `_small_config`, `make_synthetic_examples`, `_collate_pretrain`.
    """

    base = _small_config(tmp_path)
    config = replace(
        base,
        train=replace(base.train, checkpoint_interval=1, keep_step_checkpoints=True),
    )
    examples = make_synthetic_examples(config, count=2)
    loader = DataLoader(examples, batch_size=2, shuffle=False, collate_fn=_collate_pretrain)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=seed)
    loop.fit(loader, max_steps=2, epochs=1, fixed_t=1.0)
    return loop, loader


def test_mid_run_checkpoint_records_live_wall_clock_not_the_value_at_process_start(
    tmp_path: Path,
) -> None:
    """A checkpoint written DURING fit() must carry the running elapsed time.

    Regression test. `elapsed_wall_seconds` is only folded forward when fit()
    returns, so checkpoints written mid-run used to serialize the value as of
    this process's startup -- 0.0 on a fresh run. A run killed mid-fit then
    resumed from such a checkpoint restarted its wall clock from zero, which is
    why the metrics files showed the elapsed column resetting at every resume
    point. Asserting on a `step-N.pt` snapshot rather than `last.pt` matters:
    a clean fit() return rewrites `last.pt` with the correct folded total, so
    only the never-rewritten mid-run snapshot exposes the bug.
    """

    loop, _ = _fit_keeping_step_checkpoints(tmp_path, seed=23)

    mid_run_path = loop.resume_checkpoint_path.parent / "step-2.pt"
    assert mid_run_path.exists()
    mid_run = torch.load(mid_run_path, map_location="cpu", weights_only=False)
    assert mid_run["elapsed_wall_seconds"] > 0.0


def test_wall_clock_accumulates_across_a_resume_instead_of_restarting(tmp_path: Path) -> None:
    """The elapsed column must keep climbing after a resume, not rewind to ~0.

    Simulates a preempted run end to end: train, then resume a fresh loop from
    a snapshot written mid-fit (what a killed process leaves behind) and train
    again. The second process has to pick the clock up where the first one was
    cut off, so the newest metrics row is a true cumulative total for the whole
    run instead of just the latest process's slice.
    """

    first, loader = _fit_keeping_step_checkpoints(tmp_path, seed=24)
    total_after_first_process = first.elapsed_wall_seconds
    assert total_after_first_process > 0.0

    resumed_model = SC2StrategyDiffusionModel(config=first.config, vocab_size=128)
    resumed = TrainingLoop(model=resumed_model, config=first.config, seed=24)
    resumed.load_checkpoint(first.resume_checkpoint_path.parent / "step-2.pt")

    # The restored baseline is the time the first process had banked when it
    # was cut off -- non-zero, and no larger than that process's final total.
    assert resumed.elapsed_wall_seconds > 0.0
    assert resumed.elapsed_wall_seconds <= total_after_first_process

    restored_baseline = resumed.elapsed_wall_seconds
    resumed.fit(loader, max_steps=4, epochs=1, fixed_t=1.0)
    # Second process's time is ADDED to the inherited baseline, not counted
    # from zero.
    assert resumed.elapsed_wall_seconds > restored_baseline


def test_epoch_checkpoint_families_replace_best_and_retain_durable(tmp_path: Path) -> None:
    base = _small_config(tmp_path)
    config = replace(
        base,
        train=replace(
            base.train,
            resume_checkpoint_subdir="resume",
            best_checkpoint_subdir="best",
            durable_checkpoint_subdir="durable",
            save_best_checkpoint=True,
            durable_checkpoint_interval_epochs=5,
            early_stopping_patience_epochs=10,
        ),
    )
    loop = TrainingLoop(
        model=SC2StrategyDiffusionModel(config, vocab_size=128),
        config=config,
        seed=22,
    )

    loop.completed_epochs = 1
    assert loop._should_stop_early(2.0) is False
    loop._save_epoch_checkpoints(2.0)
    assert (tmp_path / "checkpoints" / "best" / "epoch-0001.pt").exists()

    loop.completed_epochs = 2
    # A strict 0.05% improvement is too small to reset the 0.1% patience
    # threshold, but it is still the best dev loss seen and must replace the
    # inference/resume checkpoint.
    assert loop._should_stop_early(1.999) is False
    assert loop.epochs_without_improvement == 1
    loop._save_epoch_checkpoints(1.999)
    assert not (tmp_path / "checkpoints" / "best" / "epoch-0001.pt").exists()
    assert (tmp_path / "checkpoints" / "best" / "epoch-0002.pt").exists()

    loop.completed_epochs = 3
    assert loop._should_stop_early(1.8) is False
    loop._save_epoch_checkpoints(1.8)
    assert not (tmp_path / "checkpoints" / "best" / "epoch-0002.pt").exists()
    assert (tmp_path / "checkpoints" / "best" / "epoch-0003.pt").exists()

    loop.completed_epochs = 5
    loop._save_epoch_checkpoints(1.9)
    durable = tmp_path / "checkpoints" / "durable" / "epoch-0005.pt"
    assert durable.exists()
    restored = TrainingLoop(
        model=SC2StrategyDiffusionModel(config, vocab_size=128),
        config=config,
        seed=22,
    )
    restored.load_checkpoint(durable)
    assert restored.completed_epochs == 5
    assert restored.best_dev_loss == pytest.approx(1.8)


def test_optimizer_steps_per_epoch_accounts_for_gradient_accumulation() -> None:
    assert optimizer_steps_per_epoch(4_763, 5) == 953
    assert optimizer_steps_per_epoch(7_144, 7) == 1_021
    assert optimizer_steps_per_epoch(10, 5) == 2
    assert optimizer_steps_per_epoch(11, 5) == 3


def test_full_resume_rejects_a_checkpoint_from_a_different_ablation_toggle_set(tmp_path: Path) -> None:
    """`load_checkpoint` (full resume) must reject a checkpoint written under a
    DIFFERENT architecture ablation toggle set, via
    `validate_checkpoint_compatibility`.

    `frozen_input_kv` and `per_segment_positions` add ZERO parameters, so
    without this gate `load_state_dict` would silently SUCCEED across
    mismatched arms and quietly corrupt the ablation study -- the
    `architecture_identity` string is the only thing that can catch it. Covers
    the user's explicit minimum case: a
    `{frozen_input_kv, segment_embeddings, per_segment_positions}` run must
    NOT resume a `{frozen_input_kv}`-only run's checkpoint. The rejection must
    be a raised error, not a warning or silent no-op.
    """

    source_config = _small_config(tmp_path, frozen_input_kv=True)
    source_loop = TrainingLoop(
        model=SC2StrategyDiffusionModel(source_config, vocab_size=128), config=source_config, seed=41
    )
    checkpoint_path = source_loop.save_checkpoint(tmp_path / "checkpoints" / "frozen_kv_only.pt")

    mismatched_config = _small_config(
        tmp_path, frozen_input_kv=True, segment_embeddings=True, per_segment_positions=True
    )
    mismatched_loop = TrainingLoop(
        model=SC2StrategyDiffusionModel(mismatched_config, vocab_size=128),
        config=mismatched_config,
        seed=41,
    )

    with pytest.raises(ValueError, match="architecture identity mismatch"):
        mismatched_loop.load_checkpoint(checkpoint_path)


def test_full_resume_accepts_a_checkpoint_from_the_matching_ablation_toggle_set(tmp_path: Path) -> None:
    """The positive counterpart of the rejection test above: a
    `{segment_embeddings}` run MUST resume a `{segment_embeddings}` run's
    checkpoint (the user's other explicitly named minimum case) -- the gate
    must accept a genuine match, not just reject mismatches.
    """

    config = _small_config(tmp_path, segment_embeddings=True)
    source_loop = TrainingLoop(
        model=SC2StrategyDiffusionModel(config, vocab_size=128), config=config, seed=42
    )
    checkpoint_path = source_loop.save_checkpoint(tmp_path / "checkpoints" / "segment_embeddings.pt")

    target_loop = TrainingLoop(
        model=SC2StrategyDiffusionModel(config, vocab_size=128), config=config, seed=42
    )
    target_loop.load_checkpoint(checkpoint_path)  # must not raise

    assert target_loop.global_step == source_loop.global_step
    for restored, saved in zip(
        target_loop.model.parameters(), source_loop.model.parameters(), strict=True
    ):
        assert torch.allclose(restored, saved)


def test_ema_tracks_training_and_validation_uses_ema_weights(tmp_path: Path) -> None:
    # The EMA window is now DERIVED from the run's step count
    # (TrainingLoop._resolve_ema_decay), and a run has to be long enough for
    # averaging to mean anything: at `ema_horizon_ratio` x total_steps <= 1 step
    # the derived decay is 0.0 and the EMA correctly just mirrors the raw weights.
    # This test is about the EMA LAGGING the raw weights, so it needs a run whose
    # window spans more than one step -- an 8-step run with the whole run as the
    # window (ratio 1.0) is the smallest thing that exercises the lag.
    base = _small_config(tmp_path)
    config = replace(base, train=replace(base.train, ema_horizon_ratio=1.0, max_steps=8))
    loop, batch = _loop_and_batch(config, seed=31)
    initial_ema = [parameter.detach().clone() for parameter in loop.ema_model.parameters()]

    loop.fit([batch], max_steps=8, fixed_t=1.0)

    assert any(
        not torch.allclose(before, after)
        for before, after in zip(initial_ema, loop.ema_model.parameters(), strict=True)
    )
    assert any(
        not torch.allclose(raw, ema)
        for raw, ema in zip(loop.model.parameters(), loop.ema_model.parameters(), strict=True)
    )

    loop.generator.manual_seed(900)
    validation = loop.validate([batch], fixed_t=1.0)
    loop.generator.manual_seed(900)
    expected = loop.compute_batch_loss(batch, fixed_t=1.0, model=loop.ema_model)
    assert validation.loss == pytest.approx(float(expected.loss.detach()))
    assert validation.per_class


def test_ema_averaging_window_is_fitted_to_the_runs_step_count(tmp_path: Path) -> None:
    """The EMA window must scale with the run, capped by `train.ema_decay`.

    An EMA with decay d averages over 1/(1-d) steps. Holding d constant pins that
    window to a fixed step count, which is the bug this replaces: a fixed 0.9999
    is a ~10,000-step window, so a 3,400-step run ended with an EMA that had never
    traversed its own window once. `_resolve_ema_decay` derives d from the run
    instead, so the window is always `ema_horizon_ratio` x the run's steps.
    """

    base = _small_config(tmp_path)
    # max_steps 100_000 seeds __init__'s decay at the 0.9999 ceiling, which is
    # what the re-fit assertion at the end of this test contrasts against.
    config = replace(
        base,
        train=replace(base.train, ema_decay=0.9999, ema_horizon_ratio=0.1, max_steps=100_000),
    )
    loop, batch = _loop_and_batch(config, seed=33)
    assert loop._ema_target_decay == pytest.approx(0.9999)

    # 3400 steps (the overfitV2 / ablation budget) x 0.1 = a 340-step window.
    assert loop._resolve_ema_decay(3400) == pytest.approx(1.0 - 1.0 / 340.0)
    # Half the run length must halve the window, not leave it unchanged.
    assert loop._resolve_ema_decay(1700) == pytest.approx(1.0 - 1.0 / 170.0)
    # At and beyond 100,000 steps the derived window hits the 0.9999 ceiling, so
    # the long-run behavior is exactly what the old fixed constant gave.
    assert loop._resolve_ema_decay(100_000) == pytest.approx(0.9999)
    assert loop._resolve_ema_decay(10_000_000) == pytest.approx(0.9999)
    # A run too short to average over collapses to "mirror the raw weights"
    # rather than producing a decay outside [0, 1).
    assert loop._resolve_ema_decay(1) == pytest.approx(0.0)
    assert loop._resolve_ema_decay(0) == pytest.approx(0.0)

    # fit() must RE-FIT the window to the budget it actually derives, not keep the
    # value __init__ computed from config.train.max_steps. A 2-step bounded run is
    # far too short to average over, so the decay must drop from the 0.9999
    # ceiling all the way to 0.0 (EMA mirrors the raw weights).
    loop.fit([batch], max_steps=2, fixed_t=1.0)
    assert loop._ema_target_decay == pytest.approx(loop._resolve_ema_decay(2))
    assert loop._ema_target_decay == pytest.approx(0.0)


def test_lr_schedule_selects_linear_or_cosine_and_both_land_on_the_floor(
    tmp_path: Path,
) -> None:
    """Both shapes span peak -> floor; linear leaves the peak sooner.

    These are the three properties the V2 / ablation schedule change relies on:
    the linear shape starts descending immediately (less time at peak), its
    constant slope is shallower than the cosine's steepest mid-run section, and
    both shapes still end exactly on `lr_floor_ratio`.
    """

    base = _small_config(tmp_path)
    horizon = 1000
    warmup = 10

    def multipliers(shape: str, floor: float) -> list[float]:
        config = replace(
            base,
            train=replace(
                base.train,
                warmup=warmup,
                max_steps=horizon,
                lr_schedule=shape,
                lr_floor_ratio=floor,
            ),
        )
        loop, _ = _loop_and_batch(config, seed=34)
        return [loop._lr_multiplier(step) for step in range(horizon + 1)]

    cosine = multipliers("cosine", 0.1)
    linear = multipliers("linear", 0.03)

    # Warmup is untouched by the shape: both ramp linearly to the peak.
    assert cosine[warmup - 1] == pytest.approx(1.0)
    assert linear[warmup - 1] == pytest.approx(1.0)

    # Less time at the peak: one tenth into the decay, linear has already given
    # up meaningfully more of the peak rate than the cosine has.
    tenth = warmup + (horizon - warmup) // 10
    assert linear[tenth] < cosine[tenth]

    # Shallower through the middle: the cosine's steepest per-step drop is
    # pi/2 ~= 1.57x the straight line's constant one.
    cosine_max_drop = max(cosine[i] - cosine[i + 1] for i in range(warmup, horizon))
    linear_max_drop = max(linear[i] - linear[i + 1] for i in range(warmup, horizon))
    assert linear_max_drop < cosine_max_drop

    # Lower end of training, and both land exactly on their configured floor.
    assert cosine[horizon] == pytest.approx(0.1)
    assert linear[horizon] == pytest.approx(0.03)
    assert linear[horizon] < cosine[horizon]
    # Monotonically non-increasing across the decay for both shapes.
    for series in (cosine, linear):
        for step in range(warmup, horizon):
            assert series[step + 1] <= series[step] + 1e-12


def test_wsd_schedule_uses_fixed_warmup_then_holds_peak_until_linear_decay(
    tmp_path: Path,
) -> None:
    base = _small_config(tmp_path)
    horizon = 1000
    config = replace(
        base,
        train=replace(
            base.train,
            max_steps=horizon,
            lr_schedule="wsd",
            lr_floor_ratio=0.01,
            warmup=100,
            lr_decay_ratio=0.20,
        ),
    )
    loop, _ = _loop_and_batch(config, seed=34)

    assert loop._schedule_phase_steps(horizon) == (100, 700, 200)
    assert loop._lr_multiplier(0) == pytest.approx(0.01)
    assert loop._lr_multiplier(99) == pytest.approx(1.0)
    assert loop._lr_multiplier(799) == pytest.approx(1.0)
    assert loop._lr_multiplier(800) == pytest.approx(1.0)
    assert loop._lr_multiplier(900) == pytest.approx(0.505)
    assert loop._lr_multiplier(1000) == pytest.approx(0.01)
    # Extending the run does not silently lengthen warmup; stable absorbs the
    # extra horizon while the final decay remains 20% of total steps.
    assert loop._schedule_phase_steps(2000) == (100, 1500, 400)


def test_confidence_loss_is_weighted_and_disableable(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    loop, batch = _loop_and_batch(config, seed=41)
    weighted = loop.compute_batch_loss(batch, fixed_t=1.0)

    off_config = replace(config, train=replace(config.train, confidence_loss_weight=0.0))
    off_loop, off_batch = _loop_and_batch(off_config, seed=41)
    disabled = off_loop.compute_batch_loss(off_batch, fixed_t=1.0)

    assert torch.allclose(disabled.loss, disabled.denoising_loss)
    assert disabled.confidence_loss.item() == pytest.approx(0.0)
    if weighted.confidence_loss.item() > 0:
        assert torch.allclose(weighted.loss, weighted.denoising_loss + weighted.confidence_loss)

    logits = torch.zeros(1, 2, 8)
    logits[0, 0, 3] = 4.0
    logits[0, 1, 2] = 4.0
    targets = torch.tensor([[3, 7]])
    scored = torch.tensor([[True, True]])
    assert auxiliary_confidence_loss(logits, targets, scored).item() > 0.0


def test_periodic_validation_logs_held_out_loss_from_ema(tmp_path: Path) -> None:
    config = replace(_small_config(tmp_path), train=replace(_small_config(tmp_path).train, val_interval=1))
    torch.manual_seed(51)
    train_examples = make_synthetic_examples(config, count=2)
    val_examples = make_synthetic_examples(config, count=4)[2:]
    assert {example.window_start for example in train_examples}.isdisjoint(
        {example.window_start for example in val_examples}
    )
    train_loader = DataLoader(train_examples, batch_size=2, shuffle=False, collate_fn=_collate_pretrain)
    val_loader = DataLoader(val_examples, batch_size=2, shuffle=False, collate_fn=_collate_pretrain)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=51)

    logs = loop.fit(train_loader, val_dataloader=val_loader, max_steps=1, fixed_t=1.0)

    assert logs[0].validation is not None
    assert logs[0].validation.loss > 0
    assert logs[0].validation.per_class


def test_seeded_smoke_runs_are_deterministic(tmp_path: Path) -> None:
    first = run_smoke_train(max_steps=5, seed=99, checkpoint_dir=tmp_path / "a")
    second = run_smoke_train(max_steps=5, seed=99, checkpoint_dir=tmp_path / "b")

    assert [log.loss for log in first] == pytest.approx([log.loss for log in second])
    assert [log.noise_fraction for log in first] == pytest.approx([log.noise_fraction for log in second])
    assert [log.per_class for log in first] == [log.per_class for log in second]


def test_epoch_metrics_csv_contains_train_dev_classes_and_throughput(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    examples = make_synthetic_examples(config, count=4)
    train_loader = DataLoader(examples[:2], batch_size=2, collate_fn=_collate_pretrain)
    dev_loader = DataLoader(examples[2:], batch_size=2, collate_fn=_collate_pretrain)
    csv_path = tmp_path / "epoch_metrics.csv"
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=71, epoch_metrics_path=csv_path)

    loop.fit(train_loader, val_dataloader=dev_loader, max_steps=2, epochs=2, fixed_t=1.0)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    assert all(float(row["train_loss"]) > 0 for row in rows)
    assert all(float(row["dev_loss"]) > 0 for row in rows)
    assert all(float(row["tokens_per_second"]) > 0 for row in rows)
    assert all(float(row["wall_clock_elapsed_seconds"]) > 0 for row in rows)
    assert all(float(row["average_cuda_device_memory_used_bytes"]) == 0 for row in rows)
    assert all(float(row["average_cuda_device_memory_gap_bytes"]) == 0 for row in rows)
    assert "average_input_timesteps" in fieldnames
    assert "average_enemy_future_timesteps" in fieldnames
    assert "input_timestep_p50" in fieldnames
    assert "enemy_future_timestep_p50" in fieldnames
    assert "train_enemy_future_loss_distance_1" in fieldnames
    assert "dev_enemy_future_loss_distance_1" in fieldnames
    # t-bucket / perspective breakdown columns: with fixed_t=1.0 every example
    # lands in the exact-t==1 bucket; the other three bucket columns exist but
    # are written as "" (the empty-bucket convention).
    assert all(float(row["train_t_bucket_loss_t_eq_1"]) > 0 for row in rows)
    assert all(float(row["dev_t_bucket_loss_t_eq_1"]) > 0 for row in rows)
    assert all(row["train_t_bucket_loss_t_0_75_to_1_0"] == "" for row in rows)
    assert all(row["train_t_bucket_loss_t_0_0_to_0_25"] == "" for row in rows)
    # Canvas-state columns. At fixed_t=1.0 every position takes the corruption
    # branch, but uniform diffusion re-draws replacements from the whole
    # vocabulary, so a few positions land back on their own target token and the
    # preserved bucket is NOT guaranteed to be populated. The noised bucket
    # always is, and both columns must exist either way.
    assert "train_canvas_state_loss_ground_truth_preserved" in fieldnames
    assert "dev_canvas_state_loss_ground_truth_preserved" in fieldnames
    assert all(float(row["train_canvas_state_loss_noised"]) > 0 for row in rows)
    assert all(float(row["dev_canvas_state_loss_noised"]) > 0 for row in rows)
    # The fixtures alternate p1/p2 perspectives, so both perspective columns
    # are populated in train and dev.
    assert all(float(row["train_perspective_loss_p1"]) > 0 for row in rows)
    assert all(float(row["train_perspective_loss_p2"]) > 0 for row in rows)
    assert all(float(row["dev_perspective_loss_p1"]) > 0 for row in rows)
    assert all(float(row["dev_perspective_loss_p2"]) > 0 for row in rows)
    # Two examples x (4 fogged input including EOS + 12 canvas) tokens per epoch.
    assert [int(row["total_tokens_ingested"]) for row in rows] == [32, 64]
    # BOS and EOS add two boundary identities to the shared synthetic content.
    assert [int(row["total_unique_tokens_seen"]) for row in rows] == [11, 11]
    # Pretraining retains observed/fogged/future reconstruction classes.
    assert "train_enemy_observed_loss" in fieldnames
    assert "dev_pad_loss" in fieldnames
    assert "train_enemy_fogged_loss" in fieldnames
    assert "train_enemy_future_loss" in fieldnames


def test_locked_metrics_csv_continues_with_full_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "epoch_metrics.csv"
    fieldnames = ["epoch", "train_loss"]
    assert _append_csv_row(
        csv_path,
        fieldnames,
        {"epoch": 1, "train_loss": 2.0},
    ) == csv_path

    original_open = Path.open

    def locked_open(self: Path, mode: str = "r", *args: object, **kwargs: object):
        if self == csv_path and "a" in mode:
            raise PermissionError(13, "simulated viewer lock", str(self))
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", locked_open)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    continuation_path = _append_csv_row(
        csv_path,
        fieldnames,
        {"epoch": 2, "train_loss": 1.0},
    )

    assert continuation_path != csv_path
    assert continuation_path.name.startswith("epoch_metrics-continued-")
    with original_open(continuation_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    assert _latest_metrics_csv_path(csv_path) == continuation_path
    output = capsys.readouterr().out
    assert "metrics_csv_locked action=continued" in output
    assert "history_rows_copied=1" in output


def test_interval_boundaries_are_evenly_spaced_and_end_on_the_epoch(tmp_path: Path) -> None:
    """Report boundaries tile the epoch and always close on its final batch.

    The last boundary MUST equal batches_per_epoch so the tenth interval row and
    the epoch row cover the same point in training and can be read together. An
    epoch with fewer batches than reports simply reports once per batch rather
    than emitting duplicate boundaries.
    """

    assert interval_boundaries(34) == [4, 7, 11, 14, 17, 21, 24, 28, 31, 34]
    assert interval_boundaries(100) == [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # Fewer batches than reports: one boundary per batch, no duplicates.
    assert interval_boundaries(4) == [1, 2, 3, 4]
    assert interval_boundaries(1) == [1]
    # Degenerate inputs produce no reports rather than raising.
    assert interval_boundaries(0) == []
    for batches in (1, 3, 7, 34, 101, 999):
        boundaries = interval_boundaries(batches)
        assert boundaries[-1] == batches
        assert boundaries == sorted(set(boundaries))
        assert len(boundaries) <= INTERVAL_REPORTS_PER_EPOCH


def test_interval_metrics_csv_reports_ten_times_per_epoch_with_train_and_dev(
    tmp_path: Path,
) -> None:
    """The intra-epoch CSV carries every loss sub-class, for train AND dev.

    Ten batches per epoch means one report per batch boundary, so two epochs
    produce 2 x 10 rows. Each row must carry a dev value too, because the dev
    pass is what makes these rows comparable across a run that only trains for a
    single epoch.
    """

    config = _small_config(tmp_path)
    examples = make_synthetic_examples(config, count=24)
    train_loader = DataLoader(examples[:20], batch_size=2, collate_fn=_collate_pretrain)
    dev_loader = DataLoader(examples[20:], batch_size=2, collate_fn=_collate_pretrain)
    csv_path = tmp_path / "interval_metrics.csv"
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(
        model=model, config=config, seed=73, interval_metrics_path=csv_path
    )

    loop.fit(train_loader, val_dataloader=dev_loader, max_steps=20, epochs=2, fixed_t=1.0)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    assert len(rows) == 2 * INTERVAL_REPORTS_PER_EPOCH
    assert [int(row["epoch"]) for row in rows] == [1] * 10 + [2] * 10
    assert [int(row["interval"]) for row in rows[:10]] == list(range(1, 11))
    # epoch_fraction walks 0.1 -> 1.0 within each epoch.
    assert float(rows[0]["epoch_fraction"]) == pytest.approx(0.1)
    assert float(rows[9]["epoch_fraction"]) == pytest.approx(1.0)
    # global_step advances monotonically across rows.
    steps = [int(row["global_step"]) for row in rows]
    assert steps == sorted(steps)
    assert all(float(row["train_loss"]) > 0 for row in rows)
    assert all(float(row["dev_loss"]) > 0 for row in rows)

    # Every diagnostic the overfit run is meant to surface has a column here,
    # for both splits.
    for split in ("train", "dev"):
        for name in ("pad", "delimiter", "end", "win_loss", "enemy_observed",
                     "enemy_fogged", "enemy_future"):
            assert f"{split}_{name}_loss" in fieldnames
        for name in ("t_eq_1", "t_0_75_to_1_0", "t_0_25_to_0_75", "t_0_0_to_0_25"):
            assert f"{split}_t_bucket_loss_{name}" in fieldnames
        for name in ("ground_truth_preserved", "noised"):
            assert f"{split}_canvas_state_loss_{name}" in fieldnames


def test_interval_rows_are_scoped_to_their_slice_not_the_epoch_so_far(
    tmp_path: Path,
) -> None:
    """Each row averages only its own slice, so rows are independent samples.

    If rows accumulated across the whole epoch instead, later rows would be
    dominated by earlier batches and the column would flatten out -- exactly the
    trend-hiding behavior these rows exist to avoid. Proven by giving the loop a
    dataset whose two halves differ and checking the rows differ too.
    """

    config = _small_config(tmp_path)
    examples = make_synthetic_examples(config, count=12)
    train_loader = DataLoader(examples[:10], batch_size=1, collate_fn=_collate_pretrain)
    csv_path = tmp_path / "interval_metrics.csv"
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(
        model=model, config=config, seed=74, interval_metrics_path=csv_path
    )

    loop.fit(train_loader, max_steps=10, epochs=1, fixed_t=1.0)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == INTERVAL_REPORTS_PER_EPOCH
    # 10 batches / 10 reports = one batch per row, so each row's train_loss is
    # exactly that batch's loss. A running average would instead force the
    # values to converge; assert they stay distinct.
    losses = [float(row["train_loss"]) for row in rows]
    assert len(set(losses)) > 1
    # No dev loader configured -> the dev column is blank, never fabricated.
    assert all(row["dev_loss"] == "" for row in rows)


def test_interval_dev_evaluation_can_be_disabled_without_losing_train_rows(
    tmp_path: Path,
) -> None:
    """`train.interval_dev_evaluation: false` drops only the interval dev pass.

    The train-side breakdown must still be reported ten times per epoch -- that
    is the part a run keeps regardless of cost. Only the dev columns go blank,
    with dev instead reported once per epoch in the epoch CSV.
    """

    config = _small_config(tmp_path)
    config = replace(config, train=replace(config.train, interval_dev_evaluation=False))
    examples = make_synthetic_examples(config, count=24)
    train_loader = DataLoader(examples[:20], batch_size=2, collate_fn=_collate_pretrain)
    dev_loader = DataLoader(examples[20:], batch_size=2, collate_fn=_collate_pretrain)
    interval_path = tmp_path / "interval_metrics.csv"
    epoch_path = tmp_path / "epoch_metrics.csv"
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(
        model=model,
        config=config,
        seed=76,
        interval_metrics_path=interval_path,
        epoch_metrics_path=epoch_path,
    )

    loop.fit(train_loader, val_dataloader=dev_loader, max_steps=10, epochs=1, fixed_t=1.0)

    with interval_path.open(newline="", encoding="utf-8") as handle:
        interval_rows = list(csv.DictReader(handle))
    # Still ten rows, still fully populated on the train side.
    assert len(interval_rows) == INTERVAL_REPORTS_PER_EPOCH
    assert all(float(row["train_loss"]) > 0 for row in interval_rows)
    assert all(float(row["train_pad_loss"]) > 0 for row in interval_rows)
    assert all(float(row["train_canvas_state_loss_noised"]) > 0 for row in interval_rows)
    # Dev columns blank despite a dev loader being configured.
    assert all(row["dev_loss"] == "" for row in interval_rows)
    assert all(row["dev_pad_loss"] == "" for row in interval_rows)

    # Dev is not lost -- it is reported once, at the epoch end.
    with epoch_path.open(newline="", encoding="utf-8") as handle:
        epoch_rows = list(csv.DictReader(handle))
    assert len(epoch_rows) == 1
    assert float(epoch_rows[0]["dev_loss"]) > 0
    assert float(epoch_rows[0]["dev_pad_loss"]) > 0


def test_epoch_metrics_reports_rare_class_loss_and_counts_per_t_bucket(
    tmp_path: Path,
) -> None:
    """The rare-class x t-bucket cross decomposition, losses AND position counts.

    The three rare classes (win/loss, [END], [DELIMITER]) are crossed with all
    four corruption buckets, so a trend that runs along the corruption axis stays
    visible instead of being averaged away by the `per_class` marginal.

    The counts are the half that makes the losses readable. Each synthetic canvas
    holds exactly one win/loss token, one [END], and three [DELIMITER]s, so with
    two examples per epoch and `fixed_t=1.0` pinning every example to the
    exact-t==1 bucket the expected counts are exact -- and every other bucket
    must report a real 0 rather than a blank, which is what distinguishes "no
    [END] landed in this bucket" from "this bucket was never evaluated".
    """

    config = _small_config(tmp_path)
    examples = make_synthetic_examples(config, count=4)
    train_loader = DataLoader(examples[:2], batch_size=2, collate_fn=_collate_pretrain)
    dev_loader = DataLoader(examples[2:], batch_size=2, collate_fn=_collate_pretrain)
    csv_path = tmp_path / "epoch_metrics.csv"
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=91, epoch_metrics_path=csv_path)

    loop.fit(train_loader, val_dataloader=dev_loader, max_steps=2, epochs=2, fixed_t=1.0)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    # All 12 cells exist as a loss column and a count column, for both splits.
    for split in ("train", "dev"):
        for name in RARE_CLASS_T_BUCKET_NAMES:
            assert f"{split}_rare_class_loss_{name}" in fieldnames
            assert f"{split}_rare_class_count_{name}" in fieldnames
    assert len(RARE_CLASS_T_BUCKET_NAMES) == 12

    for row in rows:
        for split in ("train", "dev"):
            # One win/loss + one [END] + three [DELIMITER] per canvas, two
            # canvases per epoch, all pinned to t == 1 by fixed_t.
            assert int(row[f"{split}_rare_class_count_win_loss_t_eq_1"]) == 2
            assert int(row[f"{split}_rare_class_count_end_t_eq_1"]) == 2
            assert int(row[f"{split}_rare_class_count_delimiter_t_eq_1"]) == 6
            # A populated cell carries a real loss.
            assert float(row[f"{split}_rare_class_loss_win_loss_t_eq_1"]) > 0
            assert float(row[f"{split}_rare_class_loss_end_t_eq_1"]) > 0
            assert float(row[f"{split}_rare_class_loss_delimiter_t_eq_1"]) > 0
            # Every other bucket scored nothing: the COUNT is an explicit 0 (the
            # observation) while the LOSS is blank (no positions to average).
            for bucket in ("t_0_75_to_1_0", "t_0_25_to_0_75", "t_0_0_to_0_25"):
                for rare_class in ("win_loss", "end", "delimiter"):
                    cell = f"{rare_class}_{bucket}"
                    assert int(row[f"{split}_rare_class_count_{cell}"]) == 0
                    assert row[f"{split}_rare_class_loss_{cell}"] == ""


def test_rare_class_cells_pool_by_position_count_not_by_microbatch(
    tmp_path: Path,
) -> None:
    """A cell's reported loss is its mean over every scored position.

    The loss module hands back a SUM and a COUNT per cell rather than a finished
    mean precisely so the training loop can pool this way. Averaging
    per-microbatch means instead would weight a microbatch holding 3 [DELIMITER]
    positions the same as one holding 6, which is wrong whenever the rare tokens
    are unevenly spread across batches -- the normal case for [END], which only
    appears in windows that reach game end.

    Proven directly against the loss module: the pooled mean of two unequal
    batches must equal the total-sum / total-count, not the mean of the two
    batch means.
    """

    config = _small_config(tmp_path)
    examples = make_synthetic_examples(config, count=3)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=92)

    # Deliberately unequal batches: 1 example then 2, so the delimiter cell holds
    # 3 positions and then 6.
    small = loop.compute_batch_loss(
        _collate_pretrain(examples[:1]), fixed_t=1.0
    ).loss_output
    large = loop.compute_batch_loss(
        _collate_pretrain(examples[1:]), fixed_t=1.0
    ).loss_output

    cell = "delimiter_t_eq_1"
    small_count = int(small.rare_class_t_bucket_counts[cell])
    large_count = int(large.rare_class_t_bucket_counts[cell])
    assert (small_count, large_count) == (3, 6)

    small_sum = float(small.rare_class_t_bucket_sums[cell].detach())
    large_sum = float(large.rare_class_t_bucket_sums[cell].detach())
    pooled = (small_sum + large_sum) / (small_count + large_count)
    mean_of_means = ((small_sum / small_count) + (large_sum / large_count)) / 2

    # The helper the loop uses must produce the pooled value.
    finalized = _finalize_rare_class_t_bucket(
        {cell: small_sum + large_sum},
        {cell: small_count + large_count},
    )
    assert finalized[cell] == pytest.approx(pooled)
    # And that value is genuinely different from the naive reduction, so this
    # test would fail if the pooling ever regressed to a mean of means.
    assert pooled != pytest.approx(mean_of_means)

    # A cell that scored nothing is omitted from the finalized means entirely
    # rather than dividing by zero.
    assert _finalize_rare_class_t_bucket({"end_t_eq_1": 0.0}, {"end_t_eq_1": 0}) == {}


def test_interval_train_evaluation_false_blanks_train_and_keeps_dev(
    tmp_path: Path,
) -> None:
    """`interval_train_evaluation: false` is the exact mirror of the dev knob.

    Train columns go blank while the dev pass still populates its half of every
    interval row, and train loss is instead reported once per epoch.
    """

    config = _small_config(tmp_path)
    config = replace(config, train=replace(config.train, interval_train_evaluation=False))
    examples = make_synthetic_examples(config, count=24)
    train_loader = DataLoader(examples[:20], batch_size=2, collate_fn=_collate_pretrain)
    dev_loader = DataLoader(examples[20:], batch_size=2, collate_fn=_collate_pretrain)
    interval_path = tmp_path / "interval_metrics.csv"
    epoch_path = tmp_path / "epoch_metrics.csv"
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(
        model=model,
        config=config,
        seed=93,
        interval_metrics_path=interval_path,
        epoch_metrics_path=epoch_path,
    )

    loop.fit(train_loader, val_dataloader=dev_loader, max_steps=10, epochs=1, fixed_t=1.0)

    with interval_path.open(newline="", encoding="utf-8") as handle:
        interval_rows = list(csv.DictReader(handle))
    # Rows are still emitted at the same cadence, still carrying dev.
    assert len(interval_rows) == INTERVAL_REPORTS_PER_EPOCH
    assert all(float(row["dev_loss"]) > 0 for row in interval_rows)
    assert all(float(row["dev_pad_loss"]) > 0 for row in interval_rows)
    # Train columns blank across the headline, per-class, and rare-class cells.
    assert all(row["train_loss"] == "" for row in interval_rows)
    assert all(row["train_pad_loss"] == "" for row in interval_rows)
    assert all(row["train_canvas_state_loss_noised"] == "" for row in interval_rows)
    assert all(
        row["train_rare_class_count_win_loss_t_eq_1"] == "" for row in interval_rows
    )

    # Train is not lost -- it is reported once, at the epoch end.
    with epoch_path.open(newline="", encoding="utf-8") as handle:
        epoch_rows = list(csv.DictReader(handle))
    assert len(epoch_rows) == 1
    assert float(epoch_rows[0]["train_loss"]) > 0
    assert float(epoch_rows[0]["train_pad_loss"]) > 0
    assert int(epoch_rows[0]["train_rare_class_count_win_loss_t_eq_1"]) > 0


def test_both_interval_evaluations_false_writes_no_interval_file(
    tmp_path: Path,
) -> None:
    """Both sides off means no row at all, not a file full of blank cells.

    This is the overfit profile's configuration: every number is reported once
    per epoch instead. The accumulation wiring must stay intact regardless, so
    the epoch row is still fully populated -- including the rare-class cells,
    whose per-epoch totals are exactly what the interval rows would have been
    summed over.
    """

    config = _small_config(tmp_path)
    config = replace(
        config,
        train=replace(
            config.train,
            interval_train_evaluation=False,
            interval_dev_evaluation=False,
        ),
    )
    examples = make_synthetic_examples(config, count=24)
    train_loader = DataLoader(examples[:20], batch_size=2, collate_fn=_collate_pretrain)
    dev_loader = DataLoader(examples[20:], batch_size=2, collate_fn=_collate_pretrain)
    interval_path = tmp_path / "interval_metrics.csv"
    epoch_path = tmp_path / "epoch_metrics.csv"
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(
        model=model,
        config=config,
        seed=94,
        interval_metrics_path=interval_path,
        epoch_metrics_path=epoch_path,
    )

    loop.fit(train_loader, val_dataloader=dev_loader, max_steps=10, epochs=1, fixed_t=1.0)

    # No all-blank rows, and no empty CSV left behind either.
    assert not interval_path.exists()

    # Everything still lands in the epoch row, for both splits.
    with epoch_path.open(newline="", encoding="utf-8") as handle:
        epoch_rows = list(csv.DictReader(handle))
    assert len(epoch_rows) == 1
    row = epoch_rows[0]
    assert float(row["train_loss"]) > 0
    assert float(row["dev_loss"]) > 0
    assert int(row["train_rare_class_count_delimiter_t_eq_1"]) > 0
    assert int(row["dev_rare_class_count_delimiter_t_eq_1"]) > 0
    assert float(row["train_rare_class_loss_delimiter_t_eq_1"]) > 0
    assert float(row["dev_rare_class_loss_delimiter_t_eq_1"]) > 0


def test_epoch_metrics_migrates_an_existing_narrower_schema(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    examples = make_synthetic_examples(config, count=2)
    train_loader = DataLoader(examples, batch_size=2, collate_fn=_collate_pretrain)
    csv_path = tmp_path / "epoch_metrics.csv"
    csv_path.write_text("epoch,train_loss\n0,9.0\n", encoding="utf-8")
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=75, epoch_metrics_path=csv_path)

    loop.fit(train_loader, max_steps=1, epochs=1, fixed_t=1.0)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["epoch"] == "0"
    # The migrated legacy row has no value for the newer t-bucket column; the
    # freshly-written epoch (run with fixed_t=1.0) populates the exact-t==1
    # bucket; the migration is asserted through this newer column.
    assert rows[0]["train_t_bucket_loss_t_eq_1"] == ""
    assert float(rows[1]["train_t_bucket_loss_t_eq_1"]) > 0


def test_pretraining_epoch_metrics_has_all_seven_classes_including_winloss(tmp_path: Path) -> None:

    config = _small_config(tmp_path)
    assert config.data.debut_mode is False
    examples = make_synthetic_examples(config, count=2)
    train_loader = DataLoader(examples, batch_size=2, collate_fn=_collate_pretrain)
    csv_path = tmp_path / "epoch_metrics.csv"
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=81, epoch_metrics_path=csv_path)

    loop.fit(train_loader, max_steps=1, epochs=1, fixed_t=1.0)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    expected_class_columns = {
        f"{split}_{name}_loss"
        for split in ("train", "dev")
        for name in (
            "enemy_observed",
            "enemy_fogged",
            "enemy_future",
            "delimiter",
            "end",
            "pad",
            "win_loss",
        )
    }
    present_class_columns = {name for name in fieldnames if name.endswith("_loss") and name not in {"train_loss", "dev_loss"}}
    # Exclude the t-bucket / perspective / canvas-state breakdown columns, which
    # also end in a bucket name but use distinct "t_bucket_loss_",
    # "perspective_loss_", and "canvas_state_loss_" naming schemes (they are not
    # per-class columns).
    present_class_columns = {
        name
        for name in present_class_columns
        if not any(
            marker in name
            for marker in ("t_bucket_loss", "perspective_loss", "canvas_state_loss")
        )
    }
    assert present_class_columns == expected_class_columns
    assert "train_win_loss_loss" in fieldnames
    assert "train_enemy_observed_loss" in fieldnames
    assert "train_enemy_fogged_loss" in fieldnames
    assert "train_enemy_future_loss" in fieldnames
    assert all(float(rows[0][column]) >= 0 for column in expected_class_columns if rows[0][column] != "")


def test_debut_mode_epoch_metrics_has_all_seven_classes_populated_from_epoch_one(tmp_path: Path) -> None:
    """Debut mode must log all 7 debut classes, populated from the FIRST epoch.

    Every synthetic debut canvas built by ``_make_debut_synthetic_examples``
    below contains one token of each of the 7 debut classes (visible-debut,
    fogged-debut, future-debut, delimiter, win-loss, end, pad), so every
    train_/dev_ column for those classes should be a real (non-empty) numeric
    value starting at epoch 1 -- there is no "ramp-up" period where a debut
    class is simply absent from the data.
    """

    config = _small_debut_config(tmp_path)
    examples = _make_debut_synthetic_examples(config, count=4)
    train_loader = DataLoader(examples[:2], batch_size=2, collate_fn=_collate_debut)
    dev_loader = DataLoader(examples[2:], batch_size=2, collate_fn=_collate_debut)
    csv_path = tmp_path / "epoch_metrics.csv"
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=82, epoch_metrics_path=csv_path)

    loop.fit(train_loader, val_dataloader=dev_loader, max_steps=1, epochs=1, fixed_t=1.0)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    first_row = rows[0]
    assert first_row["epoch"] == "1"
    # DEBUT_CLASS_ID_TO_NAME.values() = visible-debut, fogged-debut,
    # future-debut, delimiter, win-loss, end, pad -- sanitized the same way
    # train/loop.py's `_metric_class_name` sanitizes names for CSV headers.
    expected_debut_columns = [
        "visible_debut",
        "fogged_debut",
        "future_debut",
        "delimiter",
        "win_loss",
        "end",
        "pad",
    ]
    for name in expected_debut_columns:
        for split in ("train", "dev"):
            column = f"{split}_{name}_loss"
            assert column in first_row, f"missing column {column}"
            value = first_row[column]
            assert value != "", f"{column} is empty on the first epoch"
            assert float(value) >= 0.0


def _make_debut_synthetic_examples(config: ProjectConfig, *, count: int) -> list[DatasetExample]:
    """Build tiny synthetic debut-mode canvases containing all 7 debut classes.

    Mirrors ``thesis_ml.train.train.make_synthetic_examples`` (the
    pretraining fixture) but lays out a debut-style canvas: a single win/loss
    clamped BOS at position 0, outcome token at position 1
    (``CLASS_WINLOSS``), followed by one token of
    each of the other 6 debut classes. This lets the per-class-loss test
    above assert every debut column is populated without needing a real
    replay or the full ``_build_debut_target`` pipeline (which depends on
    on-disk metadata unavailable in unit tests).
    """

    debut_canvas = torch.tensor(
        [
            BOS_ID,
            WIN_ID,
            100,
            DELIMITER_ID,
            102,
            DELIMITER_ID,
            104,
            DELIMITER_ID,
            END_ID,
            PAD_ID,
            PAD_ID,
            PAD_ID,
        ],
        dtype=torch.long,
    )
    debut_class_labels = torch.tensor(
        [
            CLASS_CLAMPED,
            CLASS_WINLOSS,
            CLASS_ENEMY_OBSERVED,  # "visible-debut"
            CLASS_DELIMITER,  # "delimiter"
            CLASS_ENEMY_FOGGED,  # "fogged-debut"
            CLASS_DELIMITER,
            CLASS_ENEMY_FUTURE,  # "future-debut"
            CLASS_DELIMITER,
            CLASS_END,  # "end"
            CLASS_PAD,  # "pad"
            CLASS_PAD,
            CLASS_PAD,
        ],
        dtype=torch.long,
    )
    assert {label for label in debut_class_labels.tolist() if label >= 0} == set(
        DEBUT_CLASS_ID_TO_NAME.keys()
    )
    examples = []
    for example_index in range(count):
        input_records = _synthetic_input_records(example_index)
        examples.append(
            DatasetExample(
                input_records=input_records,
                input_token_ids=torch.tensor([record.token_id for record in input_records], dtype=torch.long),
                target_canvas=debut_canvas.clone(),
                class_labels=debut_class_labels.clone(),
                terminated=True,
                truncated=False,
                canvas_metadata=[
                    {"token_id": int(token_id), "timestep_index": index // 3}
                    for index, token_id in enumerate(debut_canvas.tolist())
                ],
                fogged_counts={},
                observed_counts={},
                window_start=example_index,
                perspective_player="p1" if example_index % 2 == 0 else "p2",
            )
        )
    return examples


def test_training_prints_live_epoch_and_batch_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base_config = _small_config(tmp_path)
    config = replace(
        base_config,
        train=replace(base_config.train, accumulation_steps=2),
    )
    examples = make_synthetic_examples(config, count=8)
    train_loader = DataLoader(examples, batch_size=2, shuffle=False, collate_fn=_collate_pretrain)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    step_metrics_path = tmp_path / "step_metrics.jsonl"
    epoch_metrics_path = tmp_path / "epoch_metrics.csv"
    loop = TrainingLoop(
        model=model,
        config=config,
        seed=72,
        metrics_path=step_metrics_path,
        epoch_metrics_path=epoch_metrics_path,
    )

    loop.fit(train_loader, max_steps=4, epochs=2, fixed_t=1.0)

    output = capsys.readouterr().out
    assert "phase=train epoch=1/2 batch=1/4" in output
    assert "phase=train epoch=1/2 batch=4/4" in output
    assert "phase=train epoch=2/2 batch=1/4" in output
    assert "phase=train epoch=2/2 batch=4/4" in output
    assert "step=1 step_wall_seconds=" in output
    assert "tokens_per_second=" in output
    assert "loss=" in output
    assert "epoch_loss_so_far=" in output
    # Routine progress stays readable; memory telemetry remains durable below.
    assert "cuda_max_memory_allocated_gb=" not in output
    assert "cuda_memory_reserved_gb=" not in output
    assert "cuda_device_memory_used_gb=" not in output
    assert "cuda_device_memory_gap_gb=" not in output

    step_rows = [
        json.loads(line)
        for line in step_metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    with epoch_metrics_path.open(newline="", encoding="utf-8") as handle:
        epoch_rows = list(csv.DictReader(handle))

    assert len(step_rows) == 4
    assert len(epoch_rows) == 2
    # On the first optimizer update of each epoch, the step mean and epoch mean
    # cover the same two accumulated microbatches and must therefore be exact.
    assert step_rows[0]["loss"] == pytest.approx(step_rows[0]["epoch_loss_so_far"])
    assert step_rows[2]["loss"] == pytest.approx(step_rows[2]["epoch_loss_so_far"])
    # The cumulative value resets each epoch, then converges to the exact
    # train_loss persisted for that epoch.
    assert step_rows[1]["epoch_loss_so_far"] == pytest.approx(
        float(epoch_rows[0]["train_loss"])
    )
    assert step_rows[3]["epoch_loss_so_far"] == pytest.approx(
        float(epoch_rows[1]["train_loss"])
    )
    for row in step_rows:
        assert row["loss"] > 0
        assert row["epoch_loss_so_far"] > 0
        for memory_field in (
            "cuda_max_memory_allocated_bytes",
            "cuda_memory_allocated_bytes",
            "cuda_memory_reserved_bytes",
            "cuda_inactive_split_bytes",
            "cuda_device_memory_used_bytes",
            "cuda_device_memory_gap_bytes",
        ):
            assert memory_field in row


class _RecordingDataset(Dataset):
    """Wrap a list of examples and record every index the DataLoader fetches.

    Lets a test observe the EXACT order and set of examples training touched, so
    it can prove a mid-epoch resume advances through the epoch (fetches the
    still-unseen batches) instead of replaying it from the first batch.
    """

    def __init__(self, examples: list[DatasetExample]) -> None:
        self._examples = examples
        self.served: list[int] = []

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> DatasetExample:
        self.served.append(int(index))
        return self._examples[index]


class _RaiseAfterNForwards(SC2StrategyDiffusionModel):
    """Model that raises mid-epoch to emulate a spot-preemption / Ctrl+C kill.

    A real interruption never runs the epoch-end bookkeeping; the newest
    on-disk state is whatever the periodic checkpoint wrote. Raising after the
    Nth forward reproduces that: checkpoints for the completed steps land on
    disk, then training dies partway through the epoch.
    """

    def __init__(self, config: ProjectConfig, *, vocab_size: int, raise_after: int) -> None:
        super().__init__(config, vocab_size=vocab_size)
        self._raise_after = raise_after
        self._forwards = 0

    def forward(self, *args, **kwargs):
        self._forwards += 1
        if self._forwards > self._raise_after:
            raise RuntimeError("simulated mid-epoch interruption")
        return super().forward(*args, **kwargs)


def _resumable_loader(dataset: Dataset, *, base_seed: int) -> DataLoader:
    """Build a single-example-per-batch loader backed by ResumableBatchSampler."""

    batch_sampler = ResumableBatchSampler(
        dataset_size=len(dataset),
        batch_size=1,
        base_seed=base_seed,
        drop_last=False,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        collate_fn=_collate_pretrain,
    )


def test_resume_continues_epoch_instead_of_replaying_batches(tmp_path: Path) -> None:
    # One epoch of six single-example batches; checkpoint every step so a kill
    # leaves a mid-epoch checkpoint on disk (as a real preemption would).
    base_config = _small_config(tmp_path)
    config = replace(
        base_config,
        model=replace(base_config.model, self_conditioning=False),
        train=replace(base_config.train, checkpoint_interval=1, max_steps=0),
    )
    sampler_seed = 1234

    # The deterministic per-epoch order the sampler produces for epoch 0. Both
    # the interrupted run and the resumed run must follow exactly this order.
    reference = ResumableBatchSampler(dataset_size=6, batch_size=1, base_seed=sampler_seed)
    reference.set_epoch(0)
    expected_order = [batch[0] for batch in reference]
    assert sorted(expected_order) == list(range(6))

    examples = make_synthetic_examples(config, count=6)

    # ---- Interrupted run: die after the 3rd batch's forward pass. ----------
    dataset_a = _RecordingDataset(examples)
    loader_a = _resumable_loader(dataset_a, base_seed=sampler_seed)
    model_a = _RaiseAfterNForwards(config, vocab_size=128, raise_after=3)
    loop_a = TrainingLoop(model=model_a, config=config, seed=7)
    with pytest.raises(RuntimeError, match="simulated mid-epoch interruption"):
        loop_a.fit(loader_a, epochs=1, fixed_t=1.0)

    # The interrupted run trained on exactly the first three batches, in order.
    assert dataset_a.served[:3] == expected_order[:3]

    # The on-disk checkpoint records intra-epoch progress, not just the step.
    resumed_loop = TrainingLoop(
        model=SC2StrategyDiffusionModel(config, vocab_size=128),
        config=config,
        seed=7,
    )
    resumed_loop.load_checkpoint(tmp_path / "checkpoints" / "last.pt")
    assert resumed_loop.global_step == 3
    assert resumed_loop.completed_epochs == 0
    assert resumed_loop.batches_completed_in_epoch == 3

    # ---- Resumed run: must fetch ONLY the three not-yet-seen batches. ------
    dataset_b = _RecordingDataset(examples)
    loader_b = _resumable_loader(dataset_b, base_seed=sampler_seed)
    resumed_loop.fit(loader_b, epochs=1, fixed_t=1.0)

    # This is the crux: the resume advances through the epoch (the remaining
    # tail of the deterministic order) and never re-touches the first three
    # batches. Before the fix it would have replayed from expected_order[0].
    assert dataset_b.served == expected_order[3:]
    assert set(dataset_b.served).isdisjoint(expected_order[:3])

    # Epoch finished cleanly: counters advance and the intra-epoch offset resets
    # so the next epoch would start at batch 0.
    assert resumed_loop.completed_epochs == 1
    assert resumed_loop.batches_completed_in_epoch == 0
    assert resumed_loop.global_step == 6


def test_bounded_fit_preserves_partial_epoch_resume_offset(tmp_path: Path) -> None:
    base_config = _small_config(tmp_path)
    config = replace(
        base_config,
        model=replace(base_config.model, self_conditioning=False),
        train=replace(base_config.train, checkpoint_interval=1, max_steps=0),
    )
    sampler_seed = 4321
    reference = ResumableBatchSampler(dataset_size=6, batch_size=1, base_seed=sampler_seed)
    expected_order = [batch[0] for batch in reference]
    examples = make_synthetic_examples(config, count=6)

    first_dataset = _RecordingDataset(examples)
    first_loop = TrainingLoop(
        model=SC2StrategyDiffusionModel(config, vocab_size=128),
        config=config,
        seed=9,
    )
    first_loop.fit(
        _resumable_loader(first_dataset, base_seed=sampler_seed),
        max_steps=2,
        epochs=1,
        fixed_t=1.0,
    )
    assert first_loop.completed_epochs == 0
    assert first_loop.batches_completed_in_epoch == 2

    resumed_dataset = _RecordingDataset(examples)
    resumed_loop = TrainingLoop(
        model=SC2StrategyDiffusionModel(config, vocab_size=128),
        config=config,
        seed=9,
    )
    resumed_loop.load_checkpoint(tmp_path / "checkpoints" / "last.pt")
    resumed_loop.fit(
        _resumable_loader(resumed_dataset, base_seed=sampler_seed),
        max_steps=3,
        epochs=1,
        fixed_t=1.0,
    )
    assert resumed_dataset.served == [expected_order[2]]
    assert resumed_loop.completed_epochs == 0
    assert resumed_loop.batches_completed_in_epoch == 3
    assert resumed_loop.global_step == 3


def test_cuda_reserved_memory_limit_trims_reclaimable_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = replace(
        _small_config(tmp_path),
        train=replace(_small_config(tmp_path).train, max_cuda_reserved_gb=7.0),
    )
    loop = TrainingLoop(model=SC2StrategyDiffusionModel(config, vocab_size=128), config=config, seed=74)
    loop.device = torch.device("cuda")

    empty_cache_calls = 0

    def empty_cache() -> None:
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 2 * 1024**3)

    loop._enforce_cuda_memory_limit(7 * 1024**3)

    assert empty_cache_calls == 1
    output = capsys.readouterr().out
    assert "cuda_cache_trim reason=reserved_memory_ceiling" in output
    assert "reserved_before_gb=7.000" in output
    assert "reserved_after_gb=2.000" in output


def test_cuda_reserved_memory_limit_fails_after_cache_trim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        _small_config(tmp_path),
        train=replace(_small_config(tmp_path).train, max_cuda_reserved_gb=7.0),
    )
    loop = TrainingLoop(model=SC2StrategyDiffusionModel(config, vocab_size=128), config=config, seed=74)
    loop.device = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 7 * 1024**3)

    with pytest.raises(RuntimeError, match="reserved-memory safety limit exceeded after cache trim"):
        loop._enforce_cuda_memory_limit(7 * 1024**3)


def test_relative_early_stopping_requires_consecutive_subthreshold_epochs(tmp_path: Path) -> None:
    config = replace(
        _small_config(tmp_path),
        train=replace(
            _small_config(tmp_path).train,
            early_stopping_patience_epochs=2,
            early_stopping_min_relative_improvement=0.001,
        ),
    )
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=73)

    assert loop._should_stop_early(10.0) is False
    assert loop._should_stop_early(9.995) is False
    assert loop._should_stop_early(9.994) is True
    assert loop.epochs_without_improvement == 2
    assert loop.early_stopping_best_dev_loss == pytest.approx(10.0)


def _small_config(tmp_path: Path | None = None, **model_overrides: bool) -> ProjectConfig:
    config = load_config("config/default.yaml")
    return replace(
        config,
        data=replace(config.data, input_budget_tokens=64, canvas_budget_tokens=12),
        model=replace(config.model, d_model=32, layers=2, heads=4, ffn=64, **model_overrides),
        train=replace(
            config.train,
            lr=0.01,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
            adam_eps=1e-8,
            warmup=1,
            lr_floor_ratio=0.1,
            lr_schedule="cosine",
            accumulation_steps=1,
            target_effective_batch_tokens=0,
            max_steps=8,
            val_interval=0,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path / "checkpoints") if tmp_path is not None else "checkpoints/test",
            resume_checkpoint_subdir="",
            save_best_checkpoint=False,
            durable_checkpoint_interval_epochs=0,
            ema_decay=0.9,
            confidence_loss_weight=0.1,
            precision="fp32",
        ),
    )


def _small_debut_config(tmp_path: Path | None = None) -> ProjectConfig:
    """Fine-tuning (debut_mode=True) variant of `_small_config`.

    A debut-mode config is REQUIRED (by `_validate_shared_training_sections` /
    `CanvasCrossEntropyLoss`) to carry `fog` and `loss.class_loss_weights`, so
    both are populated here with plain uniform values -- the exact numbers are
    not what tests using this helper assert.
    """

    base = _small_config(tmp_path)
    return replace(
        base,
        data=replace(base.data, debut_mode=True),
        fog=FogConfig(
            rate_distribution=UniformDistributionConfig(name="uniform", min=0.0, max=0.8)
        ),
        loss=replace(
            base.loss,
            class_loss_weights=ClassLossWeightsConfig(
                enemy_observed_reconstruction=1.0,
                enemy_fogged_reconstruction=1.0,
                enemy_future_prediction=1.0,
                delimiter=1.0,
                end=1.0,
                pad=1.0,
                win_loss=1.0,
            ),
        ),
    )


def _loop_and_batch(config: ProjectConfig, *, seed: int) -> tuple[TrainingLoop, object]:
    torch.manual_seed(seed)
    examples = make_synthetic_examples(config, count=2)
    dataloader = DataLoader(examples, batch_size=2, shuffle=False, collate_fn=_collate_pretrain)
    batch = next(iter(dataloader))
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    return TrainingLoop(model=model, config=config, seed=seed), batch


class CountingDiffusionModel(SC2StrategyDiffusionModel):
    def __init__(self, config: ProjectConfig, *, vocab_size: int) -> None:
        super().__init__(config, vocab_size=vocab_size)
        self.forward_records: list[tuple[bool, bool]] = []

    def forward(self, *args, canvas_self_conditioning=None, **kwargs):
        self.forward_records.append((torch.is_grad_enabled(), canvas_self_conditioning is not None))
        return super().forward(*args, canvas_self_conditioning=canvas_self_conditioning, **kwargs)


def _assert_optimizer_states_match(first: dict, second: dict) -> None:
    assert first["param_groups"] == second["param_groups"]
    assert first["state"].keys() == second["state"].keys()
    for key in first["state"]:
        for state_name, state_value in first["state"][key].items():
            loaded_value = second["state"][key][state_name]
            if isinstance(state_value, torch.Tensor):
                assert torch.allclose(state_value, loaded_value)
            else:
                assert state_value == loaded_value


# ---------------------------------------------------------------------------
# Token-level reporting: accuracy, macro F1, bits per token, perplexity.
#
# These columns exist so a run can be read as something other than a loss curve.
# Accuracy and macro F1 are split by canvas state on purpose (see
# _finalize_token_metrics); bits-per-token and perplexity come from a SEPARATE
# unweighted cross-entropy accumulator, not from train_loss/dev_loss, because
# those are the class-weighted objective (see _finalize_bits_per_token).
# ---------------------------------------------------------------------------


def test_macro_f1_from_counts_matches_a_hand_computed_example() -> None:
    """Verify the macro F1 arithmetic against numbers worked out by hand.

    Three token ids in a 4-token vocabulary, with the fourth neither predicted
    nor present so it must be excluded from the average entirely:

      token 0: tp=2, predicted=4, target=2 -> P=0.5,  R=1.0,  F1=2/3
      token 1: tp=1, predicted=1, target=3 -> P=1.0,  R=1/3,  F1=0.5
      token 2: tp=0, predicted=1, target=1 -> P=0.0,  R=0.0,  F1=0.0
      token 3: absent everywhere            -> excluded

      macro F1 = (2/3 + 0.5 + 0.0) / 3
    """

    counts = torch.tensor(
        [
            [2, 1, 0, 0],  # true_positive
            [4, 1, 1, 0],  # predicted
            [2, 3, 1, 0],  # target
        ],
        dtype=torch.int64,
    )

    expected = ((2.0 / 3.0) + 0.5 + 0.0) / 3.0
    assert _macro_f1_from_counts(counts) == pytest.approx(expected)


def test_macro_f1_penalises_collapse_onto_the_common_token() -> None:
    """Macro F1 must fall when accuracy is bought by ignoring rare tokens.

    This is the property that earns macro F1 its column next to accuracy. The
    model here predicts the dominant token for every single position: 90 of 100
    positions come out right, so accuracy is 0.90, but it has completely given up
    on the other two tokens and the macro average reflects that.
    """

    counts = torch.tensor(
        [
            [90, 0, 0],   # only the common token is ever right
            [100, 0, 0],  # every position predicted as the common token
            [90, 5, 5],   # the two rare tokens really do occur
        ],
        dtype=torch.int64,
    )
    accuracy, macro_f1 = _finalize_token_metrics({"noised": counts})

    assert accuracy["noised"] == pytest.approx(0.90)
    # Common token: P=0.9, R=1.0 -> F1 = 2*0.9/1.9; the two rare tokens score 0.
    expected = (2.0 * 0.9 * 1.0 / 1.9) / 3.0
    assert macro_f1["noised"] == pytest.approx(expected)
    assert macro_f1["noised"] < accuracy["noised"]


def test_finalize_token_metrics_omits_a_state_that_scored_nothing() -> None:
    """An unscored canvas state is absent, not zero.

    A zero accuracy and a state that was never evaluated are different
    observations, and the CSV writers render an absent key as a blank cell so a
    reader can tell them apart.
    """

    populated = torch.tensor([[1, 0], [1, 1], [2, 0]], dtype=torch.int64)
    empty = torch.zeros(3, 2, dtype=torch.int64)
    accuracy, macro_f1 = _finalize_token_metrics(
        {"noised": populated, "ground_truth_preserved": empty}
    )

    assert set(accuracy) == {"noised"}
    assert set(macro_f1) == {"noised"}


def test_finalize_bits_per_token_converts_nats_and_matches_perplexity() -> None:
    """bits = nats / ln 2, and perplexity is the same quantity as 2 ** bits."""

    # 8 scored positions averaging ln(4) nats each -> exactly 2 bits per token,
    # so the model is effectively choosing between 4 equally likely tokens.
    bits, perplexity = _finalize_bits_per_token(8.0 * math.log(4.0), 8.0)

    assert bits == pytest.approx(2.0)
    assert perplexity == pytest.approx(4.0)
    assert perplexity == pytest.approx(2.0 ** bits)
    # Nothing scored -> nothing to report, rendered as blank cells.
    assert _finalize_bits_per_token(0.0, 0.0) == (None, None)


def test_bits_per_token_is_not_derived_from_the_class_weighted_loss(
    tmp_path: Path,
) -> None:
    """The reported bits must come from the UNWEIGHTED cross entropy.

    ``train_loss``/``dev_loss`` are normalised by the configured class weights,
    which the project deliberately sets far from 1.0 (``end`` is boosted, ``pad``
    is damped). Exponentiating that would not be a perplexity of any
    distribution. Running the same data through two configs that differ ONLY in
    those weights must therefore move the loss while leaving bits-per-token
    alone.
    """

    def bits_and_loss(weights: ClassLossWeightsConfig, seed: int) -> tuple[float, float]:
        config = _small_config(tmp_path)
        config = replace(config, loss=replace(config.loss, class_loss_weights=weights))
        examples = make_synthetic_examples(config, count=4)
        dev_loader = DataLoader(examples, batch_size=2, collate_fn=_collate_pretrain)
        torch.manual_seed(seed)
        model = SC2StrategyDiffusionModel(config, vocab_size=128)
        loop = TrainingLoop(model=model, config=config, seed=seed)
        torch.manual_seed(seed)
        validation = loop.validate(dev_loader, fixed_t=1.0)
        return validation.bits_per_token, validation.loss

    flat = ClassLossWeightsConfig(
        enemy_observed_reconstruction=1.0,
        enemy_fogged_reconstruction=1.0,
        enemy_future_prediction=1.0,
        delimiter=1.0,
        end=1.0,
        pad=1.0,
        win_loss=1.0,
    )
    skewed = replace(flat, end=25.0, pad=0.1)

    flat_bits, flat_loss = bits_and_loss(flat, seed=1234)
    skewed_bits, skewed_loss = bits_and_loss(skewed, seed=1234)

    # Same model, same data, same corruption draw: the unweighted information
    # content is identical...
    assert skewed_bits == pytest.approx(flat_bits, rel=1e-9)
    # ...while the weighted objective genuinely moved, which is exactly why the
    # bits column cannot be a transform of the loss column.
    assert skewed_loss != pytest.approx(flat_loss, rel=1e-6)


def test_epoch_metrics_csv_reports_token_accuracy_macro_f1_and_bits(
    tmp_path: Path,
) -> None:
    """The epoch CSV carries the token-metric columns, ordered per split.

    Column order matters to a human scanning the file: everything summarising the
    train split sits between ``train_loss`` and ``dev_loss``, and everything
    summarising dev follows ``dev_loss``.
    """

    config = _small_config(tmp_path)
    examples = make_synthetic_examples(config, count=8)
    train_loader = DataLoader(examples[:4], batch_size=2, collate_fn=_collate_pretrain)
    dev_loader = DataLoader(examples[4:], batch_size=2, collate_fn=_collate_pretrain)
    csv_path = tmp_path / "epoch_metrics.csv"
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=71, epoch_metrics_path=csv_path)

    # 4 train examples at batch_size 2 is 2 batches per epoch, so 4 steps is
    # exactly the 2 epochs asked for. The loop derives its epoch limit from
    # max_steps, so an over-large budget would silently run extra epochs.
    loop.fit(train_loader, val_dataloader=dev_loader, max_steps=4, epochs=2)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    assert len(rows) == 2
    # Ordering: the train block sits immediately after train_loss, the dev block
    # immediately after dev_loss.
    assert fieldnames[:15] == [
        "epoch",
        "train_loss",
        "train_accuracy_ground_truth_preserved",
        "train_accuracy_noised",
        "train_macro_f1_ground_truth_preserved",
        "train_macro_f1_noised",
        "train_bits_per_token",
        "train_perplexity",
        "dev_loss",
        "dev_accuracy_ground_truth_preserved",
        "dev_accuracy_noised",
        "dev_macro_f1_ground_truth_preserved",
        "dev_macro_f1_noised",
        "dev_bits_per_token",
        "dev_perplexity",
    ]

    for row in rows:
        for split in ("train", "dev"):
            bits = float(row[f"{split}_bits_per_token"])
            perplexity = float(row[f"{split}_perplexity"])
            assert bits > 0
            # Perplexity is the same quantity expressed as a branching factor.
            assert perplexity == pytest.approx(2.0 ** bits)
            # An untrained-ish model on a 128-token vocabulary cannot be more
            # uncertain than uniform over the vocabulary.
            assert bits <= math.log2(128) + 1e-6
            for state in ("ground_truth_preserved", "noised"):
                for metric in ("accuracy", "macro_f1"):
                    cell = row[f"{split}_{metric}_{state}"]
                    # Blank is legitimate: a canvas state can score no positions.
                    if cell:
                        assert 0.0 <= float(cell) <= 1.0


def test_interval_metrics_csv_reports_token_accuracy_macro_f1_and_bits(
    tmp_path: Path,
) -> None:
    """The intra-epoch CSV mirrors the epoch CSV's token-metric columns.

    On a corpus large enough that pre-training runs a single epoch, the interval
    rows are the only place these metrics show a trend at all, so they must be
    present here and not only in the epoch file.
    """

    config = _small_config(tmp_path)
    examples = make_synthetic_examples(config, count=24)
    train_loader = DataLoader(examples[:20], batch_size=2, collate_fn=_collate_pretrain)
    dev_loader = DataLoader(examples[20:], batch_size=2, collate_fn=_collate_pretrain)
    csv_path = tmp_path / "interval_metrics.csv"
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(
        model=model, config=config, seed=73, interval_metrics_path=csv_path
    )

    loop.fit(train_loader, val_dataloader=dev_loader, max_steps=20, epochs=2)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    assert len(rows) == 2 * INTERVAL_REPORTS_PER_EPOCH
    for split in ("train", "dev"):
        assert f"{split}_bits_per_token" in fieldnames
        assert f"{split}_perplexity" in fieldnames
        for state in ("ground_truth_preserved", "noised"):
            assert f"{split}_accuracy_{state}" in fieldnames
            assert f"{split}_macro_f1_{state}" in fieldnames
    for row in rows:
        for split in ("train", "dev"):
            bits = float(row[f"{split}_bits_per_token"])
            assert bits > 0
            assert float(row[f"{split}_perplexity"]) == pytest.approx(2.0 ** bits)


def test_token_metrics_pool_by_position_not_by_averaging_batch_ratios() -> None:
    """Counts must be summed across batches before any ratio is taken.

    This is the whole reason the loss module returns raw counts instead of a
    finished accuracy. The two batches below are deliberately lopsided: a large
    one the model gets entirely right and a small one it gets entirely wrong.

      pooled over positions : 90 correct out of 100          -> 0.90
      averaged per batch    : (1.0 + 0.0) / 2                -> 0.50

    A pooling bug would therefore be off by a wide, obvious margin rather than by
    a rounding error, which is what makes this worth asserting.
    """

    # Batch A: 90 positions, every one correct, all on token id 0.
    large_correct_batch = torch.tensor(
        [[90, 0], [90, 0], [90, 0]], dtype=torch.int64
    )
    # Batch B: 10 positions, every one wrong -- token 1 was the target, token 0
    # was predicted.
    small_wrong_batch = torch.tensor([[0, 0], [10, 0], [0, 10]], dtype=torch.int64)

    accumulator: dict[str, torch.Tensor] = {}
    _accumulate_token_class_counts(accumulator, {"noised": large_correct_batch})
    _accumulate_token_class_counts(accumulator, {"noised": small_wrong_batch})

    accuracy, _ = _finalize_token_metrics(accumulator)

    assert accuracy["noised"] == pytest.approx(0.90)
    # The value an average-of-per-batch-ratios implementation would produce.
    assert accuracy["noised"] != pytest.approx(0.50)
    # Accumulation must not have mutated either caller-supplied tensor.
    assert large_correct_batch[0].tolist() == [90, 0]
    assert small_wrong_batch[1].tolist() == [10, 0]
