from dataclasses import replace
import csv
from functools import partial
from pathlib import Path

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
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.train.corruption import corrupt_batch, inverse_t_weights
from thesis_ml.train.loop import TrainingLoop, auxiliary_confidence_loss
from thesis_ml.train.train import _synthetic_input_records, make_synthetic_examples, run_smoke_train
from thesis_ml.vocab.special_tokens import DELIMITER_ID, END_ID, PAD_ID, WIN_ID

# All fixtures in this file (except `_make_debut_synthetic_examples`, used only
# by the one debut-mode test below) are PRE-TRAINING-shaped (see
# `make_synthetic_examples`'s docstring in train/train.py): absent input, the
# collapsed CLASS_CONTENT taxonomy. `collate_diffusion_examples` requires an
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

    # The smoke train is a PRE-TRAINING run, so its per-class log uses the
    # collapsed 5-name pre-training taxonomy (PRETRAIN_CLASS_ID_TO_NAME); the
    # synthetic fixtures carry one label of each of those ids.
    examples = make_synthetic_examples(_small_config(tmp_path), count=1)
    expected_classes = {
        PRETRAIN_CLASS_ID_TO_NAME[int(label)] for label in examples[0].class_labels.unique()
    }
    assert set(first.per_class) == expected_classes
    assert all(value > 0 for value in first.per_class.values())


def test_corruption_never_masks_input_region() -> None:
    config = _small_config()
    input_token_ids = torch.tensor([[100, 101, 102], [103, 104, 105]])
    target_canvas = torch.tensor([[100, 101, 102, 103], [104, 105, 106, 107]])
    generator = torch.Generator(device="cpu").manual_seed(1)

    for t in (0.0, 0.25, 0.75, 1.0):
        corrupted = corrupt_batch(
            input_token_ids=input_token_ids,
            target_canvas=target_canvas,
            schedule=config.diffusion.mask_schedule,
            generator=generator,
            t=t,
        )
        assert torch.equal(corrupted.input_token_ids, input_token_ids)
        assert not (corrupted.input_token_ids == 0).any()
        if t == 1.0:
            assert corrupted.masked_positions.all()
            assert (corrupted.noised_canvas == 0).all()


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
    schedule = config.diffusion.mask_schedule
    assert schedule.t_one_fraction == 0.1

    batch = 10_000
    target_canvas = torch.zeros((batch, 4), dtype=torch.long)
    input_token_ids = torch.zeros((batch, 0), dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(1234)

    corrupted = corrupt_batch(
        input_token_ids=input_token_ids,
        target_canvas=target_canvas,
        schedule=schedule,
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
        schedule=disabled_schedule,
        generator=generator,
        t=None,
    )
    assert int((disabled.t == 1.0).sum()) == 0


def test_per_epoch_reseed_makes_masking_deterministic_and_epochs_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fit() reseeds the generator to base_seed+epoch at every epoch boundary.

    Three claims (the per-epoch reseed contract from train/loop.py):
      1. Two runs with the same seed produce IDENTICAL per-epoch masking.
      2. Within one run, epoch 0's and epoch 1's masks DIFFER (the reseed is
         per-epoch, not a frozen repeat of one stream).
      3. Reseeding a fresh generator to base_seed + epoch reproduces that
         epoch's corruption stream exactly (this is what makes a mid-training
         resume replay the same draws an uninterrupted run would have made).
    """

    import thesis_ml.train.loop as loop_module

    def run_and_capture(seed: int) -> list[torch.Tensor]:
        """Run 2 epochs x 2 steps and record every corruption mask, in order."""

        captured: list[torch.Tensor] = []
        real_corrupt_batch = corrupt_batch

        def spy_corrupt_batch(**kwargs):
            output = real_corrupt_batch(**kwargs)
            captured.append(output.masked_positions.clone())
            return output

        monkeypatch.setattr(loop_module, "corrupt_batch", spy_corrupt_batch)
        config = _small_config(tmp_path)
        torch.manual_seed(seed)
        examples = make_synthetic_examples(config, count=4)
        loader = DataLoader(examples, batch_size=2, shuffle=False, collate_fn=_collate_pretrain)
        model = SC2StrategyDiffusionModel(config, vocab_size=128)
        loop = TrainingLoop(model=model, config=config, seed=seed)
        # fixed_t=None so masking consumes the loop's own generator (the
        # subject of the reseed); 2 batches/epoch x 2 epochs = 4 steps.
        loop.fit(loader, max_steps=4, epochs=2)
        monkeypatch.setattr(loop_module, "corrupt_batch", real_corrupt_batch)
        return captured

    first_run = run_and_capture(seed=123)
    second_run = run_and_capture(seed=123)
    assert len(first_run) == len(second_run) == 4

    # 1. Same seed -> identical masking, step by step, across BOTH epochs.
    for first_mask, second_mask in zip(first_run, second_run, strict=True):
        assert torch.equal(first_mask, second_mask)

    # 2. Epoch 0 (steps 0-1) and epoch 1 (steps 2-3) draw DIFFERENT masks.
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
                schedule=config.diffusion.mask_schedule,
                generator=replay_generator,
                t=None,
            )
            recorded = first_run[epoch_index * 2 + step_in_epoch]
            assert torch.equal(replayed.masked_positions, recorded)


def test_outcome_position_zero_is_masked_iid_and_scored_like_any_position(tmp_path: Path) -> None:
    """REGRESSION: canvas position 0 (the win/loss token) gets no training exemption.

    Position 0 is masked iid at rate t exactly like every other canvas
    position and contributes to the loss with a NONZERO class weight in BOTH
    training modes. `outcome_last` (denoise the outcome token last) is a
    SAMPLER-only inference constraint -- nothing in corruption, scoring, or
    loss weighting ever special-cases position 0 during training.
    """

    # t=1.0 masks EVERY canvas position -- including position 0.
    config = _small_config(tmp_path)
    loop, batch = _loop_and_batch(config, seed=93)
    result = loop.compute_batch_loss(batch, fixed_t=1.0)
    assert result.corruption.masked_positions[:, 0].all()
    assert result.scored_mask[:, 0].all()
    # The fixtures put CLASS_WINLOSS at position 0; its per-class loss is
    # therefore populated (it received loss like any other class).
    assert batch.class_labels[0, 0].item() == CLASS_WINLOSS
    assert "win-loss" in result.loss_output.per_class

    # Its loss WEIGHT is nonzero in both modes (only [PAD] is ever zeroed).
    from thesis_ml.model.loss import CanvasCrossEntropyLoss

    pretrain_weights = CanvasCrossEntropyLoss(config).class_weights
    debut_weights = CanvasCrossEntropyLoss(_small_debut_config(tmp_path)).class_weights
    assert pretrain_weights[CLASS_WINLOSS].item() > 0.0
    assert debut_weights[CLASS_WINLOSS].item() > 0.0

    # And at intermediate t the mask at position 0 is a plain iid Bernoulli(t)
    # draw: across many examples it is sometimes masked and sometimes not --
    # never always-exempt (and never always-forced).
    generator = torch.Generator(device="cpu").manual_seed(7)
    target_canvas = torch.zeros((2_000, 4), dtype=torch.long)
    corrupted = corrupt_batch(
        input_token_ids=torch.zeros((2_000, 0), dtype=torch.long),
        target_canvas=target_canvas,
        schedule=config.diffusion.mask_schedule,
        generator=generator,
        t=0.5,
    )
    position_zero_rate = float(corrupted.masked_positions[:, 0].float().mean())
    assert 0.4 <= position_zero_rate <= 0.6


def test_training_scores_exactly_masked_canvas_positions(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    loop, batch = _loop_and_batch(config, seed=9)

    result = loop.compute_batch_loss(batch, fixed_t=0.5)

    assert torch.equal(result.scored_mask, result.corruption.masked_positions)
    assert not result.scored_mask[result.corruption.masked_positions.logical_not()].any()
    assert result.canvas_logits.shape[1] == batch.target_canvas.shape[1]


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

    assert result.self_conditioning_used is True
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

    assert result.self_conditioning_used is False
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


def test_ema_tracks_training_and_validation_uses_ema_weights(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    loop, batch = _loop_and_batch(config, seed=31)
    initial_ema = [parameter.detach().clone() for parameter in loop.ema_model.parameters()]

    loop.fit([batch], max_steps=1, fixed_t=1.0)

    assert any(
        not torch.allclose(before, after)
        for before, after in zip(initial_ema, loop.ema_model.parameters(), strict=True)
    )
    assert any(
        not torch.allclose(raw, ema)
        for raw, ema in zip(loop.model.parameters(), loop.ema_model.parameters(), strict=True)
    )

    validation = loop.validate([batch], fixed_t=1.0)
    expected = loop.compute_batch_loss(batch, fixed_t=1.0, model=loop.ema_model)
    assert validation.loss == pytest.approx(float(expected.loss.detach()))
    assert validation.per_class


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
    assert [log.masked_fraction for log in first] == pytest.approx([log.masked_fraction for log in second])
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
    # Pre-training has NO input, NO fog, and NO future class, so the epoch CSV
    # must carry NONE of the input-timestep / future-distance columns at all
    # (they are fine-tuning-only, not merely empty).
    assert "average_input_timesteps" not in fieldnames
    assert "average_enemy_future_timesteps" not in fieldnames
    assert "input_timestep_p50" not in fieldnames
    assert "enemy_future_timestep_p50" not in fieldnames
    assert "train_enemy_future_loss_distance_1" not in fieldnames
    assert "dev_enemy_future_loss_distance_1" not in fieldnames
    # t-bucket / perspective breakdown columns (Worker 3): with fixed_t=1.0
    # every example lands in the exact-t==1 bucket; the other four bucket
    # columns exist but are written as "" (the empty-bucket convention).
    assert all(float(row["train_t_bucket_loss_t_eq_1"]) > 0 for row in rows)
    assert all(float(row["dev_t_bucket_loss_t_eq_1"]) > 0 for row in rows)
    assert all(row["train_t_bucket_loss_t_0_7_to_1_0"] == "" for row in rows)
    assert all(row["train_t_bucket_loss_t_0_0_to_0_3"] == "" for row in rows)
    # The fixtures alternate p1/p2 perspectives, so both perspective columns
    # are populated in train and dev.
    assert all(float(row["train_perspective_loss_p1"]) > 0 for row in rows)
    assert all(float(row["train_perspective_loss_p2"]) > 0 for row in rows)
    assert all(float(row["dev_perspective_loss_p1"]) > 0 for row in rows)
    assert all(float(row["dev_perspective_loss_p2"]) > 0 for row in rows)
    # 24 tokens per epoch, canvas only: pre-training input is literally absent
    # (zero input tokens), leaving 2 examples x 12 canvas tokens per epoch.
    assert [int(row["total_tokens_ingested"]) for row in rows] == [24, 48]
    # 10 unique ids, all from the canvas: [WIN], six content ids (100..105),
    # [DELIMITER], [END], [PAD]. No input tokens exist to add more.
    assert [int(row["total_unique_tokens_seen"]) for row in rows] == [10, 10]
    # Per-class columns use the collapsed pre-training taxonomy: "content"
    # replaces the old observed/fogged/future trio.
    assert "train_content_loss" in fieldnames
    assert "dev_pad_loss" in fieldnames
    assert "train_enemy_observed_loss" not in fieldnames
    assert "train_enemy_fogged_loss" not in fieldnames
    assert "train_enemy_future_loss" not in fieldnames


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
    # bucket. (average_input_timesteps is no longer a pre-training column, so
    # a pre-training-schema migration is asserted via the t-bucket column.)
    assert rows[0]["train_t_bucket_loss_t_eq_1"] == ""
    assert float(rows[1]["train_t_bucket_loss_t_eq_1"]) > 0


def test_pretraining_epoch_metrics_has_all_five_collapsed_classes_including_winloss(tmp_path: Path) -> None:
    """Pre-training's epoch CSV carries exactly the 5 collapsed class columns.

    With debut_mode False (pre-training), the class taxonomy is COLLAPSED: the
    old observed/fogged/future trio becomes a single "content" class (there is
    no input and no fog to distinguish them), leaving content, delimiter, end,
    pad, and win_loss (the canvas still leads with the [WIN]/[LOSS] token at
    position 0). The CSV must contain exactly those 5 class columns per split
    -- the observed/fogged/future columns must NOT exist at all.
    """

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
        for name in ("content", "delimiter", "end", "pad", "win_loss")
    }
    present_class_columns = {name for name in fieldnames if name.endswith("_loss") and name not in {"train_loss", "dev_loss"}}
    # Exclude the t-bucket / perspective breakdown columns, which also end in a
    # bucket name but use the distinct "t_bucket_loss_"/"perspective_loss_"
    # naming scheme (they are not per-class columns).
    present_class_columns = {
        name
        for name in present_class_columns
        if "t_bucket_loss" not in name and "perspective_loss" not in name
    }
    assert present_class_columns == expected_class_columns
    assert "train_win_loss_loss" in fieldnames
    # The pre-fine-tuning-only class names must be absent entirely.
    assert "train_enemy_observed_loss" not in fieldnames
    assert "train_enemy_fogged_loss" not in fieldnames
    assert "train_enemy_future_loss" not in fieldnames
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
    outcome token at position 0 (``CLASS_WINLOSS``), followed by one token of
    each of the other 6 debut classes. This lets the per-class-loss test
    above assert every debut column is populated without needing a real
    replay or the full ``_build_debut_target`` pipeline (which depends on
    on-disk metadata unavailable in unit tests).
    """

    debut_canvas = torch.tensor(
        [
            WIN_ID,
            100,
            101,
            DELIMITER_ID,
            102,
            103,
            DELIMITER_ID,
            104,
            105,
            DELIMITER_ID,
            END_ID,
            PAD_ID,
        ],
        dtype=torch.long,
    )
    debut_class_labels = torch.tensor(
        [
            CLASS_WINLOSS,
            CLASS_ENEMY_OBSERVED,  # "visible-debut"
            CLASS_ENEMY_OBSERVED,
            CLASS_DELIMITER,  # "delimiter"
            CLASS_ENEMY_FOGGED,  # "fogged-debut"
            CLASS_ENEMY_FOGGED,
            CLASS_DELIMITER,
            CLASS_ENEMY_FUTURE,  # "future-debut"
            CLASS_ENEMY_FUTURE,
            CLASS_DELIMITER,
            CLASS_END,  # "end"
            CLASS_PAD,  # "pad"
        ],
        dtype=torch.long,
    )
    assert set(debut_class_labels.tolist()) == set(DEBUT_CLASS_ID_TO_NAME.keys())
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
    config = _small_config(tmp_path)
    examples = make_synthetic_examples(config, count=4)
    train_loader = DataLoader(examples, batch_size=2, shuffle=False, collate_fn=_collate_pretrain)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    loop = TrainingLoop(model=model, config=config, seed=72)

    loop.fit(train_loader, max_steps=4, epochs=2, fixed_t=1.0)

    output = capsys.readouterr().out
    assert "phase=train epoch=1/2 batch=1/2" in output
    assert "phase=train epoch=1/2 batch=2/2" in output
    assert "phase=train epoch=2/2 batch=1/2" in output
    assert "phase=train epoch=2/2 batch=2/2" in output
    assert "step=1 step_wall_seconds=" in output
    assert "tokens_per_second=" in output
    assert "cuda_max_memory_allocated_gb=0.000" in output
    assert "cuda_memory_reserved_gb=0.000" in output
    assert "cuda_device_memory_used_gb=0.000" in output
    assert "cuda_device_memory_gap_gb=0.000" in output


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


def test_cuda_reserved_memory_limit_fails_hard(tmp_path: Path) -> None:
    config = replace(
        _small_config(tmp_path),
        train=replace(_small_config(tmp_path).train, max_cuda_reserved_gb=7.0),
    )
    loop = TrainingLoop(model=SC2StrategyDiffusionModel(config, vocab_size=128), config=config, seed=74)
    loop.device = torch.device("cuda")

    with pytest.raises(RuntimeError, match="reserved-memory safety limit exceeded"):
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


def _small_config(tmp_path: Path | None = None) -> ProjectConfig:
    config = load_config("config/default.yaml")
    return replace(
        config,
        data=replace(config.data, input_budget_tokens=64, canvas_budget_tokens=12),
        model=replace(config.model, d_model=32, layers=2, heads=4, ffn=64),
        train=replace(
            config.train,
            lr=0.01,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
            adam_eps=1e-8,
            warmup=1,
            lr_floor_ratio=0.1,
            accumulation_steps=1,
            target_effective_batch_tokens=0,
            max_steps=8,
            val_interval=0,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path / "checkpoints") if tmp_path is not None else "checkpoints/test",
            ema_decay=0.9,
            confidence_loss_weight=0.1,
            precision="fp32",
        ),
    )


def _small_debut_config(tmp_path: Path | None = None) -> ProjectConfig:
    """Fine-tuning (debut_mode=True) variant of `_small_config`.

    A debut-mode config is REQUIRED (by `_validate_debut_mode_sections` /
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
