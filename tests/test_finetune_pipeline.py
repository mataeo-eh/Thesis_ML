"""Tests for the fine-tune pipeline's warm-start mechanics and config (Worker 5).

Two things are exercised here:

1. `TrainingLoop.load_model_weights` -- the weights-only "warm start" helper
   fine-tuning uses instead of the full-resume `load_checkpoint` path. We
   prove it copies model/EMA weights from a checkpoint while leaving the
   optimizer state, `global_step`, and `completed_epochs` exactly as a
   freshly-constructed loop would have them (i.e. it does NOT also restore
   those counters the way `load_checkpoint` does).
2. The fine-tune config (`configs/local_overfit_v2_finetune.yaml`) loads with
   the expected overrides and pretraining token budgets, but owns a separate
   input-tiled window manifest.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.train.loop import TrainingLoop
from thesis_ml.train.train import make_synthetic_examples

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_config(tmp_path: Path) -> ProjectConfig:
    """A small/cheap config (tiny model + short canvas) for fast CPU tests.

    Mirrors the `_small_config` pattern used in `tests/test_training.py`, kept
    local here so this file has no cross-test-file dependency.
    """

    config = load_config(REPO_ROOT / "config" / "default.yaml")
    return replace(
        config,
        data=replace(config.data, input_budget_tokens=64, canvas_budget_tokens=12),
        model=replace(config.model, d_model=32, layers=2, heads=4, ffn=64),
        train=replace(
            config.train,
            lr=0.01,
            warmup=1,
            accumulation_steps=1,
            target_effective_batch_tokens=0,
            max_steps=4,
            val_interval=0,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            ema_decay=0.9,
            precision="fp32",
        ),
    )


def _fit_one_step(loop: TrainingLoop, config: ProjectConfig, *, seed: int) -> None:
    """Run a single optimizer step so the loop's counters/optimizer state are non-trivial."""

    torch.manual_seed(seed)
    examples = make_synthetic_examples(config, count=2)
    # make_synthetic_examples builds PRE-TRAINING fixtures, and this helper's
    # config is a pre-training one, so collate in pre-training mode.
    dataloader = DataLoader(
        examples,
        batch_size=2,
        shuffle=False,
        collate_fn=partial(collate_diffusion_examples, debut_mode=False),
    )
    batch = next(iter(dataloader))
    loop.fit([batch], max_steps=loop.global_step + 1, fixed_t=1.0)


def test_load_model_weights_restores_weights_but_leaves_optimizer_step_epoch_fresh(
    tmp_path: Path,
) -> None:
    # --- "Pretrained" producer loop: take two optimizer steps so its saved
    # checkpoint carries a NON-ZERO global_step/completed_epochs/optimizer
    # state -- this is what makes the test meaningful: if load_model_weights
    # accidentally behaved like a full resume, these non-zero values would
    # leak into the fine-tune loop below.
    config = _tiny_config(tmp_path)
    torch.manual_seed(11)
    pretrained_model = SC2StrategyDiffusionModel(config, vocab_size=128)
    pretrained_loop = TrainingLoop(model=pretrained_model, config=config, seed=11)
    _fit_one_step(pretrained_loop, config, seed=11)
    _fit_one_step(pretrained_loop, config, seed=12)
    assert pretrained_loop.global_step == 2  # sanity: checkpoint will carry step=2
    checkpoint_path = pretrained_loop.save_checkpoint(tmp_path / "pretrained.pt")
    saved_model_params = [p.detach().clone() for p in pretrained_loop.model.parameters()]
    saved_ema_params = [p.detach().clone() for p in pretrained_loop.ema_model.parameters()]

    # --- A brand new ("freshly-constructed") loop with a DIFFERENT random
    # initialization. Its optimizer/global_step/completed_epochs are the
    # "fresh" baseline we expect load_model_weights to leave untouched.
    torch.manual_seed(99)
    finetune_model = SC2StrategyDiffusionModel(config, vocab_size=128)
    finetune_loop = TrainingLoop(model=finetune_model, config=config, seed=99)
    fresh_global_step = finetune_loop.global_step
    fresh_completed_epochs = finetune_loop.completed_epochs
    fresh_optimizer_state = finetune_loop.optimizer.state_dict()
    assert fresh_global_step == 0
    assert fresh_completed_epochs == 0
    assert fresh_optimizer_state["state"] == {}  # no steps taken yet -> empty Adam state

    # Prove the copy actually happens: perturb the fresh loop's weights away
    # from both the pretrained checkpoint AND its own original init.
    with torch.no_grad():
        for parameter in finetune_loop.model.parameters():
            parameter.add_(1.0)

    finetune_loop.load_model_weights(checkpoint_path)

    # Model + EMA weights are now restored to the PRETRAINED checkpoint's values.
    for restored, saved in zip(finetune_loop.model.parameters(), saved_model_params, strict=True):
        assert torch.allclose(restored, saved)
    for restored, saved in zip(finetune_loop.ema_model.parameters(), saved_ema_params, strict=True):
        assert torch.allclose(restored, saved)

    # Optimizer / global_step / completed_epochs are UNCHANGED from the
    # freshly-constructed baseline -- specifically NOT overwritten with the
    # checkpoint's global_step=2 the way a full `load_checkpoint` resume would.
    assert finetune_loop.global_step == fresh_global_step == 0
    assert finetune_loop.completed_epochs == fresh_completed_epochs == 0
    assert finetune_loop.optimizer.state_dict()["state"] == fresh_optimizer_state["state"] == {}


def test_finetune_config_extends_overfit_v2_with_warm_start_and_debut_settings() -> None:
    config = load_config(REPO_ROOT / "configs" / "local_overfit_v2_finetune.yaml")

    # Fine-tune-specific overrides.
    assert config.data.debut_mode is True
    assert config.sampler.outcome_last is True
    assert config.train.init_from_checkpoint == "checkpoints/local-overfitV2/last.pt"
    assert config.storage.checkpoint_uri == "checkpoints/local-overfitV2-finetune"
    assert config.train.lr == 1.0e-6
    assert config.train.epochs == 150

    # Inherited from local_overfit_v2.yaml / local_overfit.yaml -- log_uri is
    # SHARED with pre-training on purpose (the pipeline prefixes filenames).
    assert config.storage.log_uri == "tests/output/overfitV2"

    # Token budgets stay aligned, but fine-tuning owns a separate input-tiled
    # manifest because its output canvases may overlap across input windows.
    assert config.data.sampling_interval_s == 1
    assert config.data.input_budget_tokens == 4096
    assert config.data.canvas_budget_tokens == 4096
    assert config.data.canvas_recon_fraction == 0.5
    assert config.data.within_type_tiebreak == "unit_id"

    # Pre-training's own config must remain completely untouched by this
    # profile's existence: debut_mode/outcome_last default False, and
    # init_from_checkpoint defaults to "" (disabled).
    pretraining_config = load_config(REPO_ROOT / "configs" / "local_overfit_v2.yaml")
    assert config.data.window_manifest_path == "data/processed/local/finetune_window_manifest.jsonl"
    assert config.data.window_manifest_path != pretraining_config.data.window_manifest_path
    assert pretraining_config.data.debut_mode is False
    assert pretraining_config.sampler.outcome_last is False
    assert pretraining_config.train.init_from_checkpoint == ""


def test_default_config_disables_warm_start_by_default() -> None:
    config = load_config(REPO_ROOT / "config" / "default.yaml")
    assert config.train.init_from_checkpoint == ""
