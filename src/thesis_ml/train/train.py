"""Training entry points, including the prompt-005 smoke-train mode."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from functools import partial

from torch.utils.data import DataLoader

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import collate_diffusion_examples
from thesis_ml.data.dataset import (
    CLASS_CONTENT,
    CLASS_DELIMITER,
    CLASS_END,
    CLASS_ENEMY_FOGGED,
    CLASS_ENEMY_FUTURE,
    CLASS_ENEMY_OBSERVED,
    CLASS_PAD,
    CLASS_WINLOSS,
    DatasetExample,
)
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.serialize import TokenRecord
from thesis_ml.train.loop import TrainStepLog, TrainingLoop
from thesis_ml.vocab.special_tokens import DELIMITER_ID, END_ID, PAD_ID, WIN_ID

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
SMOKE_VOCAB_SIZE = 128


def run_smoke_train(
    *,
    max_steps: int = 32,
    seed: int = 123,
    checkpoint_dir: str | Path | None = None,
) -> list[TrainStepLog]:
    """Run a tiny deterministic overfit check through the real loop."""

    torch.manual_seed(seed)
    config = _smoke_config(max_steps=max_steps, checkpoint_dir=checkpoint_dir)
    examples = make_synthetic_examples(config, count=4)
    dataloader = DataLoader(
        examples,
        batch_size=2,
        shuffle=False,
        # Smoke train uses the pre-training synthetic fixtures.
        collate_fn=partial(collate_diffusion_examples, debut_mode=False),
    )
    model = SC2StrategyDiffusionModel(config, vocab_size=SMOKE_VOCAB_SIZE)
    loop = TrainingLoop(model=model, config=config, seed=seed)
    return loop.fit(dataloader, max_steps=max_steps, fixed_t=1.0)


def make_synthetic_examples(config: ProjectConfig, *, count: int) -> list[DatasetExample]:
    """Build tiny clamped-input pretraining examples for the smoke path.

    Parameters:
        config: the (smoke) project config; unused fields are ignored here.
        count: how many identical-canvas examples to produce.
    Returns:
        A list of ``DatasetExample`` with fogged input and full-rollout labels.
    Calls: none (self-contained fixture builder).
    """

    examples = []
    # Position 0 is the win/loss outcome token, mirroring the real canvas built
    # by _build_artifact_target (leading outcome token). Length stays 12 so the
    # smoke canvas budget is unchanged.
    base_canvas = torch.tensor(
        [WIN_ID, 100, 101, DELIMITER_ID, 102, 103, DELIMITER_ID, 104, 105, DELIMITER_ID, END_ID, PAD_ID],
        dtype=torch.long,
    )
    class_labels = torch.tensor(
        [
            CLASS_WINLOSS,
            CLASS_ENEMY_OBSERVED,
            CLASS_ENEMY_FOGGED,
            CLASS_DELIMITER,
            CLASS_ENEMY_FUTURE,
            CLASS_ENEMY_FUTURE,
            CLASS_DELIMITER,
            CLASS_ENEMY_FUTURE,
            CLASS_ENEMY_FUTURE,
            CLASS_DELIMITER,
            CLASS_END,
            CLASS_PAD,
        ],
        dtype=torch.long,
    )
    for example_index in range(count):
        perspective = "p1" if example_index % 2 == 0 else "p2"
        input_records, clean_records = _synthetic_input_pair(example_index, perspective)
        examples.append(
            DatasetExample(
                input_records=input_records,
                input_token_ids=torch.tensor([record.token_id for record in input_records]),
                target_canvas=base_canvas.clone(),
                class_labels=class_labels.clone(),
                terminated=True,
                truncated=False,
                canvas_metadata=[
                    {"token_id": int(token_id), "timestep_index": index // 3}
                    for index, token_id in enumerate(base_canvas.tolist())
                ],
                fogged_counts={(0, "synthetic_enemy_hidden"): 1},
                observed_counts={(0, "synthetic_enemy_visible"): 1},
                window_start=example_index,
                perspective_player=perspective,
                clean_input_token_ids=torch.tensor([record.token_id for record in clean_records]),
                window_end=example_index + 2,
            )
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="SC2 clean-state diffusion training")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true", help="run the tiny synthetic smoke-train")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if not args.smoke:
        raise SystemExit("only --smoke is wired in prompt 005; full data wiring belongs to later prompts")

    logs = run_smoke_train(max_steps=args.steps, seed=args.seed)
    for log in logs:
        per_class = ", ".join(f"{name}={value:.4f}" for name, value in sorted(log.per_class.items()))
        print(f"step={log.step} loss={log.loss:.4f} lr={log.lr:.6g} noise={log.noise_fraction:.3f} {per_class}")


def _smoke_config(*, max_steps: int, checkpoint_dir: str | Path | None) -> ProjectConfig:
    config = load_config(DEFAULT_CONFIG)
    checkpoint = str(checkpoint_dir) if checkpoint_dir is not None else str(PROJECT_ROOT / "checkpoints" / "smoke")
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
            max_steps=max_steps,
            val_interval=0,
            checkpoint_interval=max(1, max_steps + 1),
            checkpoint_dir=checkpoint,
            ema_decay=0.9,
            confidence_loss_weight=0.1,
            precision="fp32",
        ),
    )


def _synthetic_input_pair(
    example_index: int,
    perspective: str,
) -> tuple[list[TokenRecord], list[TokenRecord]]:
    enemy = "p2" if perspective == "p1" else "p1"
    clean: list[TokenRecord] = []
    fogged: list[TokenRecord] = []
    for index, (owner, name) in enumerate(
        (
            (perspective, "synthetic_self"),
            (enemy, "synthetic_enemy_visible"),
            (enemy, "synthetic_enemy_hidden"),
        )
    ):
        record = TokenRecord(
            token_id=100 + index,
            token_name=name,
            token_kind="entity",
            owner=owner,
            allegiance="self" if owner == perspective else "enemy",
            game_loop=example_index,
            timestamp_seconds=float(example_index),
            entity_type=name,
            instance_id=f"{example_index}{index:03d}",
            raw_position=f"({float(index + 1)}, {float(example_index + index + 2)}, 0.0)",
            raw_attributes={"health": "45.0/45.0", "build_progress": "1.0"},
        )
        clean.append(record)
        if name != "synthetic_enemy_hidden":
            fogged.append(record)
    delimiter = TokenRecord(
        token_id=DELIMITER_ID,
        token_name="[DELIMITER]",
        token_kind="delimiter",
        owner=None,
        allegiance=None,
        game_loop=example_index,
        timestamp_seconds=float(example_index),
    )
    clean.append(delimiter)
    fogged.append(delimiter)
    return fogged, clean


def _synthetic_input_records(example_index: int) -> list[TokenRecord]:
    """Compatibility helper returning a clean p1-perspective synthetic input."""

    _fogged, clean = _synthetic_input_pair(example_index, "p1")
    return clean


if __name__ == "__main__":
    main()
