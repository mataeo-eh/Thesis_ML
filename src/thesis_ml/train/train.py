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
        # Smoke train uses the pre-training (debut_mode=False) synthetic
        # fixtures, so collate in pre-training mode (absent input, no future
        # telemetry).
        collate_fn=partial(collate_diffusion_examples, debut_mode=False),
    )
    model = SC2StrategyDiffusionModel(config, vocab_size=SMOKE_VOCAB_SIZE)
    loop = TrainingLoop(model=model, config=config, seed=seed)
    return loop.fit(dataloader, max_steps=max_steps, fixed_t=1.0)


def make_synthetic_examples(config: ProjectConfig, *, count: int) -> list[DatasetExample]:
    """Build tiny pre-training-shaped synthetic examples for the smoke path.

    These fixtures follow the PRE-TRAINING contract (the smoke config runs with
    ``debut_mode=False``):
      - Input is LITERALLY ABSENT: ``input_records`` is empty and
        ``input_token_ids`` is a zero-length tensor, so the collated batch has a
        ``[B, 0]`` input segment and the model sequence is exactly the canvas.
      - The class taxonomy is COLLAPSED: every enemy content token is
        ``CLASS_CONTENT`` (there is no observed/fogged/future split in
        pre-training); only content / delimiter / end / pad / win-loss labels
        appear.

    Parameters:
        config: the (smoke) project config; unused fields are ignored here.
        count: how many identical-canvas examples to produce.
    Returns:
        A list of ``DatasetExample`` with absent input and collapsed labels.
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
    # Collapsed pre-training labels: all six content tokens are CLASS_CONTENT
    # (no fogged/future ids ever emitted in pre-training).
    class_labels = torch.tensor(
        [
            CLASS_WINLOSS,
            CLASS_CONTENT,
            CLASS_CONTENT,
            CLASS_DELIMITER,
            CLASS_CONTENT,
            CLASS_CONTENT,
            CLASS_DELIMITER,
            CLASS_CONTENT,
            CLASS_CONTENT,
            CLASS_DELIMITER,
            CLASS_END,
            CLASS_PAD,
        ],
        dtype=torch.long,
    )
    empty_input = torch.zeros((0,), dtype=torch.long)
    for example_index in range(count):
        examples.append(
            DatasetExample(
                # Pre-training input is absent -- no input records, zero-length
                # input token tensor.
                input_records=[],
                input_token_ids=empty_input.clone(),
                target_canvas=base_canvas.clone(),
                class_labels=class_labels.clone(),
                terminated=True,
                truncated=False,
                canvas_metadata=[
                    {"token_id": int(token_id), "timestep_index": index // 3}
                    for index, token_id in enumerate(base_canvas.tolist())
                ],
                fogged_counts={},
                observed_counts={},
                window_start=example_index,
                perspective_player="p1" if example_index % 2 == 0 else "p2",
                clean_input_token_ids=empty_input.clone(),
            )
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="SC2 masked-diffusion training")
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
        print(f"step={log.step} loss={log.loss:.4f} lr={log.lr:.6g} masked={log.masked_fraction:.3f} {per_class}")


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


def _synthetic_input_records(example_index: int) -> list[TokenRecord]:
    records: list[TokenRecord] = []
    owners = ("p1", "p1", "p2", "p2", "p1", "p2", "p1", "p2")
    for index, owner in enumerate(owners):
        records.append(
            TokenRecord(
                token_id=100 + (index % 6),
                token_name=f"synthetic_{index % 6}",
                token_kind="entity" if index % 3 else "upgrade",
                owner=owner,
                allegiance="self" if owner == "p1" else "enemy",
                game_loop=index,
                timestamp_seconds=float(index * 5),
                entity_type=f"synthetic_{index % 6}",
                instance_id=f"{example_index}{index:03d}",
                raw_position=f"({float(index + 1)}, {float(example_index + index + 2)}, 0.0)",
                raw_attributes={
                    "health": "45.0/45.0",
                    "is_flying": "False",
                    "build_progress": "1.0",
                },
            )
        )
    return records


if __name__ == "__main__":
    main()
