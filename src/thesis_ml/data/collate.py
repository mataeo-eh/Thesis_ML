"""Batch collation for SC2 diffusion dataset examples."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from thesis_ml.data.dataset import CLASS_ENEMY_FUTURE, CLASS_PAD, DatasetExample
from thesis_ml.model.embedding import InputFeatures, build_input_features
from thesis_ml.vocab.special_tokens import PAD_ID


@dataclass(frozen=True)
class DiffusionBatch:
    input_token_ids: torch.Tensor
    input_attention_mask: torch.Tensor
    input_lengths: torch.Tensor
    target_canvas: torch.Tensor
    canvas_attention_mask: torch.Tensor
    class_labels: torch.Tensor
    canvas_loss_mask: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    # CPU-only telemetry retained even when raw example metadata is dropped.
    input_timestep_counts: torch.Tensor
    enemy_future_timestep_counts: torch.Tensor
    canvas_prediction_distances: torch.Tensor
    input_records: list[list[object]]
    canvas_metadata: list[list[dict[str, object]]]
    # Pre-parsed contextual encodings for the input region, built here so the
    # CPU-bound parsing runs once per batch in the DataLoader worker instead of
    # on every forward pass. The model consumes these directly; input_records
    # is retained only for non-model tooling such as the eval harness.
    input_features: InputFeatures

    def pin_memory(self) -> "DiffusionBatch":
        """Pin model-facing tensors when DataLoader pinning sees this custom type."""

        features = self.input_features
        return replace(
            self,
            input_token_ids=self.input_token_ids.pin_memory(),
            input_attention_mask=self.input_attention_mask.pin_memory(),
            input_lengths=self.input_lengths.pin_memory(),
            target_canvas=self.target_canvas.pin_memory(),
            canvas_attention_mask=self.canvas_attention_mask.pin_memory(),
            class_labels=self.class_labels.pin_memory(),
            canvas_loss_mask=self.canvas_loss_mask.pin_memory(),
            terminated=self.terminated.pin_memory(),
            truncated=self.truncated.pin_memory(),
            canvas_prediction_distances=self.canvas_prediction_distances.pin_memory(),
            input_features=InputFeatures(
                map_values=features.map_values.pin_memory(),
                stat_values=features.stat_values.pin_memory(),
                team_ids=features.team_ids.pin_memory(),
            ),
        )


def collate_diffusion_examples(
    examples: list[DatasetExample],
    *,
    retain_metadata: bool = True,
) -> DiffusionBatch:
    if not examples:
        raise ValueError("cannot collate an empty example list")
    max_input_len = max(example.input_token_ids.numel() for example in examples)
    input_token_ids = torch.full((len(examples), max_input_len), PAD_ID, dtype=torch.long)
    input_attention_mask = torch.zeros((len(examples), max_input_len), dtype=torch.bool)
    input_lengths = torch.zeros((len(examples),), dtype=torch.long)

    for row, example in enumerate(examples):
        length = example.input_token_ids.numel()
        input_token_ids[row, max_input_len - length :] = example.input_token_ids
        input_attention_mask[row, max_input_len - length :] = True
        input_lengths[row] = length

    max_canvas_len = max(example.target_canvas.numel() for example in examples)
    target_canvas = torch.full((len(examples), max_canvas_len), PAD_ID, dtype=torch.long)
    class_labels = torch.full((len(examples), max_canvas_len), CLASS_PAD, dtype=torch.long)
    canvas_attention_mask = torch.zeros((len(examples), max_canvas_len), dtype=torch.bool)
    canvas_prediction_distances = torch.full(
        (len(examples), max_canvas_len),
        -1,
        dtype=torch.long,
    )
    for row, example in enumerate(examples):
        length = example.target_canvas.numel()
        target_canvas[row, :length] = example.target_canvas
        class_labels[row, :length] = example.class_labels
        canvas_attention_mask[row, :length] = True
        canvas_prediction_distances[row, :length] = torch.tensor(
            _enemy_future_prediction_distances(example),
            dtype=torch.long,
        )
    canvas_loss_mask = canvas_attention_mask.clone()

    input_records = [example.input_records for example in examples]
    input_features = build_input_features(input_records, max_input_len, left_pad=True)

    return DiffusionBatch(
        input_token_ids=input_token_ids,
        input_attention_mask=input_attention_mask,
        input_lengths=input_lengths,
        target_canvas=target_canvas,
        canvas_attention_mask=canvas_attention_mask,
        class_labels=class_labels,
        canvas_loss_mask=canvas_loss_mask,
        terminated=torch.tensor([example.terminated for example in examples], dtype=torch.bool),
        truncated=torch.tensor([example.truncated for example in examples], dtype=torch.bool),
        input_timestep_counts=torch.tensor(
            [_input_timestep_count(example) for example in examples],
            dtype=torch.long,
        ),
        enemy_future_timestep_counts=torch.tensor(
            [_enemy_future_timestep_count(example) for example in examples],
            dtype=torch.long,
        ),
        canvas_prediction_distances=canvas_prediction_distances,
        input_records=input_records if retain_metadata else [],
        canvas_metadata=(
            [example.canvas_metadata for example in examples]
            if retain_metadata
            else []
        ),
        input_features=input_features,
    )


def _input_timestep_count(example: DatasetExample) -> int:
    if example.window_end is not None:
        return max(0, example.window_end - example.window_start)
    return len(
        {
            record.game_loop
            for record in example.input_records
            if getattr(record, "game_loop", None) is not None
        }
    )


def _enemy_future_timestep_count(example: DatasetExample) -> int:
    if example.window_end is not None:
        input_count = _input_timestep_count(example)
        return len(
            {
                int(metadata["timestep_index"])
                for metadata in example.canvas_metadata
                if metadata.get("timestep_index") is not None
                and int(metadata["timestep_index"]) >= input_count
            }
        )
    labels = example.class_labels.tolist()
    if len(example.canvas_metadata) == len(labels):
        return len(
            {
                int(metadata["timestep_index"])
                for label, metadata in zip(labels, example.canvas_metadata, strict=True)
                if label == CLASS_ENEMY_FUTURE and metadata.get("timestep_index") is not None
            }
        )
    return sum(
        label == CLASS_ENEMY_FUTURE
        and (index == 0 or labels[index - 1] != CLASS_ENEMY_FUTURE)
        for index, label in enumerate(labels)
    )


def _enemy_future_prediction_distances(example: DatasetExample) -> list[int]:
    labels = example.class_labels.tolist()
    distances = [-1] * len(labels)
    if len(example.canvas_metadata) == len(labels):
        if example.window_end is not None:
            input_count = _input_timestep_count(example)
            for index, (label, metadata) in enumerate(
                zip(labels, example.canvas_metadata, strict=True)
            ):
                timestep_index = metadata.get("timestep_index")
                if label == CLASS_ENEMY_FUTURE and timestep_index is not None:
                    distance = int(timestep_index) - input_count + 1
                    if distance > 0:
                        distances[index] = distance
            return distances

        future_timesteps = sorted(
            {
                int(metadata["timestep_index"])
                for label, metadata in zip(labels, example.canvas_metadata, strict=True)
                if label == CLASS_ENEMY_FUTURE and metadata.get("timestep_index") is not None
            }
        )
        ordinal_by_timestep = {
            timestep: ordinal
            for ordinal, timestep in enumerate(future_timesteps, start=1)
        }
        for index, (label, metadata) in enumerate(
            zip(labels, example.canvas_metadata, strict=True)
        ):
            timestep_index = metadata.get("timestep_index")
            if label == CLASS_ENEMY_FUTURE and timestep_index is not None:
                distances[index] = ordinal_by_timestep[int(timestep_index)]
        return distances

    distance = 0
    in_future_timestep = False
    for index, label in enumerate(labels):
        if label == CLASS_ENEMY_FUTURE:
            if not in_future_timestep:
                distance += 1
                in_future_timestep = True
            distances[index] = distance
        else:
            in_future_timestep = False
    return distances
