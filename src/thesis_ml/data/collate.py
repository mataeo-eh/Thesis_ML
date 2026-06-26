"""Batch collation for SC2 diffusion dataset examples."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from thesis_ml.data.dataset import DatasetExample
from thesis_ml.vocab.special_tokens import PAD_ID


@dataclass(frozen=True)
class DiffusionBatch:
    input_token_ids: torch.Tensor
    input_attention_mask: torch.Tensor
    input_lengths: torch.Tensor
    target_canvas: torch.Tensor
    class_labels: torch.Tensor
    canvas_loss_mask: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    input_records: list[list[object]]
    canvas_metadata: list[list[dict[str, object]]]


def collate_diffusion_examples(examples: list[DatasetExample]) -> DiffusionBatch:
    max_input_len = max(example.input_token_ids.numel() for example in examples)
    input_token_ids = torch.full((len(examples), max_input_len), PAD_ID, dtype=torch.long)
    input_attention_mask = torch.zeros((len(examples), max_input_len), dtype=torch.bool)
    input_lengths = torch.zeros((len(examples),), dtype=torch.long)

    for row, example in enumerate(examples):
        length = example.input_token_ids.numel()
        input_token_ids[row, :length] = example.input_token_ids
        input_attention_mask[row, :length] = True
        input_lengths[row] = length

    target_canvas = torch.stack([example.target_canvas for example in examples])
    class_labels = torch.stack([example.class_labels for example in examples])
    canvas_loss_mask = torch.ones_like(target_canvas, dtype=torch.bool)

    return DiffusionBatch(
        input_token_ids=input_token_ids,
        input_attention_mask=input_attention_mask,
        input_lengths=input_lengths,
        target_canvas=target_canvas,
        class_labels=class_labels,
        canvas_loss_mask=canvas_loss_mask,
        terminated=torch.tensor([example.terminated for example in examples], dtype=torch.bool),
        truncated=torch.tensor([example.truncated for example in examples], dtype=torch.bool),
        input_records=[example.input_records for example in examples],
        canvas_metadata=[example.canvas_metadata for example in examples],
    )
