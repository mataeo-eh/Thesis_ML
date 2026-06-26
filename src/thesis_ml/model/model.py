"""Assembled SC2 masked-diffusion transformer model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from thesis_ml.config import ProjectConfig
from thesis_ml.model.backbone import BidirectionalTransformer, RMSNorm
from thesis_ml.model.embedding import InputContextEmbedding
from thesis_ml.serialize import TokenRecord


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor


class SC2StrategyDiffusionModel(nn.Module):
    def __init__(self, config: ProjectConfig, *, vocab_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        model_config = config.model
        self.self_conditioning = model_config.self_conditioning
        # Enabling QK-norm or self-conditioning changes the architecture; pre-009 checkpoints need retraining.
        self.embedding = InputContextEmbedding(
            vocab_size,
            model_config.d_model,
            self_conditioning=model_config.self_conditioning,
        )
        self.backbone = BidirectionalTransformer(
            d_model=model_config.d_model,
            layers=model_config.layers,
            heads=model_config.heads,
            ffn_dim=model_config.ffn,
            dropout=dropout,
            qk_norm=model_config.qk_norm,
        )
        self.output_head = nn.Linear(model_config.d_model, vocab_size, bias=False)
        self._init_weights(model_config.layers)

    def forward(
        self,
        *,
        input_token_ids: torch.Tensor,
        canvas_token_ids: torch.Tensor,
        input_attention_mask: torch.Tensor | None = None,
        canvas_attention_mask: torch.Tensor | None = None,
        input_records: Sequence[Sequence[TokenRecord]] | None = None,
        canvas_self_conditioning: torch.Tensor | None = None,
    ) -> ModelOutput:
        embeddings = self.embedding(
            input_token_ids,
            canvas_token_ids,
            input_records=input_records,
            canvas_self_conditioning=canvas_self_conditioning,
        )
        attention_mask = _combine_attention_masks(
            input_token_ids,
            canvas_token_ids,
            input_attention_mask,
            canvas_attention_mask,
        )
        hidden = self.backbone(embeddings, attention_mask=attention_mask)
        logits = self.output_head(hidden)
        return ModelOutput(logits=logits, hidden_states=hidden)

    def _init_weights(self, layers: int) -> None:
        """Explicit LLaMA/GPT-style init instead of framework defaults."""

        residual_std = 0.02 / (2 * layers) ** 0.5
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                std = residual_std if name.endswith(("attn.out", "ffn.down")) else 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)


def _combine_attention_masks(
    input_token_ids: torch.Tensor,
    canvas_token_ids: torch.Tensor,
    input_attention_mask: torch.Tensor | None,
    canvas_attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if input_attention_mask is None:
        input_attention_mask = torch.ones_like(input_token_ids, dtype=torch.bool)
    if canvas_attention_mask is None:
        canvas_attention_mask = torch.ones_like(canvas_token_ids, dtype=torch.bool)
    return torch.cat([input_attention_mask.to(torch.bool), canvas_attention_mask.to(torch.bool)], dim=1)


def canvas_self_conditioning_from_logits(canvas_logits: torch.Tensor) -> torch.Tensor:
    """Convert prior clean-canvas logits to the shared self-conditioning tensor form."""

    return torch.softmax(canvas_logits.float(), dim=-1).detach()
