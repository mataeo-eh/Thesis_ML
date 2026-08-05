"""Assembled SC2 clean-state diffusion transformer model."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from thesis_ml.config import ProjectConfig
from thesis_ml.data.feature_stats import FeatureStatistics
from thesis_ml.model.backbone import BidirectionalTransformer, RMSNorm
from thesis_ml.model.embedding import InputContextEmbedding, InputFeatures

ARCHITECTURE_ID = "uniform-gemma4-dense-v1"


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor


class SC2StrategyDiffusionModel(nn.Module):
    def __init__(
        self,
        config: ProjectConfig,
        *,
        vocab_size: int,
        feature_statistics: FeatureStatistics | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        model_config = config.model
        if model_config.rope_scaling.rope_type != "llama3":
            raise ValueError(
                "model.rope_scaling.rope_type must be 'llama3', "
                f"got {model_config.rope_scaling.rope_type!r}"
            )
        self.self_conditioning = model_config.self_conditioning
        self.vocab_size = vocab_size
        self.architecture_identity = ARCHITECTURE_ID
        self.diffusion_process = config.diffusion.process
        statistics = feature_statistics or FeatureStatistics.identity_for_tests()
        self.feature_statistics_identity = statistics.identity
        # Enabling QK-norm or self-conditioning changes the architecture; pre-009 checkpoints need retraining.
        self.embedding = InputContextEmbedding(
            vocab_size,
            model_config.d_model,
            feature_statistics=statistics,
            self_conditioning=model_config.self_conditioning,
        )
        self.backbone = BidirectionalTransformer(
            d_model=model_config.d_model,
            layers=model_config.layers,
            heads=model_config.heads,
            ffn_dim=model_config.ffn,
            dropout=dropout,
            qk_norm=model_config.qk_norm,
            rope_theta=model_config.rope_theta,
            rope_scaling_factor=model_config.rope_scaling.factor,
            rope_low_freq_factor=model_config.rope_scaling.low_freq_factor,
            rope_high_freq_factor=model_config.rope_scaling.high_freq_factor,
            rope_original_context=model_config.rope_scaling.original_max_position_embeddings,
            gradient_checkpointing=model_config.gradient_checkpointing,
        )
        self.output_head = nn.Linear(model_config.d_model, vocab_size, bias=False)
        self._init_weights(model_config.layers)
        # The general initializer above intentionally initializes every Linear;
        # restore the exact zero-output joint residual after it completes.
        self.embedding.reset_joint_output()

    def forward(
        self,
        *,
        input_token_ids: torch.Tensor,
        canvas_token_ids: torch.Tensor,
        input_attention_mask: torch.Tensor | None = None,
        canvas_attention_mask: torch.Tensor | None = None,
        input_features: InputFeatures | None = None,
        canvas_self_conditioning: torch.Tensor | None = None,
    ) -> ModelOutput:
        embeddings = self.embedding(
            input_token_ids,
            canvas_token_ids,
            input_features=input_features,
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
                if module.weight is not None:
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


def canvas_self_conditioning_from_logits(
    canvas_logits: torch.Tensor,
    token_embedding_weight: torch.Tensor,
    *,
    probabilities: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert a stopped clean-state distribution to expected token embeddings."""

    probs = torch.softmax(canvas_logits.float(), dim=-1) if probabilities is None else probabilities.float()
    return torch.matmul(probs.detach(), token_embedding_weight.detach().float()).detach()


def validate_checkpoint_compatibility(
    checkpoint: dict,
    model: nn.Module,
    checkpoint_path: str,
) -> None:
    """Fail closed on retired or cross-process checkpoint metadata."""

    expected_architecture = getattr(model, "architecture_identity", None)
    observed_architecture = checkpoint.get("architecture_identity")
    if observed_architecture != expected_architecture:
        raise ValueError(
            f"checkpoint {checkpoint_path} architecture identity mismatch: "
            f"expected {expected_architecture!r}, got {observed_architecture!r}"
        )
    expected_process = getattr(model, "diffusion_process", None)
    observed_process = checkpoint.get("diffusion_process")
    if observed_process != expected_process:
        raise ValueError(
            f"checkpoint {checkpoint_path} diffusion process mismatch: "
            f"expected {expected_process!r}, got {observed_process!r}"
        )
