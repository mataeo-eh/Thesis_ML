"""Assembled SC2 clean-state diffusion transformer model."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from thesis_ml.config import ProjectConfig, toggle_fingerprint
from thesis_ml.data.feature_stats import FeatureStatistics
from thesis_ml.model.backbone import BidirectionalTransformer, FrozenInputKV, RMSNorm
from thesis_ml.model.embedding import InputContextEmbedding, InputFeatures

ARCHITECTURE_ID = "uniform-gemma4-dense-v1"


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    # Populated only when the caller asked for it via
    # `return_cached_input_kv=True`, which additionally requires the
    # `frozen_input_kv` ablation to be enabled. Defaulted so every existing
    # construction and every existing consumer is unaffected.
    cached_input_kv: FrozenInputKV | None = None


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
        # Only the toggle that changes how positions are BUILT lives here; the
        # frozen-KV toggle is owned by the backbone, which is what splits the
        # passes. Kept as a plain attribute so `forward` never reaches back into
        # the config object at step time.
        self.per_segment_positions = model_config.per_segment_positions
        # The suffix is empty for the all-off baseline, so this is byte-for-byte
        # the historical identity and every existing checkpoint still loads.
        # Any enabled ablation appends its name, which makes
        # `validate_checkpoint_compatibility` reject a cross-ablation load.
        self.architecture_identity = ARCHITECTURE_ID + toggle_fingerprint(model_config)
        self.diffusion_process = config.diffusion.process
        statistics = feature_statistics or FeatureStatistics.identity_for_tests()
        self.feature_statistics_identity = statistics.identity
        # Enabling QK-norm or self-conditioning changes the architecture; pre-009 checkpoints need retraining.
        self.embedding = InputContextEmbedding(
            vocab_size,
            model_config.d_model,
            feature_statistics=statistics,
            self_conditioning=model_config.self_conditioning,
            segment_embeddings=model_config.segment_embeddings,
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
            frozen_input_kv=model_config.frozen_input_kv,
        )
        self.output_head = nn.Linear(model_config.d_model, vocab_size, bias=False)
        self._init_weights(model_config.layers)
        # The general initializer above intentionally initializes every Linear;
        # restore the exact zero-output joint residual after it completes.
        self.embedding.reset_joint_output()
        # Same reason, for the segment-embedding ablation table: `_init_weights`
        # walks every nn.Embedding at std=0.02, which would leave the segment
        # table non-zero and change day-0 behavior away from the baseline. The
        # embedding module's own reset re-zeroes it and is a no-op when the
        # `segment_embeddings` toggle is off.
        self.embedding.reset_segment_embeddings()

    def forward(
        self,
        *,
        input_token_ids: torch.Tensor,
        canvas_token_ids: torch.Tensor,
        input_attention_mask: torch.Tensor | None = None,
        canvas_attention_mask: torch.Tensor | None = None,
        input_features: InputFeatures | None = None,
        canvas_self_conditioning: torch.Tensor | None = None,
        input_lengths: torch.Tensor | None = None,
        cached_input_kv: FrozenInputKV | None = None,
        return_cached_input_kv: bool = False,
    ) -> ModelOutput:
        """Embed both regions, run the backbone, and project to token logits.

        Parameters:
            input_token_ids: ``[batch, input_len]`` LEFT-padded input region.
            canvas_token_ids: ``[batch, canvas_len]`` right-padded canvas.
            input_attention_mask: ``[batch, input_len]``; True on real tokens.
            canvas_attention_mask: ``[batch, canvas_len]``; True on real tokens.
            input_features: batched static conditioning for the input region.
            canvas_self_conditioning: previous-step canvas estimate, if enabled.
            input_lengths: ``[batch]`` count of REAL (non-pad) input tokens.
                Consumed only by the ``per_segment_positions`` ablation, to know
                how much left padding each row carries. Two ways of obtaining it
                coexist on purpose: callers that already hold ``batch.input_lengths``
                pass it explicitly (cheapest and most direct), and callers that
                do not get it derived from the input attention mask below. The
                two are identically equal by construction, because the collater
                sets exactly ``input_lengths`` mask entries True per row.
            cached_input_kv: a ``FrozenInputKV`` from an earlier forward over
                the same input region; skips the input backbone pass.
            return_cached_input_kv: request the input pass's cache back on the
                returned ``ModelOutput``.

        Returns:
            ``ModelOutput`` whose ``logits`` and ``hidden_states`` always span
            the FULL ``[input | canvas]`` length, so the ``[:, input_len:, :]``
            slices every caller uses stay valid.

        Calls: ``self.embedding``, ``_combine_attention_masks``,
        ``_build_per_segment_position_ids``, ``self.backbone``,
        ``self.output_head``.
        """

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
        # The embedding drops the input region entirely when it is empty, so the
        # split index has to come from the token ids, which is also what every
        # caller slices with.
        input_len = input_token_ids.shape[1]

        position_ids = None
        if self.per_segment_positions and input_len > 0:
            if input_lengths is None:
                # Fallback derivation. `_combine_attention_masks` has already
                # substituted an all-True mask when the caller passed None, so
                # this branch is always well defined.
                input_lengths = attention_mask[:, :input_len].sum(dim=1)
            position_ids = _build_per_segment_position_ids(
                input_len=input_len,
                canvas_len=canvas_token_ids.shape[1],
                input_lengths=input_lengths,
                device=embeddings.device,
            )

        backbone_output = self.backbone(
            embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_len=input_len,
            cached_input_kv=cached_input_kv,
            return_cached_input_kv=return_cached_input_kv,
        )
        if return_cached_input_kv:
            hidden, produced_cache = backbone_output
        else:
            hidden, produced_cache = backbone_output, None
        logits = self.output_head(hidden)
        return ModelOutput(logits=logits, hidden_states=hidden, cached_input_kv=produced_cache)

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


def _build_per_segment_position_ids(
    *,
    input_len: int,
    canvas_len: int,
    input_lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build ``[batch, input_len + canvas_len]`` RoPE positions that restart per region.

    This is the whole point of the ``per_segment_positions`` ablation. RoPE is
    the model's only positional signal and it is purely RELATIVE, so with one
    shared ``arange`` the phase at canvas index 0 depends on how many input
    tokens happen to precede it -- a count that varies by thousands across the
    corpus. Restarting the canvas at 0 pins canvas index 0 (the win/loss token)
    to one fixed RoPE phase instead.

    Parameters:
        input_len: padded width of the input region.
        canvas_len: padded width of the canvas region.
        input_lengths: ``[batch]`` count of real, non-pad input tokens per row.
        device: device to build the position tensor on.

    Returns:
        ``[batch, input_len + canvas_len]`` long tensor of per-example positions.

    Calls: nothing beyond torch primitives.
    """

    batch_size = input_lengths.shape[0]
    lengths = input_lengths.to(device=device, dtype=torch.long)
    # Input real content is LEFT-padded, so row i's real tokens occupy slots
    # [input_len - L_i, input_len). Subtracting that offset yields 0..L_i-1 on
    # exactly the real slots.
    offsets = input_len - lengths
    input_positions = (
        torch.arange(input_len, device=device, dtype=torch.long)[None, :] - offsets[:, None]
    ).clamp_min(0)
    # The left-pad slots land negative before the clamp above. Pinning them to 0
    # is a deliberate, deterministic choice rather than an accident: those slots
    # are excluded from attention as keys by the padding mask and their logits
    # are never scored, so their rotation is unobservable -- but it must still
    # be defined, because a negative position would produce a valid-looking
    # rotation nobody intended.
    canvas_positions = torch.arange(canvas_len, device=device, dtype=torch.long)[None, :].expand(
        batch_size, canvas_len
    )
    return torch.cat([input_positions, canvas_positions], dim=1)


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
