from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.dataset import (
    CLASS_DELIMITER,
    CLASS_ENEMY_FOGGED,
    CLASS_ENEMY_OBSERVED,
    CLASS_PAD,
)
from thesis_ml.model.loss import CanvasCrossEntropyLoss
from thesis_ml.model import backbone as backbone_module
from thesis_ml.model.backbone import MultiHeadSelfAttention
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.serialize import TokenRecord


def _small_config(
    *,
    d_model: int = 32,
    layers: int = 2,
    heads: int = 4,
    ffn: int = 64,
    qk_norm: bool = True,
    self_conditioning: bool = True,
) -> ProjectConfig:
    config = load_config("config/default.yaml")
    return replace(
        config,
        model=replace(
            config.model,
            d_model=d_model,
            layers=layers,
            heads=heads,
            ffn=ffn,
            qk_norm=qk_norm,
            self_conditioning=self_conditioning,
        ),
    )


def _records(batch: int, seq_len: int, *, x_offset: float = 0.0) -> list[list[TokenRecord]]:
    rows = []
    for batch_index in range(batch):
        row = []
        for index in range(seq_len):
            row.append(
                TokenRecord(
                    token_id=6 + (index % 4),
                    token_name="scv",
                    token_kind="entity",
                    owner="p1" if index % 2 == 0 else "p2",
                    allegiance="self" if index % 2 == 0 else "enemy",
                    game_loop=index,
                    timestamp_seconds=float(index),
                    entity_type="scv",
                    instance_id=f"{index + 1:03d}",
                    raw_position=f"({x_offset + index + 1.0}, {index + 2.0}, 0.0)",
                    raw_attributes={
                        "health": "45.0/45.0",
                        "is_flying": "False",
                        "build_progress": "1.0",
                    },
                )
            )
        rows.append(row)
    return rows


def test_forward_shapes() -> None:
    config = _small_config()
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    batch = 2
    input_ids = torch.tensor([[6, 7, 8, 9, 3], [6, 7, 8, 0, 0]])
    canvas_ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    input_mask = input_ids != 0

    output = model(
        input_token_ids=input_ids,
        canvas_token_ids=canvas_ids,
        input_attention_mask=input_mask,
        input_records=_records(batch, input_ids.shape[1]),
    )

    assert output.logits.shape == (batch, input_ids.shape[1] + canvas_ids.shape[1], 128)


def test_contextual_encodings_are_input_only() -> None:
    config = _small_config(layers=1)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    input_ids = torch.tensor([[6, 7, 8]])
    canvas_ids = torch.tensor([[9, 10, 11]])

    base_records = _records(1, 3, x_offset=0.0)
    changed_records = _records(1, 3, x_offset=100.0)

    canvas_embeddings = model.embedding.embed_canvas(canvas_ids)
    pure_canvas = model.embedding.token_embedding(canvas_ids)
    assert torch.allclose(canvas_embeddings, pure_canvas)

    base_input = model.embedding.embed_input(input_ids, input_records=base_records)
    changed_input = model.embedding.embed_input(input_ids, input_records=changed_records)
    assert not torch.allclose(base_input[:, 0], changed_input[:, 0])

    base_full = model.embedding(input_ids, canvas_ids, input_records=base_records)
    changed_full = model.embedding(input_ids, canvas_ids, input_records=changed_records)
    assert torch.allclose(base_full[:, input_ids.shape[1] :], changed_full[:, input_ids.shape[1] :])


def test_self_conditioning_adds_to_canvas_only_and_null_is_equivalent() -> None:
    config = _small_config(layers=1, self_conditioning=True)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    input_ids = torch.tensor([[6, 7, 8]])
    canvas_ids = torch.tensor([[9, 10, 11]])
    records = _records(1, 3)
    conditioning = torch.zeros(1, 3, 128)
    conditioning[0, :, 12] = 1.0

    base = model.embedding(input_ids, canvas_ids, input_records=records)
    zero = model.embedding(
        input_ids,
        canvas_ids,
        input_records=records,
        canvas_self_conditioning=torch.zeros_like(conditioning),
    )
    conditioned = model.embedding(
        input_ids,
        canvas_ids,
        input_records=records,
        canvas_self_conditioning=conditioning,
    )

    input_len = input_ids.shape[1]
    assert torch.allclose(base, zero)
    assert torch.allclose(base[:, :input_len], conditioned[:, :input_len])
    assert not torch.allclose(base[:, input_len:], conditioned[:, input_len:])


def test_qk_norm_is_gated_and_applied_before_rope(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(19)
    attention = MultiHeadSelfAttention(d_model=32, heads=4, qk_norm=True)
    seen_rms: list[torch.Tensor] = []

    def spy_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        seen_rms.append(x.pow(2).mean(dim=-1).sqrt().detach())
        return x

    monkeypatch.setattr(backbone_module, "apply_rope", spy_rope)
    attention(torch.randn(2, 5, 32))

    assert attention.q_norm is not None
    assert attention.k_norm is not None
    assert len(seen_rms) == 2
    for rms in seen_rms:
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    legacy_attention = MultiHeadSelfAttention(d_model=32, heads=4, qk_norm=False)
    assert legacy_attention.q_norm is None
    assert legacy_attention.k_norm is None


def test_qk_norm_disabled_matches_legacy_attention_path() -> None:
    torch.manual_seed(23)
    attention = MultiHeadSelfAttention(d_model=32, heads=4, qk_norm=False)
    x = torch.randn(2, 5, 32)

    actual = attention(x)

    batch, seq_len, d_model = x.shape
    qkv = attention.qkv(x).view(batch, seq_len, 3, attention.heads, attention.head_dim)
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    cos, sin = attention.rope(seq_len, device=x.device, dtype=x.dtype)
    q = backbone_module.apply_rope(q, cos, sin)
    k = backbone_module.apply_rope(k, cos, sin)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    expected = expected.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
    expected = attention.out(expected)

    assert torch.allclose(actual, expected)


def test_loss_is_canvas_only() -> None:
    config = _small_config()
    loss_fn = CanvasCrossEntropyLoss(config)
    batch, input_len, canvas_len, vocab_size = 1, 3, 4, 16
    full_logits = torch.zeros(batch, input_len + canvas_len, vocab_size)
    changed_full_logits = full_logits.clone()
    changed_full_logits[:, :input_len, :] = 100.0
    target = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[CLASS_ENEMY_OBSERVED, CLASS_ENEMY_FOGGED, CLASS_DELIMITER, CLASS_PAD]])

    loss_a = loss_fn(full_logits[:, input_len:], target, labels).loss
    loss_b = loss_fn(changed_full_logits[:, input_len:], target, labels).loss

    assert torch.allclose(loss_a, loss_b)


def test_per_class_logging_populated_and_consistent() -> None:
    config = _small_config()
    loss_fn = CanvasCrossEntropyLoss(config)
    logits = torch.zeros(1, 4, 8)
    target = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[CLASS_ENEMY_OBSERVED, CLASS_ENEMY_FOGGED, CLASS_DELIMITER, CLASS_PAD]])

    result = loss_fn(logits, target, labels)

    assert set(result.per_class) == {"enemy-observed", "enemy-fogged", "[DELIMITER]", "[PAD]"}
    expected = torch.stack(list(result.per_class.values())).mean()
    assert torch.allclose(result.loss, expected)


def test_attention_is_bidirectional() -> None:
    torch.manual_seed(7)
    config = _small_config(layers=1)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    model.eval()
    input_ids = torch.tensor([[6, 7, 8, 9]])
    canvas_ids = torch.tensor([[10, 11, 12, 13]])

    with torch.no_grad():
        base = model(input_token_ids=input_ids, canvas_token_ids=canvas_ids, input_records=_records(1, 4)).logits
        changed_canvas = canvas_ids.clone()
        changed_canvas[0, -1] = 40
        changed = model(
            input_token_ids=input_ids,
            canvas_token_ids=changed_canvas,
            input_records=_records(1, 4),
        ).logits

    first_canvas_index = input_ids.shape[1]
    assert not torch.allclose(base[:, first_canvas_index], changed[:, first_canvas_index])


def test_rope_length_extrapolation_smoke() -> None:
    config = _small_config()
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    input_ids = torch.tensor([[6, 7, 8, 9, 10, 11, 12, 13]])
    canvas_ids = torch.tensor([[14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])

    output = model(
        input_token_ids=input_ids,
        canvas_token_ids=canvas_ids,
        input_records=_records(1, input_ids.shape[1]),
    )

    assert output.logits.shape == (1, input_ids.shape[1] + canvas_ids.shape[1], 128)


def test_default_provisional_shape_instantiates_on_meta_device() -> None:
    config = load_config("config/default.yaml")
    with torch.device("meta"):
        model = SC2StrategyDiffusionModel(config, vocab_size=400)

    first_layer = model.backbone.layers[0]
    assert model.embedding.token_embedding.embedding_dim == 1536
    assert len(model.backbone.layers) == 16
    assert first_layer.attn.heads == 12
    assert first_layer.attn.head_dim == 128


def test_model_uses_explicit_depth_scaled_initialization() -> None:
    torch.manual_seed(11)
    config = _small_config(layers=2)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    residual_std = 0.02 / (2 * config.model.layers) ** 0.5

    assert model.embedding.token_embedding.weight.std().item() == pytest.approx(0.02, rel=0.25)
    assert model.backbone.layers[0].attn.out.weight.std().item() == pytest.approx(residual_std, rel=0.3)
    assert torch.allclose(model.backbone.layers[0].attn_norm.weight, torch.ones_like(model.backbone.layers[0].attn_norm.weight))
