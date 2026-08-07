from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from thesis_ml.config import (
    ClassLossWeightsConfig,
    FogConfig,
    ProjectConfig,
    UniformDistributionConfig,
    load_config,
)
from thesis_ml.data.dataset import (
    CLASS_CONTENT,
    CLASS_DELIMITER,
    CLASS_END,
    CLASS_ENEMY_FOGGED,
    CLASS_ENEMY_FUTURE,
    CLASS_ENEMY_OBSERVED,
    CLASS_PAD,
    CLASS_WINLOSS,
    PRETRAIN_CLASS_ID_TO_NAME,
)
from thesis_ml.data.features import (
    BUFF_ID_MAX,
    BUFF_VALIDITY_INDEX,
    BUFF_VALUE_OFFSET,
    CLOAK_STATE_COUNT,
    CONTINUOUS_FEATURE_NAMES,
    parse_buff_ids,
)
from thesis_ml.model.loss import CanvasCrossEntropyLoss
from thesis_ml.model import backbone as backbone_module
from thesis_ml.model.backbone import GeGLU, MultiHeadSelfAttention, RotaryEmbedding, TransformerBlock
from thesis_ml.model.embedding import InputFeatures, _numeric_feature, build_input_features
from thesis_ml.model.model import SC2StrategyDiffusionModel, canvas_self_conditioning_from_logits
from thesis_ml.serialize import TokenRecord


def test_slash_form_numeric_features_are_encoded_as_fractions() -> None:
    assert _numeric_feature("6.0/45.0") == pytest.approx(6.0 / 45.0)
    assert _numeric_feature("51.23046875/200.0") == pytest.approx(51.23046875 / 200.0)
    assert _numeric_feature("0.0/0.0") == 0.0
    assert _numeric_feature("sentinel-with-123") == 0.0


def test_approved_features_preserve_valid_zero_and_encode_categories() -> None:
    records = []
    for cloak_state in range(CLOAK_STATE_COUNT):
        records.append(
            TokenRecord(
                token_id=6,
                token_name="scv",
                token_kind="entity",
                owner="p1",
                allegiance="self",
                game_loop=0,
                timestamp_seconds=0.0,
                raw_position="(0.0, 0.0, 12.0)",
                raw_attributes={
                    "health": "0.0/45.0",
                    "facing": "0.0",
                    "ideal_harvesters": "16",
                    "buff_duration_remain": "0",
                    "buff_duration_max": "1440",
                    "detect_range": "0.0",
                    "cloak": str(cloak_state),
                    "buff_ids": "[7, 27, 289, 294, 298, 299, 302]" if cloak_state == 0 else "[]",
                    # Explicitly rejected fields must not affect the encoding.
                    "assigned_harvesters": "12",
                    "radar_range": "30.0",
                },
            )
        )

    features = build_input_features([records], len(records))
    for name in (
        "map_x",
        "map_y",
        "health",
        "facing_sin",
        "facing_cos",
        "ideal_harvesters",
        "buff_duration_remain",
        "buff_duration_max",
        "detect_range",
    ):
        assert features.continuous_validity[0, 0, CONTINUOUS_FEATURE_NAMES.index(name)]
    assert not features.continuous_validity[
        0, 0, CONTINUOUS_FEATURE_NAMES.index("energy")
    ]
    assert features.continuous_values[0, 0, CONTINUOUS_FEATURE_NAMES.index("map_x")] == 0
    assert features.continuous_values[0, 0, CONTINUOUS_FEATURE_NAMES.index("facing_sin")] == 0
    assert features.continuous_values[0, 0, CONTINUOUS_FEATURE_NAMES.index("facing_cos")] == 1
    assert "assigned_harvesters" not in CONTINUOUS_FEATURE_NAMES
    assert "radar_range" not in CONTINUOUS_FEATURE_NAMES

    for cloak_state in range(CLOAK_STATE_COUNT):
        assert features.categorical_values[0, cloak_state, cloak_state] == 1
        assert features.categorical_values[
            0, cloak_state, :CLOAK_STATE_COUNT
        ].sum() == 1
        assert features.categorical_values[0, cloak_state, BUFF_VALIDITY_INDEX] == 1
    for buff_id in (7, 27, 289, 294, 298, 299, 302):
        assert features.categorical_values[0, 0, BUFF_VALUE_OFFSET + buff_id] == 1


def test_nonnumeric_stat_sentinels_are_missing_not_valid_zero() -> None:
    record = TokenRecord(
        token_id=6,
        token_name="scv",
        token_kind="entity",
        owner="p1",
        allegiance="self",
        game_loop=0,
        timestamp_seconds=0.0,
        raw_position="(1.0, 2.0, 3.0)",
        raw_attributes={
            "health": "inside refinery",
            "is_active": "destroyed",
            "cloak": "completed",
            "buff_ids": "inside refinery",
        },
    )

    features = build_input_features([[record]], 1)

    assert not features.continuous_validity[
        0, 0, CONTINUOUS_FEATURE_NAMES.index("health")
    ]
    assert not features.continuous_validity[
        0, 0, CONTINUOUS_FEATURE_NAMES.index("is_active")
    ]
    assert features.categorical_values[0, 0].sum() == 0


def test_buff_ids_preserve_corpus_range_and_reject_unseen_schema_drift() -> None:
    assert parse_buff_ids("[302, 299, 302]") == (299, 302)
    with pytest.raises(ValueError, match=rf"0\.\.{BUFF_ID_MAX}"):
        parse_buff_ids(f"[{BUFF_ID_MAX + 1}]")


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


def _small_debut_config(**kwargs) -> ProjectConfig:
    """Fine-tuning (debut_mode=True) variant of `_small_config`.

    The future-distance loss decomposition (and per-class config weighting) is
    now FINE-TUNING-ONLY, so tests of those behaviors must build the loss from
    a debut-mode config. A debut config is required (by
    `_validate_debut_mode_sections` / `CanvasCrossEntropyLoss`) to carry `fog`
    and `loss.class_loss_weights`, so both are populated with plain uniform
    values here -- their exact numbers are not what these tests assert.
    """

    base = _small_config(**kwargs)
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


def _features(records: list[list[TokenRecord]], seq_len: int) -> InputFeatures:
    return build_input_features(records, seq_len)


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
        input_features=_features(_records(batch, input_ids.shape[1]), input_ids.shape[1]),
    )

    assert output.logits.shape == (batch, input_ids.shape[1] + canvas_ids.shape[1], 128)


def test_contextual_encodings_are_input_only() -> None:
    config = _small_config(layers=1, self_conditioning=False)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    input_ids = torch.tensor([[6, 7, 8]])
    canvas_ids = torch.tensor([[9, 10, 11]])

    base_records = _records(1, 3, x_offset=0.0)
    changed_records = _records(1, 3, x_offset=100.0)

    canvas_embeddings = model.embedding.embed_canvas(canvas_ids)
    pure_canvas = model.embedding.token_embedding(canvas_ids)
    assert torch.allclose(canvas_embeddings, pure_canvas)

    base_features = _features(base_records, input_ids.shape[1])
    changed_features = _features(changed_records, input_ids.shape[1])
    base_input = model.embedding.embed_input(input_ids, input_features=base_features)
    changed_input = model.embedding.embed_input(input_ids, input_features=changed_features)
    # Exact zero final projection means the joint branch initially outputs
    # zero, so both are exactly the unobstructed type embedding.
    assert torch.equal(base_input, model.embedding.token_embedding(input_ids))
    assert torch.equal(base_input, changed_input)

    with torch.no_grad():
        model.embedding.joint_mixer[-1].weight.fill_(0.01)
    changed_input = model.embedding.embed_input(input_ids, input_features=changed_features)
    base_input = model.embedding.embed_input(input_ids, input_features=base_features)
    assert not torch.allclose(base_input[:, 0], changed_input[:, 0])

    base_full = model.embedding(input_ids, canvas_ids, input_features=base_features)
    changed_full = model.embedding(input_ids, canvas_ids, input_features=changed_features)
    assert torch.allclose(base_full[:, input_ids.shape[1] :], changed_full[:, input_ids.shape[1] :])


def test_absolute_game_time_cannot_enter_model_features() -> None:
    config = _small_config(layers=1)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    input_ids = torch.tensor([[6, 7, 8]])
    base_records = _records(1, 3)
    changed_records = [[replace(record, timestamp_seconds=record.timestamp_seconds + 10_000) for record in base_records[0]]]

    base_features = _features(base_records, input_ids.shape[1])
    changed_features = _features(changed_records, input_ids.shape[1])

    assert not hasattr(base_features, "clock_values")
    assert not hasattr(model.embedding, "clock_projection")
    assert not hasattr(model.embedding, "clock_fourier")
    assert torch.equal(base_features.continuous_values, changed_features.continuous_values)
    assert torch.equal(base_features.allegiance_values, changed_features.allegiance_values)
    assert torch.equal(base_features.feature_mask, changed_features.feature_mask)
    assert torch.equal(
        model.embedding.embed_input(input_ids, input_features=base_features),
        model.embedding.embed_input(input_ids, input_features=changed_features),
    )


def test_joint_static_embedding_zero_init_staged_gradients_and_locality() -> None:
    torch.manual_seed(31)
    model = SC2StrategyDiffusionModel(
        _small_config(layers=1, self_conditioning=False), vocab_size=128
    )
    embedding = model.embedding
    input_ids = torch.tensor([[6, 7, 8]])
    features = _features(_records(1, 3), 3)

    initial = embedding.embed_input(input_ids, input_features=features)
    assert torch.equal(initial, embedding.token_embedding(input_ids))
    assert torch.count_nonzero(embedding.joint_mixer[-1].weight) == 0
    assert torch.count_nonzero(embedding.joint_mixer[-1].bias) == 0

    optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)
    initial.square().sum().backward()
    assert embedding.token_embedding.weight.grad is not None
    assert embedding.token_embedding.weight.grad.abs().sum() > 0
    assert embedding.joint_mixer[-1].weight.grad is not None
    assert embedding.joint_mixer[-1].weight.grad.abs().sum() > 0
    assert embedding.joint_mixer[0].weight.grad is not None
    assert embedding.joint_mixer[0].weight.grad.abs().sum() == 0
    assert embedding.feature_mlp[0].weight.grad is not None
    assert embedding.feature_mlp[0].weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad()

    embedding.embed_input(input_ids, input_features=features).square().sum().backward()
    assert embedding.joint_mixer[0].weight.grad.abs().sum() > 0
    assert embedding.feature_mlp[0].weight.grad.abs().sum() > 0

    changed_continuous = features.continuous_values.clone()
    changed_continuous[0, 1, 0] += 10.0
    changed = InputFeatures(
        continuous_values=changed_continuous,
        continuous_validity=features.continuous_validity,
        categorical_values=features.categorical_values,
        allegiance_values=features.allegiance_values,
        feature_mask=features.feature_mask,
    )
    base_output = embedding.embed_input(input_ids, input_features=features)
    changed_output = embedding.embed_input(input_ids, input_features=changed)
    # The staged branch has only taken one update and is intentionally still
    # tiny; require a real feature-local change without imposing an arbitrary
    # allclose magnitude threshold on random initialization.
    assert not torch.equal(base_output[:, 1], changed_output[:, 1])
    assert torch.equal(base_output[:, 0], changed_output[:, 0])
    assert torch.equal(base_output[:, 2], changed_output[:, 2])


def test_self_conditioning_adds_to_canvas_only_and_null_is_equivalent() -> None:
    config = _small_config(layers=1, self_conditioning=True)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    input_ids = torch.tensor([[6, 7, 8]])
    canvas_ids = torch.tensor([[9, 10, 11]])
    records = _records(1, 3)
    features = _features(records, input_ids.shape[1])
    conditioning = torch.randn(1, 3, config.model.d_model)

    base = model.embedding(input_ids, canvas_ids, input_features=features)
    zero = model.embedding(
        input_ids,
        canvas_ids,
        input_features=features,
        canvas_self_conditioning=torch.zeros_like(conditioning),
    )
    conditioned = model.embedding(
        input_ids,
        canvas_ids,
        input_features=features,
        canvas_self_conditioning=conditioning,
    )

    input_len = input_ids.shape[1]
    assert torch.allclose(base, zero)
    assert torch.allclose(base[:, :input_len], conditioned[:, :input_len])
    assert not torch.allclose(base[:, input_len:], conditioned[:, input_len:])


def test_self_conditioning_expected_embedding_uses_shared_token_table() -> None:
    model = SC2StrategyDiffusionModel(_small_config(layers=1), vocab_size=8)
    logits = torch.full((1, 2, 8), -100.0)
    logits[0, 0, 3] = 100.0
    logits[0, 1, 5] = 100.0

    expected = canvas_self_conditioning_from_logits(
        logits,
        model.embedding.token_embedding.weight,
    )

    assert torch.allclose(expected[0, 0], model.embedding.token_embedding.weight[3].float())
    assert torch.allclose(expected[0, 1], model.embedding.token_embedding.weight[5].float())
    assert not hasattr(model.embedding, "self_cond_projection")


def test_geglu_uses_tanh_approximate_gelu(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    original = F.gelu

    def capture(input: torch.Tensor, approximate: str = "none") -> torch.Tensor:
        seen.append(approximate)
        return original(input, approximate=approximate)

    monkeypatch.setattr(backbone_module.F, "gelu", capture)
    layer = GeGLU(8, 16)
    assert layer(torch.randn(2, 3, 8)).shape == (2, 3, 8)
    assert seen == ["tanh"]


def test_transformer_block_has_sandwich_norms_on_both_residual_branches() -> None:
    block = TransformerBlock(d_model=16, heads=4, ffn_dim=32)
    assert hasattr(block, "pre_attn_norm")
    assert hasattr(block, "post_attn_norm")
    assert hasattr(block, "pre_ffn_norm")
    assert hasattr(block, "post_ffn_norm")
    assert isinstance(block.ffn, GeGLU)
    assert block(torch.randn(2, 5, 16)).shape == (2, 5, 16)


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


def test_padding_mask_is_boolean_key_mask_broadcast_over_heads_and_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = MultiHeadSelfAttention(d_model=32, heads=4)
    seen_mask: list[torch.Tensor] = []
    original = F.scaled_dot_product_attention

    def capture_mask(query, key, value, *, attn_mask=None, **kwargs):
        seen_mask.append(attn_mask)
        return original(query, key, value, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(backbone_module.F, "scaled_dot_product_attention", capture_mask)
    padding_mask = torch.tensor([[True, True, False], [True, True, True]])

    attention(torch.randn(2, 3, 32), attention_mask=padding_mask)

    assert len(seen_mask) == 1
    assert seen_mask[0].dtype == torch.bool
    assert seen_mask[0].shape == (2, 1, 1, 3)
    assert torch.equal(seen_mask[0][:, 0, 0, :], padding_mask)


def test_gradient_checkpointing_is_config_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _small_config(layers=1)
    config = replace(
        config,
        model=replace(config.model, gradient_checkpointing=True, self_conditioning=False),
    )
    calls = 0
    original = backbone_module.checkpoint

    def count_checkpoint(function, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(function, *args, **kwargs)

    monkeypatch.setattr(backbone_module, "checkpoint", count_checkpoint)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    input_ids = torch.tensor([[6, 7, 8]])
    canvas_ids = torch.tensor([[9, 10, 11]])

    output = model(
        input_token_ids=input_ids,
        canvas_token_ids=canvas_ids,
        input_features=_features(_records(1, 3), 3),
    )
    output.logits.sum().backward()

    assert calls == 1


def test_llama3_rope_scaling_matches_reference_frequency_bands() -> None:
    head_dim = 128
    theta = 500_000.0
    factor = 8.0
    original_context = 8192
    rope = RotaryEmbedding(
        head_dim,
        base=theta,
        scaling_factor=factor,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_context=original_context,
    )
    unscaled = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    wavelengths = 2 * torch.pi / unscaled

    high_frequency = wavelengths < original_context / 4.0
    low_frequency = wavelengths > original_context
    medium_frequency = ~(high_frequency | low_frequency)

    assert torch.equal(rope.inv_freq[high_frequency], unscaled[high_frequency])
    assert torch.allclose(rope.inv_freq[low_frequency], unscaled[low_frequency] / factor)
    assert medium_frequency.any()
    assert torch.all(rope.inv_freq[medium_frequency] < unscaled[medium_frequency])
    assert torch.all(rope.inv_freq[medium_frequency] > unscaled[medium_frequency] / factor)


def test_loss_is_canvas_only() -> None:
    config = _small_config()
    loss_fn = CanvasCrossEntropyLoss(config)
    batch, input_len, canvas_len, vocab_size = 1, 3, 4, 16
    full_logits = torch.zeros(batch, input_len + canvas_len, vocab_size)
    changed_full_logits = full_logits.clone()
    changed_full_logits[:, :input_len, :] = 100.0
    target = torch.tensor([[1, 2, 3, 4]])
    # Pretraining labels preserve the observed/fogged/future partition.
    # class (ids 1/2 are never emitted in this mode).
    labels = torch.tensor([[CLASS_CONTENT, CLASS_CONTENT, CLASS_DELIMITER, CLASS_PAD]])

    loss_a = loss_fn(full_logits[:, input_len:], target, labels).loss
    loss_b = loss_fn(changed_full_logits[:, input_len:], target, labels).loss

    assert torch.allclose(loss_a, loss_b)


def test_per_class_logging_populated_and_consistent() -> None:
    # Pre-training uses reconstruction/fog/future names with stable ids.
    config = _small_config()
    loss_fn = CanvasCrossEntropyLoss(config)
    logits = torch.zeros(1, 4, 8)
    target = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[CLASS_CONTENT, CLASS_CONTENT, CLASS_DELIMITER, CLASS_PAD]])

    result = loss_fn(logits, target, labels)

    assert set(result.per_class) == {"enemy-observed", "[DELIMITER]", "[PAD]"}
    assert set(result.per_class) <= set(PRETRAIN_CLASS_ID_TO_NAME.values())
    assert "enemy-observed" in result.per_class
    assert "enemy-fogged" not in result.per_class
    expected = torch.stack(list(result.per_class.values())).mean()
    assert torch.allclose(result.loss, expected)


def test_future_loss_is_bucketed_by_prediction_distance() -> None:
    # A debut-config loss keeps this test focused on the fine-tuning class names;
    # the same distance decomposition is also active in pre-training.
    loss_fn = CanvasCrossEntropyLoss(_small_debut_config())
    logits = torch.zeros(1, 6, 8)
    target = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.full((1, 6), CLASS_ENEMY_FUTURE, dtype=torch.long)
    distances = torch.tensor([[1, 3, 8, 20, 40, -1]])

    result = loss_fn(logits, target, labels, prediction_distances=distances)

    assert set(result.future_distance) == {"1", "2_5", "6_10", "11_30", "31_plus"}
    assert result.future_distance_counts == {
        "1": 1,
        "2_5": 1,
        "6_10": 1,
        "11_30": 1,
        "31_plus": 1,
    }
    expected = torch.log(torch.tensor(8.0)).item()
    assert all(
        value.item() == pytest.approx(expected)
        for value in result.future_distance.values()
    )


def test_pretraining_loss_emits_future_distance() -> None:

    loss_fn = CanvasCrossEntropyLoss(_small_config())
    logits = torch.zeros(1, 4, 8)
    target = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor(
        [[CLASS_ENEMY_FUTURE, CLASS_ENEMY_FUTURE, CLASS_DELIMITER, CLASS_PAD]]
    )
    distances = torch.tensor([[1, 2, -1, -1]])

    result = loss_fn(logits, target, labels, prediction_distances=distances)

    assert set(result.future_distance) == {"1", "2_5"}
    assert result.future_distance_counts == {"1": 1, "2_5": 1}


def test_t_bucket_edges_are_contiguous_exhaustive_and_perspectives_emit_both_keys() -> None:
    """Every t in [0, 1] lands in EXACTLY one t-bucket, with the documented edges.

    The four buckets partition [0, 1]: exact t==1.0 -> t_eq_1; [0.75, 1.0) ->
    t_0_75_to_1_0 (so t=0.995 goes there, NOT in t_eq_1 -- keeping full
    corruption separable from near-full corruption is the whole reason t_eq_1 is
    its own bucket); then [0.25, 0.75) and [0.0, 0.25). Also proves the
    perspective breakdown emits both `p1` and `p2` keys for a batch containing
    both perspectives.
    """

    loss_fn = CanvasCrossEntropyLoss(_small_config())

    def bucket_for(t_value: float) -> str:
        # One single-position example whose sampled t is t_value: exactly one
        # bucket key must appear in the decomposition.
        logits = torch.zeros(1, 1, 8)
        target = torch.tensor([[1]])
        labels = torch.tensor([[CLASS_CONTENT]])
        result = loss_fn(logits, target, labels, sampled_t=torch.tensor([t_value]))
        assert len(result.t_bucket) == 1, f"t={t_value} mapped to {set(result.t_bucket)}"
        return next(iter(result.t_bucket))

    # A dense grid over [0, 1] plus every boundary: each t maps to exactly one
    # bucket (asserted inside bucket_for) and to the DOCUMENTED bucket.
    def expected_bucket(t_value: float) -> str:
        if t_value == 1.0:
            return "t_eq_1"
        if t_value >= 0.75:
            return "t_0_75_to_1_0"
        if t_value >= 0.25:
            return "t_0_25_to_0_75"
        return "t_0_0_to_0_25"

    grid = [index / 100.0 for index in range(101)]
    for t_value in [*grid, 0.995, 0.2499, 0.25, 0.75, 0.7499, 0.9999]:
        assert bucket_for(t_value) == expected_bucket(t_value), f"t={t_value}"

    # The two spelled-out cases from the metric contract.
    assert bucket_for(0.995) == "t_0_75_to_1_0"
    assert bucket_for(1.0) == "t_eq_1"

    # Perspective split: a two-example batch built from p1 and p2 perspectives
    # emits BOTH keys (1 -> "p1", 2 -> "p2" per collate's integer encoding).
    logits = torch.zeros(2, 2, 8)
    target = torch.tensor([[1, 2], [3, 4]])
    labels = torch.tensor([[CLASS_CONTENT, CLASS_PAD], [CLASS_CONTENT, CLASS_PAD]])
    result = loss_fn(
        logits,
        target,
        labels,
        sampled_t=torch.tensor([0.4, 0.8]),
        perspective_ids=torch.tensor([1, 2]),
    )
    assert set(result.perspective) == {"p1", "p2"}
    # And the two different sampled ts populate their two different buckets.
    assert set(result.t_bucket) == {"t_0_25_to_0_75", "t_0_75_to_1_0"}


def test_canvas_state_splits_preserved_ground_truth_from_actually_noised() -> None:
    """The canvas-state breakdown keys on TOKEN INEQUALITY, not the noise flag.

    Four scored positions: two the model saw already-correct
    (``changed_positions`` False) and two it saw corrupted. Their per-position
    cross entropies must land in the matching bucket, and neither bucket may
    borrow from the other -- that separation is what makes
    "ground_truth_preserved" readable as "has the model learned to recognize a
    valid sequence and leave it alone".
    """

    loss_fn = CanvasCrossEntropyLoss(_small_config())
    # Row of 4 positions. Logits are shaped so the two preserved positions are
    # predicted confidently (low CE) and the two noised ones are not (high CE),
    # giving the two buckets clearly different means.
    logits = torch.zeros(1, 4, 8)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    target = torch.tensor([[1, 2, 3, 4]])
    labels = torch.full((1, 4), CLASS_CONTENT)
    changed = torch.tensor([[False, False, True, True]])

    result = loss_fn(logits, target, labels, changed_positions=changed)

    assert set(result.canvas_state) == {"ground_truth_preserved", "noised"}
    preserved = float(result.canvas_state["ground_truth_preserved"])
    noised = float(result.canvas_state["noised"])
    assert preserved < noised

    # A batch with nothing corrupted omits the "noised" key entirely rather than
    # reporting a sentinel (the same convention per_class uses).
    all_clean = loss_fn(
        logits,
        target,
        labels,
        changed_positions=torch.zeros(1, 4, dtype=torch.bool),
    )
    assert set(all_clean.canvas_state) == {"ground_truth_preserved"}

    # Omitting changed_positions omits the breakdown; it is never fabricated.
    assert loss_fn(logits, target, labels).canvas_state == {}


def test_canvas_state_ignores_positions_excluded_by_the_scored_mask() -> None:
    """Batch-shape padding must not leak into either canvas-state bucket.

    ``scored_mask`` excludes padded positions from the aggregate loss, so it must
    exclude them here too -- otherwise the preserved bucket would be diluted by
    padding, which is trivially "already correct" and would make the model look
    better at recognizing ground truth than it is.
    """

    loss_fn = CanvasCrossEntropyLoss(_small_config())
    logits = torch.zeros(1, 4, 8)
    target = torch.tensor([[1, 2, 3, 4]])
    labels = torch.full((1, 4), CLASS_CONTENT)
    # Positions 2 and 3 are batch-shape padding: unscored, and "unchanged".
    scored = torch.tensor([[True, True, False, False]])
    changed = torch.tensor([[True, True, False, False]])

    result = loss_fn(
        logits,
        target,
        labels,
        scored_mask=scored,
        changed_positions=changed,
    )

    assert set(result.canvas_state) == {"noised"}


def test_pretraining_loss_weights_are_uniform_including_semantic_pad() -> None:
    """Pre-training loss weighting is fully uniform (published MDLM style).

    The weight buffer must be length 7 (max class id 6 + 1, even though the
    pre-training map is sparse with 5 entries), hold exactly 1.0 for every
    class including semantic `[PAD]`, and be built
    WITHOUT reading `config.loss.class_loss_weights` -- which is None for a
    pre-training config, so any read would crash.
    """

    config = _small_config()
    # Precondition making the "never reads the knob" claim meaningful: the
    # pre-training config genuinely has no class-weight section to read.
    assert config.loss.class_loss_weights is None

    loss_fn = CanvasCrossEntropyLoss(config)
    weights = loss_fn.class_weights

    assert weights.shape == (7,)
    assert weights[CLASS_PAD].item() == 1.0
    for class_id in (CLASS_CONTENT, CLASS_DELIMITER, CLASS_END, CLASS_WINLOSS):
        assert weights[class_id].item() == 1.0
    # The unused ids 1/2 keep the uniform fill too (they are never emitted in
    # pre-training, but the buffer is id-indexed so they must exist).
    assert weights[CLASS_ENEMY_FOGGED].item() == 1.0
    assert weights[CLASS_ENEMY_FUTURE].item() == 1.0


def test_attention_is_bidirectional() -> None:
    torch.manual_seed(7)
    config = _small_config(layers=1)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    model.eval()
    input_ids = torch.tensor([[6, 7, 8, 9]])
    canvas_ids = torch.tensor([[10, 11, 12, 13]])

    with torch.no_grad():
        features = _features(_records(1, 4), input_ids.shape[1])
        base = model(input_token_ids=input_ids, canvas_token_ids=canvas_ids, input_features=features).logits
        changed_canvas = canvas_ids.clone()
        changed_canvas[0, -1] = 40
        changed = model(
            input_token_ids=input_ids,
            canvas_token_ids=changed_canvas,
            input_features=features,
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
        input_features=_features(_records(1, input_ids.shape[1]), input_ids.shape[1]),
    )

    assert output.logits.shape == (1, input_ids.shape[1] + canvas_ids.shape[1], 128)


def test_default_local_shape_instantiates_on_meta_device() -> None:
    config = load_config("config/default.yaml")
    with torch.device("meta"):
        model = SC2StrategyDiffusionModel(config, vocab_size=400)

    first_layer = model.backbone.layers[0]
    assert model.embedding.token_embedding.embedding_dim == 256
    assert len(model.backbone.layers) == 10
    assert first_layer.attn.heads == 4
    assert first_layer.attn.head_dim == 64


def test_model_uses_explicit_depth_scaled_initialization() -> None:
    torch.manual_seed(11)
    config = _small_config(layers=2)
    model = SC2StrategyDiffusionModel(config, vocab_size=128)
    residual_std = 0.02 / (2 * config.model.layers) ** 0.5

    assert model.embedding.token_embedding.weight.std().item() == pytest.approx(0.02, rel=0.25)
    assert model.backbone.layers[0].attn.out.weight.std().item() == pytest.approx(residual_std, rel=0.3)
    assert torch.allclose(
        model.backbone.layers[0].pre_attn_norm.weight,
        torch.ones_like(model.backbone.layers[0].pre_attn_norm.weight),
    )
    assert torch.allclose(
        model.backbone.layers[0].post_attn_norm.weight,
        torch.ones_like(model.backbone.layers[0].post_attn_norm.weight),
    )
