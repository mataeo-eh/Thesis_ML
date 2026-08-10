"""Contract checks for the standalone full-size GPU smoke fixture."""

from dataclasses import replace
from pathlib import Path

from scripts.gpu_smoke_test import build_synthetic_batch
from thesis_ml.config import load_config
from thesis_ml.data.dataset import CLASS_CLAMPED, CLASS_WINLOSS
from thesis_ml.vocab.special_tokens import BOS_ID, EOS_ID, LOSS_ID, WIN_ID


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_smoke_fixture_uses_production_boundary_grammar() -> None:
    config = load_config(ROOT / "config" / "default.yaml")
    config = replace(config, data=replace(config.data, canvas_budget_tokens=12))

    batch = build_synthetic_batch(
        config,
        batch_size=4,
        input_len=8,
        vocab_size=291,
        device="cpu",
    )

    assert batch.input_token_ids[:, -1].eq(EOS_ID).all()
    assert batch.target_canvas[:, 0].eq(BOS_ID).all()
    assert set(batch.target_canvas[:, 1].tolist()) == {WIN_ID, LOSS_ID}
    assert batch.class_labels[:, 0].eq(CLASS_CLAMPED).all()
    assert batch.class_labels[:, 1].eq(CLASS_WINLOSS).all()
    assert not batch.canvas_loss_mask[:, 0].any()
    assert batch.canvas_loss_mask[:, 1].all()
