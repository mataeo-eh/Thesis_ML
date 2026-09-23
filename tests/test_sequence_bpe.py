"""BPE frequency, hard boundaries, lossless decoding and re-windowing."""

from dataclasses import replace
from pathlib import Path
import json

import pytest
import yaml

from thesis_shared.config import ConfigError, load_sequence_preview_config
from thesis_shared.sequence_formats.bpe import fit_content_bpe
from thesis_shared.sequence_formats.types import SequenceToken
from thesis_shared.sequence_formats.preview import write_sequence_preview
from thesis_shared.data.window_policies.unconditioned_joint import iter_joint_windows
from thesis_shared.vocab.content_vocab import build_content_vocabulary
from thesis_shared.vocab.sequence_vocabulary import JOINT, build_sequence_vocabulary

ROOT = Path(__file__).resolve().parents[1]
CONTENT = build_content_vocabulary({"1": "scv", "2": "marine", "100001": "upgrade"})
VOCAB = build_sequence_vocabulary(CONTENT, sequence_format=JOINT)
ENTITIES = tuple(t.token_id for t in CONTENT.tokens if t.kind == "entity")


def tokens(*names, timestep=0, owner="p1"):
    return tuple(SequenceToken(VOCAB.token_id(n), n, timestep, owner,
                               "content" if n in CONTENT.name_to_id else "structure") for n in names)


def fit(steps, minimum=3):
    return fit_content_bpe(steps, VOCAB, VOCAB.content_ids, min_occurrences=minimum)


def test_threshold_three_accepts_and_two_rejects():
    pair = tokens("scv", "marine")
    assert not fit([pair] * 2).merges
    bpe = fit([pair] * 3)
    assert bpe.merges == ((ENTITIES[0], ENTITIES[1], VOCAB.vocab_size, 3),)
    assert bpe.decode(bpe.encode(pair)) == pair
    assert bpe.vocabulary.names[:VOCAB.vocab_size] == VOCAB.names


@pytest.mark.parametrize("boundary", [n for i, n in enumerate(VOCAB.names) if i not in VOCAB.content_ids])
def test_every_structural_id_is_a_hard_boundary(boundary):
    step = tokens("scv", "scv", boundary, "marine", "marine")
    bpe = fit([step] * 3)
    encoded = bpe.encode(step)
    assert encoded[1] == step[2]
    assert len(encoded) == 3
    assert bpe.decode(encoded) == step
    assert all(set(bpe.expansions[new]) <= set(VOCAB.content_ids) for _, _, new, _ in bpe.merges)


def test_overlap_frequency_and_deterministic_ties():
    step = tokens("scv", "scv", "scv", "scv")
    bpe = fit([step])
    assert bpe.merges[0][3] == 3
    assert len(bpe.encode(step)) == 2
    steps = [tokens("marine", "scv"), tokens("scv", "marine")] * 3
    assert fit(steps).merges == fit(reversed(steps)).merges
    assert fit(steps).merges[0][:2] == ENTITIES
    assert json.loads(json.dumps(bpe.artifact()))["identity"] == bpe.artifact()["identity"]


def test_owners_and_timesteps_cannot_be_crossed_even_without_markers():
    a, b = tokens("scv", "marine")
    assert not fit([(a, replace(b, owner="p2"))] * 3).merges
    assert not fit([(a, replace(b, timestep=1))] * 3).merges


def test_rewindow_compressed_steps_keeps_footer_end_and_budget():
    steps = [tokens("[SELF]", "scv", "scv", "scv", "scv", "[ENEMY]", "[DELIMITER]", timestep=i)
             for i in range(6)]
    bpe = fit(steps)
    args = dict(context_budget=21, perspective="p1", outcomes={"p1": 4, "p2": 5})
    before = list(iter_joint_windows(steps, VOCAB, **args))
    after = list(iter_joint_windows([bpe.encode(s) for s in steps], bpe.vocabulary, **args))
    assert after[0].end_timestep > before[0].end_timestep
    assert after[-1].end_timestep == len(steps)
    assert sum(w.reached_replay_end for w in after) == 1
    assert after[-1].tokens[-1].name == "[END]"
    assert all(len(w.tokens) <= 21 and not w.loss_mask[0] and all(w.loss_mask[1:]) for w in after)
    for w in after:
        decoded = bpe.decode(w.tokens)
        body = tuple(t for step in steps[w.start_timestep:w.end_timestep] for t in step)
        assert decoded[2:2 + len(body)] == body


@pytest.mark.parametrize("field,value", [("tokenizer", "typo"), ("bpe_min_occurrences", 1),
                                         ("bpe_min_occurrences", True), ("bpe_min_occurrences", 2.5)])
def test_preview_bpe_config_fails_closed(tmp_path, field, value):
    raw = yaml.safe_load((ROOT / "config/sequence_preview.yaml").read_text())
    raw[field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        load_sequence_preview_config(path)


def test_old_preview_profiles_stay_atomic(tmp_path):
    raw = yaml.safe_load((ROOT / "config/sequence_preview.yaml").read_text())
    del raw["tokenizer"], raw["bpe_min_occurrences"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_sequence_preview_config(path).tokenizer == "atomic"


def test_upgrades_merge_with_entities_and_render_as_one_readable_token(tmp_path):
    steps = [tokens("[SELF]", "scv", "upgrade", "marine", "[ENEMY]", "[DELIMITER]", timestep=i)
             for i in range(3)]
    bpe = fit(steps)
    assert any(CONTENT.token_id_for("upgrade") in bpe.expansions[m[2]] for m in bpe.merges)
    encoded = [bpe.encode(s) for s in steps]
    assert all(bpe.decode(e) == s for s, e in zip(steps, encoded))
    window = next(iter_joint_windows(encoded, bpe.vocabulary, context_budget=100,
                                     perspective="p1", outcomes={"p1": 4, "p2": 5}))
    write_sequence_preview(window, tmp_path, {}, bpe=bpe)
    text = (tmp_path / "sequence.txt").read_text()
    assert "[SELF] [scv upgrade marine]" in text
    assert "BPE_" not in text
    assert [line for line in text.splitlines() if line.startswith("[DELIMITER")] == [
        "[DELIMITER 1]", "[DELIMITER 2]", "[DELIMITER 3]",
        "[DELIMITER | replay end after timestep 3]"]
    assert "<BPE_" in (tmp_path / "tokens.csv").read_text()
    # Nonzero-start windows use replay timestep numbers, not per-window counts.
    later = replace(window, start_timestep=7, end_timestep=10, reached_replay_end=False,
                    tokens=tuple(replace(t, timestep=t.timestep + 7) if t.timestep is not None else t
                                 for t in window.tokens[:-1]))
    write_sequence_preview(later, tmp_path, {}, bpe=bpe)
    text = (tmp_path / "sequence.txt").read_text()
    assert "[DELIMITER 8]" in text and "[DELIMITER 9]" in text
    assert "[DELIMITER | window end after timestep 10]" in text
