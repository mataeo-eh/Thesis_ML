"""Corpus study: exact BPE equivalence, restart, leakage boundary and accounting."""

from collections import Counter
import json
from pathlib import Path
import random
import shutil

import pandas as pd
import pytest
import yaml

from scripts import tokenizer_study as study
from scripts.oscer.verify_upload import verify
from thesis_shared.config import ConfigError, load_config, load_tokenizer_study_config
from thesis_shared.sequence_formats.bpe import ContentBPE, ContentBPETrainer, _merge, fit_content_bpe
from thesis_shared.sequence_formats.unconditioned_joint import iter_joint_timesteps
from thesis_shared.vocab.content_vocab import load_content_vocabulary
from thesis_shared.vocab.sequence_vocabulary import JOINT, build_sequence_vocabulary
from tests.test_sequence_bpe import VOCAB, tokens

ROOT = Path(__file__).resolve().parents[1]


def reference_fit(words, base_size, maximum):
    """Intentionally slow full recount oracle, independent of the indexed fitter."""
    spans, merges = Counter(words), []
    while base_size + len(merges) < maximum:
        pairs = Counter()
        for word, count in spans.items():
            pairs.update({pair: frequency * count for pair, frequency in Counter(zip(word, word[1:])).items()})
        if not pairs:
            break
        pair = min(pairs, key=lambda p: (-pairs[p], p))
        if pairs[pair] < 2:
            break
        new = base_size + len(merges)
        merges.append((*pair, new, pairs[pair]))
        updated = Counter()
        for word, count in spans.items():
            updated[_merge(word, pair, new)] += count
        spans = updated
    return tuple(merges)


def test_indexed_fitter_and_fast_encoder_match_ranked_reference(tmp_path):
    rng = random.Random(104)
    alphabet = VOCAB.content_ids
    for trial in range(25):
        words = [tuple(rng.choice(alphabet) for _ in range(rng.randrange(25))) for _ in range(30)]
        # Long repeated runs exercise overlap frequency vs true replacement count.
        words += [(alphabet[0],) * 23] * 7
        cap = VOCAB.vocab_size + 20
        with ContentBPETrainer(":memory:", VOCAB, alphabet, min_occurrences=2) as trainer:
            trainer.initialize(Counter(words).items())
            bpe = trainer.fit_to(cap)
            assert bpe.merges == reference_fit(words, VOCAB.vocab_size, cap)
        for word in words:
            slow = word
            for a, b, new, _ in bpe.merges:
                slow = _merge(slow, (a, b), new)
            assert bpe.encode_ids(word) == slow
            assert tuple(i for token in slow for i in bpe.expansions[token]) == word
    with pytest.raises(ValueError, match="content-only"):
        bpe.encode_ids((VOCAB.token_id("[SELF]"),))


def test_fitting_restart_and_identity_guard(tmp_path):
    word = tuple(t.token_id for t in tokens("scv", "scv", "scv", "marine"))
    store = tmp_path / "fit.sqlite"
    with ContentBPETrainer(store, VOCAB, VOCAB.content_ids, min_occurrences=2, identity="train-A") as trainer:
        trainer.initialize([(word, 10)])
        trainer.fit_to(VOCAB.vocab_size + 1)
    with ContentBPETrainer(store, VOCAB, VOCAB.content_ids, min_occurrences=2, identity="train-A") as trainer:
        trainer.initialize(iter(()))  # persisted state, no source reread
        resumed = trainer.fit_to(VOCAB.vocab_size + 10)
        assert sum(row[0] for row in trainer.db.execute("SELECT deleted FROM merges")) == (
            len(word) - len(resumed.encode_ids(word))) * 10
    assert resumed.merges == reference_fit([word] * 10, VOCAB.vocab_size, VOCAB.vocab_size + 10)
    with pytest.raises(ValueError, match="identity"):
        ContentBPETrainer(store, VOCAB, VOCAB.content_ids, min_occurrences=2, identity="test-leak")


def test_metrics_framing_and_heldout_compositions(tmp_path):
    # Only SCV/SCV can be learned. Dev-only marine pair must stay decomposable.
    bpe = fit_content_bpe([tokens("scv", "scv")] * 5, VOCAB, VOCAB.content_ids, min_occurrences=2)
    cache = tmp_path / "words.json.gz"
    import gzip
    marine = VOCAB.token_id("marine")
    scv = VOCAB.token_id("scv")
    with gzip.open(cache, "wt") as handle:
        json.dump({"words": [[[scv, scv], 2], [[marine, marine], 1], [[], 1]]}, handle)
    replay = {"replay": "r", "split": "test", "timesteps": 2, "words": 4, "empty_words": 1,
              "cache": str(cache), "content_before": 6, "structural_tokens": 17,
              "p1_content_tokens": 4, "p2_content_tokens": 2}
    metric, rows, usage = study.measure([replay], bpe, 100)
    assert metric["before"]["total"] == 23
    assert metric["after"]["total"] == 21
    assert metric["tokens_per_word_after"] == 1
    assert usage[marine] == 2
    assert metric["structural_tokens"] == 17
    assert rows[0]["content_after"] == 4
    # Endpoint rule consumes dev only, including zero-savings atomic fallback.
    candidates = [{"vocab_size": size, "dev": {"tokens_removed": saved}, "test": {"tokens_removed": 10**size}}
                  for size, saved in ((1, 0), (2, 99), (3, 100))]
    assert study.choose_endpoint(candidates, .99)["vocab_size"] == 2
    assert study.choose_endpoint([{"vocab_size": 1, "dev": {"tokens_removed": 0}}], .99)["vocab_size"] == 1


def test_frozen_tokenizer_reload_rejects_tampering():
    original = fit_content_bpe([tokens("scv", "scv", "marine")] * 5, VOCAB, VOCAB.content_ids, min_occurrences=2)
    payload = json.loads(json.dumps(original.artifact()))
    loaded = ContentBPE.from_artifact(payload, VOCAB)
    assert loaded == original
    payload["merges"][0][3] += 1
    with pytest.raises(ValueError, match="identity"):
        ContentBPE.from_artifact(payload, VOCAB)


def test_projected_real_replay_matches_full_serializer(tmp_path):
    path = ROOT.parent / "data/quickstart/parquet/match_4745721_game_state.parquet"
    if not path.exists():
        pytest.skip("single real replay unavailable")
    project = load_config(ROOT / "config/default.yaml")
    content = load_content_vocabulary(ROOT / project.pipeline.token_dictionary_uri)
    vocab = build_sequence_vocabulary(content, sequence_format=JOINT)
    info = study.extract_replay((path, tmp_path, "test", project, content, vocab))
    full = pd.read_parquet(path).sort_values("game_loop", kind="stable")
    expected = Counter()
    for step in iter_joint_timesteps((row for _, row in full.iterrows()), project, content, vocab, perspective="p1"):
        for owner in ("p1", "p2"):
            expected[tuple(t.token_id for t in step if t.role == "content" and t.owner == owner)] += 1
    assert study.load_words(info["cache"]) == expected
    assert info["structural_tokens"] == 3 * len(full) + 11
    assert study.extract_replay((path, tmp_path, "test", project, content, vocab)) == info


def test_tiny_corpus_end_to_end_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    shutil.copyfile(ROOT / "data/Token_Dictionary.json", tmp_path / "data/Token_Dictionary.json")
    (tmp_path / "parquet").mkdir()
    (tmp_path / "json").mkdir()
    for number in range(20):
        worker, race = ("probe", "Protoss") if number % 2 == 0 else ("drone", "Zerg")
        pd.DataFrame({"game_loop": [0, 22, 45],
                      "p1_bot_scv_000_pos_(X,Y,Z)": ["(0,0,0)", "(0,0,0)", "(0,0,0)"],
                      "p1_bot_scv_001_pos_(X,Y,Z)": ["(0,0,0)", "(0,0,0)", "destroyed"],
                      f"p2_bot_{worker}_000_pos_(X,Y,Z)": ["(0,0,0)", "(0,0,0)", "(0,0,0)"]}).to_parquet(
                          tmp_path / f"parquet/match_{number}_game_state.parquet")
        (tmp_path / f"json/match_{number}_metadata.json").write_text(json.dumps({"players": {
            "p1": {"race": "Terran", "result": "Victory"},
            "p2": {"race": race, "result": "Defeat"}}}))
    project_raw = yaml.safe_load((ROOT / "config/default.yaml").read_text())
    project_raw["pipeline"].update(train_replay_count=12, validation_replay_count=4)
    project_path = tmp_path / "project.yaml"
    project_path.write_text(yaml.safe_dump(project_raw))
    raw = yaml.safe_load((ROOT / "config/tokenizer_study.yaml").read_text())
    raw.update(project_config=str(project_path), vocabulary_sizes="300,305",
               reference_vocab_size=300, selection_rule="multilingual_savings", workers=2)
    config = tmp_path / "settings.yaml"
    config.write_text(yaml.safe_dump(raw))
    output = tmp_path / "scripts/output/study"
    result = study.run(config, tmp_path / "parquet", output=output)
    summary = json.loads((result / "summary.json").read_text())
    assert summary["state"] == "complete"
    assert summary["metrics"]["all"]["replays"] == 20
    assert summary["metrics"]["all"]["before"]["total"] == 20 * (8 + 3 * 3 + 11)
    assert summary["metrics"]["all"]["tokens_removed"] == 40
    assert summary["selection_rule"] == "multilingual_savings"
    terran = summary["metrics"]["all"]["languages"]["[TERRAN]"]
    assert terran["player_replays"] == 20
    assert terran["words"] == 60
    assert sum(value["content_after"] for value in summary["metrics"]["all"]["languages"].values()) == summary["metrics"]["all"]["content_after"]
    assert summary["both_perspectives_total_after"] == 2 * summary["metrics"]["all"]["after"]["total"]
    provenance = json.loads((result / "provenance.json").read_text())
    assert not (set(provenance["split"]["train"]) & set(provenance["split"]["dev"]))
    assert (result / "compression_curve.png").is_file()
    assert study.run(config, tmp_path / "parquet", output=output) == result


@pytest.mark.parametrize("field,value", [("workers", 0), ("min_occurrences", 1),
                                          ("retained_dev_savings", 1.1), ("workers", True),
                                          ("vocabulary_sizes", "512,512"), ("vocabulary_sizes", "bad")])
def test_study_config_rejects_invalid_settings(tmp_path, field, value):
    raw = yaml.safe_load((ROOT / "config/tokenizer_study.yaml").read_text())
    raw[field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        load_tokenizer_study_config(path)


def test_upload_checksums_and_replay_membership_fail_closed(tmp_path):
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    file = parquet / "match_a_game_state.parquet"
    file.write_bytes(b"fixture")
    (tmp_path / "upload_manifest.json").write_text(json.dumps({"files": [
        {"path": "parquet/" + file.name, "sha256": study.file_hash(file)}]}))
    verify(tmp_path)
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path)


def test_random_resolves_to_actual_language_without_fourth_language():
    assert study.resolve_language("[RANDOM]", (VOCAB.token_id("scv"),), VOCAB) == "[TERRAN]"
    assert study.resolve_language("[ZERG]", (), VOCAB) == "[ZERG]"
    with pytest.raises(ValueError, match="unambiguous"):
        study.resolve_language("[RANDOM]", (), VOCAB)


def test_multilingual_selection_protects_each_language():
    candidates = []
    for size, all_saved, race_saved in ((1, 0, 0), (2, 999, 90), (3, 1000, 100)):
        candidates.append({"vocab_size": size, "dev": {"tokens_removed": all_saved,
            "languages": {race: {"words": 100, "content_tokens_removed": race_saved} for race in study.LANGUAGES}},
            "train": {"languages": {race: {"words": 100} for race in study.LANGUAGES}}})
    assert study.choose_endpoint(candidates, .99)["vocab_size"] == 2
    assert study.choose_multilingual_endpoint(candidates, .99)["vocab_size"] == 3
    candidates[0]["dev"]["languages"]["[ZERG]"]["words"] = 0
    with pytest.raises(ValueError, match="coverage"):
        study.choose_multilingual_endpoint(candidates, .99)
