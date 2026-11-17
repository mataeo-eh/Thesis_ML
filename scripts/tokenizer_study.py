"""CPU-only, restartable corpus BPE study for OSCER; no training artifacts.

Each word is one owner's sorted content IDs at one recorded timestep. Both
owners occur once per replay. Structural tokens are accounted for separately.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from thesis_shared.config import load_config, load_tokenizer_study_config
from thesis_shared.data.features import POSITION_KEY
from thesis_shared.data.replay_outcomes import resolve_replay_outcome, resolve_replay_race
from thesis_shared.data.split import split_replays
from thesis_shared.sequence_formats.bpe import ContentBPE, ContentBPETrainer
from thesis_shared.sequence_formats.unconditioned_joint import iter_joint_timesteps
from thesis_shared.serialize import parse_entity_columns
from thesis_shared.vocab.content_vocab import load_content_vocabulary
from thesis_shared.vocab.sequence_vocabulary import JOINT, build_sequence_vocabulary

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "owner_timestep_bpe_study_v2"
LANGUAGES = ("[PROTOSS]", "[TERRAN]", "[ZERG]")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_words(path: str | Path) -> Counter:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return Counter({tuple(word): count for word, count in json.load(handle)["words"]})


def resolve_language(race: str, first_word: tuple[int, ...], vocabulary) -> str:
    """Random resolves through the first recorded starting worker/townhall.

    Metadata usually already names the actual race. A truly unresolved or
    ambiguous Random start fails instead of inventing a fourth language.
    """
    if race in LANGUAGES:
        return race
    names = {vocabulary.names[i] for i in first_word}
    anchors = {"[PROTOSS]": {"probe", "nexus"}, "[TERRAN]": {"scv", "commandcenter"},
               "[ZERG]": {"drone", "hatchery"}}
    matches = [language for language, units in anchors.items() if units & names]
    if race != "[RANDOM]" or len(matches) != 1:
        raise ValueError("Random race requires an unambiguous starting worker/townhall in the first timestep")
    return matches[0]


def extract_replay(task) -> dict:
    """One replay per worker; project only positions/upgrades needed for IDs."""
    path, cache, source_identity, project, content, vocabulary = task
    metadata = path.parent.parent / "json" / path.name.replace("_game_state.parquet", "_metadata.json")
    before = (path.stat().st_size, path.stat().st_mtime_ns, metadata.stat().st_mtime_ns)
    hashes = {"parquet_sha256": file_hash(path), "metadata_sha256": file_hash(metadata)}
    identity = digest([SCHEMA, source_identity, path.name, hashes])
    destination = cache / f"{path.stem}-{identity}.json.gz"
    if destination.exists():
        with gzip.open(destination, "rt", encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved["info"]["identity"] != identity:
            raise ValueError("corrupt replay cache identity")
        return {**saved["info"], "cache": str(destination)}
    recorded_races = {owner: resolve_replay_race(path, owner) for owner in ("p1", "p2")}
    races = dict(recorded_races)
    outcomes = {owner: resolve_replay_outcome(path, owner) for owner in ("p1", "p2")}
    if set(outcomes.values()) != {4, 5}:
        raise ValueError(f"{path.name}: expected complementary recorded outcomes")
    schema_names = pq.ParquetFile(path).schema_arrow.names
    groups = parse_entity_columns(schema_names)
    columns = {"game_loop"}
    columns.update(group.attributes[POSITION_KEY] for group in groups if POSITION_KEY in group.attributes)
    columns.update(name for name in ("p1_upgrades", "p2_upgrades") if name in schema_names)
    frame = pd.read_parquet(path, columns=sorted(columns)).sort_values("game_loop", kind="stable")
    if frame.empty or frame.game_loop.isna().any() or frame.game_loop.duplicated().any():
        raise ValueError(f"{path.name}: missing/duplicate timesteps or empty replay")
    words = Counter()
    race_words = {race: Counter() for race in LANGUAGES}
    owners = Counter()
    timesteps = 0
    for step in iter_joint_timesteps((row for _, row in frame.iterrows()), project,
                                    content, vocabulary, perspective="p1"):
        # Marker-separated words preserve empty player blocks as well.
        current = []
        owner = None
        for token in step:
            if token.role == "content":
                current.append(token.token_id)
                owners[owner] += 1
            else:
                if owner is not None:
                    word = tuple(current)
                    if timesteps == 0:
                        races[owner] = resolve_language(races[owner], word, vocabulary)
                    words[word] += 1
                    race_words[races[owner]][word] += 1
                current = []
                owner = token.owner if token.name in ("[SELF]", "[ENEMY]") else None
        timesteps += 1
    if sum(words.values()) != 2 * timesteps:
        raise ValueError("owner/timestep word accounting mismatch")
    if before != (path.stat().st_size, path.stat().st_mtime_ns, metadata.stat().st_mtime_ns):
        raise ValueError(f"{path.name}: source changed while parsing")
    info = {"replay": path.name, "identity": identity, **hashes, "timesteps": timesteps,
            "words": 2 * timesteps, "empty_words": words[()], "races": races, "recorded_races": recorded_races,
            "p1_content_tokens": owners["p1"], "p2_content_tokens": owners["p2"],
            "content_before": sum(len(word) * count for word, count in words.items()),
            "structural_tokens": 3 * timesteps + 11}
    cache.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump({"info": info, "words": sorted(words.items()),
                   "race_words": {race: sorted(counts.items()) for race, counts in race_words.items()}}, handle)
    temporary.replace(destination)
    return {**info, "cache": str(destination)}


def summary_counts(values) -> dict:
    array = np.asarray(values, dtype=np.int64)
    if not len(array):
        return {"total": 0, "mean": None, "min": None, "max": None,
                **{f"p{p}": None for p in (50, 90, 95, 99)}}
    return {"total": int(array.sum()), "mean": float(array.mean()),
            "min": int(array.min()), "max": int(array.max()),
            **{f"p{p}": float(np.percentile(array, p)) for p in (50, 90, 95, 99)}}


def measure(replays: list[dict], bpe: ContentBPE, rare_threshold: int) -> tuple[dict, list, Counter]:
    """Exact round trip for every distinct word; pool counts before ratios."""
    rows, usage = [], Counter()
    languages = {race: {"player_replays": 0, "words": 0, "empty_words": 0,
                        "content_before": 0, "content_after": 0} for race in LANGUAGES}
    language_usage = {race: Counter() for race in LANGUAGES}
    expansions = bpe.expansions
    for replay in replays:
        after = 0
        with gzip.open(replay["cache"], "rt", encoding="utf-8") as handle:
            cache = json.load(handle)
        encodings = {}
        for word_list, count in cache["words"]:
            word = tuple(word_list)
            encoded = bpe.encode_ids(word)
            encodings[word] = encoded
            decoded = tuple(atom for token in encoded for atom in expansions[token])
            if decoded != word:
                raise ValueError(f"lossy BPE in {replay['replay']}")
            after += len(encoded) * count
            for token in encoded:
                usage[token] += count
        for race, race_words in cache.get("race_words", {}).items():
            value = languages[race]
            value["player_replays"] += sum(r == race for r in replay["races"].values())
            for word_list, count in race_words:
                word = tuple(word_list)
                encoded = encodings[word]
                value["words"] += count
                value["empty_words"] += (not word) * count
                value["content_before"] += len(word) * count
                value["content_after"] += len(encoded) * count
                for token in encoded:
                    language_usage[race][token] += count
        rows.append({"replay": replay["replay"], "split": replay["split"],
                     "timesteps": replay["timesteps"], "words": replay["words"],
                     "empty_words": replay["empty_words"],
                     "content_before": replay["content_before"], "content_after": after,
                     "structural_tokens": replay["structural_tokens"],
                     "full_before": replay["content_before"] + replay["structural_tokens"],
                     "full_after": after + replay["structural_tokens"],
                     "p1_content_before": replay["p1_content_tokens"],
                     "p2_content_before": replay["p2_content_tokens"]})
    before = summary_counts([row["full_before"] for row in rows])
    after = summary_counts([row["full_after"] for row in rows])
    content_before = sum(row["content_before"] for row in rows)
    content_after = sum(row["content_after"] for row in rows)
    words = sum(row["words"] for row in rows)
    nonempty = words - sum(row["empty_words"] for row in rows)
    compounds = [merge[2] for merge in bpe.merges]
    metrics = {"replays": len(rows), "before": before, "after": after,
               "content_before": content_before, "content_after": content_after,
               "structural_tokens": sum(row["structural_tokens"] for row in rows),
               "tokens_removed": before["total"] - after["total"],
               "reduction_fraction": 1 - after["total"] / before["total"] if before["total"] else 0,
               "compression_factor": before["total"] / after["total"] if after["total"] else None,
               "words": words, "empty_words": words - nonempty,
               "tokens_per_word_before": content_before / words if words else None,
               "tokens_per_word_after": content_after / words if words else None,
               "tokens_per_nonempty_word_after": content_after / nonempty if nonempty else None,
               "used_content_ids": len(usage), "unused_compound_ids": sum(usage[i] == 0 for i in compounds),
               "rare_compound_ids": sum(0 < usage[i] < rare_threshold for i in compounds),
               "roundtrip": "exact atomic ID sequence for every owner/timestep word"}
    metrics["vocab_size"] = bpe.vocabulary.vocab_size
    metrics["allocated_compound_ids"] = len(compounds)
    metrics["unused_content_ids"] = len(bpe.vocabulary.content_ids) - len(usage)
    for race, value in languages.items():
        value["content_tokens_removed"] = value["content_before"] - value["content_after"]
        value["content_reduction_fraction"] = (1 - value["content_after"] / value["content_before"]
                                                if value["content_before"] else None)
        value["tokens_per_word_after"] = value["content_after"] / value["words"] if value["words"] else None
        value["used_content_ids"] = len(language_usage[race])
        value["used_compound_ids"] = sum(i in language_usage[race] for i in compounds)
    metrics["languages"] = languages
    return metrics, rows, usage


def choose_endpoint(candidates: list[dict], retained: float) -> dict:
    """Smallest vocabulary retaining the specified share of best DEV savings."""
    best = max(row["dev"]["tokens_removed"] for row in candidates)
    return min((row for row in candidates if row["dev"]["tokens_removed"] >= retained * best),
               key=lambda row: row["vocab_size"])


def choose_multilingual_endpoint(candidates: list[dict], retained: float) -> dict:
    """Retain dev compression gains overall AND within each represented language.

    Missing train/dev language coverage cannot support a three-language choice.
    The tolerance is a project rule, not an optimum claimed by an LLM report.
    """
    missing = [race for race in LANGUAGES if not candidates[0]["dev"]["languages"][race]["words"]
               or not candidates[0]["train"]["languages"][race]["words"]]
    if missing:
        raise ValueError(f"three-language selection requires train/dev coverage for {missing}; inspect split and corpus")
    best_all = max(row["dev"]["tokens_removed"] for row in candidates)
    best_races = {race: max(row["dev"]["languages"][race]["content_tokens_removed"] for row in candidates)
                  for race in LANGUAGES}
    eligible = [row for row in candidates if row["dev"]["tokens_removed"] >= retained * best_all
                and all(row["dev"]["languages"][race]["content_tokens_removed"] >= retained * best_races[race]
                        for race in LANGUAGES)]
    return min(eligible, key=lambda row: row["vocab_size"])


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path, input_directory: Path | None = None, *, output: Path | None = None,
        replay_limit: int = 0) -> Path:
    started = time.monotonic()
    settings = load_tokenizer_study_config(config_path)
    project = load_config(config_path.parent / settings.project_config)
    content = load_content_vocabulary(ROOT / project.pipeline.token_dictionary_uri)
    vocabulary = build_sequence_vocabulary(content, sequence_format=JOINT)
    if min(settings.sizes) < vocabulary.vocab_size:
        raise ValueError("candidate sizes must include the entire base vocabulary")
    if project.pipeline.train_replay_ids or project.pipeline.dev_replay_ids:
        raise ValueError("study expects a seeded replay split; named training profiles are unsupported")
    source = input_directory or (config_path.parent / settings.input_directory)
    paths = sorted(source.resolve().glob(settings.replay_glob))
    if replay_limit:
        paths = paths[:replay_limit]
    if len(paths) < 3:
        raise ValueError("at least three replays and nonempty train/dev/test splits are required")
    split = split_replays([path.name for path in paths], seed=project.pipeline.split_seed,
                         test_fraction=project.pipeline.test_fraction, dev_fraction=project.pipeline.dev_fraction,
                         train_count=project.pipeline.train_replay_count,
                         dev_count=project.pipeline.validation_replay_count)
    if not all((split.train, split.dev, split.test)):
        raise ValueError("train/dev/test must each contain replays; use a larger corpus or explicit counts")
    base = (output or ROOT / "scripts/output/tokenizer_study").resolve()
    if not base.is_relative_to(ROOT / "scripts/output"):
        raise ValueError("output must stay below scripts/output")
    work = base / "work"
    work.mkdir(parents=True, exist_ok=True)
    # Source and merged config, not paths/mtimes, bind reusable parse caches.
    package = ROOT / "packages/thesis-shared/src/thesis_shared"
    source_identity = digest([SCHEMA, asdict(project), vocabulary.identity, file_hash(Path(__file__)),
                              [(str(p.relative_to(package)), file_hash(p)) for p in sorted(package.rglob("*.py"))]])
    tasks = ((path, work / "replays", source_identity, project, content, vocabulary) for path in paths)
    print(f"Parsing/checking {len(paths)} replays with {settings.workers} workers", flush=True)
    replays = []
    with ProcessPoolExecutor(max_workers=settings.workers) as pool:
        for index, info in enumerate(pool.map(extract_replay, tasks, chunksize=1), 1):
            replays.append(info)
            print(f"replay {index}/{len(paths)}: {info['replay']} ({info['content_before']:,} atomic content)", flush=True)
    membership = {name: group for group in ("train", "dev", "test") for name in getattr(split, group)}
    for info in replays:
        info["split"] = membership[info["replay"]]
    manifest = [{k: v for k, v in info.items() if k != "cache"} for info in replays]
    identity = digest([source_identity, asdict(settings), manifest])
    result = base / "results" / identity[:16]
    result.mkdir(parents=True, exist_ok=True)
    write_json(result / "status.json", {"state": "running", "study_identity": identity})
    write_json(result / "provenance.json", {"schema": SCHEMA, "study_identity": identity,
               "source_identity": source_identity, "settings": asdict(settings), "project_config": asdict(project),
               "base_vocabulary_identity": vocabulary.identity, "replays": manifest,
               "split": asdict(split), "partial_corpus_debugging": bool(replay_limit),
               "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__})
    train = [r for r in replays if r["split"] == "train"]
    dev = [r for r in replays if r["split"] == "dev"]
    if settings.selection_rule == "multilingual_savings":
        for label, subset in (("train", train), ("dev", dev)):
            represented = {race for replay in subset for race in replay["races"].values()}
            if missing := set(LANGUAGES) - represented:
                raise ValueError(f"{label} lacks race-language coverage {sorted(missing)}; stop before BPE fitting")
    models = {vocabulary.vocab_size: ContentBPE(vocabulary, vocabulary.content_ids, settings.min_occurrences, ())}
    candidates = []

    def evaluate(bpe):
        size = bpe.vocabulary.vocab_size
        cached = result / f"candidate-{size}.json"
        if cached.exists():
            row = json.loads(cached.read_text(encoding="utf-8"))
            if row["tokenizer_identity"] != bpe.artifact()["identity"]:
                raise ValueError("candidate checkpoint identity mismatch")
        else:
            print(f"Measuring vocabulary {size:,} on train/dev", flush=True)
            row = {"vocab_size": size, "merges": len(bpe.merges), "tokenizer_identity": bpe.artifact()["identity"],
                   "tied_embedding_parameters_at_reference_width": size * settings.reference_width}
            for label, subset in (("train", train), ("dev", dev)):
                row[label] = measure(subset, bpe, settings.rare_usage_threshold)[0]
                # Structural work indicator, not end-to-end runtime/FLOPs:
                # dense LM-head work scales with tokens * vocab * width.
                row[label]["dense_head_token_vocab_product"] = row[label]["after"]["total"] * size
            write_json(cached, row)
        candidates.append(row)
        models[size] = bpe

    def training_words():
        for replay in train:
            yield from load_words(replay["cache"]).items()

    with ContentBPETrainer(work / f"fit-{identity}.sqlite", vocabulary, vocabulary.content_ids,
                           min_occurrences=settings.min_occurrences, identity=identity) as trainer:
        print("Initializing/checking disk pair index (training replays only)", flush=True)
        trainer.initialize(training_words())
        evaluate(models[vocabulary.vocab_size])
        # Reconstruct earlier candidates from the persisted ranked prefix on resume.
        persisted = trainer.snapshot()
        for target in settings.sizes:
            if target <= persisted.vocabulary.vocab_size:
                bpe = ContentBPE(vocabulary, vocabulary.content_ids, settings.min_occurrences,
                                 persisted.merges[:target - vocabulary.vocab_size])
            else:
                bpe = trainer.fit_to(target, progress=lambda size, freq, deleted:
                                    print(f"vocab={size} pair_frequency={freq} tokens_removed={deleted}", flush=True)
                                    if size % 100 == 0 else None)
            if bpe.vocabulary.vocab_size not in models:
                evaluate(bpe)
        merge_curve = [{"rank": rank, "vocab_size": new + 1, "left": left, "right": right,
                        "new_id": new, "overlapping_pair_frequency": frequency, "train_tokens_removed": deleted}
                       for rank, (new, left, right, frequency, deleted) in enumerate(trainer.db.execute(
                           "SELECT new_id,left_id,right_id,frequency,deleted FROM merges ORDER BY new_id"), 1)]
        write_csv(result / "merge_curve.csv", merge_curve)
    compression_choice = choose_endpoint(candidates, settings.retained_dev_savings)
    # The primary AR/NLP reference is fixed in advance, independent of all metrics.
    reference_size = min(settings.reference_vocab_size, max(models))
    reference_choice = next(row for row in candidates if row["vocab_size"] == reference_size)
    if settings.selection_rule == "reference":
        chosen = reference_choice
    elif settings.selection_rule == "multilingual_savings":
        chosen = choose_multilingual_endpoint(candidates, settings.retained_dev_savings)
    else:
        chosen = compression_choice
    bpe = models[chosen["vocab_size"]]
    # Lock selection BEFORE ever measuring test. Test cannot influence the endpoint.
    write_json(result / "selection.json", {"rule": settings.selection_rule,
               "reference_requested": settings.reference_vocab_size, "reference_achieved": reference_size,
               "compression_recommendation": compression_choice,
               "retained_dev_savings": settings.retained_dev_savings, "selected": chosen,
               "scope": "shared three-language vocabulary for generic AR LLM; downstream model quality untested"})
    metrics, rows, usage = measure(replays, bpe, settings.rare_usage_threshold)
    write_csv(result / "per_replay.csv", rows)
    write_json(result / "tokenizer.json", bpe.artifact())
    expansions = bpe.expansions
    content_ids = set(bpe.vocabulary.content_ids)
    write_csv(result / "token_usage.csv", [{"id": i, "name": name, "content_occurrences_all": usage[i],
               "atomic_length": len(expansions[i])} for i, name in enumerate(bpe.vocabulary.names)
               if i in content_ids])
    flat = []
    for row in candidates:
        flat.append({"vocab_size": row["vocab_size"], "merges": row["merges"],
                     "embedding_parameters": row["tied_embedding_parameters_at_reference_width"],
                     **{f"{split_name}_{key}": row[split_name][key]
                        for split_name in ("train", "dev")
                        for key in ("content_before", "content_after", "tokens_removed", "reduction_fraction",
                                    "tokens_per_nonempty_word_after", "unused_compound_ids", "rare_compound_ids",
                                    "dense_head_token_vocab_product")},
                     **{f"dev_{race.strip('[]').lower()}_{key}": row["dev"]["languages"][race][key]
                        for race in LANGUAGES for key in ("words", "content_before", "content_after", "tokens_per_word_after")}})
    write_csv(result / "candidates.csv", flat)
    summaries = {"all": metrics}
    for group in ("train", "dev", "test"):
        summaries[group] = measure([r for r in replays if r["split"] == group], bpe, settings.rare_usage_threshold)[0]
    summary = {"state": "complete", "study_identity": identity, "base_vocab_size": vocabulary.vocab_size,
               "selected_vocab_size": chosen["vocab_size"], "tokenizer_identity": bpe.artifact()["identity"],
               "selection_rule": settings.selection_rule, "reference_requested": settings.reference_vocab_size,
               "compression_recommendation_vocab_size": compression_choice["vocab_size"],
               "intended_consumer": "generic autoregressive language model; tokenizer study does not construct a model",
               "candidates": candidates, "metrics": summaries,
               "accounting": "one full joint stream per replay; content + 3*timestep + 11; no padding or repeated window framing",
               "both_perspectives_total_before": 2 * metrics["before"]["total"],
               "both_perspectives_total_after": 2 * metrics["after"]["total"],
               "partial_corpus_debugging": bool(replay_limit), "elapsed_seconds": time.monotonic() - started,
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "requested_vocabulary_sizes": settings.sizes,
               "largest_achieved_vocab_size": max(models),
               "selected_at_search_boundary": chosen["vocab_size"] == max(models) and bool(bpe.merges),
               "boundary": "tokenizer study only; no manifests, training integration, or model-performance claims"}
    write_json(result / "summary.json", summary)
    report = ["# Tokenizer study", "", f"Selected vocabulary: {chosen['vocab_size']:,} IDs ({len(bpe.merges):,} merges).",
              f"Selection rule: {settings.selection_rule}; requested AR/NLP reference: {settings.reference_vocab_size:,} IDs.",
              f"Dev-compression recommendation ({settings.retained_dev_savings:.1%} of observed savings): {compression_choice['vocab_size']:,} IDs.",
              "One shared tokenizer for Protoss/Terran/Zerg; generic AR LLM consumer. Model quality untested.", "",
              "| Scope | Replays | Tokens before | Tokens after | Mean before | Mean after | Min before | Min after | Max before | Max after |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scope, value in summaries.items():
        before, after = value["before"], value["after"]
        report.append(f"| {scope} | {value['replays']} | {before['total']:,} | {after['total']:,} | "
                      f"{before['mean']:.1f} | {after['mean']:.1f} | {before['min']:,} | {after['min']:,} | "
                      f"{before['max']:,} | {after['max']:,} |")
    report += ["", summary["accounting"], "", "See candidates.csv for the dev sweep and per_replay.csv for individual counts.",
               "Word = one player's sorted entity and upgrade IDs at one timestep; exact decoding verified.",
               f"Elapsed job analysis time: {summary['elapsed_seconds']:.1f} seconds (not an advance runtime estimate)."]
    if summary["selected_at_search_boundary"]:
        report += ["", "Selection is at the largest achieved inventory. Inspect whether pair exhaustion ended fitting or the configured search ceiling needs expansion; this is not proof of a vocabulary optimum."]
    report += ["", "| Race language | Player replay streams | Words | Content before | Content after | Tokens/word after |",
               "|---|---:|---:|---:|---:|---:|"]
    for race, value in metrics["languages"].items():
        fertility = f"{value['tokens_per_word_after']:.3f}" if value["words"] else "n/a"
        report.append(f"| {race} | {value['player_replays']} | {value['words']:,} | {value['content_before']:,} | {value['content_after']:,} | {fertility} |")
    report += ["", "Race rows count player-block CONTENT only; shared delimiters/headers are included in full-stream totals above."]
    (result / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    plot_candidates(result, candidates, chosen["vocab_size"])
    write_json(result / "status.json", {"state": "complete", "study_identity": identity})
    print("\n".join(report), flush=True)
    print(f"Results: {result.relative_to(ROOT)}", flush=True)
    return result


def plot_candidates(result: Path, candidates: list[dict], selected: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, (axis, languages) = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)
    for split in ("train", "dev"):
        axis.plot([r["vocab_size"] for r in candidates],
                  [100 * r[split]["reduction_fraction"] for r in candidates], marker="o", label=split)
    axis.axvline(selected, color="black", linestyle="--", label=f"selected {selected:,}")
    axis.set_xscale("log", base=2)
    axis.set(xlabel="Total vocabulary IDs", ylabel="Full-stream tokens removed (%)",
             title="Owner/timestep BPE: compression shortlist")
    axis.grid(alpha=0.25)
    axis.legend()
    for race in LANGUAGES:
        points = [(row["vocab_size"], row["dev"]["languages"][race]["tokens_per_word_after"])
                  for row in candidates if row["dev"]["languages"][race]["words"]]
        if points:
            languages.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=race.strip("[]"))
    languages.set_xscale("log", base=2)
    languages.set(xlabel="Shared vocabulary IDs", ylabel="Content tokens per player/timestep word",
                  title="Dev fertility by race language (empty words included)")
    languages.axvline(selected, color="black", linestyle="--")
    languages.grid(alpha=0.25)
    if languages.lines and any(line.get_label()[0] != "_" for line in languages.lines):
        languages.legend()
    figure.savefig(result / "compression_curve.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/tokenizer_study.yaml")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-limit", type=int, default=0, help="debugging only; 0 = complete corpus")
    args = parser.parse_args()
    if args.replay_limit < 0:
        parser.error("--replay-limit must be non-negative")
    run(args.config.resolve(), args.input_dir, output=args.output, replay_limit=args.replay_limit)


if __name__ == "__main__":
    main()
