"""Few-replay inspection orchestration and readable exports; no manifests or models.

Loads a small, deterministic set of replays, builds the joint sequence through
the same shared functions future preprocessing will use, and writes readable
text/CSV/JSON views of the first window and of the first packed window.
"""

from bisect import bisect_right
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import shutil

import pandas as pd

from thesis_shared.config import load_config, load_sequence_preview_config
from thesis_shared.data.replay_outcomes import resolve_replay_outcome, resolve_replay_race
from thesis_shared.data.window_policies.unconditioned_joint import SEGMENT_OVERHEAD, iter_joint_windows
from thesis_shared.sequence_formats.types import JointReplay, SequenceWindow, WindowSegment
from thesis_shared.sequence_formats.bpe import ContentBPE, fit_content_bpe
from thesis_shared.sequence_formats.unconditioned_joint import REPLAY_HEADER_LENGTH, iter_joint_timesteps
from thesis_shared.vocab.content_vocab import load_content_vocabulary
from thesis_shared.vocab.sequence_vocabulary import JOINT, OWNERSHIP_TOKENS, RACE_TOKENS, build_sequence_vocabulary

# Tokens that start a new line in sequence.txt.
LINE_STARTS = ("[BOS]", "[SELF]", "[ENEMY]", "[DELIMITER]", "[END]")


def _display_names(window: SequenceWindow, bpe: ContentBPE | None) -> list[str]:
    """Return one human-readable label per window token.

    BPE compounds expand to their atomic names. [BOS] names its replay.
    Delimiters are numbered with one-based replay timesteps; each segment's
    final delimiter says whether the segment stopped at the window end or the
    replay end.
    """
    expansions = bpe.expansions if bpe is not None else {}
    segment_starts = [segment.start_position for segment in window.segments]
    names = []
    for position, token in enumerate(window.tokens):
        segment = window.segments[bisect_right(segment_starts, position) - 1]
        name = token.name
        if bpe is not None and token.token_id >= bpe.base.vocab_size:
            name = "[" + " ".join(bpe.base.names[i] for i in expansions[token.token_id]) + "]"
        elif token.name == "[BOS]" and position == segment.start_position:
            name = f"[BOS | {segment.replay_id}]"
        elif token.name == "[DELIMITER]":
            if position == segment.start_position + REPLAY_HEADER_LENGTH:
                name = f"[DELIMITER {segment.start_timestep + 1}]"
            elif token.timestep == segment.end_timestep - 1:
                ending = "replay end" if segment.reached_replay_end else "window end"
                name = f"[DELIMITER | {ending} after timestep {segment.end_timestep}]"
            else:
                name = f"[DELIMITER {token.timestep + 2}]"
        names.append(name)
    return names


def write_sequence_preview(window: SequenceWindow, output: Path, summary: dict,
                           *, bpe: ContentBPE | None = None) -> None:
    """Write sequence.txt, tokens.csv and summary.json for one window.

    Display formatting never changes token IDs; tokens.csv keeps the raw IDs,
    names and both arms' masks.
    """
    output.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    line: list[str] = []
    for token, name in zip(window.tokens, _display_names(window, bpe)):
        if token.name in LINE_STARTS and line:
            lines.append(" ".join(line))
            line = []
        line.append(name)
    if line:
        lines.append(" ".join(line))
    (output / "sequence.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    segment_starts = [segment.start_position for segment in window.segments]
    fieldnames = ["position", "token_id", "name", "timestep", "owner", "role", "replay_id",
                  "diffusion_clamped", "ar_scored"]
    rows = zip(window.tokens, window.diffusion_clamp_mask, window.ar_loss_mask)
    with (output / "tokens.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for position, (token, clamped, ar_scored) in enumerate(rows):
            segment = window.segments[bisect_right(segment_starts, position) - 1]
            writer.writerow({"position": position, **asdict(token), "replay_id": segment.replay_id,
                             "diffusion_clamped": clamped, "ar_scored": ar_scored})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def select_replays(anchor: Path, count: int) -> list[Path]:
    """Return the anchor replay plus the next ``count - 1`` in sorted filename order.

    Uses the same deterministic selection as ``scripts/bpe_merge_curve.py``.

    Raises:
        ValueError: If fewer than ``count`` replays are available from the anchor on.
    """
    candidates = sorted(anchor.parent.glob("*_game_state.parquet"))
    following = [path for path in candidates if path.name > anchor.name]
    replays = [anchor] + following[:count - 1]
    if len(replays) < count:
        raise ValueError(f"only {len(replays)} replays available from {anchor.name}; replay_count={count}")
    return replays


def _encode_window(window: SequenceWindow, bpe: ContentBPE) -> SequenceWindow:
    """BPE-encode an atomic window segment by segment, keeping segment positions exact."""
    bounds = [segment.start_position for segment in window.segments] + [len(window.tokens)]
    tokens = []
    segments = []
    for segment, begin, stop in zip(window.segments, bounds, bounds[1:]):
        segments.append(replace(segment, start_position=len(tokens)))
        tokens.extend(bpe.encode(window.tokens[begin:stop]))
    return replace(window, tokens=tuple(tokens), segments=tuple(segments))


def _segment_summary(segment: WindowSegment) -> dict:
    return {"replay": segment.replay_id, "start_timestep": segment.start_timestep,
            "end_timestep_exclusive": segment.end_timestep,
            "reached_replay_end": segment.reached_replay_end}


def _window_summary(index: int, window: SequenceWindow) -> dict:
    return {"window": index, "tokens": len(window.tokens),
            "unused_capacity": window.context_budget - len(window.tokens),
            "segments": [_segment_summary(segment) for segment in window.segments]}


def preview_replay(config_path: Path, *, replay_override: Path | None = None,
                   perspective_override: str | None = None) -> Path:
    """Inspect a few source replays; only exports under scripts/output are writable.

    Writes the first window to the output directory and, when packing produced
    one, the first window holding more than one replay to ``packed_boundary/``.

    Calls: select_replays, resolve_replay_race, resolve_replay_outcome,
        iter_joint_timesteps, iter_joint_windows, fit_content_bpe (BPE mode),
        write_sequence_preview.
    """
    settings = load_sequence_preview_config(config_path)
    if settings.sequence_format != JOINT:
        raise ValueError("conditioned_enemy_v1 remains available through the existing pipeline; this preview supports unconditioned_joint_v1 only")
    base = config_path.resolve().parent
    root = Path(__file__).resolve().parents[5]
    output = (base / settings.output_directory).resolve()
    if not output.is_relative_to(root / "scripts" / "output"):
        raise ValueError("preview output must stay below scripts/output")
    perspective = perspective_override or settings.perspective
    if perspective not in ("p1", "p2"):
        raise ValueError("perspective must be p1 or p2")
    project = load_config(base / settings.project_config)
    content = load_content_vocabulary(root / project.pipeline.token_dictionary_uri)
    vocabulary = build_sequence_vocabulary(content, sequence_format=settings.sequence_format)
    if replay_override is not None:
        anchor = replay_override.resolve()
    elif settings.replay_path:
        anchor = (base / settings.replay_path).resolve()
    else:
        anchor = next(iter(sorted((base / settings.input_directory).glob("*_game_state.parquet"))), None)
        if anchor is None:
            raise ValueError("no game-state replay found; provide --replay")
    replay_paths = select_replays(anchor, settings.replay_count)

    # Replay metadata comes first so a bad race/result fails before serialization.
    replays = []
    replay_timesteps = {}
    for path in replay_paths:
        races = {player: resolve_replay_race(path, player) for player in ("p1", "p2")}
        outcomes = {player: resolve_replay_outcome(path, player) for player in ("p1", "p2")}
        frame = pd.read_parquet(path).sort_values("game_loop", kind="stable").reset_index(drop=True)
        replay_timesteps[path.name] = len(frame)
        rows = (row for _, row in frame.iterrows())
        timesteps = iter_joint_timesteps(rows, project, content, vocabulary, perspective=perspective)
        replays.append(JointReplay(path.name, perspective, races, outcomes, timesteps))
    window_args = dict(context_budget=settings.context_budget_tokens, pack_replays=settings.pack_replays)

    comparison = None
    bpe = None
    if settings.tokenizer == "bpe":
        # Bounded to a few replays for debugging, not a corpus preprocessing path.
        replays = [replace(r, timesteps=list(r.timesteps)) for r in replays]
        baseline = next(iter_joint_windows(replays, vocabulary, **window_args))
        all_steps = [step for r in replays for step in r.timesteps]
        bpe = fit_content_bpe(all_steps, vocabulary, vocabulary.content_ids,
                              min_occurrences=settings.bpe_min_occurrences)
        encoded = [replace(r, timesteps=[bpe.encode(step) for step in r.timesteps]) for r in replays]
        for before, after in zip(replays, encoded):
            if any(bpe.decode(e) != b for b, e in zip(before.timesteps, after.timesteps)):
                raise ValueError(f"BPE round-trip failed for {before.replay_id}")
        vocabulary = bpe.vocabulary
        windows = list(iter_joint_windows(encoded, vocabulary, **window_args))
        window = windows[0]
        same_span = _encode_window(baseline, bpe)
        overhead = SEGMENT_OVERHEAD * len(replays)
        comparison = {
            "fit_scope": "every selected replay, one perspective, both owners; no other replay",
            "evaluation_scope": "in-sample debugging only, not held-out compression evidence",
            "min_occurrences": settings.bpe_min_occurrences,
            "merge_count": len(bpe.merges),
            "vocabulary_before": bpe.base.vocab_size, "vocabulary_after": vocabulary.vocab_size,
            "round_trip_exact": True,
            "full_replay": {"tokens_before": sum(map(len, all_steps)) + overhead,
                            "tokens_after": sum(len(s) for r in encoded for s in r.timesteps) + overhead,
                            "accounting": f"all timesteps plus {SEGMENT_OVERHEAD} header/delimiter/footer/END tokens per replay; unwindowed"},
            "original_first_window": {"tokens_before": len(baseline.tokens),
                                      "tokens_after": len(same_span.tokens),
                                      "timesteps": baseline.segments[0].end_timestep},
            "rewindowed_first_window": {"tokens": len(window.tokens),
                                        "timesteps": window.segments[0].end_timestep,
                                        "atomic_equivalent_tokens": len(bpe.decode(window.tokens))},
        }
        output.mkdir(parents=True, exist_ok=True)
        artifact = bpe.artifact()
        (output / "bpe_vocabulary.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        write_sequence_preview(baseline, output / "before", comparison["original_first_window"])
        write_sequence_preview(same_span, output / "same_window_bpe", comparison["original_first_window"], bpe=bpe)
    else:
        windows = list(iter_joint_windows(replays, vocabulary, **window_args))
        window = windows[0]

    packed = next((i for i, w in enumerate(windows) if len(w.segments) > 1), None)
    summary = {
        "status": "preview_only_not_training_integrated",
        "tokenizer": settings.tokenizer,
        "preprocessing_location": "cloud for full tokenization, windowing and manifest building; local few-replay debug only",
        "replays": [path.name for path in replay_paths],
        "replay_timesteps": replay_timesteps,
        "races": {r.replay_id: r.races for r in replays},
        "perspective": perspective,
        "sequence_format": settings.sequence_format,
        "grammar_revision": "race_header_packed_v2",
        "within_timestep_order": "per player block: content tokens in ascending content token ID (serializer order: SC2 type ID, then upgrades); see PRETRAINING_PIPELINE.md",
        "vocabulary_identity": vocabulary.identity, "vocabulary_size": vocabulary.vocab_size,
        "ownership_ids": {name: vocabulary.token_id(name) for name in OWNERSHIP_TOKENS},
        "race_ids": {name: vocabulary.token_id(name) for name in sorted(RACE_TOKENS.values())},
        "pack_replays": settings.pack_replays,
        "context_budget": window.context_budget, "emitted_tokens": len(window.tokens),
        "unused_capacity": window.context_budget - len(window.tokens),
        "segments": [_segment_summary(segment) for segment in window.segments],
        "next_timestep_tokens_including_end_if_terminal": window.next_timestep_tokens,
        "content_tokens": sum(t.role == "content" for t in window.tokens),
        "outcome_tokens": sum(t.role == "outcome" for t in window.tokens),
        "input_tokens": 0, "fog_omission": False, "feature_conditioning": False,
        "padding": "unpadded inspection export; future adapters must distinguish semantic from batch padding",
        "diffusion_contract": {
            "clamped_positions": list(range(window.clamped_prefix_length)),
            "replacement_ids": list(range(vocabulary.vocab_size)),
            "note": "window's leading [BOS] + race header clamped; later packed headers are mutable and scored; full-vocabulary replacement; not yet wired to training/sampling",
        },
        "ar_contract": "every position except 0 is a next-token target, including race headers, packed [BOS] and [END]",
        "outcome_placement": "one four-token SELF/outcome/ENEMY/outcome footer closes every replay segment; END follows only at real replay end",
        "outcome_scoring": "all four footer positions scored; no inference-time complement constraint",
        "outcome_caveat": "The second outcome can use the first under causal teacher forcing; useful for learning the pairing, but not independent evidence of outcome forecasting.",
        "packed_boundary_window": packed,
        "windows": [_window_summary(i, w) for i, w in enumerate(windows)],
    }
    if comparison is not None:
        summary["bpe_identity"] = artifact["identity"]
        summary["comparison"] = comparison
    write_sequence_preview(window, output, summary, bpe=bpe)
    if packed is not None:
        packed_summary = {"window": packed, **_window_summary(packed, windows[packed])}
        write_sequence_preview(windows[packed], output / "packed_boundary", packed_summary, bpe=bpe)
    elif (output / "packed_boundary").exists():
        # Never leave a stale packed view from an earlier run beside new output.
        shutil.rmtree(output / "packed_boundary")
    return output
