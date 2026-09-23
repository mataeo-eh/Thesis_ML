"""One-replay inspection orchestration and readable exports; no manifests or models."""

from dataclasses import asdict, replace
import csv
import json
from pathlib import Path

import pandas as pd

from thesis_shared.config import load_config, load_sequence_preview_config
from thesis_shared.data.replay_outcomes import resolve_replay_outcome
from thesis_shared.data.window_policies.unconditioned_joint import iter_joint_windows
from thesis_shared.sequence_formats.types import SequenceWindow
from thesis_shared.sequence_formats.bpe import ContentBPE, fit_content_bpe
from thesis_shared.sequence_formats.unconditioned_joint import iter_joint_timesteps
from thesis_shared.vocab.content_vocab import load_content_vocabulary
from thesis_shared.vocab.sequence_vocabulary import JOINT, build_sequence_vocabulary


def write_sequence_preview(window: SequenceWindow, output: Path, summary: dict,
                           *, bpe: ContentBPE | None = None) -> None:
    """Keep human formatting separate from sequence construction."""
    output.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    line: list[str] = []
    expansions = bpe.expansions if bpe is not None else {}
    for position, token in enumerate(window.tokens):
        if token.name in ("[SELF]", "[ENEMY]", "[DELIMITER]", "[END]") and line:
            lines.append(" ".join(line))
            line = []
        name = token.name
        if bpe is not None and token.token_id >= bpe.base.vocab_size:
            name = "[" + " ".join(bpe.base.names[i] for i in expansions[token.token_id]) + "]"
        elif token.name == "[DELIMITER]":
            if position == 1:
                name = f"[DELIMITER {window.start_timestep + 1}]"
            elif token.timestep == window.end_timestep - 1:
                ending = "replay end" if window.reached_replay_end else "window end"
                name = f"[DELIMITER | {ending} after timestep {window.end_timestep}]"
            else:
                name = f"[DELIMITER {token.timestep + 2}]"
        line.append(name)
    if line:
        lines.append(" ".join(line))
    (output / "sequence.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "tokens.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["position", "token_id", "name", "timestep", "owner", "role", "scored"])
        writer.writeheader()
        for position, (token, scored) in enumerate(zip(window.tokens, window.loss_mask)):
            writer.writerow({"position": position, **asdict(token), "scored": scored})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def preview_replay(config_path: Path, *, replay_override: Path | None = None,
                   perspective_override: str | None = None) -> Path:
    """Inspect one source replay; only exports under scripts/output are writable."""
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
        replay = replay_override.resolve()
    elif settings.replay_path:
        replay = (base / settings.replay_path).resolve()
    else:
        replay = next(iter(sorted((base / settings.input_directory).glob("*_game_state.parquet"))), None)
        if replay is None:
            raise ValueError("no game-state replay found; provide --replay")
    outcomes = {player: resolve_replay_outcome(replay, player) for player in ("p1", "p2")}
    frame = pd.read_parquet(replay).sort_values("game_loop", kind="stable").reset_index(drop=True)
    rows = (row for _, row in frame.iterrows())
    timesteps = iter_joint_timesteps(rows, project, content, vocabulary,
                                    perspective=perspective)
    window_args = dict(context_budget=settings.context_budget_tokens, perspective=perspective, outcomes=outcomes)
    comparison = None
    bpe = None
    if settings.tokenizer == "bpe":
        # Bounded to one replay for debugging, not a corpus preprocessing path.
        timesteps = list(timesteps)
        baseline = next(iter_joint_windows(timesteps, vocabulary, **window_args))
        bpe = fit_content_bpe(timesteps, vocabulary,
                            vocabulary.content_ids,
                            min_occurrences=settings.bpe_min_occurrences)
        encoded = [bpe.encode(step) for step in timesteps]
        if any(bpe.decode(after) != before for before, after in zip(timesteps, encoded)):
            raise ValueError("BPE round-trip failed")
        vocabulary = bpe.vocabulary
        window = next(iter_joint_windows(encoded, vocabulary, **window_args))
        same_span = replace(baseline, tokens=bpe.encode(baseline.tokens))
        comparison = {
            "fit_scope": "entire selected replay, one perspective, both owners; no other replay",
            "evaluation_scope": "in-sample debugging only, not held-out compression evidence",
            "min_occurrences": settings.bpe_min_occurrences,
            "merge_count": len(bpe.merges),
            "vocabulary_before": bpe.base.vocab_size, "vocabulary_after": vocabulary.vocab_size,
            "round_trip_exact": True,
            "full_replay": {"tokens_before": sum(map(len, timesteps)) + 7,
                            "tokens_after": sum(map(len, encoded)) + 7,
                            "accounting": "all timesteps plus one BOS/delimiter, one outcome footer, one END; unwindowed"},
            "original_first_window": {"tokens_before": len(baseline.tokens),
                                      "tokens_after": len(same_span.tokens),
                                      "timesteps": baseline.end_timestep},
            "rewindowed_first_window": {"tokens": len(window.tokens),
                                        "timesteps": window.end_timestep,
                                        "atomic_equivalent_tokens": len(bpe.decode(window.tokens))},
        }
        output.mkdir(parents=True, exist_ok=True)
        artifact = bpe.artifact()
        (output / "bpe_vocabulary.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        write_sequence_preview(baseline, output / "before", comparison["original_first_window"])
        write_sequence_preview(same_span, output / "same_window_bpe", comparison["original_first_window"], bpe=bpe)
    else:
        window = next(iter_joint_windows(timesteps, vocabulary, **window_args))
    summary = {
        "status": "preview_only_not_training_integrated",
        "tokenizer": settings.tokenizer,
        "preprocessing_location": "cloud for full tokenization, windowing and manifest building; local single-replay debug only",
        "replay": replay.name, "perspective": perspective,
        "sequence_format": settings.sequence_format,
        "grammar_revision": "terminal_outcome_footer_v1",
        "vocabulary_identity": vocabulary.identity, "vocabulary_size": vocabulary.vocab_size,
        "ownership_ids": {name: vocabulary.token_id(name) for name in ("[SELF]", "[ENEMY]")},
        "context_budget": window.context_budget, "emitted_tokens": len(window.tokens),
        "unused_capacity": window.context_budget - len(window.tokens),
        "start_timestep": window.start_timestep, "end_timestep_exclusive": window.end_timestep,
        "replay_timesteps": len(frame), "reached_replay_end": window.reached_replay_end,
        "next_timestep_tokens_including_end_if_terminal": window.next_timestep_tokens,
        "content_tokens": sum(t.role == "content" for t in window.tokens),
        "outcome_tokens": sum(t.role == "outcome" for t in window.tokens),
        "input_tokens": 0, "fog_omission": False, "feature_conditioning": False,
        "padding": "unpadded inspection export; future adapters must distinguish semantic from batch padding",
        "diffusion_contract": {
            "clamped_positions": [0], "replacement_ids": list(range(vocabulary.vocab_size)),
            "note": "full vocabulary including BOS/MASK/EOS; only position 0 is protected; not yet wired to training/sampling",
        },
        "outcome_placement": "one four-token SELF/outcome/ENEMY/outcome footer after the final timestep delimiter; END follows only at replay end",
        "outcome_scoring": "all four footer positions scored; no inference-time complement constraint",
        "outcome_caveat": "The second outcome can use the first under causal teacher forcing; useful for learning the pairing, but not independent evidence of outcome forecasting.",
    }
    if comparison is not None:
        summary["bpe_identity"] = artifact["identity"]
        summary["comparison"] = comparison
    write_sequence_preview(window, output, summary, bpe=bpe)
    return output
