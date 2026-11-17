"""Plot tokens deleted per BPE merge for one or more replays, to look for an elbow.

Role in the larger system:
    A local, small-sample diagnostic that sits beside ``preview_sequence.py``.
    It fits the shared content BPE (``thesis_shared.sequence_formats.bpe``) on the
    same replay the BPE preview uses (optionally pooled with more), but with the lowest legal merge threshold
    (``min_occurrences=2``), so merging runs until no repeated adjacent pair is
    left inside any hard-boundary span. It then replays the ranked merges one at
    a time and records exactly how many tokens each merge removed.

    The shape of that curve suggests where a vocabulary-size cut could sit: the
    merge rank where deletions per merge collapse is a candidate elbow. It is
    in-sample evidence from a handful of replays, not a corpus decision.

    ``--replay-count N`` pools N replays (the anchor plus the next N-1 in sorted
    filename order) into one fit, written to ``merge_curve_<N>_replays/``.

Outputs (under ``scripts/output/sequence_preview/bpe/merge_curve/`` for one replay):
    tokens_deleted_per_merge.png  three-panel plot (linear, log-log, cumulative)
    merge_curve.csv               one row per merge
    merge_curve_summary.json      totals, elbow hint, coverage milestones
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt
import pandas as pd

from thesis_shared.config import load_config, load_sequence_preview_config
from thesis_shared.sequence_formats.bpe import _merge, fit_content_bpe
from thesis_shared.sequence_formats.unconditioned_joint import iter_joint_timesteps
from thesis_shared.vocab.content_vocab import load_content_vocabulary
from thesis_shared.vocab.sequence_vocabulary import build_sequence_vocabulary

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/sequence_preview_bpe.yaml"
DEFAULT_OUTPUT = ROOT / "scripts/output/sequence_preview/bpe/merge_curve"
LINE_COLOR = "#2a6fdb"  # the single series color; text stays in neutral ink
ELBOW_COLOR = "#555555"


def load_timesteps(config_path: Path, replay_override: Path | None, perspective_override: str | None,
                   replay_count: int = 1):
    """Build the atomic joint timesteps for one or more replays, exactly as the BPE preview does.

    Args:
        config_path: sequence-preview YAML (defaults to the BPE preview profile).
        replay_override: optional replay parquet replacing the config's replay_path.
        perspective_override: optional "p1"/"p2" replacing the config's perspective.
        replay_count: how many replays to pool. The anchor replay (override or config)
            comes first; the rest are the next game-state parquets in sorted filename
            order from the same directory, so the selection is deterministic.

    Returns:
        (timesteps, vocabulary, replays, perspective) where timesteps is a list of
        per-timestep SequenceToken tuples pooled across all replays, and replays is
        the list of replay paths used.

    Calls: load_sequence_preview_config, load_config, load_content_vocabulary,
        build_sequence_vocabulary, iter_joint_timesteps (all from thesis_shared).
    """
    settings = load_sequence_preview_config(config_path)
    base = config_path.resolve().parent
    project = load_config(base / settings.project_config)
    content = load_content_vocabulary(ROOT / project.pipeline.token_dictionary_uri)
    vocabulary = build_sequence_vocabulary(content, sequence_format=settings.sequence_format)
    if replay_override is not None:
        anchor = replay_override.resolve()
    else:
        anchor = (base / settings.replay_path).resolve()
    # Anchor first, then the following files in sorted order (skipping the anchor itself).
    candidates = sorted(anchor.parent.glob("*_game_state.parquet"))
    others = [path for path in candidates if path.name > anchor.name]
    replays = [anchor] + others[:replay_count - 1]
    if len(replays) < replay_count:
        raise ValueError(f"only {len(replays)} replays available after {anchor.name}")
    perspective = perspective_override or settings.perspective

    timesteps = []
    for replay in replays:
        frame = pd.read_parquet(replay).sort_values("game_loop", kind="stable").reset_index(drop=True)
        rows = (row for _, row in frame.iterrows())
        timesteps.extend(iter_joint_timesteps(rows, project, content, vocabulary, perspective=perspective))
    return timesteps, vocabulary, replays, perspective


def exact_deletions(timesteps, merges) -> list[int]:
    """Replay ranked merges and count the tokens each one actually removes.

    The fitter records each pair's *overlapping* adjacent count, which overstates
    removals for runs like A A A (2 overlapping pairs, only 1 replacement). This
    measures the true sequence-length drop instead.

    Segments are keyed by (owner, timestep) because ContentBPE.encode never merges
    across those; merges only contain content IDs, so structural tokens inside a
    segment can never be absorbed. Identical segments are deduplicated with a
    multiplicity so each merge is applied once per distinct segment.

    Args:
        timesteps: list of per-timestep SequenceToken tuples (atomic).
        merges: ContentBPE.merges, i.e. (left, right, new_id, frequency) in rank order.

    Returns:
        List of tokens removed by each merge, in rank order.

    Calls: thesis_shared.sequence_formats.bpe._merge (the fitter's own replacement rule).
    """
    segments: dict[tuple[int, ...], int] = {}
    for step in timesteps:
        current: list[int] = []
        previous_key = None
        for token in step:
            key = (token.owner, token.timestep)
            if previous_key is not None and key != previous_key:
                segments[tuple(current)] = segments.get(tuple(current), 0) + 1
                current = []
            current.append(token.token_id)
            previous_key = key
        if current:
            segments[tuple(current)] = segments.get(tuple(current), 0) + 1

    deletions = []
    for left, right, new_id, _ in merges:
        removed = 0
        updated: dict[tuple[int, ...], int] = {}
        for segment, multiplicity in segments.items():
            merged = _merge(segment, (left, right), new_id)
            removed += (len(segment) - len(merged)) * multiplicity
            updated[merged] = updated.get(merged, 0) + multiplicity
        segments = updated
        deletions.append(removed)
    return deletions


def elbow_rank(deletions: list[int]) -> int:
    """Kneedle-style elbow hint on the cumulative-removal curve.

    Normalizes merge rank and cumulative removals to [0, 1] and returns the rank
    (1-based) farthest above the straight chord from first to last merge. It is a
    reading aid only; the plot is the evidence.
    """
    total = sum(deletions)
    count = len(deletions)
    best_rank, best_gap, running = 1, -1.0, 0
    for index, removed in enumerate(deletions):
        running += removed
        x = index / (count - 1) if count > 1 else 0.0
        gap = running / total - x  # the chord from (0,0) to (1,1) is y = x
        if gap > best_gap:
            best_rank, best_gap = index + 1, gap
    return best_rank


def coverage_milestones(deletions: list[int], fractions=(0.5, 0.8, 0.9, 0.95, 0.99)) -> dict:
    """Merges needed to achieve each fraction of the total achievable token removal."""
    total = sum(deletions)
    milestones, running, index = {}, 0, 0
    for fraction in fractions:
        while running < fraction * total:
            running += deletions[index]
            index += 1
        milestones[f"{int(fraction * 100)}%"] = index
    return milestones


def plot_curve(deletions: list[int], atomic_tokens: int, elbow: int, milestones: dict,
               title_suffix: str, destination: Path) -> None:
    """Draw three stacked panels sharing merge rank on x.

    Panels: deletions per merge (linear y), the same on log-log axes (the tail
    and any plateau are only visible there), and cumulative share of atomic
    tokens removed. Separate panels instead of a dual axis, one series each.
    """
    ranks = list(range(1, len(deletions) + 1))
    cumulative, running = [], 0
    for removed in deletions:
        running += removed
        cumulative.append(100 * running / atomic_tokens)

    figure, (linear, loglog, share) = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    figure.suptitle(f"Tokens deleted per BPE merge — {title_suffix}", fontsize=13, color="#222222")

    for axis in (linear, loglog):
        axis.plot(ranks, deletions, color=LINE_COLOR, linewidth=1.5)
        axis.set_ylabel("tokens deleted by this merge")
    linear.set_title("Linear scale", loc="left", fontsize=10, color="#444444")
    loglog.set_xscale("log")
    loglog.set_yscale("log")
    loglog.set_title("Log–log scale", loc="left", fontsize=10, color="#444444")

    share.plot(ranks, cumulative, color=LINE_COLOR, linewidth=2)
    share.set_ylabel("% of atomic tokens removed")
    share.set_ylim(0, 100)
    share.set_title("Cumulative compression", loc="left", fontsize=10, color="#444444")
    for label, rank in milestones.items():
        if label in ("90%", "99%"):
            share.annotate(f"{label} of total removal @ merge {rank:,}",
                           xy=(rank, cumulative[rank - 1]), xytext=(8, -14), textcoords="offset points",
                           fontsize=8, color="#444444")
            share.plot([rank], [cumulative[rank - 1]], "o", color=LINE_COLOR, markersize=5)

    for axis in (linear, loglog, share):
        axis.axvline(elbow, color=ELBOW_COLOR, linestyle="--", linewidth=1)
        axis.grid(True, color="#e6e6e6", linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.set_xlabel("merge rank")
    linear.annotate(f"elbow hint: merge {elbow:,} ({deletions[elbow - 1]:,} deleted)",
                    xy=(elbow, deletions[elbow - 1]), xytext=(10, 20), textcoords="offset points",
                    fontsize=8, color="#444444")
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def main() -> None:
    """Fit exhaustive BPE on the selected replay(s), measure per-merge deletions, write plot + data."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--perspective", choices=("p1", "p2"))
    parser.add_argument("--replay-count", type=int, default=1,
                        help="pool this many replays (anchor + next in sorted order)")
    parser.add_argument("--output", type=Path,
                        help="default: merge_curve/ for one replay, merge_curve_<N>_replays/ otherwise")
    args = parser.parse_args()
    if args.replay_count < 1:
        raise ValueError("--replay-count must be at least 1")

    # Separate default folders so a multi-replay run never overwrites the single-replay plot.
    if args.output is not None:
        output = args.output.resolve()
    elif args.replay_count == 1:
        output = DEFAULT_OUTPUT
    else:
        output = DEFAULT_OUTPUT.with_name(f"merge_curve_{args.replay_count}_replays")
    if not output.is_relative_to(ROOT / "scripts" / "output"):
        raise ValueError("output must stay below scripts/output")
    output.mkdir(parents=True, exist_ok=True)

    timesteps, vocabulary, replays, perspective = load_timesteps(
        args.config, args.replay, args.perspective, args.replay_count)
    if len(replays) == 1:
        replay_label = replays[0].name
    else:
        # Short range label ("match_A to match_B") so the plot title fits the figure width.
        first = replays[0].stem.removesuffix("_game_state")
        last = replays[-1].stem.removesuffix("_game_state")
        replay_label = f"{len(replays)} replays ({first} to {last})"
    atomic_tokens = sum(len(step) for step in timesteps)
    # min_occurrences=2 is the fitter's floor: merge every pair that repeats anywhere.
    bpe = fit_content_bpe(timesteps, vocabulary, vocabulary.content_ids, min_occurrences=2)
    deletions = exact_deletions(timesteps, bpe.merges)

    # Cross-check against the real encoder so the curve cannot drift from ContentBPE.
    encoded_tokens = sum(len(bpe.encode(step)) for step in timesteps)
    if atomic_tokens - sum(deletions) != encoded_tokens:
        raise ValueError(f"deletion replay disagrees with encoder: "
                         f"{atomic_tokens - sum(deletions)} vs {encoded_tokens}")

    elbow = elbow_rank(deletions)
    milestones = coverage_milestones(deletions)
    with (output / "merge_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "left_id", "right_id", "new_id", "pair_frequency_overlapping",
                         "tokens_deleted", "tokens_remaining", "atomic_expansion"])
        remaining = atomic_tokens
        expansions = bpe.expansions
        for rank, ((left, right, new_id, frequency), removed) in enumerate(zip(bpe.merges, deletions), 1):
            remaining -= removed
            names = " ".join(vocabulary.names[i] for i in expansions[new_id])
            writer.writerow([rank, left, right, new_id, frequency, removed, remaining, names])

    summary = {
        "scope": f"{len(replays)} replay(s) pooled, one perspective, both owners; in-sample fit, not corpus evidence",
        "replays": [path.name for path in replays], "perspective": perspective,
        "min_occurrences": 2, "merge_count": len(bpe.merges),
        "vocabulary_before": vocabulary.vocab_size, "vocabulary_after": bpe.vocabulary.vocab_size,
        "timestep_tokens_before": atomic_tokens, "timestep_tokens_after": encoded_tokens,
        "accounting": "per-timestep tokens only; excludes BOS, outcome footer, and END",
        "elbow_hint_rank": elbow, "elbow_hint_method": "max gap above chord of normalized cumulative removal",
        "merges_for_fraction_of_total_removal": milestones,
        "first_merges_deleted": deletions[:10],
    }
    (output / "merge_curve_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_curve(deletions, atomic_tokens, elbow, milestones, f"{replay_label} / {perspective}",
               output / "tokens_deleted_per_merge.png")

    print(f"{replay_label} / {perspective}: {atomic_tokens:,} -> {encoded_tokens:,} timestep tokens "
          f"over {len(bpe.merges):,} merges (min_occurrences=2)")
    print(f"Elbow hint: merge {elbow:,}; merges for share of total removal: {milestones}")
    print(f"Wrote {output.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
