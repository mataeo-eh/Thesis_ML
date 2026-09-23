"""Export the first joint pretraining window of one replay without preprocessing."""

import argparse
import json
from pathlib import Path

from thesis_shared.sequence_formats.preview import preview_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/sequence_preview.yaml")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--perspective", choices=("p1", "p2"))
    args = parser.parse_args()
    output = preview_replay(args.config, replay_override=args.replay, perspective_override=args.perspective)
    print(f"UNCONDITIONED JOINT PREVIEW (no fog, no features, no manifest): {output.name}")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print(f"{summary['replay']} / {summary['perspective']}: {summary['emitted_tokens']:,} tokens, "
          f"{summary['end_timestep_exclusive']} whole timesteps, {summary['unused_capacity']} slots unused")
    print("Wrote sequence.txt, tokens.csv, summary.json")
    if "comparison" in summary:
        comparison = summary["comparison"]
        for label in ("full_replay", "original_first_window"):
            counts = comparison[label]
            print(f"{label}: {counts['tokens_before']:,} -> {counts['tokens_after']:,} tokens")
        print(f"Vocabulary: {comparison['vocabulary_before']:,} -> {comparison['vocabulary_after']:,}; "
              f"{comparison['merge_count']:,} merges; minimum occurrences={comparison['min_occurrences']}")
        print("Wrote comparison.json, bpe_vocabulary.json, before/, same_window_bpe/; exact round-trip verified")


if __name__ == "__main__":
    main()
