"""Full-size GPU smoke test for the SC2 masked-diffusion training pipeline.

Purpose: BEFORE committing to a long (and expensive) cloud training run, launch
this on the target GPU instance to confirm the real-size model actually does a
forward + backward + optimizer step, and to measure peak VRAM and per-step
throughput at the configured sequence lengths. It does NOT need the dataset: it
fabricates a random batch shaped exactly like real training batches, so it
isolates the model/optimizer memory cost from data loading.

Run on a cloud GPU (after `uv sync`):
    uv run python scripts/gpu_smoke_test.py --batch-size 1 --steps 5
    uv run python scripts/gpu_smoke_test.py --batch-size 2 --input-len 2048

It exercises the genuine training path (corruption, self-conditioning two-pass,
EMA, confidence loss) via TrainingLoop.fit, so the memory and timing numbers
reflect what a real run will use. On CPU it still runs as a shape/sanity check,
but the memory figures are only meaningful on CUDA.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import tempfile
import time

import torch

from thesis_ml.config import ProjectConfig, load_config
from thesis_ml.data.collate import DiffusionBatch
from thesis_ml.model.embedding import STAT_KEYS, InputFeatures
from thesis_ml.model.model import SC2StrategyDiffusionModel
from thesis_ml.train.loop import TrainingLoop
from thesis_ml.vocab.special_tokens import CONTENT_TOKEN_OFFSET


def build_synthetic_batch(
    config: ProjectConfig,
    *,
    batch_size: int,
    input_len: int,
    vocab_size: int,
    device: str,
) -> DiffusionBatch:
    """Fabricate a random batch with the exact shapes of a real training batch.

    The token ids and contextual features are random but correctly shaped, so
    the forward/backward pass touches every parameter and allocates the same
    activations a real batch would. No dataset is required.

    Parameters:
        config: project config (drives canvas length).
        batch_size: micro-batch size to test (memory scales with this).
        input_len: number of input-region tokens (set near your real length).
        vocab_size: content vocabulary size for random token sampling.
        device: 'cuda' or 'cpu'.
    Returns:
        A DiffusionBatch ready to pass to TrainingLoop.fit.
    """

    canvas_len = config.data.canvas_budget_tokens
    generator = torch.Generator().manual_seed(0)

    input_token_ids = torch.randint(
        CONTENT_TOKEN_OFFSET, vocab_size, (batch_size, input_len), generator=generator
    )
    target_canvas = torch.randint(
        CONTENT_TOKEN_OFFSET, vocab_size, (batch_size, canvas_len), generator=generator
    )
    # Six canvas token classes (see dataset.CLASS_*); random labels are fine for
    # a memory/throughput probe.
    class_labels = torch.randint(0, 6, (batch_size, canvas_len), generator=generator)

    features = InputFeatures(
        map_values=torch.zeros(batch_size, input_len, 2),
        stat_values=torch.zeros(batch_size, input_len, len(STAT_KEYS)),
        team_ids=torch.ones(batch_size, input_len, dtype=torch.long),
    )

    return DiffusionBatch(
        input_token_ids=input_token_ids,
        input_attention_mask=torch.ones(batch_size, input_len, dtype=torch.bool),
        input_lengths=torch.full((batch_size,), input_len, dtype=torch.long),
        target_canvas=target_canvas,
        canvas_attention_mask=torch.ones(batch_size, canvas_len, dtype=torch.bool),
        class_labels=class_labels,
        canvas_loss_mask=torch.ones(batch_size, canvas_len, dtype=torch.bool),
        terminated=torch.zeros(batch_size, dtype=torch.bool),
        truncated=torch.ones(batch_size, dtype=torch.bool),
        input_records=[[] for _ in range(batch_size)],
        canvas_metadata=[[] for _ in range(batch_size)],
        input_features=features,
    )


def run_smoke(args: argparse.Namespace) -> None:
    """Build the real-size model, run a few train steps, report VRAM + timing."""

    config = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available on this machine")

    # Keep checkpoints/metrics out of the way; this is a throwaway probe.
    with tempfile.TemporaryDirectory() as scratch:
        config = replace(
            config,
            train=replace(
                config.train,
                accumulation_steps=1,
                target_effective_batch_tokens=0,  # no auto-accumulation: test one micro-batch
                warmup=1,
                val_interval=0,
                checkpoint_interval=0,  # don't checkpoint during the probe
                checkpoint_dir=str(Path(scratch) / "ckpt"),
                max_steps=args.steps,
            ),
        )

        model = SC2StrategyDiffusionModel(config, vocab_size=args.vocab_size)
        param_count = sum(parameter.numel() for parameter in model.parameters())

        batch = build_synthetic_batch(
            config,
            batch_size=args.batch_size,
            input_len=args.input_len,
            vocab_size=args.vocab_size,
            device=device,
        )

        loop = TrainingLoop(model=model, config=config, device=device, seed=0)

        print("=" * 64)
        print("SC2 diffusion GPU smoke test")
        print("=" * 64)
        print(f"device              : {device}")
        if device == "cuda":
            print(f"gpu                 : {torch.cuda.get_device_name(0)}")
        print(f"parameters          : {param_count / 1e6:.1f}M")
        print(f"d_model/layers/heads/ffn : {config.model.d_model}/{config.model.layers}"
              f"/{config.model.heads}/{config.model.ffn}")
        print(f"batch_size          : {args.batch_size}")
        print(f"input_len           : {args.input_len}")
        print(f"canvas_len          : {config.data.canvas_budget_tokens}")
        print(f"precision           : {config.train.precision}")
        print(f"steps               : {args.steps}")
        print("-" * 64)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        loop.fit([batch], max_steps=args.steps, fixed_t=None)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        per_step = elapsed / max(1, args.steps)
        print(f"total time          : {elapsed:.2f}s")
        print(f"time / step         : {per_step * 1000:.0f} ms")
        if device == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"peak allocated VRAM : {peak_alloc:.2f} GiB")
            print(f"peak reserved VRAM  : {peak_reserved:.2f} GiB")
            print(f"device total VRAM   : {total:.1f} GiB")
            print(f"headroom            : {total - peak_reserved:.2f} GiB")
        print("=" * 64)
        print("OK: forward + backward + optimizer step completed at full size.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-size GPU smoke test for cloud training")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--steps", type=int, default=5, help="number of train steps to run")
    parser.add_argument("--batch-size", type=int, default=1, help="micro-batch size to probe")
    parser.add_argument(
        "--input-len",
        type=int,
        default=2048,
        help="input-region token length (set near your real per-example length)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=400,
        help="content vocabulary size (use your real vocab size for accuracy)",
    )
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (default: auto)")
    run_smoke(parser.parse_args())


if __name__ == "__main__":
    main()
