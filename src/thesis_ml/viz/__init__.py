"""Read-only diagnostic visualisation for trained checkpoints.

This subpackage renders static figures from intermediates the evaluation
pipeline already computes (predicted vs ground-truth per-timestep entity counts
and build-order first-appearance events). Optional exports preserve raw canvases
and final-canvas top-k logits for manual inspection. It never re-implements
tokenization, sampling, decode, or oracle extraction -- it imports and calls the
existing ``thesis_ml.eval`` / ``thesis_ml.inference`` interfaces.
"""
