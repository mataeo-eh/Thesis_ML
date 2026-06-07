# experiments/

Reproducible experiments. Each experiment should be fully specified by a config
in `configs/` plus a script/entry point here, so results can be regenerated.

Run outputs (checkpoints, logs, metrics) belong in `experiments/runs/`, which is
git-ignored — track the *configs* that produced them, not the artifacts.
