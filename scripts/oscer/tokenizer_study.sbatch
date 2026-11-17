#!/bin/bash
#SBATCH --job-name=thesis-tokenizer
#SBATCH --partition=normal
#SBATCH --container=el9hw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -Eeuo pipefail
umask 077
root="${SLURM_SUBMIT_DIR:?Submit from the uploaded project root}"
cd "$root"
job_id="${SLURM_JOB_ID:?Run through sbatch}"
durable="${TOKENIZER_DURABLE_ROOT:-$HOME/thesis-tokenizer-results}/$(basename "$root")/$job_id"
mkdir -p "$durable" "$root/scripts/output/tokenizer_study" "$root/.tmp" "$root/.cache"
export TMPDIR="$root/.tmp"
export XDG_CACHE_HOME="$root/.cache"
export MPLCONFIGDIR="$root/.cache/matplotlib"
export PIP_CACHE_DIR="$root/.cache/pip"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$root/packages/thesis-shared/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1
# SLURM stdout/stderr paths are selected in the submission command, before this starts.
# Keep an independently flushed live transcript in home, including setup failures.
exec > >(tee -a "$durable/console.log") 2> >(tee -a "$durable/error.log" >&2)

finalize() {
    code=$?
    trap - EXIT
    set +e
    if [[ -d "$root/scripts/output/tokenizer_study/results" ]]; then
        cp -a "$root/scripts/output/tokenizer_study/results" "$durable/"
        copy_code=$?
        if (( copy_code != 0 )); then
            echo 'ERROR: results backup failed; retrieve results directly from scratch.' >&2
            if (( code == 0 )); then code=$copy_code; fi
        fi
    fi
    cp "$root/config/tokenizer_study.yaml" "$durable/"
    cp "$root/scripts/oscer/requirements-tokenizer.txt" "$durable/"
    if [[ -f "$root/upload_manifest.json" ]]; then cp "$root/upload_manifest.json" "$durable/"; fi
    printf 'exit_code=%s\njob_id=%s\nfinished_utc=%s\n' "$code" "$job_id" "$(date -u +%FT%TZ)" > "$durable/job_status.txt"
    echo "Job exit=$code; durable results and diagnostics: $durable"
    exit "$code"
}
trap finalize EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

module purge
module load "${TOKENIZER_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
echo "Started $(date -u +%FT%TZ) on $(hostname); job $job_id"
echo "Scratch project: $root"
if [[ ! -x .venv/bin/python ]]; then
    # Bootstrap only: no project venv exists yet. All subsequent Python uses its shim.
    python -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), "This pinned study environment requires Python 3.12"'
# A supplied wheelhouse avoids compute-node network access; otherwise pip uses PyPI.
if [[ -d wheels ]]; then
    .venv/bin/python -m pip install --no-index --find-links wheels --no-cache-dir -r scripts/oscer/requirements-tokenizer.txt
else
    .venv/bin/python -m pip install --no-cache-dir -r scripts/oscer/requirements-tokenizer.txt
fi
.venv/bin/python -m pip freeze > "$durable/environment.txt"
# No editable install needed: PYTHONPATH selects the uploaded shared source only.
.venv/bin/python scripts/oscer/verify_upload.py
.venv/bin/python scripts/tokenizer_study.py --config config/tokenizer_study.yaml --input-dir "$root/parquet"
