# OSCER vocabulary study

Run these steps yourself. Agents prepare and validate files locally; they do not
connect to OSCER. Default login: `mataeo@sooner.oscer.ou.edu`, SSH shortcut `oscer`.

The study fits lossless content-ID BPE, treating each player's sorted content at
one timestep as a word. It includes units, structures and upgrades, exactly as
the current joint preview does. It never merges owners, timesteps, replay
boundaries, race headers or outcomes. It never changes existing training.
Read [the research/design](../../research/tokenizer-vocabulary-selection.md).

This is an NLP tokenizer experiment for a future **generic AR LLM trained from
scratch**, with the three races as languages. Following the multilingual
efficiency reasoning documented by DeepSeek/Qwen/Gemma, the default chooses
the smallest shared vocabulary retaining 99% of observed dev token savings
overall **and for each race separately**, among candidates from 512 to 65,536
plus the atomic control. This tolerance is our explicit heuristic, not a number
prescribed by those teams. The report also includes a 32,768-ID comparator,
pooled compression recommendation, embedding-size accounting and a dense-head
work proxy; real model quality/throughput remains to be measured. All three
languages must be represented in train and dev. Requested sizes can exhaust
early if no pair repeats. Random resolves to its actual race, not a fourth
language. `selection_rule: reference` explicitly selects the 32k comparator;
`dev_savings` uses only the pooled compression recommendation.

Account probe supplied by the user on 2026-11-17 confirmed writable
`/scratch/mataeo`, `Python/3.12.3-GCCcore-13.3.0`, and an available `normal`
partition with a 48-hour maximum. The reported filesystem free space is shared
capacity, not the user's quota; the quota query did not establish a home quota.
To refresh this read-only check from local PowerShell (the `tr` removes Windows
pipe line endings):

```powershell
Get-Content -Raw scripts/oscer/preflight.sh | ssh oscer 'tr -d "\r" | bash -l -s'
```

## 1. Upload — local PowerShell

Working directory: the local `Thesis_ML` checkout. The dataset source is its
parent's `data/quickstart/{parquet,json}`. All matching game-state parquets and
their metadata are included; the current local directory contains 943 replays.
The uploader prints timestamped stages, replay count, local checksum progress,
and archive size. While packing, it prints an update every three seconds with
elapsed time, current archive size and the latest archive entry. The hash and
packing stages run locally and can take several minutes for the full dataset.

Normal runs check SSH login and prepare the upload directory **before hashing or packing**. Expect password/MFA
prompts at that check; password typing is hidden. Keys or an SSH agent can
authenticate without a password. The later SCP upload and SSH extraction are
separate connections and may each prompt again; answer each prompt when it
appears. The script does not store/reuse your password. There is no extra SSH
connection between packing and SCP just to create the directory.
SCP runs directly in the terminal so its transfer progress remains visible;
remote extraction prints filenames. `-ValidateOnly` skips every SSH connection
and therefore never asks for an OSCER password. Ctrl+C stops an active local
tar process before the temporary directory is cleaned up; closing/killing the
PowerShell process can prevent its cleanup handler from running.

```powershell
Set-Location '<path-to-your-Thesis_ML-checkout>'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/oscer/upload_tokenizer_study.ps1 -RunName vocab-study-01
ssh oscer
```

This creates a tar in the system temporary directory, uploads it, extracts it
under `/scratch/mataeo/thesis-tokenizer/vocab-study-01`, removes the remote tar
after successful extraction, and removes the local temporary directory in
`finally`, on success or failure. Parquet is already compressed, so the tar is
uncompressed to avoid redundant compression and CPU time. It includes only
allowlisted source/config/dictionary files, selected parquets and metadata;
never environments, credentials, Git state, checkpoints or other output.
Hashes and exact replay membership are checked on the compute node before use.

Use `-DataRoot 'D:\other\quickstart'` for a different dataset root,
`-RemoteRoot /scratch/mataeo/other-path` if needed. Choose a new `-RunName` for a
fresh upload. An existing **empty** remote directory can be reused after an
interruption during local preparation; a directory containing files is refused.
SSH connection/authentication failures are reported separately from confirmed
directory conflicts. For a local
archive check, add `-ValidateOnly -ReplayLimit 1` (it still deletes the tar).
`-ReplayLimit` is debugging only; omit it for collective-dataset metrics.

## 2. Submit — remote OSCER shell

Required working directory: the uploaded project root on scratch. Python 3.12.3
was confirmed by the user's account probe. Inspect modules again if needed,
without running analysis on a login node:

```bash
module avail Python
cd /scratch/mataeo/thesis-tokenizer/vocab-study-01
export TOKENIZER_PYTHON_MODULE=Python/3.12.3-GCCcore-13.3.0
mkdir -p "$HOME/thesis-tokenizer-results/vocab-study-01/slurm"
job=$(sbatch --parsable --export=ALL \
  --output="$HOME/thesis-tokenizer-results/vocab-study-01/slurm/%j.stdout.log" \
  --error="$HOME/thesis-tokenizer-results/vocab-study-01/slurm/%j.stderr.log" \
  scripts/oscer/tokenizer_study.sbatch)
echo "$job"
```

If that exact module is absent, select an available **Python 3.12** module and
change the export before submitting. The job validates the interpreter. It
creates/checks `.venv/bin/python` on scratch, installs a pinned analysis-only
environment there, and uses that shim for every project Python command. The
only system-Python invocation bootstraps a venv when none exists. No Torch,
TensorFlow, CUDA or GPU is needed. Dependencies normally download from PyPI
inside the job; network failure is captured in the home-directory logs. An
offline alternative is described below.

The initial request is 4 CPU cores, 32 GiB RAM and 24 hours on `normal` in
`el9hw`, following OSCER's current job examples and the supplied 48-hour limit. These are starting resource
requests, **not measured runtime or capacity claims**. Parsing uses four
workers; fitting is serial and updates a disk pair index. Set `workers` in
`config/tokenizer_study.yaml` no higher than the CPU request. You can override
resources at submission, e.g. `sbatch --time=48:00:00 --mem=64G ...`, subject to
your partition limits (`scontrol show partition normal`). If your allocation
requires an account, add its real `--account=...`; no account name is guessed.

## 3. Status, completion and retry — remote OSCER shell

`JOB_ID` means the number printed by `sbatch` above. Set it explicitly after a
new login; do not paste the literal placeholder.

```bash
job=JOB_ID
squeue -j "$job"
sacct -j "$job" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
tail -n 60 "$HOME/thesis-tokenizer-results/vocab-study-01/slurm/$job.stdout.log"
tail -n 60 "$HOME/thesis-tokenizer-results/vocab-study-01/slurm/$job.stderr.log"
cat "$HOME/thesis-tokenizer-results/vocab-study-01/$job/job_status.txt"
find "$HOME/thesis-tokenizer-results/vocab-study-01/$job/results" -name REPORT.md -print
```

Completion means SLURM `COMPLETED` / exit `0:0`, `job_status.txt` exit code 0,
and a study `summary.json` and `status.json` marked `complete`. Disappearing
from `squeue` alone does not establish success. A hard kill, OOM, or node failure
can prevent the exit handler from writing `job_status.txt` or copying results;
use `sacct`, durable Slurm logs, and the scratch fallback retrieval below.

To retry, run the same `cd`, exports, and `sbatch` command from step 2. Completed
per-replay parse caches, each committed BPE merge and completed candidate
measurements are reused when their corpus/source/config identities match.
An interrupted merge rolls back. Initial pair-index construction is atomic and
repeats if interrupted. Never run two study jobs simultaneously in the same
uploaded directory. A changed corpus or settings gets a new study identity;
source changes invalidate parse caches as well. To change a staged config after
upload, upload a fresh run through the helper so its checksum manifest matches.

## 4. Download success OR failure — local PowerShell

Run this regardless of job outcome; it retrieves setup errors, stdout/stderr,
the live transcript, exit status when available, environment versions, upload
manifest, and all finished compact reports/vocabularies. It excludes the large
scratch pair database, replay cache, raw data, and environment.

```powershell
Set-Location '<path-to-your-Thesis_ML-checkout>'
New-Item -ItemType Directory -Force scripts/output/oscer-downloads | Out-Null
scp -r 'oscer:thesis-tokenizer-results/vocab-study-01' scripts/output/oscer-downloads/
Get-ChildItem scripts/output/oscer-downloads/vocab-study-01 -Recurse -Filter REPORT.md
```

If the job died before copying results to home, additionally retrieve any
finished results directly from scratch (a missing directory simply means none
were written):

```powershell
scp -r 'oscer:/scratch/mataeo/thesis-tokenizer/vocab-study-01/scripts/output/tokenizer_study/results' scripts/output/oscer-downloads/vocab-study-01/scratch-results
```

Start with `REPORT.md`: total/mean/min/max tokens before and after for the full
corpus and each split. `summary.json` adds p50/p90/p95/p99 lengths, compression
factor, reduction percentage, empty/nonempty words, tokens per word, content
versus structural totals, distinct used IDs, rare/unused compounds and exact
round-trip status. `per_replay.csv` has individual counts. `candidates.csv` and
`compression_curve.png` show the train/dev sweep; candidate JSON/CSV and the
selected summary include Protoss/Terran/Zerg content counts and tokens per word.
`merge_curve.csv` separates
overlapping pair counts from actual tokens removed. `tokenizer.json` contains
the selected frozen vocabulary, ordered merges, atomic expansions and identity;
`token_usage.csv` shows selected content frequencies. Provenance records every
source hash, split and setting without absolute workstation paths.

Counts represent **one complete unpadded joint stream per original replay**:
atomic or BPE content plus `3 × timesteps + 11` structural tokens. They exclude
repeated window headers/footers and differ from conditioned input/canvas counts
in the older estimator. Both-perspective expansion doubles these totals;
`summary.json` reports that explicitly. No replays are silently skipped.

## Storage and regeneration

Scratch holds the raw copies, projected per-replay word caches, transactional
SQLite pair index, venv and pip/Matplotlib/tmp caches. Local source data remains
the durable original. Home holds job logs and only compact study results. Home
copies happen at ordinary exit (including errors), with live transcripts during
the job. Download the results promptly. OSCER documents scratch as unbacked,
temporary storage with an autodelete policy; confirm site policy for your
particular filesystem. The workflow does not touch timestamps to evade it.

After scratch loss, repeat upload and submission; pinned dependencies, source
checksums and splits reproduce the work. Retain the downloaded tokenizer and
provenance so you need not refit merely to encode data. Parsing/fitting elapsed
time is recorded by the first run; regeneration speed cannot be promised before
that measurement. Corpus tokenization/manifests and training integration remain
separate future steps.

## Optional offline dependency upload

If compute nodes cannot reach PyPI, create Linux CPython 3.12 wheels on your
local machine and pass that wheelhouse to the uploader. Use a temporary venv,
bootstrapped from the confirmed project shim, so the project environment is
unchanged. Everything temporary is removed even on failure:

```powershell
$project = (Get-Location).Path
if (-not (Test-Path .venv/Scripts/python.exe)) { throw 'Project venv missing' }
$stage = Join-Path ([IO.Path]::GetTempPath()) ('tokenizer-wheels-' + [guid]::NewGuid().ToString('N'))
try {
    .venv/Scripts/python.exe -m venv "$stage/venv"
    if ($LASTEXITCODE) { throw 'Temporary venv creation failed' }
    & "$stage/venv/Scripts/python.exe" -m pip download --only-binary=:all: --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --python-version 312 --implementation cp --abi cp312 --dest "$stage/wheels" -r scripts/oscer/requirements-tokenizer.txt
    if ($LASTEXITCODE) { throw 'Linux wheel download failed' }
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/oscer/upload_tokenizer_study.ps1 -RunName vocab-study-offline-01 -Wheelhouse "$stage/wheels"
    if ($LASTEXITCODE) { throw 'Upload failed' }
} finally {
    $resolved = [IO.Path]::GetFullPath($stage)
    $prefix = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetFileName($resolved) -notlike 'tokenizer-wheels-*') { throw 'Unsafe cleanup target' }
    if (Test-Path -LiteralPath $resolved) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
```

Use the same submission/retrieval commands with `vocab-study-offline-01`
substituted everywhere. The job automatically detects `wheels/` and installs
without network. Linux installation itself still requires verification on OSCER.

Site references (checked 2026-11-17): [Python setup](https://www.ou.edu/oscer/applications/python/python-basic-setup),
[job submission](https://www.ou.edu/oscer/using-the-cluster/submitting-jobs),
[scratch policy](https://ou.edu/oscer/storage/scratch-folder/).
