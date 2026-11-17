# OSCER handoff contract

## Purpose

- Own locally prepared, user-operated OSCER upload and batch workflows. Agents never connect to OSCER.

## Ownership

- `preflight.sh` inspects account/host, available modules, scratch paths and Slurm partitions without Python or file changes. The 2026-11-17 user-supplied probe confirmed the default scratch path/module and normal partition's 48-hour maximum; it did not establish a personal home quota.

- `upload_tokenizer_study.ps1` builds an allowlisted temporary archive, optionally validates it locally, and on user execution uploads/extracts it. Local temp files are removed in `finally`; resolved cleanup targets must remain below the system temp directory.
- Upload preparation reports timestamped stages, hashing file/byte progress, and three-second archive heartbeats with size/elapsed time/latest entry. Normal runs check SSH authentication before hashing/packing; SSH/SCP inherit the interactive terminal for password/MFA prompts and transfer progress. Tar subprocesses are stopped on interruption before temporary-file cleanup. `ValidateOnly` never connects remotely.
- Initial SSH login also prepares the remote destination before local hashing/packing, reusing an empty directory only. Nonempty destinations are rejected with remote exit 73, inaccessible destinations with 74; SSH transport/authentication exit 255 must not be reported as a directory collision. SCP upload and SSH extraction remain independent interactive authentication steps; no password is stored or replayed.
- `tokenizer_study.sbatch` owns CPU resources, scratch environment setup and small home-directory result/log backups. `requirements-tokenizer.txt` pins only the analysis dependencies.
- `verify_upload.py` checks source/data hashes and exact replay membership before parsing.
- `README.md` owns complete local/remote submission, status, retry, success and failure retrieval commands and storage/regeneration boundaries.

## Local Contracts

- Raw copies, venvs, SQLite fitting stores and replay caches live on scratch; downloaded reports live in ignored `scripts/output/oscer-downloads/`. Keep the local source dataset as the durable original.
- Use the remote project's `.venv/bin/python` for every project command; only initial venv creation can use module Python after checking no shim exists. Do not install either model framework for the tokenizer study.
- Declare resource requests as unmeasured until the user provides remote evidence. Persist stdout/stderr and failure status alongside successful results.

## Verification

- `upload_tokenizer_study.ps1 -ValidateOnly -ReplayLimit 1` must validate the temporary archive and clean it without any SSH call.
- Byte totals and progress arithmetic must support multi-gigabyte bundles under Windows PowerShell 5.1: explicitly select floating-point `Math.Min`/`Math.Max` overloads instead of narrowing byte counts to Int32. Check with totals above 2 GiB.
- `tests/test_tokenizer_study.py` covers algorithm equivalence, restart, source integrity, metrics, dev-only endpoint selection and a bounded synthetic end-to-end run.

## Child DOX Index

- No child contracts.
