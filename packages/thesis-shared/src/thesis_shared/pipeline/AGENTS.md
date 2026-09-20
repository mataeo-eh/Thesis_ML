# thesis_shared.pipeline Contract — Shared Acquisition and Storage

## Purpose

- Own data acquisition and the local/remote storage abstraction, which are identical for both arms and must stay that way so both train from the same bytes.

## FRAMEWORK CONTRACT

- No `torch`, no `tensorflow`.

## Ownership

- `acquire_data.py` owns config-driven dataset acquisition and the `thesis-acquire` console script.
- `storage.py` owns `StorageResolver` and the local/remote path abstraction.

## Local Contracts

- Acquisition is config-only; paths, credentials sources, and dataset identifiers are read from YAML, never hardcoded.
- Credentials must never be echoed into logs, command records, or error messages. The dry-run path exists to verify command construction without exposing secrets.
- Both arms acquire through this module. Never add an arm-specific acquisition path — if the arms ever pull different data, the comparison is void.
- Training orchestration is arm-owned and does NOT belong here; only acquisition and storage resolution are shared.

## Verification

- Changes require `tests/test_pipeline.py`, including the credential-redaction and dry-run coverage.
