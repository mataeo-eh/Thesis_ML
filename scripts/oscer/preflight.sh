#!/bin/bash
# Read-only login-node inspection. No Python, installs, jobs or file changes.
echo '=== ACCOUNT / HOST ==='
id -un
hostname
echo '=== PYTHON MODULES ==='
module avail Python 2>&1
echo '=== SCRATCH CANDIDATES ==='
for location in "/scratch/$USER" "${SCRATCH:-}" "${WORK:-}"; do
    [[ -n "$location" ]] || continue
    printf 'path=%s exists=' "$location"
    if [[ -d "$location" ]]; then
        printf 'yes writable='
        if [[ -w "$location" ]]; then echo yes; else echo no; fi
        df -h "$location"
    else
        echo no
    fi
done
echo '=== HOME SPACE ==='
df -h "$HOME"
if command -v quota >/dev/null 2>&1; then timeout 10 quota -s 2>&1; fi
echo '=== PARTITIONS ==='
sinfo -h -o '%P %a %l %D'
echo '=== NORMAL PARTITION ==='
scontrol show partition normal 2>&1
