[CmdletBinding()]
param(
    [string]$DataRoot = "",
    [string]$Destination = "oscer",
    [string]$RemoteRoot = "/scratch/mataeo/thesis-tokenizer",
    [string]$RunName = ("study-" + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$Wheelhouse = "",
    [switch]$ValidateOnly,
    [int]$ReplayLimit = 0
)
$ErrorActionPreference = 'Stop'
$runClock = [Diagnostics.Stopwatch]::StartNew()
function Write-UploadStatus([string]$Message) {
    Write-Host ("[{0} | elapsed {1:hh\:mm\:ss}] {2}" -f (Get-Date -Format 'HH:mm:ss'), $runClock.Elapsed, $Message)
}
function Format-UploadSize([long]$Bytes) {
    if ($Bytes -ge 1GB) { return ('{0:N2} GiB' -f ($Bytes / 1GB)) }
    return ('{0:N1} MiB' -f ($Bytes / 1MB))
}
function Get-UploadPreparationCommand([string]$Parent, [string]$Target) {
    # Paths are validated as simple /scratch paths before this is called.
    # An interrupted local preparation may leave an empty directory; reuse it.
    return "set -e; printf 'OSCER authentication succeeded on '; hostname; " +
        "mkdir -p '$Parent'; " +
        "if test -e '$Target'; then " +
        "test -d '$Target' || { echo 'REMOTE_DESTINATION_EXISTS: destination is not a directory' >&2; exit 73; }; " +
        "test -r '$Target' && test -w '$Target' && test -x '$Target' || { echo 'REMOTE_DESTINATION_NOT_WRITABLE' >&2; exit 74; }; " +
        "if find '$Target' -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then " +
        "echo 'REMOTE_DESTINATION_EXISTS: destination contains files; choose a new RunName' >&2; exit 73; fi; " +
        "echo 'Reusing empty upload directory'; " +
        "else mkdir '$Target'; fi; " +
        "test -w '$Target' || { echo 'REMOTE_DESTINATION_NOT_WRITABLE' >&2; exit 74; }; " +
        "echo 'Remote upload destination ready: $Target'"
}
function Assert-UploadRemoteExit([int]$ExitCode, [string]$Operation) {
    if ($ExitCode -eq 0) { return }
    if ($ExitCode -eq 255) {
        throw "SSH connection/authentication failed during $Operation (exit 255). The connection may have closed or authentication may have timed out. This does not establish a directory conflict. Run 'ssh $Destination' to check login, then retry; respond to each password/MFA prompt when it appears."
    }
    if ($ExitCode -eq 73) { throw 'Remote destination already contains files or is not a directory. Choose a new RunName; existing files were not overwritten.' }
    if ($ExitCode -eq 74) { throw 'Remote destination is not accessible/writable. Check your scratch directory permissions.' }
    throw "Remote command failed during $Operation (exit $ExitCode). See the SSH output above for the actual cause."
}
function Invoke-VisibleTar([string]$Label, [string[]]$TarArguments, [long]$ExpectedBytes = 0) {
    # Redirect only tar's output. SSH/SCP below inherit the interactive terminal.
    $logName = [guid]::NewGuid().ToString('N')
    $stdout = Join-Path $temporary "$logName.stdout"
    $stderr = Join-Path $temporary "$logName.stderr"
    # Start-Process joins ArgumentList into one native command line on Windows.
    $quoted = foreach ($argument in $TarArguments) {
        if ($argument.Contains('"')) { throw 'Tar arguments may not contain double quotes.' }
        '"' + $argument + '"'
    }
    Write-UploadStatus "$Label started. Archive activity is reported every 3 seconds."
    $stageClock = [Diagnostics.Stopwatch]::StartNew()
    $process = Start-Process -FilePath $tarExecutable -ArgumentList $quoted -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    # Keep the native handle: Windows PowerShell can otherwise lose ExitCode
    # for a fast subprocess returned by Start-Process -PassThru.
    $null = $process.Handle
    try {
        while (-not $process.WaitForExit(3000)) {
            $item = Get-Item -LiteralPath $bundle -ErrorAction SilentlyContinue
            $bytes = if ($item) { $item.Length } else { 0 }
            $percent = if ($ExpectedBytes -gt 0) { [Math]::Min(99.0, 100.0 * [double]$bytes / [double]$ExpectedBytes) } else { 0 }
            $lastFile = Get-Content -LiteralPath $stderr -Tail 1 -ErrorAction SilentlyContinue
            $message = "{0}: bundle {1}; stage elapsed {2:hh\:mm\:ss}" -f $Label, (Format-UploadSize $bytes), $stageClock.Elapsed
            if ($lastFile) { $message += "; latest archive entry: $lastFile" }
            Write-Progress -Id 2 -Activity $Label -Status $message -PercentComplete $percent
            Write-UploadStatus $message
        }
        $process.WaitForExit()
        $process.Refresh()
        if ($process.ExitCode -ne 0) {
            Get-Content -LiteralPath $stderr -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
            throw "$Label failed (tar exit $($process.ExitCode))."
        }
        Write-UploadStatus ("{0} finished in {1:hh\:mm\:ss}." -f $Label, $stageClock.Elapsed)
    }
    finally {
        # Ctrl+C must stop tar before the outer temporary-directory cleanup.
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            $process.WaitForExit()
        }
        $process.Dispose()
        Write-Progress -Id 2 -Activity $Label -Completed
    }
}
Write-UploadStatus 'Starting tokenizer upload preparation. Ctrl+C cancels and cleans up local temporary files.'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (-not $DataRoot) { $DataRoot = Join-Path $projectRoot '../data/quickstart' }
$dataPath = (Resolve-Path -LiteralPath $DataRoot).Path
if ($RunName -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]*$') { throw 'RunName must be a simple directory name.' }
if ($RemoteRoot -notmatch '^/scratch/[A-Za-z0-9_/-]+$' -or $RemoteRoot.Contains('..')) {
    throw 'RemoteRoot must be an absolute path below /scratch with simple directory names.'
}
if ($Destination -notmatch '^[A-Za-z0-9_.@-]+$') { throw 'Destination must be an SSH alias or user@hostname.' }
if ($ReplayLimit -lt 0) { throw 'ReplayLimit must be nonnegative (0 means all replays).' }
$remote = "$RemoteRoot/$RunName"
Write-UploadStatus "[1/6] Finding replay files and checking metadata in $dataPath"
$replays = @(Get-ChildItem -LiteralPath (Join-Path $dataPath 'parquet') -Filter '*_game_state.parquet' -File | Sort-Object Name)
if ($ReplayLimit -gt 0) { $replays = @($replays | Select-Object -First $ReplayLimit) }
if (-not $replays.Count) { throw 'No game-state parquets found.' }
$dataMembers = [System.Collections.Generic.List[string]]::new()
foreach ($replay in $replays) {
    $metadataName = $replay.Name.Replace('_game_state.parquet', '_metadata.json')
    if (-not (Test-Path -LiteralPath (Join-Path $dataPath "json/$metadataName"))) {
        throw "Missing metadata for $($replay.Name)"
    }
    $dataMembers.Add("parquet/$($replay.Name)")
    $dataMembers.Add("json/$metadataName")
}
# Exact allowlist: no git state, credentials, venvs, checkpoints, or other datasets.
$codeMembers = [System.Collections.Generic.List[string]]::new()
foreach ($directory in @('packages/thesis-shared/src/thesis_shared')) {
    foreach ($file in Get-ChildItem -LiteralPath (Join-Path $projectRoot $directory) -Recurse -File -Filter '*.py') {
        $codeMembers.Add($file.FullName.Substring($projectRoot.Length + 1).Replace('\', '/'))
    }
}
foreach ($member in @('scripts/tokenizer_study.py', 'scripts/oscer/tokenizer_study.sbatch',
        'scripts/oscer/verify_upload.py', 'scripts/oscer/requirements-tokenizer.txt',
        'config/default.yaml', 'config/tokenizer_study.yaml', 'data/Token_Dictionary.json')) {
    $codeMembers.Add($member)
}
$temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$temporary = Join-Path $temporaryRoot ('thesis-tokenizer-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    $tarExecutable = (Get-Command tar -CommandType Application -ErrorAction Stop).Source
    Write-UploadStatus "Selected $($replays.Count) replays; temporary workspace: $temporary"
    if ($ValidateOnly) {
        Write-UploadStatus '[2/6] ValidateOnly: no SSH connections or password prompts will occur.'
    }
    else {
        Write-UploadStatus "[2/6] Connecting to $Destination and preparing $remote NOW, before hashing or building the archive."
        Write-Host 'SSH may ask for your OSCER password/MFA now. Password typing is hidden.'
        Write-Host 'If SSH keys or an agent authenticate successfully, no password prompt is necessary.'
        & ssh -o ConnectTimeout=30 $Destination (Get-UploadPreparationCommand -Parent $RemoteRoot -Target $remote)
        Assert-UploadRemoteExit -ExitCode $LASTEXITCODE -Operation 'initial login/destination preparation'
        Write-UploadStatus 'Login and destination preparation succeeded. Next: local hashing/packing, then SCP upload, then SSH extraction.'
        Write-Host 'Each new SCP/SSH connection may ask for your password/MFA again; answer when prompted. No password is stored or reused by this script.'
    }
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [IO.File]::WriteAllLines((Join-Path $temporary 'code.txt'), $codeMembers, $utf8)
    [IO.File]::WriteAllLines((Join-Path $temporary 'data.txt'), $dataMembers, $utf8)
    $hashFiles = @(
        foreach ($member in $codeMembers) {
            Get-Item -LiteralPath (Join-Path $projectRoot $member)
        }
        foreach ($member in $dataMembers) {
            Get-Item -LiteralPath (Join-Path $dataPath $member)
        }
    )
    $hashMembers = @($codeMembers.ToArray()) + @($dataMembers.ToArray())
    $totalBytes = [long](($hashFiles | Measure-Object -Property Length -Sum).Sum)
    Write-UploadStatus ("[3/6] Computing SHA-256 checksums: {0} files, {1}. This reads the local dataset; no transfer yet." -f $hashFiles.Count, (Format-UploadSize $totalBytes))
    $entries = [System.Collections.Generic.List[object]]::new()
    $hashedBytes = [long]0
    $lastHashUpdate = -3.0
    for ($index = 0; $index -lt $hashFiles.Count; $index++) {
        $file = $hashFiles[$index]
        $status = "File $($index + 1)/$($hashFiles.Count): $($hashMembers[$index]); $(Format-UploadSize $hashedBytes) / $(Format-UploadSize $totalBytes)"
        Write-Progress -Id 1 -Activity 'Computing upload checksums (local disk)' -Status $status `
            -PercentComplete ([Math]::Min(100.0, 100.0 * [double]$hashedBytes / [Math]::Max(1.0, [double]$totalBytes)))
        if ($runClock.Elapsed.TotalSeconds - $lastHashUpdate -ge 3) {
            Write-UploadStatus $status
            $lastHashUpdate = $runClock.Elapsed.TotalSeconds
        }
        $entries.Add(@{ path = $hashMembers[$index]; sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant() })
        $hashedBytes += $file.Length
    }
    Write-Progress -Id 1 -Activity 'Computing upload checksums (local disk)' -Completed
    Write-UploadStatus "Checksums complete: $($hashFiles.Count) files, $(Format-UploadSize $hashedBytes)."
    $manifest = @{ schema = 'tokenizer_upload_v1'; replay_count = $replays.Count; files = $entries }
    [IO.File]::WriteAllText((Join-Path $temporary 'upload_manifest.json'), ($manifest | ConvertTo-Json -Depth 6), $utf8)
    $bundle = Join-Path $temporary 'payload.tar'
    Write-UploadStatus '[4/6] Building the temporary tar locally. Large replay files can take several minutes.'
    Invoke-VisibleTar -Label 'Packing source files' -TarArguments @('-cvf', $bundle, '-C', $projectRoot, '-T', (Join-Path $temporary 'code.txt'))
    Invoke-VisibleTar -Label 'Packing replay data' -ExpectedBytes $totalBytes -TarArguments @('-rvf', $bundle, '-C', $dataPath, '-T', (Join-Path $temporary 'data.txt'))
    Invoke-VisibleTar -Label 'Adding checksum manifest' -TarArguments @('-rf', $bundle, '-C', $temporary, 'upload_manifest.json')
    if ($Wheelhouse) {
        $wheelPath = (Resolve-Path -LiteralPath $Wheelhouse).Path
        $wheelStage = Join-Path $temporary 'wheels'
        New-Item -ItemType Directory -Path $wheelStage | Out-Null
        Write-UploadStatus "Copying optional dependency wheels from $wheelPath"
        Get-ChildItem -LiteralPath $wheelPath -Filter '*.whl' -File | ForEach-Object {
            Write-UploadStatus "Copying dependency: $($_.Name) ($(Format-UploadSize $_.Length))"
            Copy-Item -LiteralPath $_.FullName -Destination $wheelStage
        }
        Invoke-VisibleTar -Label 'Packing dependency wheels' -TarArguments @('-rvf', $bundle, '-C', $temporary, 'wheels')
    }
    Write-UploadStatus 'Checking the completed archive member list.'
    $members = @(& tar -tf $bundle)
    if ($LASTEXITCODE -ne 0) { throw 'Tar listing failed.' }
    if ($members.Count -lt ($codeMembers.Count + $dataMembers.Count + 1)) { throw 'Incomplete tar.' }
    Write-UploadStatus ("Bundle ready: {0} replays, {1} (temporary)." -f $replays.Count, (Format-UploadSize (Get-Item $bundle).Length))
    if ($ValidateOnly) { Write-UploadStatus '[5/6] Archive validated; upload skipped.'; return }
    # This script is run manually by the user; agents do not execute remote steps.
    Write-UploadStatus '[5/6] Starting SCP upload now. The destination was already prepared during the initial login.'
    Write-UploadStatus "Uploading $(Format-UploadSize (Get-Item $bundle).Length) to ${Destination}:$remote/payload.tar"
    Write-Host 'SCP runs in this terminal: watch its transfer percentage/rate/ETA and any password/MFA prompt.'
    & scp -o ConnectTimeout=30 $bundle "${Destination}:$remote/payload.tar"
    if ($LASTEXITCODE -ne 0) { throw "Upload failed (SCP exit $LASTEXITCODE). Check the connection/authentication output above. Local temp files will be removed. A partial remote upload requires a new RunName; an empty destination can be reused." }
    Write-UploadStatus '[6/6] Upload finished. Extracting on OSCER; remote filenames below show extraction activity.'
    & ssh -o ConnectTimeout=30 $Destination "cd '$remote' && tar -xvf payload.tar && rm -- payload.tar && echo 'Remote extraction complete; remote tar removed.'"
    Assert-UploadRemoteExit -ExitCode $LASTEXITCODE -Operation "archive extraction (remote archive: $remote/payload.tar)"
    Write-UploadStatus "Uploaded and extracted successfully: $remote"
    Write-Host "Next: ssh $Destination"
    Write-Host "Then: cd '$remote'"
    Write-Host 'See scripts/oscer/README.md in your local checkout for submission and retrieval commands.'
}
finally {
    Write-Progress -Id 1 -Activity 'Computing upload checksums (local disk)' -Completed
    Write-UploadStatus 'Cleaning up the local temporary archive and workspace.'
    # Verify the resolved absolute deletion target stays within the intended temp directory.
    $resolved = [IO.Path]::GetFullPath($temporary)
    $prefix = $temporaryRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($resolved) -notlike 'thesis-tokenizer-*') {
        throw "Unsafe temporary cleanup target: $resolved"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
    Write-UploadStatus 'Removed local temporary archive and bundle files.'
}
