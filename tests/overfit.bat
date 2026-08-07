@echo off
REM ---------------------------------------------------------------------------
REM Overfit (learnability probe) launcher -- PRE-TRAINING pathway.
REM
REM Runs configs\local_overfit_v2.yaml through thesis_ml.pipeline.train_pipeline:
REM 30 epochs over an explicitly named 10-train / 3-dev replay subset chosen at
REM the corpus median token count. The point is to confirm the full current
REM pipeline can drive loss down at all before spending a long run on it.
REM
REM This file stays a THIN launcher on purpose. Every training knob -- replay
REM selection, epochs, model scale, learning rate, budgets -- is owned by the
REM YAML profile, never by this script. Extra arguments are forwarded to the
REM Python entry point, so a bounded check is just:
REM     overfit.bat --max-steps 20
REM and a one-off learning rate is:
REM     overfit.bat --lr 1e-4
REM
REM Outputs land in tests\output\overfitV2\:
REM     console.log            full stdout+stderr of the LATEST run
REM     console-<timestamp>.log archived copy of each completed run
REM     interval_metrics.csv   10 diagnostic rows per epoch (train + dev)
REM     epoch_metrics.csv      1 row per epoch
REM     step_metrics.jsonl     1 line per optimizer step
REM     replay_selection.json  exactly which replays were used
REM ---------------------------------------------------------------------------
setlocal
set "PROJECT_ROOT=%~dp0.."
set "OUTPUT_DIR=%~dp0output\overfitV2"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Fail with a readable message instead of a confusing Python error when the
REM project virtual environment has not been created yet.
if not exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    echo ERROR: %PROJECT_ROOT%\.venv\Scripts\python.exe not found.
    echo Create the project virtual environment before running this launcher.
    exit /b 1
)

REM Unbuffered stdio. The training loop already flushes its own prints, but this
REM also covers Python's own startup/traceback output so a crash reaches
REM console.log instead of dying in a buffer.
set "PYTHONUNBUFFERED=1"

pushd "%PROJECT_ROOT%"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -Command "& { & '.venv\Scripts\python.exe' -m thesis_ml.pipeline.train_pipeline --config configs\local_overfit_v2.yaml %* 2>&1 | Tee-Object -FilePath '%OUTPUT_DIR%\console.log'; exit $LASTEXITCODE }"
set "EXIT_CODE=%ERRORLEVEL%"
popd

REM Archive this run's console output under a timestamped name. Tee-Object
REM TRUNCATES console.log at every launch, so without this copy the previous
REM run's log -- often the one holding the failure worth comparing against -- is
REM destroyed the moment the next run starts.
for /f %%T in ('powershell.exe -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "RUN_STAMP=%%T"
if exist "%OUTPUT_DIR%\console.log" copy /y "%OUTPUT_DIR%\console.log" "%OUTPUT_DIR%\console-%RUN_STAMP%.log" >nul

echo.
echo overfit run finished with exit code %EXIT_CODE%
echo   console : %OUTPUT_DIR%\console-%RUN_STAMP%.log
echo   metrics : %OUTPUT_DIR%\interval_metrics.csv
exit /b %EXIT_CODE%
