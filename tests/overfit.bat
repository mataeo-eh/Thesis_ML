@echo off
REM ---------------------------------------------------------------------------
REM Overfit (learnability probe) launcher -- PRE-TRAINING pathway.
REM
REM Runs configs\local_overfit_v2.yaml through thesis_ml.pipeline.train_pipeline:
REM 100 epochs (= 3400 optimizer steps at 34/epoch) over an explicitly named
REM 10-train / 3-dev replay subset chosen at the corpus median token count. The
REM point is to confirm the full current pipeline can drive loss down at all
REM before spending a long run on it. 100 epochs is also exactly what the five
REM configs\ablation_*.yaml arms run for, so this baseline and every arm are
REM comparable epoch for epoch.
REM
REM HEADS UP (2026-08-09): model.frozen_input_kv became a project-wide default,
REM and configs\local_overfit_v2.yaml now inherits it, so this launcher's
REM architecture_identity is dense-multinomial-SC2-v2+frozen_input_kv. The
REM PRE-PROMOTION checkpoints\local-overfitV2\last.pt will NOT resume into it --
REM the launch aborts on a fingerprint mismatch instead of training on the wrong
REM architecture. Archive tests\output\overfitV2\ and move that last.pt aside
REM before relaunching. The old all-toggles-false condition is preserved in
REM configs\ablation_00_baseline.yaml.
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
REM     epoch_metrics.csv      1 row per epoch, train + dev, every loss
REM                            decomposition including the rare-class x
REM                            corruption-bucket losses and position counts
REM     step_metrics.jsonl     1 line per optimizer step
REM     replay_selection.json  exactly which replays were used
REM   (no interval_metrics.csv: this profile reports train AND dev once per
REM    epoch -- see the interval_*_evaluation notes in configs\local_overfit.yaml)
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
echo   metrics : %OUTPUT_DIR%\epoch_metrics.csv
exit /b %EXIT_CODE%
