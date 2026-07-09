@echo off
REM Debut FINE-TUNING pathway launcher (the separate counterpart to
REM smallTrainingTestV2.bat, which runs PRE-TRAINING). This warm-starts from a
REM pre-trained checkpoint and trains the debut-units-only target via the
REM finetune_pipeline entry point -- NOT train_pipeline. Config selects the
REM pathway: local_overfit_v2_finetune.yaml sets debut_mode: true and
REM init_from_checkpoint.
setlocal
set "PROJECT_ROOT=%~dp0.."
set "OUTPUT_DIR=%~dp0output\smallFinetuneTest"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

pushd "%PROJECT_ROOT%"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -Command "& { & '.venv\Scripts\python.exe' -m thesis_ml.pipeline.finetune_pipeline --config configs\local_overfit_v2_finetune.yaml %* 2>&1 | Tee-Object -FilePath '%OUTPUT_DIR%\console.log'; exit $LASTEXITCODE }"
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
