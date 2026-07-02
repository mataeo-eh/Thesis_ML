@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
set "OUTPUT_DIR=%~dp0output\overfit"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

pushd "%PROJECT_ROOT%"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -Command "& { & '.venv\Scripts\python.exe' -m thesis_ml.pipeline.train_pipeline --config configs\local_overfit.yaml %* 2>&1 | Tee-Object -FilePath '%OUTPUT_DIR%\console.log'; exit $LASTEXITCODE }"
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
