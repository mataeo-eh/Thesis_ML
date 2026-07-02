@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
set "OUTPUT_DIR=%~dp0output\smallTrainingTest"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

pushd "%PROJECT_ROOT%"
".venv\Scripts\python.exe" -m thesis_ml.pipeline.train_pipeline --config configs\local_full.yaml %* > "%OUTPUT_DIR%\console.log" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
