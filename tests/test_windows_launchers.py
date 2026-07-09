from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_windows_launchers_are_thin_config_driven_wrappers() -> None:
    expected = {
        "overfit.bat": ("configs\\local_overfit_v2.yaml", "overfitv2"),
        # V2: renamed from smallTrainingTest.bat when the pre-training pipeline
        # was reworked (no-input pre-training); its output dir is versioned too
        # so old smallTrainingTest results are never mixed with new runs.
        "smallTrainingTestV2.bat": ("configs\\local_full.yaml", "smalltrainingtestv2"),
    }
    for filename, (config_path, output_dir) in expected.items():
        text = (ROOT / "tests" / filename).read_text(encoding="utf-8").lower()
        assert 'set "project_root=%~dp0.."' in text
        assert f"--config {config_path.lower()}" in text
        assert "%*" in text
        # Both launchers tee stdout+stderr so per-step metrics stream to the
        # terminal AND land in console.log (the redirect-only form hid all
        # output from the console the .bat opened).
        assert f'set "output_dir=%~dp0output\\{output_dir}"' in text
        assert "'.venv\\scripts\\python.exe' -m thesis_ml.pipeline.train_pipeline" in text
        assert "tee-object -filepath '%output_dir%\\console.log'" in text
        assert "exit $lastexitcode" in text
        assert "python -c" not in text
        assert "torch" not in text
