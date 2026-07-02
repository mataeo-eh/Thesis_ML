from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_windows_launchers_are_thin_config_driven_wrappers() -> None:
    expected = {
        "overfit.bat": "configs\\local_overfit.yaml",
        "smallTrainingTest.bat": "configs\\local_full.yaml",
    }
    for filename, config_path in expected.items():
        text = (ROOT / "tests" / filename).read_text(encoding="utf-8").lower()
        assert 'set "project_root=%~dp0.."' in text
        assert f"--config {config_path.lower()}" in text
        assert "%*" in text
        if filename == "overfit.bat":
            assert "'.venv\\scripts\\python.exe' -m thesis_ml.pipeline.train_pipeline" in text
            assert "tee-object -filepath '%output_dir%\\console.log'" in text
            assert "exit $lastexitcode" in text
        else:
            assert '".venv\\scripts\\python.exe" -m thesis_ml.pipeline.train_pipeline' in text
            assert '> "%output_dir%\\console.log" 2>&1' in text
        assert "python -c" not in text
        assert "torch" not in text
