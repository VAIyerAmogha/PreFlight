# tests/test_repo_hygiene.py
import subprocess
from pathlib import Path

def test_no_scratch_files_in_root():
    root = Path(".")
    forbidden_patterns = ["debug.py", "scratch.py"]
    for pattern in forbidden_patterns:
        assert not list(root.glob(pattern)), f"Found stray file matching {pattern}"

def test_no_tmp_output_artifacts_in_root():
    root = Path(".")
    assert not list(root.glob("tmp*_prepared.csv"))
    assert not list(root.glob("tmp*_pipeline.joblib"))
    assert not list(root.glob("tmp*_report.json"))

def test_gitignore_covers_scratch_patterns():
    gitignore = Path(".gitignore").read_text()
    for pattern in ["tmp*", "*.joblib", "scratch.py", "debug.py"]:
        assert pattern in gitignore, f"{pattern} not in .gitignore"

def test_pytest_testpaths_configured():
    pyproject = Path("pyproject.toml").read_text()
    assert "testpaths" in pyproject
    assert "integration" in pyproject  # marker registered

def test_full_unit_suite_passes():
    result = subprocess.run(
        ["pytest", "tests/", "-m", "not integration", "-q"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr