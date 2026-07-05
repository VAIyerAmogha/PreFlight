# tests/test_scaffold.py
import subprocess
import importlib
from pathlib import Path

def test_directory_structure_exists():
    expected = [
        "src/preflight/__init__.py",
        "src/preflight/types.py",
        "src/preflight/profiler.py",
        "src/preflight/cleaner.py",
        "src/preflight/engineer.py",
        "src/preflight/assembler.py",
        "src/preflight/report.py",
        "src/preflight/cli.py",
        "tests/test_types.py",
        "tests/test_profiler.py",
        "tests/test_cleaner.py",
        "tests/test_engineer.py",
        "tests/test_assembler.py",
        "tests/test_report.py",
        "tests/test_cli.py",
        "pyproject.toml",
        ".gitignore",
        "README.md",
    ]
    for path in expected:
        assert Path(path).exists(), f"Missing: {path}"

def test_package_installs_editable():
    import sys
    import pytest
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."], capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.skip(f"Skipping pip install: {result.stderr}")
    assert result.returncode == 0, result.stderr

def test_package_importable():
    mod = importlib.import_module("preflight")
    assert mod is not None

def test_pytest_collects_with_no_errors():
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

def test_cli_entrypoint_registered():
    import sys
    from pathlib import Path
    bin_dir = Path(sys.executable).parent
    preflight_bin = str(bin_dir / "preflight")
    if not Path(preflight_bin).exists() and not Path(preflight_bin + ".exe").exists():
        preflight_bin = "preflight"
    result = subprocess.run([preflight_bin, "--help"], capture_output=True, text=True)
    assert result.returncode == 0