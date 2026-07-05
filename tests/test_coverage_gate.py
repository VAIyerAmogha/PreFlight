# tests/test_coverage_gate.py
import subprocess
import sys

def test_coverage_meets_80_percent_threshold():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-m", "not integration",
         "--ignore=tests/test_coverage_gate.py",
         "--cov=src/preflight", "--cov-report=term-missing",
         "--cov-fail-under=80"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr