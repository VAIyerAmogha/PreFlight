# tests/test_coverage_gate.py
import subprocess

def test_coverage_meets_80_percent_threshold():
    result = subprocess.run(
        ["pytest", "tests/", "-m", "not integration",
         "--cov=src/preflight", "--cov-report=term-missing",
         "--cov-fail-under=80"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr