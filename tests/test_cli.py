# tests/test_cli.py (part 3 — hardening & edge cases)
import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from typer.testing import CliRunner
from preflight.cli import app

runner = CliRunner()

def write_csv(content: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write(content)
        return f.name

def test_empty_csv_fails_cleanly():
    path = write_csv("a,b,target\n")  # header only, 0 rows
    try:
        result = runner.invoke(app, ["prepare", path, "--target", "target"])
        assert result.exit_code == 1
        assert "traceback" not in result.output.lower()
    finally:
        os.unlink(path)

def test_all_null_target_fails_cleanly():
    path = write_csv("a,target\n1,\n2,\n3,\n")
    try:
        result = runner.invoke(app, ["prepare", path, "--target", "target"])
        assert result.exit_code == 1
        assert "traceback" not in result.output.lower()
    finally:
        os.unlink(path)

def test_malformed_csv_fails_cleanly():
    path = write_csv('a,b\n1,2,3,4,5\n"unterminated')
    try:
        result = runner.invoke(app, ["prepare", path, "--target", "b"])
        assert result.exit_code == 1
        assert "traceback" not in result.output.lower()
    finally:
        os.unlink(path)

def test_drop_threshold_out_of_range_rejected():
    n = 20
    df = pd.DataFrame({"a": np.random.normal(0, 1, n), "target": np.random.choice([0, 1], n)})
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
        path = f.name
    try:
        result = runner.invoke(app, ["prepare", path, "--target", "target", "--drop-threshold", "1.5"])
        assert result.exit_code == 1
    finally:
        os.unlink(path)

def test_cardinality_threshold_zero_rejected():
    n = 20
    df = pd.DataFrame({"a": np.random.normal(0, 1, n), "target": np.random.choice([0, 1], n)})
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
        path = f.name
    try:
        result = runner.invoke(app, ["prepare", path, "--target", "target", "--cardinality-threshold", "0"])
        assert result.exit_code == 1
    finally:
        os.unlink(path)

def test_verbose_flag_prints_report(tmp_path):
    n = 30
    df = pd.DataFrame({
        "age": np.random.normal(40, 10, n),
        "target": np.random.choice([0, 1], n),
    })
    csv_path = str(tmp_path / "data.csv")
    df.to_csv(csv_path, index=False)
    result = runner.invoke(app, [
        "prepare", csv_path, "--target", "target", "--output-dir", str(tmp_path), "--verbose"
    ])
    assert result.exit_code == 0
    assert len(result.output) > 0  # report content printed in addition to paths

def test_non_verbose_flag_still_succeeds(tmp_path):
    n = 30
    df = pd.DataFrame({
        "age": np.random.normal(40, 10, n),
        "target": np.random.choice([0, 1], n),
    })
    csv_path = str(tmp_path / "data.csv")
    df.to_csv(csv_path, index=False)
    result = runner.invoke(app, [
        "prepare", csv_path, "--target", "target", "--output-dir", str(tmp_path)
    ])
    assert result.exit_code == 0

def test_valid_drop_threshold_boundary_values(tmp_path):
    n = 20
    df = pd.DataFrame({"a": np.random.normal(0, 1, n), "target": np.random.choice([0, 1], n)})
    csv_path = str(tmp_path / "data.csv")
    df.to_csv(csv_path, index=False)
    result = runner.invoke(app, [
        "prepare", csv_path, "--target", "target", "--drop-threshold", "0.0", "--output-dir", str(tmp_path)
    ])
    assert result.exit_code == 0