import pytest
import os
import tempfile
from typer.testing import CliRunner
from preflight.cli import app

runner = CliRunner()

def test_cli_input_path_not_exist():
    result = runner.invoke(app, ["prepare", "nonexistent.csv", "--target", "price"])
    assert result.exit_code == 1
    assert "input_path must exist" in result.stderr or "input_path must exist" in result.stdout

def test_cli_input_path_not_csv():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"a,b,c\n1,2,3")
        path = f.name
    try:
        result = runner.invoke(app, ["prepare", path, "--target", "price"])
        assert result.exit_code == 1
        assert "be a .csv file" in result.stderr or "be a .csv file" in result.stdout
    finally:
        os.remove(path)

def test_cli_value_error_from_prepare():
    # Provide a valid CSV but with a target that doesn't exist to trigger ValueError
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"a,b,c\n1,2,3")
        path = f.name
    try:
        result = runner.invoke(app, ["prepare", path, "--target", "nonexistent"])
        assert result.exit_code == 1
    finally:
        os.remove(path)

def test_cli_oserror_on_write(monkeypatch):
    import pandas as pd
    def mock_to_csv(*args, **kwargs):
        raise OSError("Permission denied")
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"price,b,c\n1,2,3\n1,2,3")
        path = f.name
    try:
        monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv)
        result = runner.invoke(app, ["prepare", path, "--target", "price"])
        assert result.exit_code == 1
        assert "Error writing output file" in result.stderr or "Error writing output file" in result.stdout
    finally:
        os.remove(path)
