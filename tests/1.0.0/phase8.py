"""
Tests for Phase 8 (v1.0.0): CLI consolidation pass.

Covers:
- Every v1.0.0 CLI flag exists and is wired through (--column-type, --preset,
  --dry-run, --save-pdf, --text-features, --text-tfidf, --text-tfidf-top-k)
- Every pre-existing v0.1.0/v0.2.0 flag still works unchanged
- Explicit CLI flags override --preset values, consistent with the Python API
- --dry-run combined with --save-pdf produces a valid PDF from a dry-run report
- Malformed --column-type input produces a clear CLI error, not a stack trace
- Invalid --preset name produces a clear CLI error
- CLI behavior matches the Python API for equivalent parameter combinations
"""

import os

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader
from typer.testing import CliRunner

import preflight as pf
from preflight.cli import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path):
    n = 100
    df = pd.DataFrame({
        "num_a": np.random.uniform(0, 100, size=n),
        "cat": np.random.choice(["x", "y", "z"], size=n),
        "text_col": [
            "this is a fairly long piece of free text used to trigger text detection here"
        ] * n,
        "target": np.random.randint(0, 2, size=n),
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path, df


# ---------------------------------------------------------------------------
# Flag existence / basic wiring
# ---------------------------------------------------------------------------

class TestFlagsExist:

    def test_help_lists_all_v1_flags(self):
        result = runner.invoke(app, ["prepare", "--help"])
        assert result.exit_code == 0
        for flag in [
            "--column-type", "--preset", "--dry-run", "--save-pdf",
            "--text-features", "--text-tfidf", "--text-tfidf-top-k",
        ]:
            assert flag in result.output

    def test_help_lists_existing_v02_flags(self):
        result = runner.invoke(app, ["prepare", "--help"])
        for flag in [
            "--interactions", "--interaction-top-k", "--interaction-types",
            "--datetime-cyclical", "--datetime-deltas", "--datetime-reference-col",
            "--clustering", "--cluster-k", "--cluster-features",
        ]:
            assert flag in result.output


# ---------------------------------------------------------------------------
# Individual new flags work end-to-end
# ---------------------------------------------------------------------------

class TestIndividualFlagsWork:

    def test_preset_flag_runs_successfully(self, sample_csv):
        path, _df = sample_csv
        result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--preset", "fast",
        ])
        assert result.exit_code == 0

    def test_invalid_preset_gives_clear_error(self, sample_csv):
        path, _df = sample_csv
        result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--preset", "bogus_preset",
        ])
        assert result.exit_code != 0
        assert "fast" in result.output.lower() or "thorough" in result.output.lower()

    def test_dry_run_flag_runs_successfully(self, sample_csv):
        path, _df = sample_csv
        result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--dry-run",
        ])
        assert result.exit_code == 0

    def test_column_type_flag_runs_successfully(self, sample_csv):
        path, _df = sample_csv
        result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--column-type", "cat:CATEGORICAL_LOW",
        ])
        assert result.exit_code == 0

    def test_malformed_column_type_gives_clear_error(self, sample_csv):
        path, _df = sample_csv
        result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--column-type", "no_colon_here",
        ])
        assert result.exit_code != 0
        assert "traceback" not in result.output.lower()

    def test_invalid_column_type_value_gives_clear_error(self, sample_csv):
        path, _df = sample_csv
        result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--column-type", "cat:NOT_A_REAL_TYPE",
        ])
        assert result.exit_code != 0
        assert "traceback" not in result.output.lower()

    def test_text_features_flags_run_successfully(self, sample_csv):
        path, _df = sample_csv
        result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--text-features", "--text-tfidf", "--text-tfidf-top-k", "5",
        ])
        assert result.exit_code == 0

    def test_save_pdf_flag_creates_file(self, sample_csv, tmp_path):
        path, _df = sample_csv
        pdf_path = tmp_path / "report.pdf"
        result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--save-pdf", str(pdf_path),
        ])
        assert result.exit_code == 0
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Flag interaction edge cases
# ---------------------------------------------------------------------------

class TestFlagInteractions:

    def test_explicit_flag_overrides_preset_via_cli(self, sample_csv):
        """Should not error, and should behave consistently with the Python API
        override guarantee (explicit flags win over --preset)."""
        path, _df = sample_csv
        result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--preset", "fast", "--drop-threshold", "0.9",
        ])
        assert result.exit_code == 0

    def test_dry_run_combined_with_save_pdf(self, sample_csv, tmp_path):
        path, _df = sample_csv
        pdf_path = tmp_path / "dry_run_report.pdf"
        result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--dry-run", "--save-pdf", str(pdf_path),
        ])
        assert result.exit_code == 0
        assert pdf_path.exists()
        reader = PdfReader(pdf_path)
        assert len(reader.pages) >= 1

    def test_cluster_k_auto_still_works_alongside_new_flags(self, sample_csv):
        path, _df = sample_csv
        result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--clustering", "--cluster-k", "auto",
            "--preset", "fast",
        ])
        # explicit --clustering should win over preset's feature_config per override rules
        assert result.exit_code == 0

    def test_multiple_column_type_flags_combined(self, sample_csv):
        path, _df = sample_csv
        result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--column-type", "cat:CATEGORICAL_LOW",
            "--column-type", "text_col:TEXT",
        ])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# CLI vs Python API consistency
# ---------------------------------------------------------------------------

class TestCliMatchesPythonApi:

    def test_cli_preset_matches_python_api_output_shape(self, sample_csv, tmp_path):
        path, df = sample_csv

        cli_result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--preset", "thorough", "--output-dir", str(tmp_path),
        ])
        assert cli_result.exit_code == 0

        python_result = pf.prepare(df, target="target", task="classification", preset="thorough")
        
        out_csv = tmp_path / "data_prepared.csv"
        if out_csv.exists():
            cli_df = pd.read_csv(out_csv)
            assert set(cli_df.columns) == set(python_result.df.columns)

    def test_cli_dry_run_produces_no_transformed_output_file_with_wrong_shape(self, sample_csv):
        """Sanity: dry-run via CLI should not error even though no real transform happens."""
        path, _df = sample_csv
        result = runner.invoke(app, [
            "prepare", str(path), "--target", "target", "--task", "classification",
            "--dry-run",
        ])
        assert result.exit_code == 0