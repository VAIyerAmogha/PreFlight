"""
Phase 5 tests — CLI flags wiring FeatureConfig into `preflight prepare`.

NOTE: written against the Phase 5 spec. Adjust option names/output paths if
the actual implementation differs slightly.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from preflight.cli import app

runner = CliRunner()


@pytest.fixture
def sample_csv(tmp_path):
    rng = np.random.default_rng(11)
    n = 100
    df = pd.DataFrame({
        "sqft": rng.normal(1500, 300, n),
        "rooms": rng.integers(1, 6, n),
        "price": rng.normal(300000, 50000, n),
    })
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Backward compatibility: no new flags used
# ---------------------------------------------------------------------------

class TestCliBackwardCompatibility:

    def test_prepare_without_feature_flags_produces_same_columns_as_before(self, sample_csv, tmp_path):
        result = runner.invoke(app, [
            "prepare", str(sample_csv),
            "--target", "price",
            "--task", "regression",
            "--output-dir", str(tmp_path),
        ])
        assert result.exit_code == 0

        output_csv = tmp_path / "train_prepared.csv"
        assert output_csv.exists()
        df_out = pd.read_csv(output_csv)
        forbidden = ["_div_", "_times_", "_minus_", "cluster_label",
                     "cluster_dist_to_centroid", "month_sin", "month_cos"]
        for col in df_out.columns:
            for f in forbidden:
                assert f not in col

    def test_help_text_lists_new_flags(self):
        result = runner.invoke(app, ["prepare", "--help"])
        assert result.exit_code == 0
        for flag in ["--interactions", "--datetime-cyclical", "--clustering", "--cluster-k"]:
            assert flag in result.output


# ---------------------------------------------------------------------------
# Feature flags produce expected output
# ---------------------------------------------------------------------------

class TestCliFeatureFlags:

    def test_interactions_flag_adds_columns(self, sample_csv, tmp_path):
        result = runner.invoke(app, [
            "prepare", str(sample_csv),
            "--target", "price",
            "--task", "regression",
            "--output-dir", str(tmp_path),
            "--interactions",
            "--interaction-top-k", "2",
        ])
        assert result.exit_code == 0

        df_out = pd.read_csv(tmp_path / "train_prepared.csv")
        assert any("_div_" in c or "_times_" in c for c in df_out.columns)

    def test_clustering_flag_with_explicit_k(self, sample_csv, tmp_path):
        result = runner.invoke(app, [
            "prepare", str(sample_csv),
            "--target", "price",
            "--task", "regression",
            "--output-dir", str(tmp_path),
            "--clustering",
            "--cluster-k", "3",
        ])
        assert result.exit_code == 0

        df_out = pd.read_csv(tmp_path / "train_prepared.csv")
        assert "cluster_label" in df_out.columns
        assert df_out["cluster_label"].nunique() <= 3

    def test_clustering_flag_with_auto_k(self, sample_csv, tmp_path):
        result = runner.invoke(app, [
            "prepare", str(sample_csv),
            "--target", "price",
            "--task", "regression",
            "--output-dir", str(tmp_path),
            "--clustering",
            "--cluster-k", "auto",
        ])
        assert result.exit_code == 0
        df_out = pd.read_csv(tmp_path / "train_prepared.csv")
        assert "cluster_label" in df_out.columns

    def test_datetime_flags_do_not_error_when_no_datetime_columns_present(self, sample_csv, tmp_path):
        # sample_csv has no datetime columns -- should no-op gracefully, not crash
        result = runner.invoke(app, [
            "prepare", str(sample_csv),
            "--target", "price",
            "--task", "regression",
            "--output-dir", str(tmp_path),
            "--datetime-cyclical",
            "--datetime-deltas",
        ])
        assert result.exit_code == 0

    def test_multiple_feature_flags_combined(self, sample_csv, tmp_path):
        result = runner.invoke(app, [
            "prepare", str(sample_csv),
            "--target", "price",
            "--task", "regression",
            "--output-dir", str(tmp_path),
            "--interactions",
            "--clustering",
            "--cluster-k", "2",
        ])
        assert result.exit_code == 0
        df_out = pd.read_csv(tmp_path / "train_prepared.csv")
        assert "cluster_label" in df_out.columns
        assert any("_div_" in c or "_times_" in c for c in df_out.columns)


# ---------------------------------------------------------------------------
# Input parsing / error handling
# ---------------------------------------------------------------------------

class TestCliFlagParsingErrors:

    def test_invalid_cluster_k_gives_clear_error_not_raw_traceback(self, sample_csv):
        result = runner.invoke(app, [
            "prepare", str(sample_csv),
            "--target", "price",
            "--task", "regression",
            "--clustering",
            "--cluster-k", "xyz",
        ])
        assert result.exit_code != 0
        assert "cluster-k" in result.output.lower()
        assert "traceback" not in result.output.lower()

    def test_interaction_types_comma_parsing(self, sample_csv, tmp_path):
        result = runner.invoke(app, [
            "prepare", str(sample_csv),
            "--target", "price",
            "--task", "regression",
            "--output-dir", str(tmp_path),
            "--interactions",
            "--interaction-types", "ratio",
        ])
        assert result.exit_code == 0
        df_out = pd.read_csv(tmp_path / "train_prepared.csv")
        # only ratio requested -> no product columns should appear
        assert any("_div_" in c for c in df_out.columns)
        assert not any("_times_" in c for c in df_out.columns)

    def test_invalid_interaction_type_gives_clear_error(self, sample_csv):
        result = runner.invoke(app, [
            "prepare", str(sample_csv),
            "--target", "price",
            "--task", "regression",
            "--interactions",
            "--interaction-types", "logarithm",
        ])
        assert result.exit_code != 0
        assert "traceback" not in result.output.lower()


# ---------------------------------------------------------------------------
# Report / JSON output includes new entries
# ---------------------------------------------------------------------------

class TestCliReportOutputIncludesNewFeatures:

    def test_report_json_contains_feature_entries(self, sample_csv, tmp_path):
        result = runner.invoke(app, [
            "prepare", str(sample_csv),
            "--target", "price",
            "--task", "regression",
            "--output-dir", str(tmp_path),
            "--interactions",
            "--interaction-top-k", "2",
        ])
        assert result.exit_code == 0

        report_json = tmp_path / "train_report.json"
        assert report_json.exists()
        with open(report_json) as f:
            report_data = json.load(f)

        entries = report_data.get("entries", report_data) if isinstance(report_data, dict) else report_data
        actions = [e.get("action") for e in entries] if isinstance(entries, list) else []
        assert any(a == "created_interaction" for a in actions)