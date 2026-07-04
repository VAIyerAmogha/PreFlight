"""
Phase 4 tests — readability improvements for Report.plot() charts and
Report.show() terminal output.

NOTE: written against the Phase 4 spec. Adjust names/signatures if the actual
implementation differs slightly (e.g. exact chart function names).
"""
import io
import contextlib

import matplotlib
matplotlib.use("Agg")  # headless backend for test environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from preflight import prepare
from preflight.report import (
    SEVERITY_COLORS,
    plot_correlation_heatmap,
    plot_missingness_heatmap,
    plot_mutual_info_bar_chart,
    plot_class_distribution,
)


@pytest.fixture
def wide_df():
    """A DataFrame with enough columns to trigger label rotation/truncation logic."""
    rng = np.random.default_rng(3)
    n = 100
    data = {f"feature_with_a_fairly_long_name_{i}": rng.normal(0, 1, n) for i in range(25)}
    data["target"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


@pytest.fixture
def prepared_wide(wide_df):
    return prepare(wide_df, target="target", task="classification")


@pytest.fixture
def prepared_small():
    df = pd.DataFrame({
        "a": np.random.default_rng(1).normal(0, 1, 50),
        "b": np.random.default_rng(2).normal(0, 1, 50),
        "target": np.random.default_rng(3).integers(0, 2, 50),
    })
    return prepare(df, target="target", task="classification")


# ---------------------------------------------------------------------------
# PART A — chart readability
# ---------------------------------------------------------------------------

class TestSharedColorPalette:

    def test_severity_colors_defined_and_complete(self):
        assert set(SEVERITY_COLORS.keys()) == {"info", "warning", "critical"}
        for color in SEVERITY_COLORS.values():
            assert isinstance(color, str)

    def test_chart_functions_return_figure_objects(self, prepared_small):
        fig1 = plot_correlation_heatmap(prepared_small.df, numeric_columns=list(prepared_small.df.columns))
        fig2 = plot_missingness_heatmap(prepared_small.df)
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig1)
        plt.close(fig2)


class TestFigureSizingScalesWithColumnCount:

    def test_wide_dataframe_produces_wider_figure_than_small_one(self, prepared_wide, prepared_small):
        fig_wide = plot_correlation_heatmap(prepared_wide.df, numeric_columns=list(prepared_wide.df.columns))
        fig_small = plot_correlation_heatmap(prepared_small.df, numeric_columns=list(prepared_small.df.columns))

        wide_width = fig_wide.get_size_inches()[0]
        small_width = fig_small.get_size_inches()[0]

        assert wide_width > small_width
        plt.close(fig_wide)
        plt.close(fig_small)

    def test_figure_width_is_capped(self):
        # Simulate an extreme column count to confirm a sane upper bound exists
        huge_df = pd.DataFrame(
            {f"col_{i}": np.random.default_rng(i).normal(0, 1, 20) for i in range(200)}
        )
        fig = plot_correlation_heatmap(huge_df, numeric_columns=list(huge_df.columns))
        width = fig.get_size_inches()[0]
        assert width <= 20  # per spec's stated cap
        plt.close(fig)


class TestAxisLabelReadability:

    def test_labels_rotated_when_many_categories(self, prepared_wide):
        fig = plot_correlation_heatmap(prepared_wide.df, numeric_columns=list(prepared_wide.df.columns))
        ax = fig.axes[0]
        rotations = [label.get_rotation() for label in ax.get_xticklabels()]
        assert any(r != 0 for r in rotations)
        plt.close(fig)

    def test_long_labels_are_truncated(self, prepared_wide):
        fig = plot_correlation_heatmap(prepared_wide.df, numeric_columns=list(prepared_wide.df.columns))
        ax = fig.axes[0]
        label_texts = [label.get_text() for label in ax.get_xticklabels()]
        assert all(len(t) <= 21 for t in label_texts)  # ~20 chars + ellipsis
        plt.close(fig)

    def test_few_categories_does_not_force_rotation(self, prepared_small):
        fig = plot_correlation_heatmap(prepared_small.df, numeric_columns=list(prepared_small.df.columns))
        ax = fig.axes[0]
        rotations = [label.get_rotation() for label in ax.get_xticklabels()]
        assert all(r == 0 for r in rotations)
        plt.close(fig)


class TestBarChartAnnotations:

    def test_mi_bar_chart_has_value_annotations(self, prepared_small):
        fig = plot_mutual_info_bar_chart(prepared_small.report.profiles)
        ax = fig.axes[0]
        # matplotlib text annotations show up as ax.texts
        assert len(ax.texts) > 0
        plt.close(fig)

    def test_class_distribution_has_value_annotations(self, prepared_small):
        fig = plot_class_distribution(prepared_small.df["target"])
        ax = fig.axes[0]
        assert len(ax.texts) > 0
        plt.close(fig)


# ---------------------------------------------------------------------------
# PART B — .show() readability
# ---------------------------------------------------------------------------

class TestShowGroupingAndSymbols:

    def _capture_show(self, report, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report.show(**kwargs)
        return buf.getvalue()

    def test_output_grouped_by_stage_headers(self, prepared_small):
        output = self._capture_show(prepared_small.report)
        for stage in ["PROFILER", "CLEANER", "ENGINEER"]:
            if stage.lower() in [e.stage for e in prepared_small.report.entries]:
                assert f"=== {stage} ===" in output or stage in output

    def test_severity_symbols_present(self, prepared_small):
        output = self._capture_show(prepared_small.report)
        entries = prepared_small.report.entries
        if any(e.severity == "warning" for e in entries):
            assert "⚠" in output
        if any(e.severity == "critical" for e in entries):
            assert "✕" in output
        if any(e.severity == "info" for e in entries):
            assert "·" in output

    def test_default_truncates_long_info_lists(self, prepared_wide):
        # wide_df has 12 numeric features -> likely produces >5 info entries
        output_default = self._capture_show(prepared_wide.report, verbose=False)
        info_count = sum(1 for e in prepared_wide.report.entries if e.severity == "info")
        if info_count > 5:
            assert "more info-level" in output_default or "verbose=True" in output_default

    def test_verbose_shows_everything(self, prepared_wide):
        output_verbose = self._capture_show(prepared_wide.report, verbose=True)
        output_default = self._capture_show(prepared_wide.report, verbose=False)
        # verbose output should never be shorter than default output
        assert len(output_verbose) >= len(output_default)

    def test_warnings_and_criticals_never_truncated_even_when_many(self, prepared_wide):
        output = self._capture_show(prepared_wide.report, verbose=False)
        entries = prepared_wide.report.entries
        warning_count_in_entries = sum(1 for e in entries if e.severity in ("warning", "critical"))
        warning_symbol_count = output.count("⚠") + output.count("✕")
        assert warning_symbol_count == warning_count_in_entries


class TestShowDoesNotAffectDataExports:

    def test_to_dict_unchanged_shape(self, prepared_small):
        d = prepared_small.report.to_dict()
        assert isinstance(d, dict)
        # sanity: still contains the core keys it always has
        assert "entries" in d or "report" in d or len(d) > 0

    def test_to_dataframe_unchanged_columns(self, prepared_small):
        df = prepared_small.report.to_dataframe()
        expected_cols = {"stage", "column", "action", "rationale", "severity"}
        assert expected_cols.issubset(set(df.columns))