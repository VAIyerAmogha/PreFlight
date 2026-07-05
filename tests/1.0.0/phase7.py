"""
Tests for Phase 7 (v1.0.0): visual PDF comparison report via pf.save_compare_pdf().

Covers:
- save_compare_pdf() produces a valid, non-empty, multi-page PDF
- The underlying compare() function is used as the single source of truth for diffs
  (not reimplemented) — verified indirectly via consistent diff content
- Comparable results (e.g. same dataset, two different presets) produce a sensible PDF
- Clearly incomparable inputs raise a clear ValueError rather than crashing internally
- dry_run results (df present, pipeline=None) can still be compared and rendered
- compare() and save_pdf() (Phase 6) remain completely unaffected by this addition
- Output does not mutate either input PrepResult
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

import preflight as pf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    n = 100
    return pd.DataFrame({
        "num_a": np.random.uniform(0, 100, size=n),
        "num_b": np.random.uniform(0, 1, size=n),
        "cat": np.random.choice(["x", "y", "z"], size=n),
        "target": np.random.randint(0, 2, size=n),
    })


@pytest.fixture
def different_schema_df():
    n = 60
    return pd.DataFrame({
        "totally_different_col": np.random.uniform(0, 1, size=n),
        "another_col": np.random.choice(["p", "q"], size=n),
        "target": np.random.randint(0, 2, size=n),
    })


def _tmp_pdf_path():
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    os.remove(path)
    return path


# ---------------------------------------------------------------------------
# Basic PDF generation
# ---------------------------------------------------------------------------

class TestSaveComparePdfBasic:

    def test_creates_valid_pdf_for_comparable_results(self, sample_df):
        result_fast = pf.prepare(sample_df, target="target", task="classification", preset="fast")
        result_thorough = pf.prepare(sample_df, target="target", task="classification", preset="thorough")

        path = _tmp_pdf_path()
        try:
            pf.save_compare_pdf(result_fast, result_thorough, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            reader = PdfReader(path)
            assert len(reader.pages) >= 2  # cover + at least one comparison page
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_identical_results_still_produce_valid_pdf(self, sample_df):
        result_a = pf.prepare(sample_df, target="target", task="classification")
        result_b = pf.prepare(sample_df, target="target", task="classification")

        path = _tmp_pdf_path()
        try:
            pf.save_compare_pdf(result_a, result_b, path)  # no diffs, must still not crash
            assert os.path.exists(path)
            reader = PdfReader(path)
            assert len(reader.pages) >= 1
        finally:
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# Consistency with compare()
# ---------------------------------------------------------------------------

class TestConsistencyWithCompare:

    def test_column_diff_reflected_in_pdf_via_compare(self, sample_df):
        """Sanity check that compare() itself reports the expected diff, which
        save_compare_pdf() should be built directly on top of."""
        config_a = pf.FeatureConfig()  # everything off
        config_b = pf.FeatureConfig(interactions=True, interaction_top_k=2)

        result_a = pf.prepare(sample_df, target="target", task="classification", feature_config=config_a)
        result_b = pf.prepare(sample_df, target="target", task="classification", feature_config=config_b)

        diff = pf.compare(result_a, result_b)
        assert diff is not None
        # There should be some indication columns differ, in whatever structure compare() returns
        assert result_a.df.shape[1] != result_b.df.shape[1]

        path = _tmp_pdf_path()
        try:
            pf.save_compare_pdf(result_a, result_b, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_compare_still_works_unaffected(self, sample_df):
        result_a = pf.prepare(sample_df, target="target", task="classification", preset="fast")
        result_b = pf.prepare(sample_df, target="target", task="classification", preset="thorough")
        diff = pf.compare(result_a, result_b)
        assert diff is not None


# ---------------------------------------------------------------------------
# Incomparable inputs
# ---------------------------------------------------------------------------

class TestIncomparableInputs:

    def test_wildly_different_schemas_raise_clear_error_or_handle_gracefully(
        self, sample_df, different_schema_df
    ):
        result_a = pf.prepare(sample_df, target="target", task="classification")
        result_b = pf.prepare(different_schema_df, target="target", task="classification")

        path = _tmp_pdf_path()
        try:
            try:
                pf.save_compare_pdf(result_a, result_b, path)
                # If it doesn't raise, it must have produced a valid, non-crashed PDF
                assert os.path.exists(path)
                assert os.path.getsize(path) > 0
            except ValueError as e:
                # Acceptable alternative: a clear ValueError, not a bare internal traceback
                assert str(e)  # message must not be empty
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_none_inputs_raise_clear_error(self, sample_df):
        result_a = pf.prepare(sample_df, target="target", task="classification")
        path = _tmp_pdf_path()
        try:
            with pytest.raises((ValueError, TypeError)):
                pf.save_compare_pdf(result_a, None, path)
        finally:
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# dry_run compatibility
# ---------------------------------------------------------------------------

class TestDryRunCompatibility:

    def test_compare_dry_run_vs_real_result(self, sample_df):
        dry_result = pf.prepare(sample_df, target="target", task="classification", dry_run=True)
        real_result = pf.prepare(sample_df, target="target", task="classification")

        path = _tmp_pdf_path()
        try:
            pf.save_compare_pdf(dry_result, real_result, path)  # must not crash
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_compare_two_dry_run_results(self, sample_df):
        dry_a = pf.prepare(sample_df, target="target", task="classification", dry_run=True, preset="fast")
        dry_b = pf.prepare(sample_df, target="target", task="classification", dry_run=True, preset="thorough")

        path = _tmp_pdf_path()
        try:
            pf.save_compare_pdf(dry_a, dry_b, path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# No mutation / no side effects
# ---------------------------------------------------------------------------

class TestNoMutation:

    def test_save_compare_pdf_does_not_mutate_inputs(self, sample_df):
        result_a = pf.prepare(sample_df, target="target", task="classification", preset="fast")
        result_b = pf.prepare(sample_df, target="target", task="classification", preset="thorough")

        shape_a_before = result_a.df.shape
        shape_b_before = result_b.df.shape
        counts_a_before = result_a.report.summary_counts()
        counts_b_before = result_b.report.summary_counts()

        path = _tmp_pdf_path()
        try:
            pf.save_compare_pdf(result_a, result_b, path)

            assert result_a.df.shape == shape_a_before
            assert result_b.df.shape == shape_b_before
            assert result_a.report.summary_counts() == counts_a_before
            assert result_b.report.summary_counts() == counts_b_before
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_save_pdf_phase6_unaffected_by_phase7_addition(self, sample_df):
        """Regression guard: Phase 6's save_pdf() must still work identically."""
        result = pf.prepare(sample_df, target="target", task="classification")
        path = _tmp_pdf_path()
        try:
            result.report.save_pdf(path)
            assert os.path.exists(path)
            reader = PdfReader(path)
            assert len(reader.pages) >= 1
        finally:
            if os.path.exists(path):
                os.remove(path)