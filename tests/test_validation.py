"""
Tests for the new task/target mismatch validation logic in
preflight.validation and the _validate_inputs helper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import preflight as pf
from preflight.validation import _validate_inputs, _validate_task_target_match


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def continuous_target_df():
    """300-row DataFrame with a clearly continuous numeric target (prices)."""
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.uniform(0, 100, n),
        "price": rng.uniform(50_000, 500_000, n),  # 300 nearly-unique floats
    })


@pytest.fixture
def binary_target_df():
    """100-row DataFrame with a binary integer target."""
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame({
        "age": rng.normal(35, 10, n),
        "city": rng.choice(["NYC", "LA", "SF"], n),
        "survived": rng.choice([0, 1], n),
    })


@pytest.fixture
def low_cardinality_regression_df():
    """DataFrame with an integer score target (1–5) used for regression."""
    rng = np.random.default_rng(7)
    n = 200
    return pd.DataFrame({
        "feature": rng.normal(0, 1, n),
        "score": rng.choice([1, 2, 3, 4, 5], n),
    })


@pytest.fixture
def string_target_regression_df():
    """DataFrame with a non-numeric target passed as task='regression'."""
    rng = np.random.default_rng(99)
    n = 80
    return pd.DataFrame({
        "feature": rng.normal(0, 1, n),
        "grade": rng.choice(["A", "B", "C"], n),
    })


# ---------------------------------------------------------------------------
# _validate_task_target_match — unit tests
# ---------------------------------------------------------------------------

class TestValidateTaskTargetMatch:
    """Direct unit tests for the heuristic helper."""

    def test_classification_continuous_raises(self, continuous_target_df):
        """300 unique float values + classification must raise."""
        with pytest.raises(ValueError, match="looks continuous"):
            _validate_task_target_match(continuous_target_df["price"], task="classification")

    def test_classification_binary_target_ok(self, binary_target_df):
        """Binary integer target must NOT raise for classification."""
        result = _validate_task_target_match(binary_target_df["survived"], task="classification")
        assert result is None

    def test_regression_low_cardinality_returns_warning(self, low_cardinality_regression_df):
        """Low-cardinality integer target + regression must return a warning string."""
        result = _validate_task_target_match(
            low_cardinality_regression_df["score"], task="regression"
        )
        assert isinstance(result, str)
        assert "regression" in result

    def test_regression_non_numeric_returns_warning(self, string_target_regression_df):
        """Non-numeric target + regression must return a warning string, not raise."""
        result = _validate_task_target_match(
            string_target_regression_df["grade"], task="regression"
        )
        assert isinstance(result, str)
        assert "regression" in result

    def test_regression_continuous_returns_none(self, continuous_target_df):
        """Continuous float target + regression is perfectly fine → None."""
        result = _validate_task_target_match(continuous_target_df["price"], task="regression")
        assert result is None

    def test_classification_exactly_at_threshold_is_ok(self):
        """A target with exactly _UNIQUE_COUNT_HARD_THRESHOLD unique ints must not raise."""
        from preflight.validation import _UNIQUE_COUNT_HARD_THRESHOLD
        n = 1000
        series = pd.Series(list(range(_UNIQUE_COUNT_HARD_THRESHOLD)) * (n // _UNIQUE_COUNT_HARD_THRESHOLD), name="label")
        # 20 unique values but fraction = 20/1000 = 0.02 < 0.05, so should pass
        result = _validate_task_target_match(series, task="classification")
        assert result is None

    def test_classification_error_message_contains_column_name(self, continuous_target_df):
        """Error message must include the column name."""
        with pytest.raises(ValueError, match="price"):
            _validate_task_target_match(continuous_target_df["price"], task="classification")

    def test_classification_error_message_contains_unique_count(self, continuous_target_df):
        """Error message must mention unique count."""
        with pytest.raises(ValueError, match=r"\d+ unique values"):
            _validate_task_target_match(continuous_target_df["price"], task="classification")

    def test_classification_error_message_suggests_regression(self, continuous_target_df):
        """Error message must suggest task='regression'."""
        with pytest.raises(ValueError, match="regression"):
            _validate_task_target_match(continuous_target_df["price"], task="classification")


# ---------------------------------------------------------------------------
# _validate_inputs — unit tests (structural checks)
# ---------------------------------------------------------------------------

class TestValidateInputs:
    """Tests for the centralized structural + task/target mismatch checks."""

    def test_empty_df_raises(self):
        df = pd.DataFrame({"a": [], "b": []})
        with pytest.raises(ValueError, match="empty"):
            _validate_inputs(df, target="b", task="classification")

    def test_missing_target_raises(self, binary_target_df):
        with pytest.raises(ValueError, match="not found"):
            _validate_inputs(binary_target_df, target="nonexistent", task="classification")

    def test_single_column_raises(self):
        df = pd.DataFrame({"only": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least one feature"):
            _validate_inputs(df, target="only", task="classification")

    def test_duplicate_columns_raises(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])
        with pytest.raises(ValueError, match="duplicate"):
            _validate_inputs(df, target="b", task="classification")

    def test_invalid_task_raises(self, binary_target_df):
        with pytest.raises(ValueError, match="task must be"):
            _validate_inputs(binary_target_df, target="survived", task="clustering")

    def test_invalid_model_hint_raises(self, binary_target_df):
        with pytest.raises(ValueError, match="model_hint must be"):
            _validate_inputs(binary_target_df, target="survived", task="classification", model_hint="xgboost")

    def test_no_model_hint_does_not_raise(self, binary_target_df):
        """When model_hint is None, check #6 should be skipped entirely."""
        result = _validate_inputs(binary_target_df, target="survived", task="classification", model_hint=None)
        assert result is None

    def test_returns_none_for_clean_input(self, binary_target_df):
        result = _validate_inputs(binary_target_df, target="survived", task="classification")
        assert result is None

    def test_returns_warning_string_for_low_cardinality_regression(self, low_cardinality_regression_df):
        result = _validate_inputs(low_cardinality_regression_df, target="score", task="regression")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Public API integration — mismatch check fires BEFORE profiler/engineer
# ---------------------------------------------------------------------------

class TestPublicAPIValidation:
    """Verify mismatch checks are raised before any internal stage runs."""

    def test_prepare_raises_before_profiler_for_continuous_classification(
        self, continuous_target_df, monkeypatch
    ):
        """prepare() must raise ValueError before calling run_assembler."""
        import preflight.assembler as asm

        called = []

        def mock_run_assembler(*args, **kwargs):
            called.append(True)
            raise RuntimeError("should not reach assembler")

        monkeypatch.setattr(asm, "run_assembler", mock_run_assembler)

        with pytest.raises(ValueError, match="looks continuous"):
            pf.prepare(continuous_target_df, target="price", task="classification")

        assert called == [], "run_assembler must not be reached when mismatch is detected"

    def test_profile_raises_before_run_profiler_for_continuous_classification(
        self, continuous_target_df, monkeypatch
    ):
        """profile() must raise before run_profiler is invoked."""
        import preflight as _pf
        from preflight import profiler as prof

        called = []

        def mock_run_profiler(*args, **kwargs):
            called.append(True)
            raise RuntimeError("should not reach profiler")

        monkeypatch.setattr(prof, "run_profiler", mock_run_profiler)

        with pytest.raises(ValueError, match="looks continuous"):
            pf.profile(continuous_target_df, target="price", task="classification")

        assert called == [], "run_profiler must not be reached when mismatch is detected"

    def test_clean_raises_for_continuous_classification(self, continuous_target_df):
        with pytest.raises(ValueError, match="looks continuous"):
            pf.clean(continuous_target_df, target="price", task="classification")

    def test_engineer_raises_for_continuous_classification(self, continuous_target_df):
        with pytest.raises(ValueError, match="looks continuous"):
            pf.engineer(continuous_target_df, target="price", task="classification")

    def test_regression_low_cardinality_produces_warning_entry_in_prepare(
        self, low_cardinality_regression_df
    ):
        """regression + low-cardinality target must produce a ReportEntry warning, not raise."""
        result = pf.prepare(low_cardinality_regression_df, target="score", task="regression")
        warning_entries = [
            e for e in result.report.entries
            if e.severity == "warning" and e.action == "task_target_mismatch_warning"
        ]
        assert len(warning_entries) == 1
        assert warning_entries[0].stage == "profiler"
        assert warning_entries[0].column == "score"

    def test_regression_low_cardinality_produces_warning_entry_in_profile(
        self, low_cardinality_regression_df
    ):
        result = pf.profile(low_cardinality_regression_df, target="score", task="regression")
        warning_entries = [
            e for e in result.report.entries
            if e.severity == "warning" and e.action == "task_target_mismatch_warning"
        ]
        assert len(warning_entries) == 1

    def test_regression_low_cardinality_produces_warning_entry_in_clean(
        self, low_cardinality_regression_df
    ):
        result = pf.clean(low_cardinality_regression_df, target="score", task="regression")
        warning_entries = [
            e for e in result.report.entries
            if e.severity == "warning" and e.action == "task_target_mismatch_warning"
        ]
        assert len(warning_entries) == 1

    def test_regression_low_cardinality_produces_warning_entry_in_engineer(
        self, low_cardinality_regression_df
    ):
        result = pf.engineer(low_cardinality_regression_df, target="score", task="regression")
        warning_entries = [
            e for e in result.report.entries
            if e.severity == "warning" and e.action == "task_target_mismatch_warning"
        ]
        assert len(warning_entries) == 1

    def test_clean_input_does_not_produce_mismatch_warning(self, binary_target_df):
        """A proper classification task on a binary target must not emit a mismatch warning."""
        result = pf.profile(binary_target_df, target="survived", task="classification")
        mismatch_entries = [
            e for e in result.report.entries
            if e.action == "task_target_mismatch_warning"
        ]
        assert len(mismatch_entries) == 0

    def test_continuous_regression_does_not_produce_mismatch_warning(self, continuous_target_df):
        """A proper regression task on a continuous target must not emit a mismatch warning."""
        result = pf.profile(continuous_target_df, target="price", task="regression")
        mismatch_entries = [
            e for e in result.report.entries
            if e.action == "task_target_mismatch_warning"
        ]
        assert len(mismatch_entries) == 0
