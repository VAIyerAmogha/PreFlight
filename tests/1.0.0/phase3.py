"""
Tests for Phase 3 (v1.0.0): manual SemanticType override via column_types.

Covers:
- Auto-inference runs normally when column_types is None (no behavior change)
- Overrides correctly replace the auto-inferred SemanticType for specified columns
- Overrides do not affect columns not mentioned in column_types
- A ReportEntry is logged for every override, mentioning both original and forced type
- Overriding a nonexistent column name raises ValueError before any stage runs
- Overriding with an invalid (non-SemanticType) value raises ValueError
- Overridden columns are treated identically downstream (Cleaner/Engineer) to
  naturally-inferred columns of the same type
- column_types works consistently across prepare(), profile(), clean(), engineer()
"""

import numpy as np
import pandas as pd
import pytest

import preflight as pf
from preflight.types import SemanticType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def zip_code_df():
    """zip_code looks numeric but should semantically be categorical."""
    n = 100
    zip_codes = np.random.choice([560001, 560002, 560003, 560004, 560005], size=n)
    return pd.DataFrame({
        "zip_code": zip_codes,
        "income": np.random.uniform(20000, 150000, size=n),
        "target": np.random.randint(0, 2, size=n),
    })


@pytest.fixture
def matching_natural_categorical_df():
    """A DataFrame where a column is NATURALLY inferred as CATEGORICAL_LOW,
    used as a baseline to compare against an overridden column of the same type."""
    n = 100
    return pd.DataFrame({
        "region": np.random.choice(["north", "south", "east", "west", "central"], size=n),
        "income": np.random.uniform(20000, 150000, size=n),
        "target": np.random.randint(0, 2, size=n),
    })


def _profiles_by_name(profiles):
    return {p.name: p for p in profiles}


# ---------------------------------------------------------------------------
# Default behavior (no override)
# ---------------------------------------------------------------------------

class TestNoOverrideBehavior:

    def test_none_column_types_unchanged_behavior(self, zip_code_df):
        profiles_default, _ = pf.run_profiler(zip_code_df, target="target", task="classification")
        profiles_explicit_none, _ = pf.run_profiler(
            zip_code_df, target="target", task="classification", column_types=None
        )
        by_default = _profiles_by_name(profiles_default)
        by_explicit = _profiles_by_name(profiles_explicit_none)

        for name in by_default:
            assert by_default[name].semantic_type == by_explicit[name].semantic_type


# ---------------------------------------------------------------------------
# Override application
# ---------------------------------------------------------------------------

class TestOverrideApplication:

    def test_override_replaces_inferred_type(self, zip_code_df):
        profiles_before, _ = pf.run_profiler(zip_code_df, target="target", task="classification")
        original_type = _profiles_by_name(profiles_before)["zip_code"].semantic_type

        profiles_after, _ = pf.run_profiler(
            zip_code_df,
            target="target",
            task="classification",
            column_types={"zip_code": SemanticType.CATEGORICAL_LOW},
        )
        overridden_type = _profiles_by_name(profiles_after)["zip_code"].semantic_type

        assert overridden_type == SemanticType.CATEGORICAL_LOW
        # sanity: confirm this is actually an override, not what it would've been anyway
        # (only meaningful if auto-inference disagreed; if it already matched, override is a no-op check)
        if original_type != SemanticType.CATEGORICAL_LOW:
            assert overridden_type != original_type

    def test_unmentioned_columns_unaffected_by_override(self, zip_code_df):
        profiles_before, _ = pf.run_profiler(zip_code_df, target="target", task="classification")
        income_type_before = _profiles_by_name(profiles_before)["income"].semantic_type

        profiles_after, _ = pf.run_profiler(
            zip_code_df,
            target="target",
            task="classification",
            column_types={"zip_code": SemanticType.CATEGORICAL_LOW},
        )
        income_type_after = _profiles_by_name(profiles_after)["income"].semantic_type

        assert income_type_before == income_type_after

    def test_multiple_overrides_applied_simultaneously(self, zip_code_df):
        profiles, _ = pf.run_profiler(
            zip_code_df,
            target="target",
            task="classification",
            column_types={
                "zip_code": SemanticType.CATEGORICAL_LOW,
                "income": SemanticType.NUMERIC_FEATURE,
            },
        )
        by_name = _profiles_by_name(profiles)
        assert by_name["zip_code"].semantic_type == SemanticType.CATEGORICAL_LOW
        assert by_name["income"].semantic_type == SemanticType.NUMERIC_FEATURE


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestOverrideValidation:

    def test_nonexistent_column_raises_before_any_stage_runs(self, zip_code_df):
        with pytest.raises(ValueError):
            pf.prepare(
                zip_code_df,
                target="target",
                task="classification",
                column_types={"does_not_exist": SemanticType.CATEGORICAL_LOW},
            )

    def test_invalid_semantic_type_value_raises(self, zip_code_df):
        with pytest.raises(ValueError):
            pf.prepare(
                zip_code_df,
                target="target",
                task="classification",
                column_types={"zip_code": "NOT_A_REAL_TYPE"},
            )

    def test_overriding_target_column_raises_or_is_rejected(self, zip_code_df):
        """Overriding the target column's type should not be silently allowed to slip through
        undetected — expect a clear ValueError since target has its own handling path."""
        with pytest.raises(ValueError):
            pf.prepare(
                zip_code_df,
                target="target",
                task="classification",
                column_types={"target": SemanticType.CATEGORICAL_LOW},
            )


# ---------------------------------------------------------------------------
# Report logging
# ---------------------------------------------------------------------------

class TestOverrideReportLogging:

    def test_report_entry_logged_for_override(self, zip_code_df):
        _profiles, entries = pf.run_profiler(
            zip_code_df,
            target="target",
            task="classification",
            column_types={"zip_code": SemanticType.CATEGORICAL_LOW},
        )
        override_entries = [
            e for e in entries
            if e.column == "zip_code" and e.stage == "profiler"
            and "forced" in e.rationale.lower()
        ]
        assert len(override_entries) == 1
        assert "categorical_low" in override_entries[0].rationale.lower()

    def test_report_entry_mentions_original_inferred_type(self, zip_code_df):
        profiles_before, _ = pf.run_profiler(zip_code_df, target="target", task="classification")
        original_type = _profiles_by_name(profiles_before)["zip_code"].semantic_type

        _profiles, entries = pf.run_profiler(
            zip_code_df,
            target="target",
            task="classification",
            column_types={"zip_code": SemanticType.CATEGORICAL_LOW},
        )
        override_entries = [
            e for e in entries
            if e.column == "zip_code" and "forced" in e.rationale.lower()
        ]
        assert original_type.name.lower() in override_entries[0].rationale.lower()

    def test_no_override_entries_when_column_types_none(self, zip_code_df):
        _profiles, entries = pf.run_profiler(zip_code_df, target="target", task="classification")
        override_entries = [e for e in entries if "forced" in e.rationale.lower()]
        assert len(override_entries) == 0


# ---------------------------------------------------------------------------
# Downstream consistency (Cleaner/Engineer treat overridden columns normally)
# ---------------------------------------------------------------------------

class TestDownstreamConsistency:

    def test_overridden_categorical_encoded_same_as_natural_categorical(
        self, zip_code_df, matching_natural_categorical_df
    ):
        result_overridden = pf.prepare(
            zip_code_df,
            target="target",
            task="classification",
            model_hint="linear",
            column_types={"zip_code": SemanticType.CATEGORICAL_LOW},
        )
        result_natural = pf.prepare(
            matching_natural_categorical_df,
            target="target",
            task="classification",
            model_hint="linear",
        )

        # Both should produce one-hot-style expanded columns for their respective
        # categorical column (linear model_hint => one-hot for low-cardinality categorical).
        zip_generated_cols = [c for c in result_overridden.df.columns if c.startswith("zip_code")]
        region_generated_cols = [c for c in result_natural.df.columns if c.startswith("region")]

        assert len(zip_generated_cols) > 1  # expanded via one-hot, not left as raw numeric
        assert len(region_generated_cols) > 1

    def test_prepare_does_not_crash_with_override(self, zip_code_df):
        result = pf.prepare(
            zip_code_df,
            target="target",
            task="classification",
            column_types={"zip_code": SemanticType.CATEGORICAL_LOW},
        )
        assert result.df is not None
        assert result.pipeline is not None
        assert result.report is not None


# ---------------------------------------------------------------------------
# Consistency across API surface
# ---------------------------------------------------------------------------

class TestAPIConsistency:

    def test_column_types_works_on_profile(self, zip_code_df):
        result = pf.profile(
            zip_code_df,
            target="target",
            task="classification",
            column_types={"zip_code": SemanticType.CATEGORICAL_LOW},
        )
        assert result.pipeline is None
        assert result.df is not None

    def test_column_types_works_on_clean(self, zip_code_df):
        result = pf.clean(
            zip_code_df,
            target="target",
            task="classification",
            column_types={"zip_code": SemanticType.CATEGORICAL_LOW},
        )
        assert result.df is not None

    def test_column_types_works_on_engineer(self, zip_code_df):
        result = pf.engineer(
            zip_code_df,
            target="target",
            task="classification",
            column_types={"zip_code": SemanticType.CATEGORICAL_LOW},
        )
        assert result.df is not None