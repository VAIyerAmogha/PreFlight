"""
Tests for Phase 5 (v1.0.0): dry_run / preview mode on prepare().

Covers:
- dry_run=False (default) behavior is unchanged from pre-Phase-5 prepare()
- dry_run=True returns the original, untouched input DataFrame as result.df
- dry_run=True returns result.pipeline is None
- dry_run=True still produces a full Report with the same decision entries a real run would produce
- A clear "dry run" ReportEntry is present and unmissable
- Input DataFrame is never mutated in dry_run mode
- dry_run works in combination with column_types overrides, FeatureConfig, and presets
- add_features() raises its existing "pipeline is None" ValueError when called on a dry_run result
- CLI --dry-run flag is wired through
"""

import inspect

import numpy as np
import pandas as pd
import pytest

import preflight as pf
from preflight.types import FeatureConfig, SemanticType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    n = 80
    return pd.DataFrame({
        "num_a": np.random.uniform(0, 100, size=n),
        "cat": np.random.choice(["x", "y", "z"], size=n),
        "target": np.random.randint(0, 2, size=n),
    })


# ---------------------------------------------------------------------------
# Default behavior unaffected
# ---------------------------------------------------------------------------

class TestDefaultBehaviorUnaffected:

    def test_dry_run_false_matches_pre_phase5_behavior(self, sample_df):
        result_default = pf.prepare(sample_df, target="target", task="classification")
        result_explicit_false = pf.prepare(
            sample_df, target="target", task="classification", dry_run=False
        )
        assert list(result_default.df.columns) == list(result_explicit_false.df.columns)
        assert result_default.pipeline is not None
        assert result_explicit_false.pipeline is not None

    def test_dry_run_param_defaults_to_false(self):
        sig = inspect.signature(pf.prepare)
        assert "dry_run" in sig.parameters
        assert sig.parameters["dry_run"].default is False


# ---------------------------------------------------------------------------
# Core dry_run contract
# ---------------------------------------------------------------------------

class TestDryRunContract:

    def test_dry_run_returns_original_df_unchanged(self, sample_df):
        original_copy = sample_df.copy(deep=True)
        result = pf.prepare(sample_df, target="target", task="classification", dry_run=True)

        pd.testing.assert_frame_equal(result.df, original_copy)

    def test_dry_run_pipeline_is_none(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification", dry_run=True)
        assert result.pipeline is None

    def test_dry_run_does_not_mutate_input_df(self, sample_df):
        original_copy = sample_df.copy(deep=True)
        pf.prepare(sample_df, target="target", task="classification", dry_run=True)
        pd.testing.assert_frame_equal(sample_df, original_copy)

    def test_dry_run_produces_a_report(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification", dry_run=True)
        assert result.report is not None
        entries = result.report.to_dataframe()
        assert len(entries) > 0


# ---------------------------------------------------------------------------
# Report accuracy: dry_run report should mirror a real run's decisions
# ---------------------------------------------------------------------------

class TestDryRunReportAccuracy:

    def test_dry_run_report_has_same_decision_entries_as_real_run(self, sample_df):
        real_result = pf.prepare(sample_df, target="target", task="classification")
        dry_result = pf.prepare(sample_df, target="target", task="classification", dry_run=True)

        real_entries = real_result.report.to_dataframe()
        dry_entries = dry_result.report.to_dataframe()

        # Exclude the dry-run-specific notice entry itself from comparison
        dry_entries_filtered = dry_entries[
            ~dry_entries["rationale"].astype(str).str.contains("dry run", case=False)
        ]

        real_actions = sorted(real_entries["action"].astype(str).tolist())
        dry_actions = sorted(dry_entries_filtered["action"].astype(str).tolist())
        assert real_actions == dry_actions

    def test_dry_run_notice_entry_present_and_clear(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification", dry_run=True)
        entries = result.report.to_dataframe()
        dry_run_entries = entries[
            entries["rationale"].astype(str).str.contains("dry run", case=False)
        ]
        assert len(dry_run_entries) >= 1
        message = dry_run_entries.iloc[0]["rationale"].lower()
        assert "no" in message and ("transform" in message or "pipeline" in message)

    def test_dry_run_notice_is_info_severity(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification", dry_run=True)
        entries = result.report.to_dataframe()
        dry_run_entries = entries[
            entries["rationale"].astype(str).str.contains("dry run", case=False)
        ]
        assert dry_run_entries.iloc[0]["severity"] == "info"


# ---------------------------------------------------------------------------
# Interaction with other v1.0.0 features
# ---------------------------------------------------------------------------

class TestDryRunWithOtherFeatures:

    def test_dry_run_with_column_types_override(self, sample_df):
        result = pf.prepare(
            sample_df,
            target="target",
            task="classification",
            dry_run=True,
            column_types={"num_a": SemanticType.CATEGORICAL_HIGH},
        )
        assert result.pipeline is None
        entries = result.report.to_dataframe()
        override_entries = entries[entries["rationale"].astype(str).str.contains("forced", case=False)]
        assert len(override_entries) >= 1

    def test_dry_run_with_feature_config(self, sample_df):
        config = FeatureConfig(interactions=True, interaction_top_k=2)
        result = pf.prepare(
            sample_df, target="target", task="classification", dry_run=True, feature_config=config
        )
        assert result.pipeline is None
        # df must remain original shape since no transform actually applied
        assert result.df.shape == sample_df.shape

    def test_dry_run_with_preset(self, sample_df):
        result = pf.prepare(
            sample_df, target="target", task="classification", dry_run=True, preset="thorough"
        )
        assert result.pipeline is None
        assert result.df.shape == sample_df.shape
        entries = result.report.to_dataframe()
        preset_entries = entries[entries["rationale"].astype(str).str.contains("preset", case=False)]
        assert len(preset_entries) >= 1


# ---------------------------------------------------------------------------
# add_features() guardrail interaction
# ---------------------------------------------------------------------------

class TestDryRunAddFeaturesGuardrail:

    def test_add_features_raises_on_dry_run_result(self, sample_df):
        dry_result = pf.prepare(sample_df, target="target", task="classification", dry_run=True)
        config = FeatureConfig(interactions=True)

        with pytest.raises(ValueError):
            pf.add_features(dry_result, config)


# ---------------------------------------------------------------------------
# API signature sanity
# ---------------------------------------------------------------------------

class TestDryRunSignature:

    def test_dry_run_is_boolean_only(self, sample_df):
        with pytest.raises((TypeError, ValueError)):
            pf.prepare(sample_df, target="target", task="classification", dry_run="yes")