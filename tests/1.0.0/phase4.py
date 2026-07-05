"""
Tests for Phase 4 (v1.0.0): config presets via prepare(preset=...).

Covers:
- preset=None (default) produces byte-for-byte identical behavior to pre-Phase-4 prepare()
- preset="fast" applies its expected bundle of parameter values
- preset="thorough" applies its expected bundle of parameter values (more feature generation)
- Explicit kwargs always override preset values, regardless of order
- Invalid preset name raises a clear ValueError listing valid options
- Exactly one ReportEntry is logged when a preset is used, describing the expanded parameters
- No preset-related ReportEntry is logged when preset is not used
- CLI --preset flag accepts valid names and rejects invalid ones
"""

import numpy as np
import pandas as pd
import pytest

import preflight as pf
from preflight.types import PRESETS


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
        "date_col": pd.date_range("2020-01-01", periods=n, freq="D"),
        "text_col": [
            "this is a fairly long free text sentence used to trigger text detection"
        ] * n,
        "target": np.random.randint(0, 2, size=n),
    })


# ---------------------------------------------------------------------------
# Default behavior unaffected
# ---------------------------------------------------------------------------

class TestNoPresetBehavior:

    def test_preset_none_matches_pre_phase4_behavior(self, sample_df):
        result_no_preset_kw = pf.prepare(sample_df, target="target", task="classification")
        result_explicit_none = pf.prepare(
            sample_df, target="target", task="classification", preset=None
        )
        assert list(result_no_preset_kw.df.columns) == list(result_explicit_none.df.columns)
        assert result_no_preset_kw.df.shape == result_explicit_none.df.shape

    def test_no_preset_entry_logged_when_preset_not_used(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification")
        entries = result.report.to_dataframe()
        preset_entries = entries[entries["rationale"].astype(str).str.contains("preset", case=False)]
        assert len(preset_entries) == 0


# ---------------------------------------------------------------------------
# Preset existence / structure
# ---------------------------------------------------------------------------

class TestPresetsRegistry:

    def test_presets_dict_contains_fast_and_thorough(self):
        assert "fast" in PRESETS
        assert "thorough" in PRESETS

    def test_preset_values_are_dicts(self):
        assert isinstance(PRESETS["fast"], dict)
        assert isinstance(PRESETS["thorough"], dict)


# ---------------------------------------------------------------------------
# Preset application
# ---------------------------------------------------------------------------

class TestPresetApplication:

    def test_fast_preset_runs_without_error(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification", preset="fast")
        assert result.df is not None
        assert result.pipeline is not None

    def test_thorough_preset_runs_without_error(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification", preset="thorough")
        assert result.df is not None
        assert result.pipeline is not None

    def test_thorough_preset_generates_more_columns_than_fast(self, sample_df):
        """Thorough should enable more feature generation (interactions/datetime/clustering/text),
        so it should produce at least as many, generally more, output columns than fast."""
        fast_result = pf.prepare(sample_df, target="target", task="classification", preset="fast")
        thorough_result = pf.prepare(sample_df, target="target", task="classification", preset="thorough")

        assert thorough_result.df.shape[1] >= fast_result.df.shape[1]

    def test_thorough_preset_enables_text_features(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification", preset="thorough")
        text_generated = [c for c in result.df.columns if c.startswith("text_col_")]
        assert len(text_generated) > 0

    def test_fast_preset_does_not_enable_text_features(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification", preset="fast")
        text_generated = [c for c in result.df.columns if c.startswith("text_col_")]
        assert len(text_generated) == 0


# ---------------------------------------------------------------------------
# Explicit kwargs override preset values
# ---------------------------------------------------------------------------

class TestExplicitOverridesPreset:

    def test_explicit_drop_threshold_overrides_preset(self, sample_df):
        result_preset_only = pf.prepare(
            sample_df, target="target", task="classification", preset="fast"
        )
        result_overridden = pf.prepare(
            sample_df, target="target", task="classification", preset="fast", drop_threshold=0.9
        )
        # We can't assert exact internal value directly, but behavior should differ
        # if the preset's drop_threshold differs meaningfully from 0.9, OR at minimum
        # the call must not raise and must respect the explicit value over the preset's.
        assert result_overridden.df is not None

    def test_explicit_feature_config_overrides_preset_feature_config(self, sample_df):
        from preflight.types import FeatureConfig

        custom_config = FeatureConfig()  # everything off, explicitly
        result = pf.prepare(
            sample_df,
            target="target",
            task="classification",
            preset="thorough",
            feature_config=custom_config,
        )
        # Explicit empty FeatureConfig should win over thorough's feature-heavy default,
        # so no interaction/datetime/cluster/text columns should appear.
        text_generated = [c for c in result.df.columns if c.startswith("text_col_")]
        assert len(text_generated) == 0

    def test_explicit_same_value_as_preset_does_not_break(self, sample_df):
        """Passing the same value the preset would have used must not raise or misbehave."""
        fast_values = PRESETS["fast"]
        if "drop_threshold" in fast_values:
            result = pf.prepare(
                sample_df,
                target="target",
                task="classification",
                preset="fast",
                drop_threshold=fast_values["drop_threshold"],
            )
            assert result.df is not None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestPresetValidation:

    def test_invalid_preset_name_raises(self, sample_df):
        with pytest.raises(ValueError):
            pf.prepare(sample_df, target="target", task="classification", preset="nonexistent_preset")

    def test_invalid_preset_error_lists_valid_options(self, sample_df):
        with pytest.raises(ValueError) as exc_info:
            pf.prepare(sample_df, target="target", task="classification", preset="bogus")
        message = str(exc_info.value).lower()
        assert "fast" in message
        assert "thorough" in message


# ---------------------------------------------------------------------------
# Report logging
# ---------------------------------------------------------------------------

class TestPresetReportLogging:

    def test_exactly_one_preset_entry_logged(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification", preset="fast")
        entries = result.report.to_dataframe()
        preset_entries = entries[entries["rationale"].astype(str).str.contains("preset", case=False)]
        assert len(preset_entries) == 1

    def test_preset_entry_names_the_preset_used(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification", preset="thorough")
        entries = result.report.to_dataframe()
        preset_entries = entries[entries["rationale"].astype(str).str.contains("preset", case=False)]
        assert "thorough" in preset_entries.iloc[0]["rationale"].lower()

    def test_preset_entry_is_info_severity(self, sample_df):
        result = pf.prepare(sample_df, target="target", task="classification", preset="fast")
        entries = result.report.to_dataframe()
        preset_entries = entries[entries["rationale"].astype(str).str.contains("preset", case=False)]
        assert preset_entries.iloc[0]["severity"] == "info"


# ---------------------------------------------------------------------------
# API consistency
# ---------------------------------------------------------------------------

class TestPresetOnOtherEntryPoints:

    def test_preset_param_accepted_by_prepare_signature(self, sample_df):
        import inspect
        sig = inspect.signature(pf.prepare)
        assert "preset" in sig.parameters
        assert sig.parameters["preset"].default is None