"""
Phase 3 tests — add_features(): applying FeatureConfig to an already-prepared
PrepResult without rerunning Profiler/Cleaner.

NOTE: written against the Phase 3 spec. Adjust names/signatures if the actual
implementation differs slightly.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from preflight import prepare, profile, add_features
from preflight.types import FeatureConfig, PrepResult


@pytest.fixture
def base_df():
    rng = np.random.default_rng(7)
    n = 150
    sqft = rng.normal(1500, 300, n)
    rooms = rng.integers(1, 6, n)
    price = sqft * 150 + rooms * 5000 + rng.normal(0, 5000, n)
    return pd.DataFrame({"sqft": sqft, "rooms": rooms, "price": price})


@pytest.fixture
def prepared_result(base_df):
    return prepare(base_df, target="price", task="regression", model_hint="tree")


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------

class TestAddFeaturesBasic:

    def test_returns_new_prepresult_with_new_columns(self, prepared_result):
        config = FeatureConfig(interactions=True, interaction_top_k=2)
        new_result = add_features(prepared_result, config)

        assert isinstance(new_result, PrepResult)
        assert new_result is not prepared_result
        assert any("_div_" in c or "_times_" in c for c in new_result.df.columns)

    def test_original_result_is_not_mutated(self, prepared_result):
        original_columns = list(prepared_result.df.columns)
        original_entry_count = len(prepared_result.report.to_dataframe())

        config = FeatureConfig(clustering=True, cluster_k=3)
        add_features(prepared_result, config)

        assert list(prepared_result.df.columns) == original_columns
        assert len(prepared_result.report.to_dataframe()) == original_entry_count

    def test_all_flags_off_returns_unchanged_columns_with_info_entry(self, prepared_result):
        config = FeatureConfig()  # everything off
        new_result = add_features(prepared_result, config)

        assert list(new_result.df.columns) == list(prepared_result.df.columns)
        entries_df = new_result.report.to_dataframe()
        assert (entries_df["severity"] == "info").any()

    def test_multiple_feature_types_can_be_requested_together(self, prepared_result):
        config = FeatureConfig(interactions=True, interaction_top_k=2, clustering=True, cluster_k=3)
        new_result = add_features(prepared_result, config)

        cols = new_result.df.columns
        assert any("_div_" in c or "_times_" in c for c in cols)
        assert "cluster_label" in cols


# ---------------------------------------------------------------------------
# Guardrails / error handling
# ---------------------------------------------------------------------------

class TestAddFeaturesGuardrails:

    def test_raises_if_result_has_no_pipeline(self, base_df):
        # profile() returns a PrepResult with pipeline=None
        partial_result = profile(base_df, target="price", task="regression")
        config = FeatureConfig(interactions=True)

        with pytest.raises(ValueError, match=r"(?i)prepare"):
            add_features(partial_result, config)

    def test_raises_if_profiles_and_target_unavailable(self, prepared_result):
        # Simulate a result whose report doesn't carry profiles/target metadata
        stripped_report = copy.deepcopy(prepared_result.report)
        stripped_report.profiles = None
        stripped_report.target = None
        stripped_result = PrepResult(
            df=prepared_result.df, pipeline=prepared_result.pipeline, report=stripped_report
        )
        config = FeatureConfig(interactions=True)

        with pytest.raises(ValueError, match=r"(?i)profiles|target"):
            add_features(stripped_result, config)

    def test_explicit_profiles_and_target_override_missing_metadata(self, prepared_result, base_df):
        stripped_report = copy.deepcopy(prepared_result.report)
        stripped_report.profiles = None
        stripped_report.target = None
        stripped_result = PrepResult(
            df=prepared_result.df, pipeline=prepared_result.pipeline, report=stripped_report
        )
        config = FeatureConfig(interactions=True, interaction_top_k=2)

        # Should succeed when profiles/target passed explicitly
        new_result = add_features(
            stripped_result, config,
            profiles=prepared_result.report.profiles,
            target="price",
        )
        assert any("_div_" in c or "_times_" in c for c in new_result.df.columns)

    def test_duplicate_column_name_is_skipped_not_overwritten(self, prepared_result):
        # Pre-seed a column name that a feature generator would also produce
        prepared_result.df["cluster_label"] = 0
        config = FeatureConfig(clustering=True, cluster_k=3)

        new_result = add_features(prepared_result, config)
        entries_df = new_result.report.to_dataframe()

        assert (entries_df["action"] == "skipped_duplicate_feature").any()
        # original column values preserved, not silently overwritten
        assert (new_result.df["cluster_label"] == 0).all()


# ---------------------------------------------------------------------------
# Pipeline reproducibility
# ---------------------------------------------------------------------------

class TestAddFeaturesPipelineReuse:

    def test_original_pipeline_object_not_mutated(self, prepared_result):
        original_step_count = len(prepared_result.pipeline.steps)
        config = FeatureConfig(interactions=True, interaction_top_k=2)

        add_features(prepared_result, config)

        assert len(prepared_result.pipeline.steps) == original_step_count

    def test_new_pipeline_has_additional_step(self, prepared_result):
        original_step_count = len(prepared_result.pipeline.steps)
        config = FeatureConfig(interactions=True, interaction_top_k=2)

        new_result = add_features(prepared_result, config)

        assert len(new_result.pipeline.steps) == original_step_count + 1

    def test_new_pipeline_transforms_new_incoming_data_consistently(self, base_df, prepared_result):
        config = FeatureConfig(clustering=True, cluster_k=3)
        new_result = add_features(prepared_result, config)

        # Simulate new incoming data with the same raw schema as the original input
        incoming = base_df.sample(10, random_state=1).drop(columns=["price"])
        transformed = new_result.pipeline.transform(incoming)

        assert "cluster_label" in transformed.columns
        # cluster assignment should be deterministic given frozen centroids
        transformed_again = new_result.pipeline.transform(incoming)
        pd.testing.assert_series_equal(
            transformed["cluster_label"].reset_index(drop=True),
            transformed_again["cluster_label"].reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Composability
# ---------------------------------------------------------------------------

class TestAddFeaturesComposability:

    def test_can_be_called_twice_with_different_configs(self, prepared_result):
        step1 = add_features(prepared_result, FeatureConfig(interactions=True, interaction_top_k=2))
        step2 = add_features(step1, FeatureConfig(clustering=True, cluster_k=3))

        cols = step2.df.columns
        assert any("_div_" in c or "_times_" in c for c in cols)
        assert "cluster_label" in cols
        # both feature sets coexist, neither step overwrote the other
        assert len(step2.pipeline.steps) == len(prepared_result.pipeline.steps) + 2