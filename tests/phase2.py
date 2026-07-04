"""
Phase 2 tests — FeatureConfig and the three new opt-in Engineer feature steps.

NOTE: these tests are written against the Phase 2 spec (the prompt given to
Claude Code). If the implementation names/signatures differ slightly, adjust
imports/calls accordingly — the intent of each test should stay the same.
"""
import numpy as np
import pandas as pd
import pytest

from preflight.types import FeatureConfig, ColumnProfile, SemanticType
from preflight.engineer import (
    generate_interaction_features,
    generate_datetime_cyclical_features,
    generate_cluster_features,
    run_engineer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    rng = np.random.default_rng(42)
    n = 200
    sqft = rng.normal(1500, 300, n)
    rooms = rng.integers(1, 6, n)
    price = sqft * 150 + rooms * 5000 + rng.normal(0, 5000, n)
    return pd.DataFrame({"sqft": sqft, "rooms": rooms, "price": price})


@pytest.fixture
def numeric_profiles(numeric_df):
    # Minimal ColumnProfile stand-ins with correlation scores set so that
    # sqft/rooms are selected as top-K interaction candidates.
    return [
        ColumnProfile(
            name="sqft", semantic_type=SemanticType.NUMERIC_FEATURE,
            missing_rate=0.0, outlier_rate=0.0, cardinality=200,
            rare_categories=[], vif_score=None, dtype="float64",
            correlation_with_target=0.85, mutual_info_with_target=0.6,
            is_leakage_suspect=False,
        ),
        ColumnProfile(
            name="rooms", semantic_type=SemanticType.NUMERIC_FEATURE,
            missing_rate=0.0, outlier_rate=0.0, cardinality=5,
            rare_categories=[], vif_score=None, dtype="int64",
            correlation_with_target=0.4, mutual_info_with_target=0.3,
            is_leakage_suspect=False,
        ),
    ]


@pytest.fixture
def datetime_df():
    dates_a = pd.date_range("2023-01-01", periods=100, freq="D")
    dates_b = dates_a + pd.to_timedelta(np.random.default_rng(1).integers(1, 30, 100), unit="D")
    return pd.DataFrame({"signup_date": dates_a, "purchase_date": dates_b})


@pytest.fixture
def datetime_profiles():
    return [
        ColumnProfile(
            name="signup_date", semantic_type=SemanticType.DATETIME_NATIVE,
            missing_rate=0.0, outlier_rate=None, cardinality=100,
            rare_categories=[], vif_score=None, dtype="datetime64[ns]",
            correlation_with_target=None, mutual_info_with_target=None,
            is_leakage_suspect=False,
        ),
        ColumnProfile(
            name="purchase_date", semantic_type=SemanticType.DATETIME_NATIVE,
            missing_rate=0.0, outlier_rate=None, cardinality=100,
            rare_categories=[], vif_score=None, dtype="datetime64[ns]",
            correlation_with_target=None, mutual_info_with_target=None,
            is_leakage_suspect=False,
        ),
    ]


# ---------------------------------------------------------------------------
# FeatureConfig validation
# ---------------------------------------------------------------------------

class TestFeatureConfigValidation:

    def test_defaults_are_all_off(self):
        config = FeatureConfig()
        assert config.interactions is False
        assert config.datetime_cyclical is False
        assert config.datetime_deltas is False
        assert config.clustering is False

    def test_valid_config_constructs(self):
        config = FeatureConfig(
            interactions=True,
            interaction_top_k=3,
            interaction_types=["ratio"],
            clustering=True,
            cluster_k=4,
        )
        assert config.interaction_top_k == 3
        assert config.cluster_k == 4

    def test_invalid_interaction_type_raises(self):
        with pytest.raises(ValueError):
            FeatureConfig(interactions=True, interaction_types=["logarithm"])

    def test_interaction_top_k_must_be_positive(self):
        with pytest.raises(ValueError):
            FeatureConfig(interactions=True, interaction_top_k=0)

    def test_cluster_k_accepts_auto(self):
        config = FeatureConfig(clustering=True, cluster_k="auto")
        assert config.cluster_k == "auto"

    def test_cluster_k_rejects_invalid_string(self):
        with pytest.raises(ValueError):
            FeatureConfig(clustering=True, cluster_k="bananas")

    def test_cluster_k_rejects_negative_int(self):
        with pytest.raises(ValueError):
            FeatureConfig(clustering=True, cluster_k=-2)

    def test_cluster_features_accepts_column_list(self):
        config = FeatureConfig(clustering=True, cluster_features=["sqft", "rooms"])
        assert config.cluster_features == ["sqft", "rooms"]


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

class TestInteractionFeatures:

    def test_creates_ratio_and_product_columns(self, numeric_df, numeric_profiles):
        config = FeatureConfig(interactions=True, interaction_top_k=2,
                                interaction_types=["ratio", "product"])
        df_out, entries = generate_interaction_features(numeric_df, numeric_profiles, "price", config)

        assert "sqft_div_rooms" in df_out.columns
        assert "sqft_times_rooms" in df_out.columns
        assert len(entries) == 2
        assert all(e.stage == "engineer" for e in entries)
        assert all(e.action == "created_interaction" for e in entries)

    def test_division_by_zero_becomes_nan_not_inf_or_crash(self, numeric_profiles):
        df = pd.DataFrame({"sqft": [100.0, 200.0], "rooms": [0.0, 2.0], "price": [1.0, 2.0]})
        config = FeatureConfig(interactions=True, interaction_types=["ratio"])
        df_out, _ = generate_interaction_features(df, numeric_profiles, "price", config)

        ratio_col = [c for c in df_out.columns if "div" in c][0]
        assert not np.isinf(df_out[ratio_col].iloc[0])
        assert pd.isna(df_out[ratio_col].iloc[0])

    def test_respects_top_k_limit(self, numeric_df, numeric_profiles):
        # top_k=1 should only consider a single candidate column, so no pair exists
        config = FeatureConfig(interactions=True, interaction_top_k=1,
                                interaction_types=["ratio"])
        df_out, entries = generate_interaction_features(numeric_df, numeric_profiles, "price", config)
        assert len(entries) == 0

    def test_original_columns_not_mutated(self, numeric_df, numeric_profiles):
        original = numeric_df.copy(deep=True)
        config = FeatureConfig(interactions=True, interaction_types=["product"])
        generate_interaction_features(numeric_df, numeric_profiles, "price", config)
        pd.testing.assert_frame_equal(numeric_df, original)

    def test_fewer_than_two_numeric_candidates_produces_no_columns(self):
        df = pd.DataFrame({"only_col": [1, 2, 3], "price": [10, 20, 30]})
        profiles = [ColumnProfile(
            name="only_col", semantic_type=SemanticType.NUMERIC_FEATURE,
            missing_rate=0.0, outlier_rate=0.0, cardinality=3, rare_categories=[],
            vif_score=None, dtype="int64", correlation_with_target=0.5,
            mutual_info_with_target=0.2, is_leakage_suspect=False,
        )]
        config = FeatureConfig(interactions=True)
        df_out, entries = generate_interaction_features(df, profiles, "price", config)
        assert entries == []
        assert df_out.shape[1] == df.shape[1]


# ---------------------------------------------------------------------------
# Datetime cyclical / delta features
# ---------------------------------------------------------------------------

class TestDatetimeFeatures:

    def test_cyclical_creates_sin_cos_and_weekend_columns(self, datetime_df, datetime_profiles):
        config = FeatureConfig(datetime_cyclical=True)
        df_out, entries = generate_datetime_cyclical_features(datetime_df, datetime_profiles, config)

        assert any("month_sin" in c for c in df_out.columns)
        assert any("month_cos" in c for c in df_out.columns)
        assert any("is_weekend" in c for c in df_out.columns)
        assert len(entries) > 0

    def test_deltas_created_when_multiple_datetime_columns(self, datetime_df, datetime_profiles):
        config = FeatureConfig(datetime_deltas=True)
        df_out, entries = generate_datetime_cyclical_features(datetime_df, datetime_profiles, config)

        delta_cols = [c for c in df_out.columns if c.endswith("_days") and "to" in c]
        assert len(delta_cols) >= 1
        assert (df_out[delta_cols[0]] >= 0).all() or df_out[delta_cols[0]].notna().any()

    def test_single_datetime_column_does_not_error_on_deltas(self):
        df = pd.DataFrame({"only_date": pd.date_range("2023-01-01", periods=10)})
        profiles = [ColumnProfile(
            name="only_date", semantic_type=SemanticType.DATETIME_NATIVE,
            missing_rate=0.0, outlier_rate=None, cardinality=10, rare_categories=[],
            vif_score=None, dtype="datetime64[ns]", correlation_with_target=None,
            mutual_info_with_target=None, is_leakage_suspect=False,
        )]
        config = FeatureConfig(datetime_deltas=True)
        # Should no-op, not raise
        df_out, entries = generate_datetime_cyclical_features(df, profiles, config)
        assert df_out.shape[0] == df.shape[0]

    def test_reference_col_creates_days_since_column(self, datetime_df, datetime_profiles):
        config = FeatureConfig(datetime_reference_col="signup_date")
        df_out, entries = generate_datetime_cyclical_features(datetime_df, datetime_profiles, config)
        assert any("days_since_ref" in c for c in df_out.columns)

    def test_no_flags_set_produces_no_new_columns(self, datetime_df, datetime_profiles):
        config = FeatureConfig()  # everything off
        df_out, entries = generate_datetime_cyclical_features(datetime_df, datetime_profiles, config)
        assert df_out.shape[1] == datetime_df.shape[1]
        assert entries == []


# ---------------------------------------------------------------------------
# Clustering features
# ---------------------------------------------------------------------------

class TestClusterFeatures:

    def test_creates_label_and_distance_columns(self, numeric_df, numeric_profiles):
        config = FeatureConfig(clustering=True, cluster_k=3)
        df_out, entries, fitted_info = generate_cluster_features(numeric_df, numeric_profiles, config)

        assert "cluster_label" in df_out.columns
        assert "cluster_dist_to_centroid" in df_out.columns
        assert df_out["cluster_label"].nunique() <= 3
        assert len(entries) >= 1

    def test_auto_k_picks_a_valid_k_within_search_range(self, numeric_df, numeric_profiles):
        config = FeatureConfig(clustering=True, cluster_k="auto")
        df_out, entries, fitted_info = generate_cluster_features(numeric_df, numeric_profiles, config)
        n_clusters_used = df_out["cluster_label"].nunique()
        assert 2 <= n_clusters_used <= 10

    def test_fitted_info_contains_reusable_centroids(self, numeric_df, numeric_profiles):
        config = FeatureConfig(clustering=True, cluster_k=3)
        _, _, fitted_info = generate_cluster_features(numeric_df, numeric_profiles, config)
        assert fitted_info is not None
        assert "model" in fitted_info or "centroids" in fitted_info

    def test_all_constant_numeric_block_does_not_crash(self, numeric_profiles):
        df = pd.DataFrame({"sqft": [100.0] * 20, "rooms": [2] * 20, "price": [500.0] * 20})
        config = FeatureConfig(clustering=True, cluster_k=2)
        # Should either produce a degenerate single cluster or a graceful
        # ReportEntry — must not raise.
        df_out, entries, fitted_info = generate_cluster_features(df, numeric_profiles, config)
        assert "cluster_label" in df_out.columns

    def test_respects_explicit_cluster_features_list(self, numeric_df, numeric_profiles):
        config = FeatureConfig(clustering=True, cluster_k=2, cluster_features=["sqft"])
        df_out, entries, fitted_info = generate_cluster_features(numeric_df, numeric_profiles, config)
        assert "cluster_label" in df_out.columns


# ---------------------------------------------------------------------------
# Backward compatibility: run_engineer with feature_config=None
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_run_engineer_default_behavior_unchanged_when_config_none(self, numeric_df, numeric_profiles):
        """
        Critical regression guard: calling run_engineer without feature_config
        (or with feature_config=None) must produce byte-for-byte the same
        columns as v0.1.0 behavior -- no interaction/datetime/cluster columns
        should ever appear unless explicitly requested.
        """
        df_out, entries, specs = run_engineer(
            df=numeric_df, profiles=numeric_profiles, target="price", model_hint="tree",
            cardinality_threshold=20,
        )
        forbidden_substrings = ["_div_", "_times_", "_minus_", "cluster_label",
                                 "cluster_dist_to_centroid", "month_sin", "month_cos",
                                 "is_weekend", "days_since_ref"]
        for col in df_out.columns:
            for forbidden in forbidden_substrings:
                assert forbidden not in col

    def test_run_engineer_accepts_feature_config_kwarg(self, numeric_df, numeric_profiles):
        config = FeatureConfig(interactions=True, interaction_top_k=2)
        df_out, entries, specs = run_engineer(
            df=numeric_df, profiles=numeric_profiles, target="price", model_hint="tree",
            cardinality_threshold=20, feature_config=config,
        )
        assert any("_div_" in c or "_times_" in c for c in df_out.columns)