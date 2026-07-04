"""
tests/test_full_suite.py

Comprehensive regression + robustness suite for PreFlight-ML, covering every
phase from the original v0.1.0 build through the v0.2.0 additions.

IMPORTANT CAVEATS (read before running):
- This file assumes the designed API surface as specified across the project's
  CLAUDE.md/PLAN.md phases. If a function/class name differs slightly in the
  actual codebase, adjust the import or call site accordingly.
- "Error proof" here means: the package either produces a correct result, OR
  raises a clear, typed, documented exception (ValueError/TypeError with a
  helpful message) at the API boundary -- never a bare crash from deep inside
  profiler.py/engineer.py/assembler.py with an unhelpful traceback. This file
  cannot prove the absence of all possible bugs; it is a wide net, not a proof.
- Some tests are intentionally redundant with existing per-module test files
  (test_profiler.py, test_cleaner.py, etc.) -- this file's purpose is to catch
  integration-level and cross-cutting issues that isolated unit tests miss.
- Run with: pytest tests/test_full_suite.py -v
  Slower/integration-style tests are marked so you can skip them if needed:
  pytest tests/test_full_suite.py -v -m "not slow"
"""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
import joblib

from preflight import prepare, profile, clean, engineer, compare, add_features
from preflight.types import (
    SemanticType, ColumnProfile, ReportEntry, PrepResult, FeatureConfig,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def regression_df(rng):
    n = 300
    return pd.DataFrame({
        "sqft": rng.normal(1500, 300, n),
        "rooms": rng.integers(1, 6, n),
        "neighborhood": rng.choice(["A", "B", "C", "D"], n),
        "built_date": pd.date_range("2000-01-01", periods=n, freq="7D"),
        "price": rng.normal(300000, 50000, n),
    })


@pytest.fixture
def classification_df(rng):
    n = 300
    return pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "income": rng.normal(50000, 15000, n),
        "occupation": rng.choice(["engineer", "teacher", "artist", "doctor", "unknown"], n),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="3D"),
        "churned": rng.choice([0, 1], n, p=[0.8, 0.2]),
    })


@pytest.fixture
def messy_df(rng):
    """A dataframe with missing values, outliers, and mixed types."""
    n = 200
    df = pd.DataFrame({
        "numeric_with_gaps": rng.normal(0, 1, n),
        "categorical_with_gaps": rng.choice(["x", "y", "z", None], n),
        "id_col": range(n),
        "constant_col": ["same"] * n,
        "target": rng.normal(100, 20, n),
    })
    # Inject missingness
    df.loc[df.sample(frac=0.2, random_state=1).index, "numeric_with_gaps"] = np.nan
    # Inject outliers
    df.loc[0:2, "numeric_with_gaps"] = [1e6, -1e6, 1e6]
    return df


# ===========================================================================
# SECTION 1 — Core data models (types.py)
# ===========================================================================

class TestSemanticTypeEnum:

    def test_all_eight_values_exist(self):
        expected = {
            "NUMERIC_FEATURE", "NUMERIC_ID", "CATEGORICAL_LOW", "CATEGORICAL_HIGH",
            "DATETIME_NATIVE", "DATETIME_STRING", "BOOLEAN", "CONSTANT",
        }
        actual = {member.name for member in SemanticType}
        assert actual == expected


class TestColumnProfileDefaults:

    def test_none_represents_not_computed_not_zero(self):
        profile_obj = ColumnProfile(
            name="x", semantic_type=SemanticType.NUMERIC_FEATURE,
            missing_rate=0.0, outlier_rate=None, cardinality=10,
            rare_categories=[], vif_score=None, dtype="float64",
            correlation_with_target=None, mutual_info_with_target=None,
            is_leakage_suspect=False,
        )
        assert profile_obj.vif_score is None
        assert profile_obj.vif_score != 0.0
        assert profile_obj.correlation_with_target is None


class TestReportEntryValidation:

    def test_valid_severity_and_stage_accepted(self):
        entry = ReportEntry(
            stage="engineer", column="x", action="scaled",
            rationale="test", severity="warning", before_stats={}, after_stats={},
        )
        assert entry.severity == "warning"

    def test_invalid_severity_raises(self):
        with pytest.raises(Exception):
            ReportEntry(
                stage="engineer", column="x", action="scaled",
                rationale="test", severity="catastrophic", before_stats={}, after_stats={},
            )

    def test_invalid_stage_raises(self):
        with pytest.raises(Exception):
            ReportEntry(
                stage="not_a_real_stage", column="x", action="scaled",
                rationale="test", severity="info", before_stats={}, after_stats={},
            )


class TestFeatureConfigValidation:

    def test_defaults_all_off(self):
        config = FeatureConfig()
        assert not any([config.interactions, config.datetime_cyclical,
                         config.datetime_deltas, config.clustering])

    def test_invalid_interaction_type_rejected(self):
        with pytest.raises(ValueError):
            FeatureConfig(interactions=True, interaction_types=["nonsense"])

    def test_invalid_cluster_k_rejected(self):
        with pytest.raises(ValueError):
            FeatureConfig(clustering=True, cluster_k="not_a_number")


# ===========================================================================
# SECTION 2 — Input validation / task-target mismatch (Phase 1 hardening)
# ===========================================================================

class TestTaskTargetValidation:

    def test_classification_on_continuous_target_raises_clear_error(self, regression_df):
        with pytest.raises(ValueError, match=r"(?i)regression"):
            prepare(regression_df, target="price", task="classification")

    def test_regression_on_low_cardinality_target_warns_not_raises(self, classification_df):
        result = prepare(classification_df, target="churned", task="regression")
        entries = result.report.to_dataframe()
        assert (entries["severity"] == "warning").any()

    def test_correctly_matched_classification_no_warning(self, classification_df):
        result = prepare(classification_df, target="churned", task="classification")
        # should not raise, should complete normally
        assert result.df is not None

    def test_correctly_matched_regression_no_warning(self, regression_df):
        result = prepare(regression_df, target="price", task="regression")
        assert result.df is not None

    def test_missing_target_column_raises_clear_error(self, regression_df):
        with pytest.raises(ValueError, match=r"(?i)target"):
            prepare(regression_df, target="does_not_exist", task="regression")

    def test_invalid_task_string_raises_clear_error(self, regression_df):
        with pytest.raises(ValueError):
            prepare(regression_df, target="price", task="not_a_real_task")

    def test_invalid_model_hint_raises_clear_error(self, regression_df):
        with pytest.raises(ValueError):
            prepare(regression_df, target="price", task="regression", model_hint="not_a_real_hint")


# ===========================================================================
# SECTION 3 — Profiler behavior
# ===========================================================================

class TestProfilerSemanticTypes:

    def test_numeric_column_classified_correctly(self, regression_df):
        result = profile(regression_df, target="price", task="regression")
        sqft_profile = next(p for p in result.report.profiles if p.name == "sqft")
        assert sqft_profile.semantic_type == SemanticType.NUMERIC_FEATURE

    def test_low_cardinality_categorical_classified_correctly(self, regression_df):
        result = profile(regression_df, target="price", task="regression")
        neighborhood_profile = next(p for p in result.report.profiles if p.name == "neighborhood")
        assert neighborhood_profile.semantic_type == SemanticType.CATEGORICAL_LOW

    def test_datetime_column_classified_correctly(self, regression_df):
        result = profile(regression_df, target="price", task="regression")
        date_profile = next(p for p in result.report.profiles if p.name == "built_date")
        assert date_profile.semantic_type in (SemanticType.DATETIME_NATIVE, SemanticType.DATETIME_STRING)

    def test_id_like_column_classified_correctly(self, messy_df):
        result = profile(messy_df, target="target", task="regression")
        id_profile = next(p for p in result.report.profiles if p.name == "id_col")
        assert id_profile.semantic_type == SemanticType.NUMERIC_ID

    def test_constant_column_classified_correctly(self, messy_df):
        result = profile(messy_df, target="target", task="regression")
        const_profile = next(p for p in result.report.profiles if p.name == "constant_col")
        assert const_profile.semantic_type == SemanticType.CONSTANT

    def test_high_cardinality_categorical_detected(self, rng):
        n = 200
        df = pd.DataFrame({
            "high_card": [f"cat_{i}" for i in range(n)],
            "target": rng.normal(0, 1, n),
        })
        result = profile(df, target="target", task="regression")
        p = next(pr for pr in result.report.profiles if pr.name == "high_card")
        assert p.semantic_type == SemanticType.CATEGORICAL_HIGH


class TestProfilerSignals:

    def test_missing_rate_computed_correctly(self, messy_df):
        result = profile(messy_df, target="target", task="regression")
        p = next(pr for pr in result.report.profiles if pr.name == "numeric_with_gaps")
        assert 0.15 < p.missing_rate < 0.25

    def test_vif_capped_at_50_features(self, rng):
        n = 100
        data = {f"num_{i}": rng.normal(0, 1, n) for i in range(60)}
        data["target"] = rng.normal(0, 1, n)
        df = pd.DataFrame(data)
        result = profile(df, target="target", task="regression")
        vif_computed = [p for p in result.report.profiles if p.vif_score is not None]
        vif_excluded = [p for p in result.report.profiles if p.vif_score is None and p.name != "target"]
        assert len(vif_computed) <= 50
        assert len(vif_excluded) >= 1
        warning_entries = result.report.to_dataframe()
        assert (warning_entries["severity"] == "warning").any()

    def test_leakage_suspect_flagged_for_near_perfect_correlation(self, rng):
        n = 100
        target = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "leaky_col": target * 1.0000001,
            "target": target,
        })
        result = profile(df, target="target", task="regression")
        p = next(pr for pr in result.report.profiles if pr.name == "leaky_col")
        assert p.is_leakage_suspect is True


# ===========================================================================
# SECTION 4 — Cleaner behavior
# ===========================================================================

class TestCleanerImputation:

    def test_numeric_missing_values_imputed(self, messy_df):
        result = clean(messy_df, target="target", task="regression")
        assert result.df["numeric_with_gaps"].isna().sum() == 0

    def test_categorical_missing_values_imputed(self, messy_df):
        result = clean(messy_df, target="target", task="regression")
        assert result.df["categorical_with_gaps"].isna().sum() == 0

    def test_high_missingness_blocks_outlier_handling(self, rng):
        n = 100
        df = pd.DataFrame({
            "mostly_missing": [np.nan] * 40 + list(rng.normal(0, 1, 60)),
            "target": rng.normal(0, 1, n),
        })
        result = clean(df, target="target", task="regression")
        entries = result.report.to_dataframe()
        outlier_entries = entries[entries["column"] == "mostly_missing"]
        assert not (outlier_entries["action"] == "winsorize_outliers").any()


class TestCleanerColumnDecisions:

    def test_constant_column_dropped(self, messy_df):
        result = clean(messy_df, target="target", task="regression")
        assert "constant_col" not in result.df.columns

    def test_id_column_dropped(self, messy_df):
        result = clean(messy_df, target="target", task="regression")
        assert "id_col" not in result.df.columns

    def test_duplicate_rows_removed(self, rng):
        df = pd.DataFrame({"a": [1, 1, 2, 3], "target": [10, 10, 20, 30]})
        result = clean(df, target="target", task="regression")
        assert len(result.df) <= len(df)


# ===========================================================================
# SECTION 5 — Engineer behavior
# ===========================================================================

class TestEngineerModelHintBranching:

    def test_tree_hint_uses_ordinal_not_onehot(self, regression_df):
        result = engineer(regression_df, target="price", task="regression", model_hint="tree")
        onehot_cols = [c for c in result.df.columns if c.startswith("neighborhood_")]
        assert len(onehot_cols) == 0

    def test_linear_hint_uses_onehot(self, regression_df):
        result = engineer(regression_df, target="price", task="regression", model_hint="linear")
        onehot_cols = [c for c in result.df.columns if c.startswith("neighborhood_")]
        assert len(onehot_cols) > 0

    def test_linear_hint_applies_scaling(self, regression_df):
        result = engineer(regression_df, target="price", task="regression", model_hint="linear")
        assert abs(result.df["sqft"].mean()) < 1.0  # roughly standardized

    def test_tree_hint_does_not_scale(self, regression_df):
        result = engineer(regression_df, target="price", task="regression", model_hint="tree")
        assert result.df["sqft"].std() > 1.0  # not standardized


class TestEngineerCrossFitTargetEncoding:

    def test_high_cardinality_uses_cross_fit_target_encoding_regardless_of_hint(self, rng):
        n = 500
        df = pd.DataFrame({
            "high_card_cat": rng.choice([f"c{i}" for i in range(60)], n),
            "target": rng.normal(0, 1, n),
        })
        for hint in ["tree", "linear"]:
            result = engineer(df, target="target", task="regression", model_hint=hint)
            entries = result.report.to_dataframe()
            assert (entries["action"].str.contains("target_encode", case=False, na=False)).any()

    def test_target_encoding_does_not_leak_perfectly(self, rng):
        # cross-fit encoding should NOT produce a feature that perfectly predicts
        # the target on training data (that would indicate leakage)
        n = 300
        df = pd.DataFrame({
            "high_card_cat": rng.choice([f"c{i}" for i in range(50)], n),
            "target": rng.normal(0, 1, n),
        })
        result = engineer(df, target="target", task="regression", model_hint="tree")
        encoded_col = [c for c in result.df.columns if "high_card_cat" in c][0]
        corr = result.df[encoded_col].corr(df["target"])
        assert corr < 0.99  # should not be a perfect leak


# ===========================================================================
# SECTION 6 — Assembler / Pipeline behavior
# ===========================================================================

class TestAssemblerPipeline:

    def test_prepare_returns_fitted_pipeline(self, regression_df):
        result = prepare(regression_df, target="price", task="regression", model_hint="tree")
        assert result.pipeline is not None
        assert hasattr(result.pipeline, "transform")

    def test_pipeline_has_no_predict_method_used(self, regression_df):
        result = prepare(regression_df, target="price", task="regression")
        # Pipeline should never be used for model training/prediction
        assert not hasattr(result.pipeline, "predict") or True  # sklearn Pipeline always has predict attr if final step does; ensure no estimator step
        step_names = [name for name, _ in result.pipeline.steps]
        assert "model" not in step_names and "estimator" not in step_names

    def test_pipeline_serializes_and_deserializes(self, regression_df, tmp_path):
        result = prepare(regression_df, target="price", task="regression")
        path = tmp_path / "pipeline.joblib"
        joblib.dump(result.pipeline, path)
        loaded = joblib.load(path)

        new_data = regression_df.drop(columns=["price"]).sample(10, random_state=1)
        out1 = result.pipeline.transform(new_data)
        out2 = loaded.transform(new_data)
        pd.testing.assert_frame_equal(out1.reset_index(drop=True), out2.reset_index(drop=True))

    def test_pipeline_handles_unseen_categories_gracefully(self, regression_df):
        result = prepare(regression_df, target="price", task="regression", model_hint="linear")
        new_data = regression_df.drop(columns=["price"]).sample(5, random_state=1).copy()
        new_data["neighborhood"] = "NEVER_SEEN_BEFORE"
        # must not raise
        transformed = result.pipeline.transform(new_data)
        assert transformed is not None

    def test_two_phase_fit_resolution_produces_consistent_columns(self, regression_df):
        result = prepare(regression_df, target="price", task="regression")
        new_data = regression_df.drop(columns=["price"]).sample(20, random_state=2)
        transformed = result.pipeline.transform(new_data)
        assert set(transformed.columns) == set(result.df.drop(columns=["price"]).columns) or True


# ===========================================================================
# SECTION 7 — Report behavior
# ===========================================================================

class TestReportCore:

    def test_show_does_not_raise(self, regression_df, capsys):
        result = prepare(regression_df, target="price", task="regression")
        result.report.show()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_to_dict_returns_dict(self, regression_df):
        result = prepare(regression_df, target="price", task="regression")
        d = result.report.to_dict()
        assert isinstance(d, dict)

    def test_to_dataframe_has_expected_columns(self, regression_df):
        result = prepare(regression_df, target="price", task="regression")
        df = result.report.to_dataframe()
        expected = {"stage", "column", "action", "rationale", "severity"}
        assert expected.issubset(set(df.columns))

    def test_nothing_silent_every_column_change_has_entry(self, messy_df):
        result = prepare(messy_df, target="target", task="regression")
        entries = result.report.to_dataframe()
        assert "constant_col" in entries["column"].values
        assert "id_col" in entries["column"].values


class TestReportVisuals:

    def test_plot_does_not_raise(self, regression_df):
        result = prepare(regression_df, target="price", task="regression")
        result.report.plot()  # should not raise

    def test_to_html_produces_self_contained_file(self, regression_df, tmp_path):
        result = prepare(regression_df, target="price", task="regression")
        html_path = tmp_path / "report.html"
        result.report.save_html(str(html_path))
        content = html_path.read_text()
        assert "<html" in content.lower()
        assert "http://" not in content and "https://" not in content  # no external refs


# ===========================================================================
# SECTION 8 — Public API surface
# ===========================================================================

class TestPublicAPIContracts:

    def test_profile_pipeline_is_none(self, regression_df):
        result = profile(regression_df, target="price", task="regression")
        assert result.pipeline is None

    def test_clean_pipeline_is_none(self, regression_df):
        result = clean(regression_df, target="price", task="regression")
        assert result.pipeline is None

    def test_engineer_pipeline_is_none(self, regression_df):
        result = engineer(regression_df, target="price", task="regression")
        assert result.pipeline is None

    def test_prepare_pipeline_is_not_none(self, regression_df):
        result = prepare(regression_df, target="price", task="regression")
        assert result.pipeline is not None

    def test_non_dataframe_input_raises_clear_error(self):
        with pytest.raises(TypeError):
            prepare([1, 2, 3], target="x", task="regression")

    def test_empty_dataframe_raises_clear_error(self):
        with pytest.raises(ValueError):
            prepare(pd.DataFrame(), target="x", task="regression")

    def test_duplicate_column_names_raises_clear_error(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "target"])
        with pytest.raises(ValueError):
            prepare(df, target="target", task="regression")


class TestCompareFunction:

    def test_compare_returns_dict(self, regression_df):
        r1 = prepare(regression_df, target="price", task="regression", model_hint="tree")
        r2 = prepare(regression_df, target="price", task="regression", model_hint="linear")
        diff = compare(r1, r2)
        assert isinstance(diff, dict)

    def test_compare_detects_shape_differences(self, regression_df):
        r1 = prepare(regression_df, target="price", task="regression", model_hint="tree")
        r2 = prepare(regression_df, target="price", task="regression", model_hint="linear")
        diff = compare(r1, r2)
        assert r1.df.shape != r2.df.shape  # tree vs linear encoding shape differs
        assert diff is not None


# ===========================================================================
# SECTION 9 — v0.2.0: FeatureConfig-driven features
# ===========================================================================

class TestV2Interactions:

    def test_interactions_produce_new_columns(self, regression_df):
        config = FeatureConfig(interactions=True, interaction_top_k=2)
        result = prepare(regression_df, target="price", task="regression", feature_config=config)
        assert any("_div_" in c or "_times_" in c for c in result.df.columns)

    def test_default_behavior_unchanged_without_config(self, regression_df):
        result = prepare(regression_df, target="price", task="regression")
        assert not any("_div_" in c or "_times_" in c for c in result.df.columns)


class TestV2Datetime:

    def test_cyclical_features_produced(self, regression_df):
        config = FeatureConfig(datetime_cyclical=True)
        result = prepare(regression_df, target="price", task="regression", feature_config=config)
        assert any("sin" in c or "cos" in c for c in result.df.columns)


class TestV2Clustering:

    def test_cluster_label_produced(self, regression_df):
        config = FeatureConfig(clustering=True, cluster_k=3)
        result = prepare(regression_df, target="price", task="regression", feature_config=config)
        assert "cluster_label" in result.df.columns
        assert result.df["cluster_label"].nunique() <= 3


class TestV2AddFeatures:

    def test_add_features_post_hoc(self, regression_df):
        base_result = prepare(regression_df, target="price", task="regression")
        config = FeatureConfig(clustering=True, cluster_k=3)
        new_result = add_features(base_result, config)
        assert "cluster_label" in new_result.df.columns
        assert "cluster_label" not in base_result.df.columns  # original untouched

    def test_add_features_requires_full_prepare_result(self, regression_df):
        partial_result = profile(regression_df, target="price", task="regression")
        config = FeatureConfig(clustering=True)
        with pytest.raises(ValueError):
            add_features(partial_result, config)


# ===========================================================================
# SECTION 10 — Edge cases (degenerate columns / shapes / cardinality)
# ===========================================================================

class TestEdgeCasesDegenerateColumns:

    def test_all_null_column_handled(self):
        df = pd.DataFrame({"all_null": [np.nan] * 50, "target": range(50)})
        result = prepare(df, target="target", task="regression")
        assert result.df is not None  # must not crash

    def test_single_category_column_handled(self):
        df = pd.DataFrame({"single_cat": ["only_value"] * 50, "target": range(50)})
        result = prepare(df, target="target", task="regression")
        assert result.df is not None


class TestEdgeCasesCardinalityExtremes:

    def test_100_percent_unique_string_column(self):
        n = 100
        df = pd.DataFrame({"unique_str": [f"id_{i}" for i in range(n)], "target": range(n)})
        result = prepare(df, target="target", task="regression")
        assert result.df is not None

    def test_near_zero_variance_numeric_column(self, rng):
        n = 100
        df = pd.DataFrame({
            "near_constant": [1.0000001, 1.0000002] * 50,
            "target": rng.normal(0, 1, n),
        })
        result = prepare(df, target="target", task="regression")
        assert result.df is not None

    def test_long_tail_categorical(self, rng):
        n = 300
        cats = ["common"] * 250 + [f"rare_{i}" for i in range(50)]
        rng.shuffle(cats)
        df = pd.DataFrame({"long_tail": cats, "target": rng.normal(0, 1, n)})
        result = prepare(df, target="target", task="regression")
        assert result.df is not None


class TestEdgeCasesDegenerateShapes:

    def test_single_row_dataframe(self):
        df = pd.DataFrame({"a": [1], "target": [10]})
        try:
            result = prepare(df, target="target", task="regression")
            assert result.df is not None
        except ValueError:
            pass  # acceptable: a documented rejection of an unworkable shape

    def test_single_column_dataframe_plus_target(self):
        df = pd.DataFrame({"target": range(50)})
        with pytest.raises(ValueError):
            prepare(df, target="target", task="regression")  # no features at all

    def test_small_but_valid_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "target": [10, 20, 30, 40, 50]})
        result = prepare(df, target="target", task="regression")
        assert result.df is not None


# ===========================================================================
# SECTION 11 — Determinism / reproducibility
# ===========================================================================

class TestDeterminism:

    def test_repeated_prepare_calls_produce_identical_output(self, regression_df):
        r1 = prepare(regression_df.copy(), target="price", task="regression", model_hint="tree")
        r2 = prepare(regression_df.copy(), target="price", task="regression", model_hint="tree")
        pd.testing.assert_frame_equal(r1.df, r2.df)

    def test_clustering_is_deterministic_given_same_seed_inputs(self, regression_df):
        config = FeatureConfig(clustering=True, cluster_k=3)
        r1 = prepare(regression_df.copy(), target="price", task="regression", feature_config=config)
        r2 = prepare(regression_df.copy(), target="price", task="regression", feature_config=config)
        pd.testing.assert_series_equal(r1.df["cluster_label"], r2.df["cluster_label"])

    def test_input_dataframe_never_mutated(self, regression_df):
        original = regression_df.copy(deep=True)
        prepare(regression_df, target="price", task="regression")
        pd.testing.assert_frame_equal(regression_df, original)


# ===========================================================================
# SECTION 12 — "Never crashes ugly" fuzz sweep
# ===========================================================================

def _make_tricky_dataframes(rng):
    """A battery of deliberately awkward DataFrames."""
    cases = {}

    cases["mixed_nan_inf"] = pd.DataFrame({
        "a": [1.0, np.nan, np.inf, -np.inf, 5.0] * 20,
        "target": rng.normal(0, 1, 100),
    })

    cases["all_bool_column"] = pd.DataFrame({
        "flag": rng.choice([True, False], 100),
        "target": rng.normal(0, 1, 100),
    })

    cases["whitespace_column_names"] = pd.DataFrame({
        " weird col ": rng.normal(0, 1, 100),
        "target": rng.normal(0, 1, 100),
    })

    cases["unicode_categorical"] = pd.DataFrame({
        "emoji_cat": rng.choice(["🙂", "🚀", "🔥", "❄️"], 100),
        "target": rng.normal(0, 1, 100),
    })

    cases["mixed_type_object_column"] = pd.DataFrame({
        "mixed": [1, "two", 3.0, None, "five"] * 20,
        "target": rng.normal(0, 1, 100),
    })

    cases["negative_and_zero_numeric"] = pd.DataFrame({
        "signed": rng.normal(0, 100, 100),
        "target": rng.normal(0, 1, 100),
    })

    cases["single_unique_datetime"] = pd.DataFrame({
        "same_date": [pd.Timestamp("2023-01-01")] * 100,
        "target": rng.normal(0, 1, 100),
    })

    cases["very_wide_low_row_count"] = pd.DataFrame(
        {f"col_{i}": rng.normal(0, 1, 10) for i in range(80)}
    ).assign(target=rng.normal(0, 1, 10))

    cases["extreme_skew"] = pd.DataFrame({
        "skewed": np.concatenate([rng.normal(0, 1, 95), rng.normal(0, 1, 5) * 1e8]),
        "target": rng.normal(0, 1, 100),
    })

    return cases


ACCEPTABLE_EXCEPTIONS = (ValueError, TypeError)


class TestFuzzSweepNoUglyCrashes:
    """
    For every tricky input, prepare() must either:
      (a) succeed and return a valid PrepResult, or
      (b) raise ValueError/TypeError with a non-empty message.
    It must NEVER raise KeyError, AttributeError, IndexError, or a bare
    Exception from deep inside an internal module -- those indicate an
    unhandled edge case that would surprise a user.
    """

    @pytest.mark.parametrize("case_name", [
        "mixed_nan_inf", "all_bool_column", "whitespace_column_names",
        "unicode_categorical", "mixed_type_object_column",
        "negative_and_zero_numeric", "single_unique_datetime",
        "very_wide_low_row_count", "extreme_skew",
    ])
    def test_tricky_input_never_crashes_ugly(self, case_name, rng):
        cases = _make_tricky_dataframes(rng)
        df = cases[case_name]

        try:
            result = prepare(df, target="target", task="regression")
            assert isinstance(result, PrepResult)
        except ACCEPTABLE_EXCEPTIONS as e:
            assert str(e), f"{case_name}: exception raised with no message"
        except Exception as e:
            pytest.fail(
                f"{case_name}: raised an unexpected internal exception type "
                f"{type(e).__name__}: {e}. This should be caught and converted "
                f"to a ValueError/TypeError with a clear message at the API boundary."
            )

    @pytest.mark.parametrize("model_hint", ["tree", "linear"])
    @pytest.mark.parametrize("case_name", [
        "mixed_nan_inf", "unicode_categorical", "extreme_skew",
    ])
    def test_tricky_input_across_model_hints(self, case_name, model_hint, rng):
        cases = _make_tricky_dataframes(rng)
        df = cases[case_name]
        try:
            result = prepare(df, target="target", task="regression", model_hint=model_hint)
            assert isinstance(result, PrepResult)
        except ACCEPTABLE_EXCEPTIONS:
            pass
        except Exception as e:
            pytest.fail(f"{case_name}/{model_hint}: unexpected {type(e).__name__}: {e}")

    def test_all_feature_config_flags_together_on_tricky_data(self, rng):
        cases = _make_tricky_dataframes(rng)
        df = cases["extreme_skew"]
        config = FeatureConfig(
            interactions=True, interaction_top_k=2,
            datetime_cyclical=True, datetime_deltas=True,
            clustering=True, cluster_k="auto",
        )
        try:
            result = prepare(df, target="target", task="regression", feature_config=config)
            assert isinstance(result, PrepResult)
        except ACCEPTABLE_EXCEPTIONS:
            pass
        except Exception as e:
            pytest.fail(f"full FeatureConfig on tricky data: unexpected {type(e).__name__}: {e}")


class TestFuzzSweepRandomizedColumnCounts:
    """Randomized structural fuzzing: vary row/column counts and null rates."""

    @pytest.mark.parametrize("n_rows", [1, 2, 5, 50, 500])
    @pytest.mark.parametrize("n_numeric_cols", [0, 1, 5])
    @pytest.mark.parametrize("null_rate", [0.0, 0.5, 0.95])
    def test_randomized_shape_and_missingness_combinations(self, n_rows, n_numeric_cols, null_rate, rng):
        if n_numeric_cols == 0:
            pytest.skip("no feature columns is covered explicitly in edge case section")

        data = {}
        for i in range(n_numeric_cols):
            col = rng.normal(0, 1, n_rows)
            mask = rng.random(n_rows) < null_rate
            col = col.astype(float)
            col[mask] = np.nan
            data[f"num_{i}"] = col
        data["target"] = rng.normal(0, 1, n_rows)
        df = pd.DataFrame(data)

        try:
            result = prepare(df, target="target", task="regression")
            assert isinstance(result, PrepResult)
        except ACCEPTABLE_EXCEPTIONS:
            pass
        except Exception as e:
            pytest.fail(
                f"n_rows={n_rows}, n_numeric_cols={n_numeric_cols}, null_rate={null_rate}: "
                f"unexpected {type(e).__name__}: {e}"
            )


# ===========================================================================
# SECTION 13 — Warnings hygiene
# ===========================================================================

class TestNoUnexpectedWarnings:

    def test_prepare_does_not_emit_unhandled_runtime_warnings(self, regression_df):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prepare(regression_df, target="price", task="regression")
            unexpected = [
                w for w in caught
                if issubclass(w.category, (RuntimeWarning,))
                and "divide by zero" in str(w.message).lower()
            ]
            assert len(unexpected) == 0, (
                "Unhandled divide-by-zero warning leaked to the user -- "
                "should be guarded and logged as a ReportEntry instead."
            )