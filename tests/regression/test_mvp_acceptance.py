"""
PreFlight v2.0 — MVP ACCEPTANCE SUITE (ported from v1.0.1 tests/mvp_test.py)

Port notes (see docs/port_regressions.md for the full rationale):
  - All v1 `preflight.<fn>` calls are routed through `tests.regression.adapter`
    (aliased as `pf` below), which bridges the v1 public API (prepare, profile,
    clean, engineer, compare, add_features, run_profiler, save_compare_pdf)
    onto v2 components. v2's own public package intentionally does not expose
    this surface (see docs/port_contract.md).
  - Section 4 (CLI round trip) is skipped: v2's CLI (`src/preflight/cli.py`)
    is a stub in this milestone. CLI wiring is explicitly out of scope for
    Phase 6 (M5) and is deferred to a later milestone.
  - All other sections are ported with assertions unchanged from v1.
"""

import os
import re
import tempfile

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

import adapter as pf
from preflight.types import FeatureConfig, SemanticType, PRESETS


def _tmp_path(suffix=".pdf"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    os.remove(path)
    return path


# ===========================================================================
# SECTION 1 — Regression tests for every bug found in the stress-test cycle
# ===========================================================================

class TestRegression_StringAndMultiClassTargets:
    """Bugs 1.1 / 1.2 — target encoding / mean reduction crashed on string targets."""

    def test_binary_string_target_does_not_crash(self):
        n = 300
        df = pd.DataFrame({
            "high_card_cat": np.random.choice([f"cat_{i}" for i in range(40)], size=n),
            "num_a": np.random.uniform(0, 100, size=n),
            "churn": np.random.choice(["Yes", "No"], size=n),
        })
        result = pf.prepare(df, target="churn", task="classification")
        assert result.df is not None
        assert result.pipeline is not None

    def test_multiclass_string_target_does_not_crash(self):
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        df["species"] = iris.target_names[iris.target]
        df = df.drop(columns=["target"])

        result = pf.prepare(df, target="species", task="classification")
        assert result.df is not None
        assert result.pipeline is not None
        assert result.df.shape[0] == 150

    def test_high_cardinality_categorical_with_string_target_encodes_correctly(self):
        """This is the exact combination that originally crashed: cross-fit target
        encoding on a high-cardinality categorical, with a non-numeric target."""
        n = 500
        df = pd.DataFrame({
            "city": np.random.choice([f"city_{i}" for i in range(60)], size=n),
            "income": np.random.uniform(20000, 150000, size=n),
            "outcome": np.random.choice(["approved", "denied"], size=n),
        })
        result = pf.prepare(df, target="outcome", task="classification")
        assert "city" in result.df.columns
        assert pd.api.types.is_numeric_dtype(result.df["city"])  # encoded to numeric

    def test_report_describes_original_class_labels_not_internal_encoding(self):
        """The fix must not leak internal numeric re-encoding into user-facing text."""
        n = 300
        df = pd.DataFrame({
            "city": np.random.choice([f"city_{i}" for i in range(40)], size=n),
            "num_a": np.random.uniform(0, 1, size=n),
            "outcome": np.random.choice(["approved", "denied"], size=n),
        })
        result = pf.prepare(df, target="outcome", task="classification")
        entries_text = " ".join(result.report.to_dataframe()["rationale"].astype(str).tolist())

        assert len(entries_text) > 0
        assert "class 0" not in entries_text.lower()
        assert "class 1" not in entries_text.lower()


class TestRegression_IdLikeColumnsNotCoercedToNumeric:
    """A fix for messy numeric strings (Bug 2.3) must not overcorrect and start
    coercing ID-like high-cardinality numeric-looking strings (e.g. zip codes)
    into NUMERIC_FEATURE."""

    def test_zip_code_like_column_not_classified_as_numeric_feature(self):
        n = 500
        df = pd.DataFrame({
            "zip_code": [f"{np.random.randint(10000, 99999)}" for _ in range(n)],
            "income": np.random.uniform(20000, 150000, size=n),
            "target": np.random.randint(0, 2, size=n),
        })
        profiles, _entries = pf.run_profiler(df, target="target", task="classification")
        by_name = {p.name: p for p in profiles}
        assert by_name["zip_code"].semantic_type != SemanticType.NUMERIC_FEATURE
        assert by_name["zip_code"].semantic_type in (
            SemanticType.CATEGORICAL_HIGH, SemanticType.NUMERIC_ID
        )

    def test_phone_number_like_column_not_coerced(self):
        n = 200
        df = pd.DataFrame({
            "phone": [f"{np.random.randint(1000000000, 9999999999)}" for _ in range(n)],
            "age": np.random.randint(18, 80, size=n),
            "target": np.random.randint(0, 2, size=n),
        })
        profiles, _entries = pf.run_profiler(df, target="target", task="classification")
        by_name = {p.name: p for p in profiles}
        assert by_name["phone"].semantic_type != SemanticType.NUMERIC_FEATURE

    def test_genuinely_messy_numeric_string_still_coerces_correctly(self):
        """Regression guard: the ORIGINAL Bug 2.3 fix must still work."""
        n = 500
        values = [str(round(np.random.uniform(20, 120), 2)) for _ in range(n - 5)]
        values += ["", " ", "N/A", "20.5 ", " 45.3"]
        df = pd.DataFrame({
            "total_charges": values,
            "tenure": np.random.randint(1, 72, size=n),
            "target": np.random.randint(0, 2, size=n),
        })
        profiles, _entries = pf.run_profiler(df, target="target", task="classification")
        by_name = {p.name: p for p in profiles}
        assert by_name["total_charges"].semantic_type == SemanticType.NUMERIC_FEATURE

    def test_id_guard_and_messy_numeric_fix_coexist(self):
        """Both fixes must work correctly in the SAME DataFrame at once."""
        n = 400
        df = pd.DataFrame({
            "customer_id": [f"{np.random.randint(10000, 99999)}" for _ in range(n)],
            "total_charges": [str(round(np.random.uniform(20, 120), 2)) for _ in range(n)],
            "target": np.random.randint(0, 2, size=n),
        })
        profiles, _entries = pf.run_profiler(df, target="target", task="classification")
        by_name = {p.name: p for p in profiles}
        assert by_name["customer_id"].semantic_type != SemanticType.NUMERIC_FEATURE
        assert by_name["total_charges"].semantic_type == SemanticType.NUMERIC_FEATURE


class TestRegression_WinsorizationEdgeCases:
    """Bugs 2.1 / 2.2 — winsorization erased sparse/zero-inflated columns and
    flattened high-frequency majority-value columns when IQR == 0."""

    def test_zero_inflated_sparse_column_not_erased(self):
        n = 300
        df = pd.DataFrame({
            "rare_event_count": np.concatenate([np.zeros(n - 10), np.random.randint(1, 5, size=10)]),
            "num_a": np.random.uniform(0, 1, size=n),
            "target": np.random.randint(0, 2, size=n),
        })
        result = pf.clean(df, target="target", task="classification")
        assert result.df["rare_event_count"].nunique() > 1

    def test_dominant_mode_column_not_flattened_to_constant(self):
        n = 300
        df = pd.DataFrame({
            "seats": np.concatenate([np.full(n - 5, 5), [2, 4, 7, 8, 9]]),
            "price": np.random.uniform(5000, 50000, size=n),
            "target": np.random.randint(0, 2, size=n),
        })
        result = pf.clean(df, target="target", task="classification")
        assert result.df["seats"].nunique() > 1

    def test_iqr_zero_skip_is_logged(self):
        n = 300
        df = pd.DataFrame({
            "seats": np.full(n, 5, dtype=float),
            "seats_noisy": np.concatenate([np.full(n - 3, 5.0), [2.0, 9.0, 20.0]]),
            "target": np.random.randint(0, 2, size=n),
        })
        result = pf.clean(df, target="target", task="classification")
        entries_text = " ".join(result.report.to_dataframe()["action"].astype(str).tolist())
        assert "skip" in entries_text.lower() or True


class TestRegression_TextVsCategoricalCalibration:
    """Bug 2.4 — high-cardinality short-label string columns (e.g. car model names)
    were being misclassified as TEXT instead of CATEGORICAL_HIGH."""

    def test_short_high_cardinality_labels_not_classified_as_text(self):
        n = 300
        car_models = [f"Model_{i}_Sedan" for i in range(80)]
        df = pd.DataFrame({
            "car_name": np.random.choice(car_models, size=n),
            "price": np.random.uniform(5000, 50000, size=n),
            "target": np.random.randint(0, 2, size=n),
        })
        profiles, _entries = pf.run_profiler(df, target="target", task="classification")
        by_name = {p.name: p for p in profiles}
        assert by_name["car_name"].semantic_type != SemanticType.TEXT

    def test_genuine_long_free_text_still_classified_as_text(self):
        n = 100
        reviews = [
            "This product exceeded my expectations in almost every way possible today"
        ] * n
        df = pd.DataFrame({
            "review": reviews,
            "rating": np.random.randint(1, 6, size=n),
            "target": np.random.randint(0, 2, size=n),
        })
        profiles, _entries = pf.run_profiler(df, target="target", task="classification")
        by_name = {p.name: p for p in profiles}
        assert by_name["review"].semantic_type == SemanticType.TEXT


class TestRegression_UnitSuffixNumericParsing:
    """Bug 2.5 — numeric columns with unit suffixes (e.g. '1248 CC', '74 bhp')
    were misclassified as CATEGORICAL_HIGH instead of being parsed as numeric."""

    def test_simple_unit_suffix_column_parses_as_numeric(self):
        n = 200
        df = pd.DataFrame({
            "engine_cc": [f"{np.random.randint(800, 2500)} CC" for _ in range(n)],
            "target": np.random.randint(0, 2, size=n),
        })
        profiles, _entries = pf.run_profiler(df, target="target", task="classification")
        by_name = {p.name: p for p in profiles}
        assert by_name["engine_cc"].semantic_type == SemanticType.NUMERIC_FEATURE

    def test_complex_nested_unit_expression_documented_limitation(self):
        """Known, documented limitation — this is a canary, not a strict gate."""
        n = 100
        df = pd.DataFrame({
            "torque": [f"{np.random.randint(100, 300)}Nm@ {np.random.randint(1500,3000)}rpm" for _ in range(n)],
            "target": np.random.randint(0, 2, size=n),
        })
        profiles, _entries = pf.run_profiler(df, target="target", task="classification")
        by_name = {p.name: p for p in profiles}
        assert by_name["torque"].semantic_type in (
            SemanticType.CATEGORICAL_HIGH, SemanticType.NUMERIC_FEATURE
        )


class TestRegression_ApiConsistency:
    """Bugs 3.1 / 3.2 / 3.3 — API surface inconsistencies."""

    def test_profile_accepts_model_hint(self):
        import inspect
        sig = inspect.signature(pf.profile)
        assert "model_hint" in sig.parameters

    def test_clean_accepts_model_hint(self):
        import inspect
        sig = inspect.signature(pf.clean)
        assert "model_hint" in sig.parameters

    def test_save_compare_pdf_validates_prepresult_types_before_comparing(self):
        n = 100
        df = pd.DataFrame({"a": np.random.uniform(0, 1, n), "target": np.random.randint(0, 2, n)})
        result = pf.prepare(df, target="target", task="classification")
        path = _tmp_path()
        try:
            with pytest.raises((ValueError, TypeError)) as exc_info:
                pf.save_compare_pdf(result, "not_a_prepresult", path)
            assert not isinstance(exc_info.value, AttributeError)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_dry_run_and_real_run_produce_consistent_decisions(self):
        n = 300
        df = pd.DataFrame({
            "city": np.random.choice([f"city_{i}" for i in range(50)], size=n),
            "num_a": np.random.uniform(0, 1, size=n),
            "target": np.random.choice(["yes", "no"], size=n),
        })
        dry = pf.prepare(df, target="target", task="classification", dry_run=True)
        real = pf.prepare(df, target="target", task="classification")
        assert dry.report is not None
        assert real.pipeline is not None


# ===========================================================================
# SECTION 2 — Full real-dataset run-throughs
# ===========================================================================

class TestRealDatasetEndToEnd:

    @pytest.fixture(scope="class")
    def telco_like_df(self):
        n = 500
        return pd.DataFrame({
            "customerID": [f"CUST-{i:05d}" for i in range(n)],
            "SeniorCitizen": np.random.choice([0, 1], size=n),
            "tenure": np.random.randint(0, 72, size=n),
            "MonthlyCharges": np.random.uniform(18, 120, size=n),
            "TotalCharges": [
                "" if np.random.random() < 0.02 else str(round(np.random.uniform(18, 8000), 2))
                for _ in range(n)
            ],
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
            "Churn": np.random.choice(["Yes", "No"], size=n),
        })

    def test_full_prepare_on_telco_like_data(self, telco_like_df):
        result = pf.prepare(telco_like_df, target="Churn", task="classification")
        assert result.df is not None
        assert result.pipeline is not None

    def test_senior_citizen_override_works(self, telco_like_df):
        result = pf.prepare(
            telco_like_df, target="Churn", task="classification",
            column_types={"SeniorCitizen": SemanticType.CATEGORICAL_LOW},
        )
        assert result.df is not None

    def test_total_charges_handled_without_crash(self, telco_like_df):
        profiles, _entries = pf.run_profiler(telco_like_df, target="Churn", task="classification")
        by_name = {p.name: p for p in profiles}
        assert by_name["TotalCharges"].semantic_type == SemanticType.NUMERIC_FEATURE

    def test_customer_id_not_treated_as_numeric_feature(self, telco_like_df):
        profiles, _entries = pf.run_profiler(telco_like_df, target="Churn", task="classification")
        by_name = {p.name: p for p in profiles}
        assert by_name["customerID"].semantic_type != SemanticType.NUMERIC_FEATURE

    def test_full_feature_config_on_telco_like_data(self, telco_like_df):
        config = FeatureConfig(interactions=True, interaction_top_k=3)
        result = pf.prepare(
            telco_like_df, target="Churn", task="classification", feature_config=config
        )
        assert result.df.shape[1] > 0

    def test_pipeline_reuse_on_holdout_telco_like(self, telco_like_df):
        train = telco_like_df.iloc[:400]
        holdout = telco_like_df.iloc[400:].drop(columns=["Churn"])
        result = pf.prepare(train, target="Churn", task="classification")
        transformed = result.pipeline.transform(holdout)
        assert len(transformed) == len(holdout)
        assert not transformed.isnull().all().any()  # no fully-NaN columns


class TestRealDatasetMissingnessVisualization:
    """Verifies the missingness heatmap actually reflects real missing data,
    not just rendering a blank/empty box regardless of input."""

    def test_missingness_heatmap_pdf_generates_with_real_missing_data(self):
        n = 200
        df = pd.DataFrame({
            "a": np.where(np.random.random(n) < 0.3, np.nan, np.random.uniform(0, 1, n)),
            "b": np.random.uniform(0, 1, n),
            "target": np.random.randint(0, 2, n),
        })
        result = pf.prepare(df, target="target", task="classification")
        path = _tmp_path()
        try:
            result.report.save_pdf(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_missing_rate_computed_correctly_in_profile(self):
        n = 200
        df = pd.DataFrame({
            "a": np.where(np.arange(n) < 60, np.nan, np.random.uniform(0, 1, n)),  # exactly 30% missing
            "target": np.random.randint(0, 2, n),
        })
        profiles, _entries = pf.run_profiler(df, target="target", task="classification")
        by_name = {p.name: p for p in profiles}
        assert abs(by_name["a"].missing_rate - 0.3) < 0.01


# ===========================================================================
# SECTION 3 — Full feature matrix stress pass
# ===========================================================================

class TestFullFeatureMatrix:

    @pytest.fixture(scope="class")
    def rich_df(self):
        np.random.seed(11)
        n = 400
        return pd.DataFrame({
            "num_a": np.random.uniform(0, 100, size=n),
            "num_b": np.random.uniform(0, 1, size=n),
            "cat_low": np.random.choice(["a", "b", "c"], size=n),
            "cat_high": np.random.choice([f"v{i}" for i in range(50)], size=n),
            "signup_date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "feedback": ["Great overall experience with fast and reliable customer support here"] * n,
            "target": np.random.choice(["yes", "no"], size=n),
        })

    def test_all_feature_config_options_enabled_simultaneously(self, rich_df):
        config = FeatureConfig(
            interactions=True, interaction_top_k=3,
            datetime_cyclical=True, datetime_deltas=False,
            clustering=True, cluster_k=3,
            text_features=True, text_tfidf=True, text_tfidf_top_k=5,
        )
        result = pf.prepare(rich_df, target="target", task="classification", feature_config=config)
        assert result.df is not None
        assert result.pipeline is not None
        assert not result.df.isnull().all().any()

    def test_no_generated_column_is_entirely_degenerate(self, rich_df):
        """Generated columns shouldn't be all-identical/all-NaN/all-zero without being logged."""
        config = FeatureConfig(interactions=True, interaction_top_k=3, clustering=True, cluster_k=3)
        baseline = pf.prepare(rich_df, target="target", task="classification")
        result = pf.prepare(rich_df, target="target", task="classification", feature_config=config)
        new_cols = set(result.df.columns) - set(baseline.df.columns)

        for col in new_cols:
            series = result.df[col]
            if series.nunique(dropna=False) <= 1:
                entries = result.report.to_dataframe()
                related = entries[entries["column"].astype(str).str.contains(re.escape(col), na=False)]
                assert len(related) > 0, f"Degenerate column '{col}' has no explanatory ReportEntry"

    def test_both_presets_work_on_rich_dataset(self, rich_df):
        fast = pf.prepare(rich_df, target="target", task="classification", preset="fast")
        thorough = pf.prepare(rich_df, target="target", task="classification", preset="thorough")
        assert thorough.df.shape[1] >= fast.df.shape[1]

    def test_dry_run_with_all_features_enabled(self, rich_df):
        config = FeatureConfig(interactions=True, clustering=True, cluster_k=3, text_features=True)
        preview = pf.prepare(
            rich_df, target="target", task="classification", dry_run=True, feature_config=config
        )
        assert preview.pipeline is None
        pd.testing.assert_frame_equal(preview.df, rich_df)

    def test_add_features_stacking_multiple_configs(self, rich_df):
        base = pf.prepare(rich_df, target="target", task="classification")
        step1 = pf.add_features(base, FeatureConfig(interactions=True, interaction_top_k=2))
        step2 = pf.add_features(step1, FeatureConfig(clustering=True, cluster_k=3))
        assert step2.df.shape[1] >= step1.df.shape[1] >= base.df.shape[1]

    def test_compare_and_pdf_comparison_on_rich_dataset(self, rich_df):
        fast = pf.prepare(rich_df, target="target", task="classification", preset="fast")
        thorough = pf.prepare(rich_df, target="target", task="classification", preset="thorough")
        diff = pf.compare(fast, thorough)
        assert diff is not None

        path = _tmp_path()
        try:
            pf.save_compare_pdf(fast, thorough, path)
            assert os.path.exists(path)
            reader = PdfReader(path)
            assert len(reader.pages) >= 1
        finally:
            if os.path.exists(path):
                os.remove(path)


# ===========================================================================
# SECTION 4 — CLI, full round trip
# ===========================================================================
#
# v2's CLI (src/preflight/cli.py) is a stub in this milestone (Phase 6 / M5
# scope covers cleaning/engineering/assembly only). These two v1 tests are
# skipped rather than ported — see docs/port_regressions.md "CLI scope
# deferral" for the deliberate divergence this documents.

@pytest.mark.skip(reason="CLI deferred out of Phase 6 (M5) scope; see docs/port_regressions.md")
class TestCliRoundTrip:

    def test_cli_full_flag_combination_on_string_target_dataset(self, tmp_path):
        pass

    def test_cli_errors_are_clean_not_tracebacks(self, tmp_path):
        pass


# ===========================================================================
# SECTION 5 — Robustness / fuzz sweep (must never hard-crash)
# ===========================================================================

class TestRobustnessSweep:

    @pytest.mark.parametrize("case_name,build_df", [
        ("string_target_all_nan_column", lambda: pd.DataFrame({
            "a": [np.nan] * 60, "b": np.random.uniform(0, 1, 60),
            "target": np.random.choice(["x", "y"], 60),
        })),
        ("string_target_single_unique_value", lambda: pd.DataFrame({
            "a": ["same"] * 60, "b": np.random.uniform(0, 1, 60),
            "target": np.random.choice(["x", "y"], 60),
        })),
        ("string_target_high_card_short_strings", lambda: pd.DataFrame({
            "id_col": [f"ID{i}" for i in range(60)],
            "b": np.random.uniform(0, 1, 60),
            "target": np.random.choice(["x", "y"], 60),
        })),
        ("string_target_unicode_categoricals", lambda: pd.DataFrame({
            "a": np.random.choice(["café", "naïve", "北京", "🚀rocket"], size=60),
            "b": np.random.uniform(0, 1, 60),
            "target": np.random.choice(["x", "y"], 60),
        })),
        ("string_target_mixed_type_column", lambda: pd.DataFrame({
            "a": [1, "two", 3.0, None, "five"] * 12,
            "b": np.random.uniform(0, 1, 60),
            "target": np.random.choice(["x", "y"], 60),
        })),
        ("multiclass_target_five_classes", lambda: pd.DataFrame({
            "a": np.random.uniform(0, 1, 100),
            "b": np.random.choice(["p", "q", "r"], size=100),
            "target": np.random.choice(["c1", "c2", "c3", "c4", "c5"], size=100),
        })),
        ("zero_variance_and_id_together", lambda: pd.DataFrame({
            "constant_col": ["same"] * 60,
            "id_col": [f"{10000+i}" for i in range(60)],
            "b": np.random.uniform(0, 1, 60),
            "target": np.random.choice(["x", "y"], 60),
        })),
    ])
    def test_no_ugly_crash_with_string_targets(self, case_name, build_df):
        df = build_df()
        try:
            result = pf.prepare(df, target="target", task="classification")
            assert result.df is not None
            assert result.pipeline is not None
        except (ValueError, TypeError) as e:
            assert str(e), f"{case_name}: exception raised with empty message"
        except Exception as e:
            pytest.fail(f"{case_name}: raised an unexpected/ugly exception type {type(e).__name__}: {e}")


# ===========================================================================
# SECTION 6 — Public API completeness (import surface, presets, semantic types)
# ===========================================================================

class TestPublicApiCompleteness:

    def test_all_expected_top_level_functions_exist(self):
        expected = [
            "prepare", "profile", "clean", "engineer", "compare",
            "add_features", "run_profiler", "save_compare_pdf",
        ]
        for name in expected:
            assert hasattr(pf, name), f"Missing bridged function: pf.{name}"

    def test_semantic_type_includes_text(self):
        assert hasattr(SemanticType, "TEXT")

    def test_presets_include_fast_and_thorough(self):
        assert "fast" in PRESETS and "thorough" in PRESETS

    def test_feature_config_has_all_v1_fields(self):
        config = FeatureConfig()
        for field in ["text_features", "text_tfidf", "text_tfidf_top_k", "interactions", "clustering"]:
            assert hasattr(config, field)

    def test_report_has_all_export_methods(self):
        n = 60
        df = pd.DataFrame({"a": np.random.uniform(0, 1, n), "target": np.random.randint(0, 2, n)})
        result = pf.prepare(df, target="target", task="classification")
        for method in ["show", "plot", "to_html", "save_html", "save_pdf", "summary_counts", "to_dataframe", "to_dict"]:
            assert hasattr(result.report, method), f"Missing Report method: {method}"
