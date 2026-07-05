"""
Phase 8.2 (v1.0.0): FULL LIBRARY REGRESSION SUITE.

This is a single end-to-end test file exercising every major capability of
PreFlight-ML from v0.1.0 through v1.0.0. Unlike the phase-specific test files,
PDF outputs generated here are NOT deleted — they are written to
./test_outputs/ (created automatically) so they can be opened and visually
inspected after the test run.

Run with:
    pytest tests/test_full_v1_regression.py -v

After running, check the ./test_outputs/ folder for:
    - basic_report.pdf
    - thorough_preset_report.pdf
    - dry_run_report.pdf
    - compare_fast_vs_thorough.pdf
"""

import os
import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import preflight as pf
from preflight.types import FeatureConfig, SemanticType
from preflight.cli import app


runner = CliRunner()

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# Shared fixtures — a realistic, messy dataset covering every column type
# ===========================================================================

@pytest.fixture(scope="module")
def full_dataset():
    np.random.seed(42)
    n = 400

    df = pd.DataFrame({
        # numeric, clean
        "age": np.random.randint(18, 80, size=n),
        # numeric, skewed + some outliers
        "income": np.concatenate([
            np.random.normal(50000, 15000, size=n - 5),
            [500000, 600000, 700000, -1000, 0],  # outliers + a couple of odd values
        ]),
        # numeric with missingness
        "credit_score": np.where(
            np.random.random(n) < 0.1, np.nan, np.random.uniform(300, 850, size=n)
        ),
        # low-cardinality categorical
        "region": np.random.choice(["north", "south", "east", "west"], size=n),
        # high-cardinality categorical (target-encoding candidate)
        "city": np.random.choice([f"city_{i}" for i in range(60)], size=n),
        # boolean-like
        "is_member": np.random.choice([True, False], size=n),
        # constant column (edge case)
        "country": ["IN"] * n,
        # datetime
        "signup_date": pd.date_range("2019-01-01", periods=n, freq="D"),
        # free text
        "feedback": [
            "The overall experience was quite good and I would recommend this to others",
            "Not satisfied with the service, response times were too slow for my liking",
            "Average experience overall, nothing particularly stood out either way",
            "Excellent support team, resolved my issue within a few minutes flat",
            "Would not use again, too many hidden fees and unclear pricing structure",
        ] * (n // 5),
        # numeric ID
        "customer_id": [f"CUST-{i:06d}" for i in range(n)],
        # target
        "churned": np.random.randint(0, 2, size=n),
    })
    return df


@pytest.fixture(scope="module")
def regression_dataset():
    np.random.seed(7)
    n = 250
    return pd.DataFrame({
        "sqft": np.random.uniform(500, 4000, size=n),
        "bedrooms": np.random.randint(1, 6, size=n),
        "neighborhood": np.random.choice(["a", "b", "c", "d"], size=n),
        "listing_notes": [
            "Spacious home with updated kitchen and large backyard perfect for families"
        ] * n,
        "price": np.random.uniform(100000, 900000, size=n),
    })


# ===========================================================================
# 1. v0.1.0 core API surface
# ===========================================================================

class TestV0_1_0_CoreAPI:

    def test_prepare_returns_full_prepresult(self, full_dataset):
        result = pf.prepare(full_dataset, target="churned", task="classification")
        assert result.df is not None
        assert result.pipeline is not None
        assert result.report is not None

    def test_profile_returns_no_pipeline(self, full_dataset):
        result = pf.profile(full_dataset, target="churned", task="classification")
        assert result.pipeline is None
        assert result.df is not None

    def test_clean_returns_no_pipeline(self, full_dataset):
        result = pf.clean(full_dataset, target="churned", task="classification")
        assert result.pipeline is None

    def test_engineer_returns_no_pipeline(self, full_dataset):
        result = pf.engineer(full_dataset, target="churned", task="classification")
        assert result.pipeline is None

    def test_compare_two_results(self, full_dataset):
        result_a = pf.prepare(full_dataset, target="churned", task="classification", model_hint="tree")
        result_b = pf.prepare(full_dataset, target="churned", task="classification", model_hint="linear")
        diff = pf.compare(result_a, result_b)
        assert diff is not None

    def test_model_hint_tree_vs_linear_differ(self, full_dataset):
        result_tree = pf.prepare(full_dataset, target="churned", task="classification", model_hint="tree")
        result_linear = pf.prepare(full_dataset, target="churned", task="classification", model_hint="linear")
        # linear (one-hot + scaling) should generally produce a different column count than tree (ordinal)
        assert result_tree.df.shape[1] != result_linear.df.shape[1] or \
               list(result_tree.df.columns) != list(result_linear.df.columns)

    def test_pipeline_reusable_on_holdout(self, full_dataset):
        train = full_dataset.iloc[:300]
        holdout = full_dataset.iloc[300:].drop(columns=["churned"])
        result = pf.prepare(train, target="churned", task="classification")
        transformed_holdout = result.pipeline.transform(holdout)
        assert transformed_holdout is not None
        assert len(transformed_holdout) == len(holdout)

    def test_regression_task_end_to_end(self, regression_dataset):
        result = pf.prepare(regression_dataset, target="price", task="regression")
        assert result.df is not None
        assert result.pipeline is not None


# ===========================================================================
# 2. v0.2.0 features
# ===========================================================================

class TestV0_2_0_Features:

    def test_task_target_mismatch_raises(self, regression_dataset):
        with pytest.raises(ValueError):
            pf.prepare(regression_dataset, target="price", task="classification")

    def test_feature_config_all_off_matches_v010_behavior(self, full_dataset):
        baseline = pf.prepare(full_dataset, target="churned", task="classification")
        explicit_off = pf.prepare(
            full_dataset, target="churned", task="classification", feature_config=FeatureConfig()
        )
        assert list(baseline.df.columns) == list(explicit_off.df.columns)

    def test_interactions_generate_columns(self, full_dataset):
        config = FeatureConfig(interactions=True, interaction_top_k=3)
        result = pf.prepare(full_dataset, target="churned", task="classification", feature_config=config)
        assert result.df.shape[1] > 0

    def test_datetime_cyclical_generates_columns(self, full_dataset):
        config = FeatureConfig(datetime_cyclical=True)
        result = pf.prepare(full_dataset, target="churned", task="classification", feature_config=config)
        cyclical_cols = [c for c in result.df.columns if "sin" in c.lower() or "cos" in c.lower()]
        assert len(cyclical_cols) > 0

    def test_clustering_generates_columns(self, full_dataset):
        config = FeatureConfig(clustering=True, cluster_k=3)
        result = pf.prepare(full_dataset, target="churned", task="classification", feature_config=config)
        cluster_cols = [c for c in result.df.columns if "cluster" in c.lower()]
        assert len(cluster_cols) > 0

    def test_add_features_post_hoc(self, full_dataset):
        base_result = pf.prepare(full_dataset, target="churned", task="classification")
        config = FeatureConfig(interactions=True, interaction_top_k=2)
        new_result = pf.add_features(base_result, config)
        assert new_result.df.shape[1] >= base_result.df.shape[1]
        # original must be untouched
        assert base_result.df.shape[1] == pf.prepare(full_dataset, target="churned", task="classification").df.shape[1]

    def test_add_features_requires_pipeline(self, full_dataset):
        profile_only = pf.profile(full_dataset, target="churned", task="classification")
        with pytest.raises(ValueError):
            pf.add_features(profile_only, FeatureConfig(interactions=True))

    def test_report_methods_all_present(self, full_dataset):
        result = pf.prepare(full_dataset, target="churned", task="classification")
        assert isinstance(result.report.summary_counts(), dict)
        assert isinstance(result.report.to_dict(), dict)
        assert isinstance(result.report.to_dataframe(), pd.DataFrame)
        figures = result.report.plot(kind="all")
        assert isinstance(figures, list)


# ===========================================================================
# 3. v1.0.0 Phase 1-2 — text detection + text features
# ===========================================================================

class TestV1_TextSupport:

    def test_text_column_detected(self, full_dataset):
        profiles, _entries = pf.run_profiler(full_dataset, target="churned", task="classification")
        by_name = {p.name: p for p in profiles}
        assert by_name["feedback"].semantic_type == SemanticType.TEXT

    def test_text_stats_populated(self, full_dataset):
        profiles, _entries = pf.run_profiler(full_dataset, target="churned", task="classification")
        by_name = {p.name: p for p in profiles}
        assert by_name["feedback"].text_avg_length is not None
        assert by_name["feedback"].text_avg_word_count is not None

    def test_text_features_off_by_default(self, full_dataset):
        result = pf.prepare(full_dataset, target="churned", task="classification")
        text_generated = [c for c in result.df.columns if c.startswith("feedback_")]
        assert text_generated == []

    def test_text_features_on_generates_columns(self, full_dataset):
        config = FeatureConfig(text_features=True, text_tfidf=True, text_tfidf_top_k=5)
        result = pf.prepare(full_dataset, target="churned", task="classification", feature_config=config)
        text_generated = [c for c in result.df.columns if c.startswith("feedback_")]
        assert len(text_generated) > 0
        assert any("char_length" in c for c in text_generated)
        assert any("word_count" in c for c in text_generated)
        assert any("has_text" in c for c in text_generated)


# ===========================================================================
# 4. v1.0.0 Phase 3 — column_types override
# ===========================================================================

class TestV1_ColumnTypeOverride:

    def test_override_changes_type(self, full_dataset):
        result = pf.prepare(
            full_dataset, target="churned", task="classification",
            column_types={"customer_id": SemanticType.CATEGORICAL_HIGH},
        )
        assert result.df is not None

    def test_nonexistent_column_override_raises(self, full_dataset):
        with pytest.raises(ValueError):
            pf.prepare(
                full_dataset, target="churned", task="classification",
                column_types={"not_a_column": SemanticType.TEXT},
            )


# ===========================================================================
# 5. v1.0.0 Phase 4 — presets
# ===========================================================================

class TestV1_Presets:

    def test_fast_preset_runs(self, full_dataset):
        result = pf.prepare(full_dataset, target="churned", task="classification", preset="fast")
        assert result.df is not None

    def test_thorough_preset_runs_and_generates_more_columns(self, full_dataset):
        fast_result = pf.prepare(full_dataset, target="churned", task="classification", preset="fast")
        thorough_result = pf.prepare(full_dataset, target="churned", task="classification", preset="thorough")
        assert thorough_result.df.shape[1] >= fast_result.df.shape[1]

    def test_explicit_kwarg_overrides_preset(self, full_dataset):
        config = FeatureConfig()  # everything off, explicit
        result = pf.prepare(
            full_dataset, target="churned", task="classification",
            preset="thorough", feature_config=config,
        )
        text_generated = [c for c in result.df.columns if c.startswith("feedback_")]
        assert text_generated == []

    def test_invalid_preset_raises(self, full_dataset):
        with pytest.raises(ValueError):
            pf.prepare(full_dataset, target="churned", task="classification", preset="not_a_real_preset")


# ===========================================================================
# 6. v1.0.0 Phase 5 — dry_run
# ===========================================================================

class TestV1_DryRun:

    def test_dry_run_returns_untouched_df(self, full_dataset):
        original_copy = full_dataset.copy(deep=True)
        result = pf.prepare(full_dataset, target="churned", task="classification", dry_run=True)
        pd.testing.assert_frame_equal(result.df, original_copy)
        assert result.pipeline is None

    def test_dry_run_report_has_entries(self, full_dataset):
        result = pf.prepare(full_dataset, target="churned", task="classification", dry_run=True)
        entries = result.report.to_dataframe()
        assert len(entries) > 0

    def test_add_features_rejects_dry_run_result(self, full_dataset):
        dry_result = pf.prepare(full_dataset, target="churned", task="classification", dry_run=True)
        with pytest.raises(ValueError):
            pf.add_features(dry_result, FeatureConfig(interactions=True))


# ===========================================================================
# 7. v1.0.0 Phase 6-7 — PDF report + PDF comparison (VISIBLE OUTPUT)
# ===========================================================================

class TestV1_PdfReports:

    def test_basic_save_pdf_visible_output(self, full_dataset):
        result = pf.prepare(full_dataset, target="churned", task="classification")
        path = os.path.join(OUTPUT_DIR, "basic_report.pdf")
        result.report.save_pdf(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        print(f"\n[VISIBLE OUTPUT] Basic PDF report saved to: {path}")

    def test_thorough_preset_save_pdf_visible_output(self, full_dataset):
        result = pf.prepare(full_dataset, target="churned", task="classification", preset="thorough")
        path = os.path.join(OUTPUT_DIR, "thorough_preset_report.pdf")
        result.report.save_pdf(path)
        assert os.path.exists(path)
        print(f"\n[VISIBLE OUTPUT] Thorough-preset PDF report saved to: {path}")

    def test_dry_run_save_pdf_visible_output(self, full_dataset):
        result = pf.prepare(full_dataset, target="churned", task="classification", dry_run=True)
        path = os.path.join(OUTPUT_DIR, "dry_run_report.pdf")
        result.report.save_pdf(path)
        assert os.path.exists(path)
        print(f"\n[VISIBLE OUTPUT] Dry-run PDF report saved to: {path}")

    def test_save_compare_pdf_visible_output(self, full_dataset):
        result_fast = pf.prepare(full_dataset, target="churned", task="classification", preset="fast")
        result_thorough = pf.prepare(full_dataset, target="churned", task="classification", preset="thorough")
        path = os.path.join(OUTPUT_DIR, "compare_fast_vs_thorough.pdf")
        pf.save_compare_pdf(result_fast, result_thorough, path)
        assert os.path.exists(path)
        print(f"\n[VISIBLE OUTPUT] Comparison PDF saved to: {path}")


# ===========================================================================
# 8. v1.0.0 Phase 8 — CLI, full flag matrix
# ===========================================================================

class TestV1_CliFullMatrix:

    def test_cli_basic_prepare(self, full_dataset, tmp_path):
        csv_path = tmp_path / "data.csv"
        full_dataset.to_csv(csv_path, index=False)
        result = runner.invoke(app, [
            "prepare", str(csv_path), "--target", "churned", "--task", "classification",
        ])
        assert result.exit_code == 0

    def test_cli_all_v1_flags_combined(self, full_dataset, tmp_path):
        csv_path = tmp_path / "data.csv"
        full_dataset.to_csv(csv_path, index=False)
        pdf_path = os.path.join(OUTPUT_DIR, "cli_full_flags_report.pdf")

        result = runner.invoke(app, [
            "prepare", str(csv_path), "--target", "churned", "--task", "classification",
            "--text-features", "--text-tfidf", "--text-tfidf-top-k", "5",
            "--column-type", "customer_id:CATEGORICAL_HIGH",
            "--save-pdf", pdf_path,
        ])
        assert result.exit_code == 0
        assert os.path.exists(pdf_path)
        print(f"\n[VISIBLE OUTPUT] CLI full-flags PDF report saved to: {pdf_path}")

    def test_cli_invalid_preset_clean_error(self, full_dataset, tmp_path):
        csv_path = tmp_path / "data.csv"
        full_dataset.to_csv(csv_path, index=False)
        result = runner.invoke(app, [
            "prepare", str(csv_path), "--target", "churned", "--task", "classification",
            "--preset", "not_real",
        ])
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output


# ===========================================================================
# 9. v1.0.0 Phase 8.1 — error message clarity spot checks
# ===========================================================================

class TestV1_ErrorMessageClarity:

    FORBIDDEN_JARGON = [
        "columnprofile", "semantic_type", "two-phase fit", "nonetype", "traceback",
    ]

    def _check_clean(self, message):
        lowered = message.lower()
        for term in self.FORBIDDEN_JARGON:
            assert term not in lowered

    def test_task_target_error_clean(self, regression_dataset):
        with pytest.raises(ValueError) as exc_info:
            pf.prepare(regression_dataset, target="price", task="classification")
        self._check_clean(str(exc_info.value))

    def test_preset_error_clean(self, full_dataset):
        with pytest.raises(ValueError) as exc_info:
            pf.prepare(full_dataset, target="churned", task="classification", preset="bogus")
        self._check_clean(str(exc_info.value))

    def test_column_types_error_clean(self, full_dataset):
        with pytest.raises(ValueError) as exc_info:
            pf.prepare(
                full_dataset, target="churned", task="classification",
                column_types={"not_a_column": SemanticType.TEXT},
            )
        self._check_clean(str(exc_info.value))


# ===========================================================================
# 10. Fuzz / robustness sweep — edge case DataFrames should never hard-crash
# ===========================================================================

class TestRobustnessSweep:

    @pytest.mark.parametrize("case_name,build_df", [
        ("all_nan_column", lambda: pd.DataFrame({
            "a": [np.nan] * 50, "b": np.random.uniform(0, 1, 50), "target": np.random.randint(0, 2, 50)
        })),
        ("single_unique_value", lambda: pd.DataFrame({
            "a": ["same"] * 50, "b": np.random.uniform(0, 1, 50), "target": np.random.randint(0, 2, 50)
        })),
        ("very_wide_low_rows", lambda: pd.DataFrame(
            {**{f"col_{i}": np.random.uniform(0, 1, 10) for i in range(30)},
             "target": np.random.randint(0, 2, 10)}
        )),
        ("unicode_categoricals", lambda: pd.DataFrame({
            "a": np.random.choice(["café", "naïve", "北京", "🚀rocket"], size=50),
            "b": np.random.uniform(0, 1, 50),
            "target": np.random.randint(0, 2, 50),
        })),
        ("mixed_type_object_column", lambda: pd.DataFrame({
            "a": [1, "two", 3.0, None, "five"] * 10,
            "b": np.random.uniform(0, 1, 50),
            "target": np.random.randint(0, 2, 50),
        })),
    ])
    def test_no_ugly_crash(self, case_name, build_df):
        df = build_df()
        try:
            result = pf.prepare(df, target="target", task="classification")
            assert result.df is not None
        except (ValueError, TypeError) as e:
            # acceptable — must be a clear, intentional error, not a bare crash
            assert str(e), f"{case_name}: raised an exception with an empty message"