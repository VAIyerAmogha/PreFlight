# tests/test_cleaner.py (part 4 — orchestration)
import pandas as pd
import numpy as np
import pytest
from preflight.cleaner import run_cleaner
from preflight.types import ColumnProfile, ReportEntry, SemanticType

def make_profile(name, semantic_type, missing_rate=0.0, cardinality=5, rare_categories=None):
    return ColumnProfile(
        name=name, semantic_type=semantic_type, missing_rate=missing_rate,
        outlier_rate=None, cardinality=cardinality, rare_categories=rare_categories or [],
        vif_score=None, correlation_with_target=None, mutual_info_with_target=None,
        is_leakage_suspect=False, dtype="object",
    )

@pytest.fixture
def sample_df_and_profiles():
    n = 100
    df = pd.DataFrame({
        "user_id": range(n),
        "age": [30] * 90 + [None] * 10,
        "junk": [None] * 70 + list(range(30)),  # 70% missing, should be dropped
        "city": ["NYC", " la ", "SF"] * 33 + ["NYC"],
        "target": np.random.choice([0, 1], n),
    })
    profiles = [
        make_profile("user_id", SemanticType.NUMERIC_ID),
        make_profile("age", SemanticType.NUMERIC_FEATURE, missing_rate=0.10),
        make_profile("junk", SemanticType.NUMERIC_FEATURE, missing_rate=0.70),
        make_profile("city", SemanticType.CATEGORICAL_LOW, missing_rate=0.0, cardinality=3),
    ]
    return df, profiles

def test_duplicates_removed_first(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    df_with_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned, _, entries, _ = run_cleaner(df_with_dupe, profiles, target="target")
    assert len(cleaned) == len(df)  # dupe removed

def test_high_missingness_column_dropped(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    cleaned, remaining_profiles, entries, _ = run_cleaner(df, profiles, target="target")
    assert "junk" not in cleaned.columns
    assert not any(p.name == "junk" for p in remaining_profiles)
    assert any(e.column == "junk" and e.severity == "warning" for e in entries)

def test_numeric_id_dropped(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    cleaned, remaining_profiles, entries, _ = run_cleaner(df, profiles, target="target")
    assert "user_id" not in cleaned.columns

def test_numeric_feature_imputed_no_nulls(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    cleaned, _, _, _ = run_cleaner(df, profiles, target="target")
    assert cleaned["age"].isnull().sum() == 0

def test_missing_indicator_added_when_missingness_present(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    cleaned, _, _, _ = run_cleaner(df, profiles, target="target")
    assert "age_missing" in cleaned.columns

def test_categorical_normalized(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    cleaned, _, _, _ = run_cleaner(df, profiles, target="target")
    assert "la" in cleaned["city"].values
    assert " la " not in cleaned["city"].values

def test_outlier_handling_skipped_above_30pct_missingness():
    n = 100
    df = pd.DataFrame({
        "sparse_numeric": [None] * 40 + list(range(60)) + [10000],  # >30% missing + outlier
        "target": np.random.choice([0, 1], n + 1),
    })
    profiles = [make_profile("sparse_numeric", SemanticType.NUMERIC_FEATURE, missing_rate=0.40)]
    cleaned, _, entries, _ = run_cleaner(df, profiles, target="target")
    assert any(
        e.column == "sparse_numeric" and "skip" in e.rationale.lower()
        for e in entries
    )

def test_transformer_specs_returned(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    _, _, _, specs = run_cleaner(df, profiles, target="target")
    assert "age" in specs
    assert isinstance(specs["age"], dict) or specs["age"] is not None

def test_report_entries_emitted_for_actions(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    _, _, entries, _ = run_cleaner(df, profiles, target="target")
    assert len(entries) > 0
    assert all(e.stage == "cleaner" for e in entries)

def test_original_df_not_mutated(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    original = df.copy()
    run_cleaner(df, profiles, target="target")
    pd.testing.assert_frame_equal(df, original)

def test_target_column_untouched(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    cleaned, _, _, _ = run_cleaner(df, profiles, target="target")
    assert "target" in cleaned.columns
    pd.testing.assert_series_equal(
        cleaned["target"].reset_index(drop=True),
        df["target"].reset_index(drop=True),
    )