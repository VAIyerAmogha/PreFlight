# tests/test_engineer.py (part 4 — orchestration)
import pandas as pd
import numpy as np
import pytest
from preflight.engineer import run_engineer
from preflight.types import ColumnProfile, ReportEntry, SemanticType

def make_profile(name, semantic_type, cardinality=5, rare_categories=None):
    return ColumnProfile(
        name=name, semantic_type=semantic_type, missing_rate=0.0,
        outlier_rate=None, cardinality=cardinality, rare_categories=rare_categories or [],
        vif_score=None, correlation_with_target=None, mutual_info_with_target=None,
        is_leakage_suspect=False, dtype="object",
    )

@pytest.fixture
def sample_df_and_profiles():
    n = 100
    df = pd.DataFrame({
        "city": np.random.choice(["NYC", "LA", "SF"], n),
        "zip_high_card": np.random.choice([f"z{i}" for i in range(50)], n),
        "income": np.random.exponential(50000, n),  # skewed
        "signup": pd.date_range("2023-01-01", periods=n, freq="D"),
        "is_verified": np.random.choice([True, False], n),
        "target": np.random.uniform(0, 1, n),
    })
    profiles = [
        make_profile("city", SemanticType.CATEGORICAL_LOW, cardinality=3),
        make_profile("zip_high_card", SemanticType.CATEGORICAL_HIGH, cardinality=50),
        make_profile("income", SemanticType.NUMERIC_FEATURE),
        make_profile("signup", SemanticType.DATETIME_NATIVE),
        make_profile("is_verified", SemanticType.BOOLEAN),
    ]
    return df, profiles

def test_invalid_model_hint_raises(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    with pytest.raises(ValueError):
        run_engineer(df, profiles, target="target", model_hint="bogus")

def test_tree_hint_uses_ordinal_not_onehot(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    result, _, _ = run_engineer(df, profiles, target="target", model_hint="tree")
    assert "city" in result.columns
    assert "city_NYC" not in result.columns  # no one-hot expansion
    assert pd.api.types.is_numeric_dtype(result["city"])

def test_linear_hint_uses_onehot(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    result, _, _ = run_engineer(df, profiles, target="target", model_hint="linear")
    onehot_cols = [c for c in result.columns if c.startswith("city_")]
    assert len(onehot_cols) > 0
    assert "city" not in result.columns

def test_high_cardinality_target_encoded_both_hints(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    for hint in ("tree", "linear"):
        result, _, _ = run_engineer(df, profiles, target="target", model_hint=hint)
        assert pd.api.types.is_numeric_dtype(result["zip_high_card"])

def test_tree_hint_no_scaling_on_numeric(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    result, _, _ = run_engineer(df, profiles, target="target", model_hint="tree")
    # unscaled income should retain original scale (large values, not ~N(0,1))
    assert result["income"].std() > 10

def test_linear_hint_scales_numeric(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    result, _, _ = run_engineer(df, profiles, target="target", model_hint="linear")
    assert abs(result["income"].mean()) < 1.0  # scaled to ~mean 0

def test_datetime_expanded_both_hints(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    for hint in ("tree", "linear"):
        result, _, _ = run_engineer(df, profiles, target="target", model_hint=hint)
        assert "signup" not in result.columns
        assert "signup_year" in result.columns

def test_target_column_untouched(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    result, _, _ = run_engineer(df, profiles, target="target", model_hint="tree")
    pd.testing.assert_series_equal(
        result["target"].reset_index(drop=True),
        df["target"].reset_index(drop=True),
    )

def test_report_entries_emitted(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    _, entries, _ = run_engineer(df, profiles, target="target", model_hint="linear")
    assert len(entries) > 0
    assert all(e.stage == "engineer" for e in entries)

def test_target_encoding_leakage_note_logged(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    _, entries, _ = run_engineer(df, profiles, target="target", model_hint="tree")
    zip_entries = [e for e in entries if e.column == "zip_high_card"]
    assert any("leak" in e.rationale.lower() or "cross-fit" in e.rationale.lower() for e in zip_entries)

def test_transform_specs_returned(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    _, _, specs = run_engineer(df, profiles, target="target", model_hint="linear")
    assert "income" in specs
    assert "city" in specs

def test_original_df_not_mutated(sample_df_and_profiles):
    df, profiles = sample_df_and_profiles
    original = df.copy()
    run_engineer(df, profiles, target="target", model_hint="tree")
    pd.testing.assert_frame_equal(df, original)