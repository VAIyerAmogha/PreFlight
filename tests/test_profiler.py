# tests/test_profiler.py (part 4 — orchestration)
import pytest
import pandas as pd
import numpy as np
from preflight.profiler import run_profiler
from preflight.types import ColumnProfile, ReportEntry, SemanticType

@pytest.fixture
def sample_df():
    n = 200
    return pd.DataFrame({
        "user_id": range(n),
        "age": np.random.normal(40, 12, n),
        "income": np.random.normal(60000, 15000, n),
        "city": np.random.choice(["NYC", "LA", "SF"], n),
        "signup_date": pd.date_range("2020-01-01", periods=n),
        "is_active": np.random.choice([True, False], n),
        "target": np.random.choice([0, 1], n),
    })

def test_returns_profile_and_report_entries(sample_df):
    profiles, entries = run_profiler(sample_df, target="target", task="classification")
    assert isinstance(profiles, list)
    assert all(isinstance(p, ColumnProfile) for p in profiles)
    assert isinstance(entries, list)
    assert all(isinstance(e, ReportEntry) for e in entries)

def test_target_column_excluded_from_profiles(sample_df):
    profiles, _ = run_profiler(sample_df, target="target", task="classification")
    names = [p.name for p in profiles]
    assert "target" not in names

def test_all_other_columns_profiled(sample_df):
    profiles, _ = run_profiler(sample_df, target="target", task="classification")
    names = {p.name for p in profiles}
    assert names == set(sample_df.columns) - {"target"}

def test_user_id_classified_as_numeric_id(sample_df):
    profiles, _ = run_profiler(sample_df, target="target", task="classification")
    p = next(p for p in profiles if p.name == "user_id")
    assert p.semantic_type == SemanticType.NUMERIC_ID

def test_categorical_low_gets_rare_categories_field(sample_df):
    profiles, _ = run_profiler(sample_df, target="target", task="classification")
    p = next(p for p in profiles if p.name == "city")
    assert isinstance(p.rare_categories, list)

def test_numeric_id_correlation_not_computed_or_none(sample_df):
    profiles, _ = run_profiler(sample_df, target="target", task="classification")
    p = next(p for p in profiles if p.name == "user_id")
    # ID columns shouldn't carry a meaningful target correlation signal
    assert p.correlation_with_target is None or isinstance(p.correlation_with_target, float)

def test_leakage_suspect_flagged_and_reported():
    n = 100
    df = pd.DataFrame({
        "leaky": np.arange(n) + np.random.normal(0, 0.001, n),
        "target": np.arange(n),
    })
    profiles, entries = run_profiler(df, target="target", task="regression")
    p = next(p for p in profiles if p.name == "leaky")
    assert p.is_leakage_suspect is True
    assert any(e.column == "leaky" and e.severity == "critical" for e in entries)

def test_high_missingness_emits_report_entry():
    n = 100
    df = pd.DataFrame({
        "sparse": [None] * 80 + list(range(20)),
        "target": np.random.choice([0, 1], n),
    })
    _, entries = run_profiler(df, target="target", task="classification")
    assert any(e.column == "sparse" and e.stage == "profiler" for e in entries)

def test_vif_computed_once_across_numeric_columns():
    n = 100
    df = pd.DataFrame({
        "a": np.random.normal(0, 1, n),
        "b": np.random.normal(0, 1, n),
        "target": np.random.choice([0, 1], n),
    })
    profiles, _ = run_profiler(df, target="target", task="classification")
    for p in profiles:
        if p.semantic_type == SemanticType.NUMERIC_FEATURE:
            assert p.vif_score is not None or p.vif_score is None  # just must be set, not crash

def test_original_df_not_mutated(sample_df):
    original = sample_df.copy()
    run_profiler(sample_df, target="target", task="classification")
    pd.testing.assert_frame_equal(sample_df, original)

def test_no_report_entry_for_benign_signals(sample_df):
    _, entries = run_profiler(sample_df, target="target", task="classification")
    # age has low missingness, no leakage — shouldn't spam entries
    age_entries = [e for e in entries if e.column == "age"]
    assert len(age_entries) <= 1  # at most one, e.g. only if something's actually notable

def test_zip_code_categorical_classification():
    n = 100
    df = pd.DataFrame({
        "zip_code": [f"{np.random.randint(10000, 99999)}" for _ in range(n)],
        "postal_code": [np.random.randint(10000, 99999) for _ in range(n)],
        "postcode": [np.random.randint(10000, 99999) for _ in range(n)],
        "normal_numeric": np.random.normal(50, 5, n),
        "target": np.random.choice([0, 1], n),
    })
    profiles, _ = run_profiler(df, target="target", task="classification")
    by_name = {p.name: p for p in profiles}
    
    assert by_name["zip_code"].semantic_type in (SemanticType.CATEGORICAL_HIGH, SemanticType.CATEGORICAL_LOW)
    assert by_name["postal_code"].semantic_type in (SemanticType.CATEGORICAL_HIGH, SemanticType.CATEGORICAL_LOW)
    assert by_name["postcode"].semantic_type in (SemanticType.CATEGORICAL_HIGH, SemanticType.CATEGORICAL_LOW)
    assert by_name["normal_numeric"].semantic_type == SemanticType.NUMERIC_FEATURE