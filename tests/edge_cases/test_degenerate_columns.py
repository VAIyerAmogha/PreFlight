# tests/edge_cases/test_degenerate_columns.py
import pandas as pd
import numpy as np
import pytest
import preflight as pf

@pytest.fixture
def base_df():
    n = 100
    return pd.DataFrame({
        "age": np.random.normal(40, 10, n),
        "city": np.random.choice(["NYC", "LA", "SF"], n),
        "target": np.random.choice([0, 1], n),
    })

def test_all_null_column_dropped_or_ignored(base_df):
    df = base_df.copy()
    df["all_null"] = None
    result = pf.prepare(df, target="target", model_hint="tree")
    assert "all_null" not in result.df.columns

def test_all_null_column_no_crash_linear(base_df):
    df = base_df.copy()
    df["all_null"] = None
    result = pf.prepare(df, target="target", model_hint="linear")
    assert result.df is not None

def test_single_category_categorical_no_crash(base_df):
    df = base_df.copy()
    df["country"] = "USA"
    result = pf.prepare(df, target="target", model_hint="tree")
    assert result.df is not None
    # if retained at all, must not contain nulls
    if "country" in result.df.columns:
        assert result.df["country"].isnull().sum() == 0

def test_numeric_constant_column_not_scaled(base_df):
    df = base_df.copy()
    df["const_num"] = 42.0
    result = pf.prepare(df, target="target", model_hint="linear")
    scale_entries = [
        e for e in result.report.entries
        if e.column == "const_num" and "scale" in e.action.lower()
    ]
    assert scale_entries == []

def test_numeric_constant_column_no_crash(base_df):
    df = base_df.copy()
    df["const_num"] = 42.0
    result = pf.prepare(df, target="target", model_hint="linear")
    assert result.df is not None

def test_all_columns_degenerate_tree(base_df):
    n = len(base_df)
    df = pd.DataFrame({
        "all_null": [None] * n,
        "single_cat": ["X"] * n,
        "const_num": [7.0] * n,
        "target": base_df["target"],
    })
    result = pf.prepare(df, target="target", model_hint="tree")
    assert result.df is not None
    assert "target" in result.df.columns

def test_all_columns_degenerate_linear(base_df):
    n = len(base_df)
    df = pd.DataFrame({
        "all_null": [None] * n,
        "single_cat": ["X"] * n,
        "const_num": [7.0] * n,
        "target": base_df["target"],
    })
    result = pf.prepare(df, target="target", model_hint="linear")
    assert result.df is not None
    assert "target" in result.df.columns

def test_degenerate_df_pipeline_transform_works(base_df):
    n = len(base_df)
    df = pd.DataFrame({
        "all_null": [None] * n,
        "const_num": [7.0] * n,
        "target": base_df["target"],
    })
    result = pf.prepare(df, target="target", model_hint="tree")
    holdout = df.drop(columns=["target"]).iloc[:5]
    transformed = result.pipeline.transform(holdout)
    assert isinstance(transformed, pd.DataFrame)