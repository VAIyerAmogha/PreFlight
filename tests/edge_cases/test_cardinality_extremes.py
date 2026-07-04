# tests/edge_cases/test_cardinality_extremes.py
import pandas as pd
import numpy as np
import pytest
import preflight as pf

@pytest.fixture
def base_target():
    n = 100
    return pd.Series(np.random.choice([0, 1], n), name="target")

def test_fully_unique_string_column_target_encoded(base_target):
    n = len(base_target)
    df = pd.DataFrame({
        "notes": [f"unique_text_{i}" for i in range(n)],
        "age": np.random.normal(40, 10, n),
        "target": base_target,
    })
    result = pf.prepare(df, target="target", model_hint="tree")
    assert "notes" in result.df.columns
    assert pd.api.types.is_numeric_dtype(result.df["notes"])

def test_fully_unique_string_column_low_variance_after_smoothing(base_target):
    n = len(base_target)
    df = pd.DataFrame({
        "notes": [f"unique_text_{i}" for i in range(n)],
        "target": base_target.astype(float),
    })
    result = pf.prepare(df, target="target", task="regression", model_hint="tree")
    # every category occurs once -> smoothing should pull all values near global mean
    assert result.df["notes"].std() < result.df["target"].std() + 1e-6

def test_fully_unique_numeric_column_not_dropped_as_id(base_target):
    n = len(base_target)
    df = pd.DataFrame({
        "measurement": np.random.normal(0, 1, n) + np.arange(n) * 1e-6,  # unique floats
        "target": base_target,
    })
    result = pf.prepare(df, target="target", model_hint="tree")
    assert "measurement" in result.df.columns

def test_near_zero_variance_numeric_no_crash_linear(base_target):
    n = len(base_target)
    df = pd.DataFrame({
        "flat": 5.0 + np.random.normal(0, 1e-8, n),
        "target": base_target,
    })
    result = pf.prepare(df, target="target", model_hint="linear")
    assert result.df is not None
    assert not result.df["flat"].isnull().any()

def test_near_zero_variance_no_inf_or_nan_after_scaling(base_target):
    n = len(base_target)
    df = pd.DataFrame({
        "flat": 5.0 + np.random.normal(0, 1e-8, n),
        "target": base_target,
    })
    result = pf.prepare(df, target="target", model_hint="linear")
    assert np.isfinite(result.df["flat"]).all()

def test_long_tail_categorical_common_vs_rare_differ(base_target):
    n = len(base_target)
    common = ["common_a"] * 60 + ["common_b"] * 20
    rare = [f"rare_{i}" for i in range(20)]
    values = common + rare
    df = pd.DataFrame({
        "field": values,
        "target": base_target,
    })
    result = pf.prepare(df, target="target", model_hint="tree")
    encoded = result.df["field"]
    # common categories should retain more distinct/spread values than rare ones collapse to
    assert encoded.nunique() > 1