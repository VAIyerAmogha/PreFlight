# tests/test_init.py (continued — compare())
import pandas as pd
import numpy as np
import pytest
import preflight as pf

@pytest.fixture
def sample_df():
    n = 100
    return pd.DataFrame({
        "age": np.random.normal(40, 10, n),
        "city": np.random.choice(["NYC", "LA", "SF"], n),
        "target": np.random.choice([0, 1], n),
    })

def test_compare_returns_dict(sample_df, capsys):
    tree_result = pf.prepare(sample_df, target="target", model_hint="tree")
    linear_result = pf.prepare(sample_df, target="target", model_hint="linear")
    diff = pf.compare(tree_result, linear_result)
    assert isinstance(diff, dict)

def test_compare_prints_summary(sample_df, capsys):
    tree_result = pf.prepare(sample_df, target="target", model_hint="tree")
    linear_result = pf.prepare(sample_df, target="target", model_hint="linear")
    pf.compare(tree_result, linear_result)
    captured = capsys.readouterr()
    assert len(captured.out) > 0

def test_compare_shapes_included(sample_df):
    tree_result = pf.prepare(sample_df, target="target", model_hint="tree")
    linear_result = pf.prepare(sample_df, target="target", model_hint="linear")
    diff = pf.compare(tree_result, linear_result)
    assert diff["shape_a"] == tree_result.df.shape
    assert diff["shape_b"] == linear_result.df.shape

def test_compare_column_diffs(sample_df):
    tree_result = pf.prepare(sample_df, target="target", model_hint="tree")
    linear_result = pf.prepare(sample_df, target="target", model_hint="linear")
    diff = pf.compare(tree_result, linear_result)
    assert isinstance(diff["columns_only_in_a"], list)
    assert isinstance(diff["columns_only_in_b"], list)
    assert isinstance(diff["columns_in_both"], list)

def test_compare_columns_only_in_linear_includes_onehot():
    n = 100
    df = pd.DataFrame({
        "city": np.random.choice(["NYC", "LA", "SF"], n),
        "target": np.random.choice([0, 1], n),
    })
    tree_result = pf.prepare(df, target="target", model_hint="tree")
    linear_result = pf.prepare(df, target="target", model_hint="linear")
    diff = pf.compare(tree_result, linear_result)
    onehot_cols = [c for c in diff["columns_only_in_b"] if c.startswith("city_")]
    assert len(onehot_cols) > 0

def test_compare_report_entry_counts_included(sample_df):
    tree_result = pf.prepare(sample_df, target="target", model_hint="tree")
    linear_result = pf.prepare(sample_df, target="target", model_hint="linear")
    diff = pf.compare(tree_result, linear_result)
    assert diff["report_entry_counts_a"] is not None
    assert diff["report_entry_counts_b"] is not None

def test_compare_handles_none_report_gracefully(sample_df):
    from preflight.types import PrepResult
    result_a = PrepResult(df=sample_df, pipeline=None, report=None)
    result_b = pf.prepare(sample_df, target="target", model_hint="tree")
    diff = pf.compare(result_a, result_b)
    assert diff["report_entry_counts_a"] is None
    assert diff["decision_diff"] == []

def test_compare_decision_diff_detects_different_encoding(sample_df):
    tree_result = pf.prepare(sample_df, target="target", model_hint="tree")
    linear_result = pf.prepare(sample_df, target="target", model_hint="linear")
    diff = pf.compare(tree_result, linear_result)
    assert isinstance(diff["decision_diff"], list)

def test_compare_exposed_at_top_level():
    assert hasattr(pf, "compare")
    assert callable(pf.compare)

def test_compare_identical_results_no_column_diff(sample_df):
    result_a = pf.prepare(sample_df, target="target", model_hint="tree")
    result_b = pf.prepare(sample_df, target="target", model_hint="tree")
    diff = pf.compare(result_a, result_b)
    assert diff["columns_only_in_a"] == []
    assert diff["columns_only_in_b"] == []