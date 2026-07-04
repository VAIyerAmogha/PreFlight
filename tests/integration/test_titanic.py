# tests/integration/test_titanic.py
import pandas as pd
import numpy as np
import pytest
import preflight as pf

pytestmark = pytest.mark.integration

def _load_titanic() -> pd.DataFrame:
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        # normalize to a stable minimal schema for this test
        df = df.rename(columns={"survived": "target"})
        return df
    except Exception:
        # offline fallback reproducing Titanic's known problem shape
        n = 200
        rng = np.random.default_rng(42)
        age = rng.normal(30, 12, n)
        age[rng.choice(n, size=int(n * 0.2), replace=False)] = np.nan
        cabin = np.array([None] * int(n * 0.77) + ["C85"] * (n - int(n * 0.77)))
        rng.shuffle(cabin)
        return pd.DataFrame({
            "pclass": rng.choice([1, 2, 3], n),
            "sex": rng.choice(["male", "female"], n),
            "age": age,
            "fare": rng.exponential(30, n),
            "embarked": rng.choice(["S", "C", "Q"], n),
            "cabin": cabin,
            "target": rng.choice([0, 1], n),
        })

@pytest.fixture(scope="module")
def titanic_df():
    return _load_titanic()

def test_full_prepare_tree_completes(titanic_df):
    result = pf.prepare(titanic_df, target="target", task="classification", model_hint="tree")
    assert result.df is not None

def test_full_prepare_linear_completes(titanic_df):
    result = pf.prepare(titanic_df, target="target", task="classification", model_hint="linear")
    assert result.df is not None

def test_no_nulls_outside_target(titanic_df):
    result = pf.prepare(titanic_df, target="target", task="classification", model_hint="tree")
    non_target = result.df.drop(columns=["target"])
    assert non_target.isnull().sum().sum() == 0

def test_high_missingness_column_dropped_or_indicator_added(titanic_df):
    result = pf.prepare(titanic_df, target="target", task="classification", model_hint="tree")
    cabin_present = "cabin" in result.df.columns
    cabin_indicator_present = any(c.startswith("cabin") and c.endswith("_missing") for c in result.df.columns)
    cabin_dropped = "cabin" not in titanic_df.columns or not cabin_present
    assert cabin_dropped or cabin_indicator_present

def test_pipeline_transform_on_holdout(titanic_df):
    result = pf.prepare(titanic_df, target="target", task="classification", model_hint="tree")
    holdout = titanic_df.drop(columns=["target"]).iloc[:20]
    transformed = result.pipeline.transform(holdout)
    expected_cols = set(result.df.columns) - {"target"}
    assert set(transformed.columns) == expected_cols

def test_report_has_all_three_stages(titanic_df):
    result = pf.prepare(titanic_df, target="target", task="classification", model_hint="tree")
    stages = {e.stage for e in result.report.entries}
    assert "profiler" in stages
    assert "cleaner" in stages
    assert "engineer" in stages

def test_no_unexpected_critical_entries(titanic_df):
    result = pf.prepare(titanic_df, target="target", task="classification", model_hint="tree")
    critical = result.report.filter_by_severity("critical")
    assert len(critical) == 0

def test_compare_tree_vs_linear_shows_column_diff(titanic_df):
    tree_result = pf.prepare(titanic_df, target="target", task="classification", model_hint="tree")
    linear_result = pf.prepare(titanic_df, target="target", task="classification", model_hint="linear")
    diff = pf.compare(tree_result, linear_result)
    assert len(diff["columns_only_in_b"]) > 0