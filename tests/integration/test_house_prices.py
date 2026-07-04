# tests/integration/test_house_prices.py
import pandas as pd
import numpy as np
from scipy.stats import skew
import pytest
import preflight as pf

pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def house_prices_df():
    rng = np.random.default_rng(7)
    n = 300
    neighborhoods = [f"neigh_{i}" for i in range(35)]  # high cardinality
    return pd.DataFrame({
        "neighborhood": rng.choice(neighborhoods, n),
        "lot_size": rng.exponential(5000, n),          # heavily skewed
        "num_bedrooms": rng.normal(3, 1, n).round(),    # roughly normal
        "price": rng.lognormal(mean=12, sigma=0.4, size=n),  # skewed target
    })

def test_full_prepare_linear_completes(house_prices_df):
    result = pf.prepare(house_prices_df, target="price", task="regression", model_hint="linear")
    assert result.df is not None

def test_skewed_feature_transformed_toward_normal_linear(house_prices_df):
    result = pf.prepare(house_prices_df, target="price", task="regression", model_hint="linear")
    original_skew = abs(skew(house_prices_df["lot_size"]))
    transformed_skew = abs(skew(result.df["lot_size"]))
    assert transformed_skew < original_skew

def test_skewed_feature_untouched_in_tree_mode(house_prices_df):
    result = pf.prepare(house_prices_df, target="price", task="regression", model_hint="tree")
    orig = house_prices_df["lot_size"]
    out = result.df["lot_size"]
    assert out.mean() == pytest.approx(orig.mean(), rel=0.15)
    # The max value will be reduced by Cleaner's winsorization, but median is robust
    assert out.median() == pytest.approx(orig.median(), rel=0.01)

def test_high_cardinality_becomes_single_numeric_column(house_prices_df):
    result = pf.prepare(house_prices_df, target="price", task="regression", model_hint="linear")
    onehot_expansion = [c for c in result.df.columns if c.startswith("neighborhood_")]
    assert onehot_expansion == []
    assert "neighborhood" in result.df.columns
    assert pd.api.types.is_numeric_dtype(result.df["neighborhood"])

def test_target_never_transformed(house_prices_df):
    result = pf.prepare(house_prices_df, target="price", task="regression", model_hint="linear")
    pd.testing.assert_series_equal(
        result.df["price"].reset_index(drop=True),
        house_prices_df["price"].reset_index(drop=True),
    )

def test_pipeline_uses_frozen_target_encoding_on_holdout(house_prices_df):
    result = pf.prepare(house_prices_df, target="price", task="regression", model_hint="tree")
    holdout = house_prices_df.drop(columns=["price"]).iloc[:5].copy()
    t1 = result.pipeline.transform(holdout)
    t2 = result.pipeline.transform(holdout)
    pd.testing.assert_frame_equal(t1, t2)  # deterministic reuse of frozen mapping

def test_report_queryable_for_numeric_columns(house_prices_df):
    result = pf.prepare(house_prices_df, target="price", task="regression", model_hint="linear")
    lot_size_entries = result.report.filter_by_column("lot_size")
    assert isinstance(lot_size_entries, list)  # no crash, queryable regardless of VIF firing