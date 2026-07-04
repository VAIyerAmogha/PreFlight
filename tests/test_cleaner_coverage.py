import pytest
import pandas as pd
from preflight.cleaner import coerce_string_dates_to_datetime, winsorize_outliers, median_impute

def test_coerce_string_dates_invalid():
    # Contains a truly invalid date to hit the exception block
    s = pd.Series(["2021-01-01", "not_a_date_at_all"])
    res = coerce_string_dates_to_datetime(s)
    assert pd.isnull(res.iloc[1])

def test_winsorize_outliers_invalid_method():
    s = pd.Series([1, 2, 3, 100])
    with pytest.raises(ValueError):
        winsorize_outliers(s, method="invalid")

def test_median_impute_non_numeric():
    s = pd.Series(["a", "b", "c"])
    with pytest.raises(ValueError):
        median_impute(s)

def test_winsorize_outliers_non_numeric():
    s = pd.Series(["a", "b", "c"])
    with pytest.raises(ValueError):
        winsorize_outliers(s)
