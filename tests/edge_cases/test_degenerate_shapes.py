# tests/edge_cases/test_degenerate_shapes.py
import pandas as pd
import numpy as np
import pytest
import preflight as pf

def test_empty_dataframe_raises_value_error():
    df = pd.DataFrame({"age": pd.Series([], dtype=float), "target": pd.Series([], dtype=int)})
    with pytest.raises(ValueError):
        pf.prepare(df, target="target")

def test_single_row_dataframe_no_raw_crash():
    df = pd.DataFrame({
        "age": [30.0],
        "city": ["NYC"],
        "target": [1],
    })
    try:
        result = pf.prepare(df, target="target", model_hint="tree")
        assert result.df is not None
    except ValueError as e:
        assert "insufficient" in str(e).lower() or "data" in str(e).lower()
    except Exception as e:
        pytest.fail(f"Raised an unclear/raw exception type: {type(e).__name__}: {e}")

def test_single_column_dataframe_raises_clear_error():
    df = pd.DataFrame({"target": [0, 1, 0, 1, 1]})
    with pytest.raises(ValueError):
        pf.prepare(df, target="target")

def test_duplicate_column_names_no_ambiguous_crash():
    df = pd.DataFrame(
        np.random.normal(0, 1, (20, 2)),
        columns=["age", "age"],
    )
    df["target"] = np.random.choice([0, 1], 20)
    try:
        pf.prepare(df, target="target", model_hint="tree")
    except ValueError:
        pass  # acceptable: clean rejection
    except Exception as e:
        pytest.fail(f"Raised an unclear exception on duplicate columns: {type(e).__name__}: {e}")

def test_small_but_valid_dataframe_completes_successfully():
    n = 8
    df = pd.DataFrame({
        "age": np.random.normal(35, 5, n),
        "city": np.random.choice(["NYC", "LA"], n),
        "signup": pd.date_range("2023-01-01", periods=n, freq="D"),
        "target": np.random.choice([0, 1], n),
    })
    result = pf.prepare(df, target="target", task="classification", model_hint="tree")
    assert result.df is not None
    assert len(result.df) == n
    assert result.pipeline is not None