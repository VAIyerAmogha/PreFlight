"""
Regression: text tidying on a pandas Categorical column.

The typed table (ApplyResult.data) holds low-variety text columns as `category`.
Lowercasing such a column changes its category set ('Male' -> 'male'), and pandas
refuses to compare two Categoricals with different category sets. Planning the
tidy step must compare the plain values instead of raising TypeError.
"""

import tempfile
from pathlib import Path

import pandas as pd

import preflight
from preflight.assemble import build_pipeline_two_phase
from preflight.clean.normalize import plan_normalization


def _categorical_frame(n: int = 60) -> pd.DataFrame:
    sex = pd.Series((["Female", "Male", " male "] * n)[:n], dtype="category")
    day = pd.Series((["Sun", "Sat", "Thur"] * n)[:n], dtype="category")
    return pd.DataFrame({"sex": sex, "day": day, "y": [float(i) for i in range(n)]})


def test_plan_normalization_handles_categorical_dtype() -> None:
    df = _categorical_frame()
    decisions = plan_normalization(df, target="y")
    tidy = {d.columns[0]: d for d in decisions if d.kind == "normalize_categories"}
    assert set(tidy) == {"sex", "day"}
    # Every value has an upper-case letter or extra spaces, so every entry changes.
    assert tidy["sex"].evidence["entries_changed"] == len(df)
    assert tidy["day"].evidence["entries_changed"] == len(df)


def test_two_phase_pipeline_on_typed_table_with_categoricals() -> None:
    df = _categorical_frame().rename(columns={"y": "target"})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cats.csv"
        df.to_csv(path, index=False)
        plan = preflight.plan(path, target="target", task="regression")
        result = preflight.apply(plan, mode="defaults", out_dir=Path(tmp) / "out")
        assert any(isinstance(dt, pd.CategoricalDtype) for dt in result.data.dtypes)
        X = result.data.drop(columns=["target"])
        y = result.data["target"]
        pipe = build_pipeline_two_phase(
            X, y, "target", decisions=result.decisions, schema=plan.schema, task="regression"
        )
        out = pipe.transform(X)
        assert len(out) == len(X)
