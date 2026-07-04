# tests/integration/test_adult_income.py
import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from typer.testing import CliRunner
import preflight as pf
from preflight.cli import app

pytestmark = pytest.mark.integration
runner = CliRunner()

@pytest.fixture(scope="module")
def adult_income_df():
    rng = np.random.default_rng(11)
    n = 400
    n_high = int(n * 0.92)  # 92/8 split ensures ratio > 10.0 to trigger profiler flag
    income = np.array(["<=50K"] * n_high + [">50K"] * (n - n_high))
    rng.shuffle(income)
    educations = [f"edu_{i}" for i in range(14)]  # near cardinality_threshold=20
    occupations = [f"occ_{i}" for i in range(45)]  # clearly high cardinality
    return pd.DataFrame({
        "age": rng.normal(38, 12, n).clip(18, 90),
        "hours_per_week": rng.normal(40, 10, n).clip(1, 99),
        "sex": rng.choice(["Male", "Female"], n),
        "education": rng.choice(educations, n),
        "occupation": rng.choice(occupations, n),
        "income": income,
    })

def _binarize_target(df):
    df = df.copy()
    df["income"] = (df["income"] == ">50K").astype(int)
    return df

def test_full_prepare_tree_completes(adult_income_df):
    df = _binarize_target(adult_income_df)
    result = pf.prepare(df, target="income", task="classification", model_hint="tree")
    assert result.df is not None

def test_full_prepare_linear_completes(adult_income_df):
    df = _binarize_target(adult_income_df)
    result = pf.prepare(df, target="income", task="classification", model_hint="linear")
    assert result.df is not None

def test_class_imbalance_flagged(adult_income_df):
    df = _binarize_target(adult_income_df)
    result = pf.prepare(df, target="income", task="classification", model_hint="tree")
    imbalance_entries = [
        e for e in result.report.entries
        if "imbalance" in e.action.lower() or "imbalance" in e.rationale.lower()
    ]
    assert len(imbalance_entries) > 0

def test_medium_cardinality_column_handled_consistently(adult_income_df):
    df = _binarize_target(adult_income_df)
    result = pf.prepare(df, target="income", task="classification", model_hint="tree")
    holdout = df.drop(columns=["income"]).iloc[:10]
    t1 = result.pipeline.transform(holdout)
    t2 = result.pipeline.transform(holdout)
    pd.testing.assert_frame_equal(t1, t2)
    assert "education" in t1.columns or any(c.startswith("education_") for c in t1.columns)

def test_pipeline_has_no_predict_method(adult_income_df):
    df = _binarize_target(adult_income_df)
    result = pf.prepare(df, target="income", task="classification", model_hint="tree")
    assert hasattr(result.pipeline, "transform")
    assert hasattr(result.pipeline, "fit")
    assert not hasattr(result.pipeline, "predict")

def test_cli_end_to_end_on_realistic_data(adult_income_df, tmp_path):
    df = _binarize_target(adult_income_df)
    csv_path = str(tmp_path / "adult.csv")
    df.to_csv(csv_path, index=False)
    result = runner.invoke(app, [
        "prepare", csv_path, "--target", "income",
        "--task", "classification", "--model-hint", "linear",
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    files = os.listdir(tmp_path)
    assert any(f.endswith("_prepared.csv") for f in files)
    assert any(f.endswith("_pipeline.joblib") for f in files)
    assert any(f.endswith("_report.json") for f in files)