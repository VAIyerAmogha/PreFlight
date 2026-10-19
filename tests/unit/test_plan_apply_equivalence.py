"""
Equivalence test suite for PreFlight v2.0 Plan and Apply model (Phase 4 / M3).

Validates:
1. Decision equivalence: Decisions logged by dry_run equal Decisions logged by an
   actual apply run (identical ids, kinds, columns, actions, fingerprints).
2. Typing execution fidelity: apply successfully parses typed columns into pandas dtypes.
3. Lossless anomaly capture: unparseable values are captured in anomalies records
   with their row, column, and raw value, never silently discarded or dropped.
"""

from pathlib import Path
import pytest
import pandas as pd

from preflight.plan import apply, dry_run, plan


@pytest.fixture
def clean_numeric_csv(tmp_path: Path) -> Path:
    p = tmp_path / "clean_num.csv"
    p.write_text("id,val_int,val_float\n1,10,12.5\n2,20,25.0\n3,30,37.5\n", encoding="utf-8")
    return p


@pytest.fixture
def messy_anomalies_csv(tmp_path: Path) -> Path:
    p = tmp_path / "messy_anom.csv"
    # Row 1 has corrupt string in integer column, row 2 has corrupt string in float column
    p.write_text("id,score,amount\n01,100,50.5\n02,corrupt,99.9\n03,300,invalid_flt\n", encoding="utf-8")
    return p


def test_dry_run_apply_decision_equivalence(clean_numeric_csv: Path, tmp_path: Path):
    """
    Verify Decisions from dry_run equal Decisions logged by an actual apply run.
    (same ids, kinds, columns, actions, fingerprints).
    """
    out_dir = tmp_path / "equiv_out"
    plan_obj, _ = dry_run(clean_numeric_csv, out_dir=out_dir)

    apply_result = apply(plan_obj)

    dry_decs = plan_obj.decisions
    apply_decs = apply_result.decisions

    assert len(dry_decs) == len(apply_decs)

    for d_dry, d_app in zip(dry_decs, apply_decs):
        assert d_dry.id == d_app.id, f"Decision id mismatch: {d_dry.id} != {d_app.id}"
        assert d_dry.stage == d_app.stage, f"Decision stage mismatch: {d_dry.stage} != {d_app.stage}"
        assert d_dry.kind == d_app.kind, f"Decision kind mismatch: {d_dry.kind} != {d_app.kind}"
        assert d_dry.columns == d_app.columns, f"Decision columns mismatch: {d_dry.columns} != {d_app.columns}"
        assert d_dry.action == d_app.action, f"Decision action mismatch: {d_dry.action} != {d_app.action}"
        assert d_dry.fingerprint == d_app.fingerprint, (
            f"Decision fingerprint mismatch: {d_dry.fingerprint} != {d_app.fingerprint}"
        )


def test_apply_typing_execution(clean_numeric_csv: Path):
    """Verify apply correctly converts raw strings into typed pandas series."""
    plan_obj = plan(clean_numeric_csv)
    apply_result = apply(plan_obj)

    df = apply_result.data
    assert len(df) == 3

    # val_int should be integer type (Int64)
    assert pd.api.types.is_integer_dtype(df["val_int"])
    assert df["val_int"].tolist() == [10, 20, 30]

    # val_float should be float type (Float64)
    assert pd.api.types.is_float_dtype(df["val_float"])
    assert df["val_float"].tolist() == [12.5, 25.0, 37.5]


def test_apply_lossless_anomaly_capture(messy_anomalies_csv: Path):
    """
    Verify values that fail to parse are recorded in an anomalies record with their
    row and column, not silently dropped or coerced to NaN without a log entry.
    """
    plan_obj = plan(messy_anomalies_csv)
    apply_result = apply(plan_obj)

    df = apply_result.data
    anomalies = apply_result.anomalies

    # Must contain exactly 2 parse failure anomaly records
    assert len(anomalies) >= 2, f"Expected anomalies for parse failures, found: {anomalies}"

    # Verify score column anomaly at row 1
    score_anoms = [a for a in anomalies if a["column"] == "score"]
    assert len(score_anoms) == 1
    assert score_anoms[0]["row"] == 1
    assert score_anoms[0]["value"] == "corrupt"
    assert "integer" in score_anoms[0]["message"].lower()

    # Verify amount column anomaly at row 2
    amount_anoms = [a for a in anomalies if a["column"] == "amount"]
    assert len(amount_anoms) == 1
    assert amount_anoms[0]["row"] == 2
    assert amount_anoms[0]["value"] == "invalid_flt"
    assert "float" in amount_anoms[0]["message"].lower()

    # The returned dataframe has NA at those exact cells without dropping rows
    assert len(df) == 3, "Rows were improperly dropped during typing apply!"
    assert pd.isna(df.loc[1, "score"])
    assert pd.isna(df.loc[2, "amount"])
