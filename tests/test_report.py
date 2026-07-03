# tests/test_report.py (part 3 — to_dict / to_dataframe)
import json
import numpy as np
import pandas as pd
import pytest
from preflight.report import Report
from preflight.types import ReportEntry

def make_entries_with_numpy_stats():
    return [
        ReportEntry(stage="cleaner", column="income", action="median_impute",
                    rationale="12% missing", severity="warning",
                    before_stats={"missing_rate": np.float64(0.12)},
                    after_stats={"missing_rate": np.float64(0.0)}),
        ReportEntry(stage="profiler", column="age", action="flag_leakage",
                    rationale="corr high", severity="critical",
                    before_stats={}, after_stats={}),
    ]

def test_to_dict_structure():
    r = Report(make_entries_with_numpy_stats())
    d = r.to_dict()
    assert "summary" in d
    assert "total_entries" in d
    assert "entries" in d
    assert d["total_entries"] == 2

def test_to_dict_json_serializable():
    r = Report(make_entries_with_numpy_stats())
    d = r.to_dict()
    serialized = json.dumps(d)  # must not raise
    assert isinstance(serialized, str)

def test_to_dict_numpy_types_converted():
    r = Report(make_entries_with_numpy_stats())
    d = r.to_dict()
    entry = next(e for e in d["entries"] if e["column"] == "income")
    assert isinstance(entry["before_stats"]["missing_rate"], float)
    assert not isinstance(entry["before_stats"]["missing_rate"], np.floating)

def test_to_dict_entry_fields_present():
    r = Report(make_entries_with_numpy_stats())
    d = r.to_dict()
    entry = d["entries"][0]
    for field in ("stage", "column", "action", "rationale", "severity",
                  "before_stats", "after_stats"):
        assert field in entry

def test_to_dataframe_row_count():
    r = Report(make_entries_with_numpy_stats())
    df = r.to_dataframe()
    assert len(df) == 2

def test_to_dataframe_columns():
    r = Report(make_entries_with_numpy_stats())
    df = r.to_dataframe()
    expected_cols = {"stage", "column", "action", "rationale", "severity",
                      "before_stats", "after_stats"}
    assert expected_cols == set(df.columns)

def test_to_dataframe_empty_report_has_columns_not_zero_cols():
    r = Report([])
    df = r.to_dataframe()
    assert len(df) == 0
    expected_cols = {"stage", "column", "action", "rationale", "severity",
                      "before_stats", "after_stats"}
    assert expected_cols == set(df.columns)

def test_to_dataframe_values_correct():
    r = Report(make_entries_with_numpy_stats())
    df = r.to_dataframe()
    row = df[df["column"] == "age"].iloc[0]
    assert row["stage"] == "profiler"
    assert row["severity"] == "critical"