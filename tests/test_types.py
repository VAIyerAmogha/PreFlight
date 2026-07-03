# tests/test_types.py
import pytest
import pandas as pd
from preflight.types import SemanticType, ColumnProfile, ReportEntry, PrepResult

def test_semantic_type_has_exactly_8_members():
    expected = {
        "NUMERIC_FEATURE", "NUMERIC_ID", "CATEGORICAL_LOW", "CATEGORICAL_HIGH",
        "DATETIME_NATIVE", "DATETIME_STRING", "BOOLEAN", "CONSTANT",
    }
    actual = {m.name for m in SemanticType}
    assert actual == expected

def test_column_profile_construction_minimal():
    cp = ColumnProfile(
        name="age",
        semantic_type=SemanticType.NUMERIC_FEATURE,
        missing_rate=0.1,
        outlier_rate=0.02,
        cardinality=45,
        rare_categories=[],
        vif_score=None,
        correlation_with_target=0.3,
        mutual_info_with_target=0.2,
        is_leakage_suspect=False,
        dtype="float64",
    )
    assert cp.name == "age"
    assert cp.semantic_type == SemanticType.NUMERIC_FEATURE
    assert cp.vif_score is None  # not conflated with 0

def test_column_profile_vif_not_defaulted_to_zero():
    cp = ColumnProfile(
        name="cat_col", semantic_type=SemanticType.CATEGORICAL_LOW,
        missing_rate=0.0, outlier_rate=0.0, cardinality=3,
        rare_categories=["x"], vif_score=None,
        correlation_with_target=None, mutual_info_with_target=0.1,
        is_leakage_suspect=False, dtype="object",
    )
    assert cp.vif_score is None

def test_report_entry_severity_values():
    entry = ReportEntry(
        stage="cleaner", column="income", action="median_impute",
        rationale="12% missing, numeric", severity="warning",
        before_stats={"missing_rate": 0.12}, after_stats={"missing_rate": 0.0},
    )
    assert entry.severity in ("info", "warning", "critical")
    assert entry.stage in ("profiler", "cleaner", "engineer")

def test_report_entry_invalid_severity_rejected():
    with pytest.raises((ValueError, TypeError)):
        ReportEntry(
            stage="cleaner", column="income", action="x", rationale="y",
            severity="not_a_real_severity", before_stats={}, after_stats={},
        )

def test_report_entry_invalid_stage_rejected():
    with pytest.raises((ValueError, TypeError)):
        ReportEntry(
            stage="not_a_real_stage", column="income", action="x",
            rationale="y", severity="info", before_stats={}, after_stats={},
        )

def test_prep_result_holds_df_pipeline_report():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = PrepResult(df=df, pipeline=None, report=None)
    assert result.df.equals(df)
    assert result.pipeline is None
    assert result.report is None

def test_type_hints_present_on_all_fields():
    for cls in (ColumnProfile, ReportEntry, PrepResult):
        hints = cls.__annotations__
        assert len(hints) > 0, f"{cls.__name__} has no type hints"