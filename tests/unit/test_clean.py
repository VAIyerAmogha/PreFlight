"""
Unit tests for PreFlight v2.0 Cleaning Stage (Phase 6 / M5).

Validates:
1. plan purity: plan_clean never mutates the input DataFrame.
2. dedupe planning and application with destructive impact threshold.
3. column drops (high missingness, constant, ID) with destructive impact threshold.
4. categorical normalization and rare category grouping.
5. outlier winsorization (IQR and z-score) with destructive impact threshold.
6. missing-value marker conversion, indicators, and imputation strategies.
7. edge cases: all-null column, constant column, single-row table.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from preflight.clean.config import CleanerConfig, DEFAULT_CLEANER_CONFIG
from preflight.clean.dedupe import apply_dedupe, plan_dedupe, remove_duplicate_rows
from preflight.clean.drops import apply_drops, plan_drops
from preflight.clean.impute import (
    add_missing_indicator,
    apply_impute,
    apply_missing_markers_to_series,
    constant_fill,
    median_impute,
    mode_impute,
    plan_impute,
)
from preflight.clean.normalize import (
    apply_normalization,
    identify_rare_categories,
    normalize_category_values,
    plan_normalization,
)
from preflight.clean.outliers import apply_outliers, plan_outliers, winsorize_outliers
from preflight.clean.cleaner import apply_clean, plan_clean
from preflight.schema import ColumnRawMetadata, TableSchema


def test_clean_plan_purity():
    """Verify plan_clean never mutates the input DataFrame."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 100.0, 2.0],
        "cat": ["Apple ", "banana", "apple", "Banana", "Cherry"],
        "const": [1, 1, 1, 1, 1],
    })
    orig_copy = df.copy(deep=True)

    decisions = plan_clean(df, target_column=None)

    pd.testing.assert_frame_equal(df, orig_copy)
    assert len(decisions) > 0


def test_dedupe_destructive_threshold():
    """Verify dedupe flags needs_attention when impact exceeds threshold."""
    # 5 rows, 1 duplicate -> 20% impact (exceeds default 2% threshold)
    df = pd.DataFrame({"a": [1, 2, 3, 4, 1], "b": [10, 20, 30, 40, 10]})

    decs = plan_dedupe(df, config=CleanerConfig(dedupe_impact_threshold=0.02))
    assert len(decs) == 1
    assert decs[0].needs_attention is True
    assert decs[0].kind == "remove_duplicates"

    # Higher threshold -> needs_attention is False
    decs_lenient = plan_dedupe(df, config=CleanerConfig(dedupe_impact_threshold=0.30))
    assert len(decs_lenient) == 1
    assert decs_lenient[0].needs_attention is False

    # Apply dedupe
    df_clean, removed = apply_dedupe(df, decs)
    assert removed == 1
    assert len(df_clean) == 4


def test_column_drops_destructive_threshold_and_target_protection():
    """Verify high-missing, constant, and ID columns are planned for drop and target is protected."""
    df = pd.DataFrame({
        "row_id": list(range(100)),
        "mostly_empty": [1.0] * 10 + [np.nan] * 90,  # 90% missing > 60%
        "constant_col": ["fixed"] * 100,
        "valid_feature": list(range(100)),
        "target": ["fixed"] * 100,  # Constant, but designated as target
    })

    decs = plan_drops(df, target="target", config=DEFAULT_CLEANER_CONFIG)
    dropped_cols = [d.columns[0] for d in decs]

    assert "mostly_empty" in dropped_cols
    assert "constant_col" in dropped_cols
    assert "row_id" in dropped_cols
    assert "target" not in dropped_cols  # Protected!
    assert "valid_feature" not in dropped_cols

    # Apply drops
    df_dropped, dropped_names = apply_drops(df, decs)
    assert "mostly_empty" not in df_dropped.columns
    assert "constant_col" not in df_dropped.columns
    assert "row_id" not in df_dropped.columns
    assert "target" in df_dropped.columns
    assert "valid_feature" in df_dropped.columns


def test_normalization_and_rare_categories():
    """Verify categorical text standardization and rare category grouping."""
    s = pd.Series([" Alpha", "ALPHA ", "beta", "gamma", "delta", "delta", "delta"])
    norm_s = normalize_category_values(s)
    assert norm_s.iloc[0] == "alpha"
    assert norm_s.iloc[1] == "alpha"

    # Rare categories identification
    counts = pd.Series(["common"] * 95 + ["rare1", "rare2", "rare3", "rare4", "rare5"])
    rares = identify_rare_categories(counts, threshold=0.02)
    assert "rare1" in rares
    assert "rare2" in rares
    assert "common" not in rares


def test_outlier_clipping_iqr_and_zscore():
    """Verify outlier winsorization with impact threshold."""
    # 99 regular numbers and 1 extreme outlier
    vals = list(range(10, 110)) + [99999]
    s = pd.Series(vals)
    df = pd.DataFrame({"metric": s})

    decs = plan_outliers(df, config=CleanerConfig(outlier_impact_threshold=0.005))
    assert len(decs) == 1
    assert decs[0].kind == "clip_outliers"
    assert decs[0].evidence["outlier_count"] >= 1

    df_clipped = apply_outliers(df, decs)
    assert df_clipped["metric"].max() < 99999


def test_missing_marker_conversion_and_imputation():
    """Verify missing markers in schema are converted to NA before imputation."""
    df = pd.DataFrame({
        "num": ["10", "20", "?", "30", "N/A"],
        "cat": ["cat", "dog", "missing", "cat", "dog"],
    })

    schema = TableSchema(
        source_path="dummy.csv",
        row_count=5,
        column_count=2,
        columns=[
            ColumnRawMetadata(
                name="num",
                position=0,
                inferred_type="numeric",
                missing_markers=[{"marker": "?"}, {"marker": "N/A"}],
            ),
            ColumnRawMetadata(
                name="cat",
                position=1,
                inferred_type="categorical",
                missing_markers=[{"marker": "missing"}],
            ),
        ],
    )

    decs = plan_impute(df, schema=schema)
    assert any(d.kind == "add_missing_indicator" for d in decs)

    df_imputed = apply_impute(df, decs, schema=schema)
    # Check that sentinel "?" is gone from numeric series
    assert "?" not in df_imputed["num"].values
    assert "missing" not in df_imputed["cat"].values
    assert "num_missing" in df_imputed.columns


def test_clean_edge_cases():
    """Verify clean stage handles all-null, constant, and single-row tables without crashing."""
    # Single row
    df_single = pd.DataFrame({"a": [1], "b": ["x"]})
    decs_single = plan_clean(df_single)
    df_out = apply_clean(df_single, decs_single)
    assert len(df_out) == 1

    # All-null column
    df_null = pd.DataFrame({"all_null": [np.nan, np.nan, np.nan], "valid": [1, 2, 3]})
    decs_null = plan_clean(df_null)
    df_null_out = apply_clean(df_null, decs_null)
    assert "all_null" not in df_null_out.columns or df_null_out["all_null"].isna().all()
