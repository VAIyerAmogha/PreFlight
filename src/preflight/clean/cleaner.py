"""
Cleaner orchestration and backwards-compatible API for PreFlight v2.0.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import dateutil.parser
import numpy as np
import pandas as pd

from preflight.decisions import Decision
from preflight.schema import ColumnRawMetadata, TableSchema
from preflight.clean.config import CleanerConfig, DEFAULT_CLEANER_CONFIG
from preflight.clean.dedupe import apply_dedupe, plan_dedupe, remove_duplicate_rows
from preflight.clean.drops import (
    apply_drops,
    drop_constant_columns,
    drop_high_missingness_columns,
    drop_numeric_id_columns,
    plan_drops,
    should_drop_column,
)
from preflight.clean.impute import (
    add_missing_indicator,
    apply_impute,
    constant_fill,
    median_impute,
    mode_impute,
    plan_impute,
)
from preflight.clean.normalize import (
    apply_normalization,
    group_rare_categories,
    identify_rare_categories,
    normalize_category_values,
    plan_normalization,
)
from preflight.clean.outliers import apply_outliers, plan_outliers, winsorize_outliers


def coerce_string_dates_to_datetime(series: pd.Series) -> pd.Series:
    """
    Uses python-dateutil to parse a DATETIME_STRING column into proper datetime64.
    Unparseable individual values become NaT rather than raising.
    """
    def _parse(val):
        if pd.isnull(val):
            return pd.NaT
        try:
            return dateutil.parser.parse(str(val))
        except (ValueError, TypeError, OverflowError):
            return pd.NaT

    parsed = series.apply(_parse)
    return pd.to_datetime(parsed, errors="coerce")


def coerce_numeric_string_column(series: pd.Series) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Tries to parse an object/string column as numeric.
    Detects and strips whitespace and common unit suffixes (like CC, kmpl, bhp, Nm).
    Returns (coerced_series, common_unit_suffix) if at least 95% of non-missing values
    can be parsed, otherwise (None, None).
    """
    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return None, None

    non_null = series.dropna()
    if len(non_null) == 0:
        return None, None

    non_null_str = non_null.astype(str).str.strip()
    missing_indicators = {"", "nan", "none", "null", "na", "n/a", "-", "?"}
    non_empty = non_null_str[~non_null_str.str.lower().isin(missing_indicators)]
    if len(non_empty) == 0:
        return None, None

    num_pat = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(.*)$")

    parsed_vals = []
    suffixes = []
    success_count = 0

    for val in non_empty:
        m = num_pat.match(val)
        if m:
            success_count += 1
            parsed_vals.append(float(m.group(1)))
            suffix = m.group(2).strip()
            if suffix:
                suffixes.append(suffix)

    if success_count / len(non_empty) >= 0.95:
        if suffixes:
            suffix_counts = pd.Series(suffixes).value_counts()
            most_common_ratio = suffix_counts.iloc[0] / len(suffixes)
            if most_common_ratio < 0.90 or len(suffix_counts) > 5:
                return None, None

        coerced = pd.Series(index=series.index, dtype=float, name=series.name)

        for idx, val in series.items():
            if pd.isna(val):
                coerced.loc[idx] = np.nan
            else:
                val_str = str(val).strip()
                if val_str.lower() in missing_indicators:
                    coerced.loc[idx] = np.nan
                else:
                    m = num_pat.match(val_str)
                    if m:
                        coerced.loc[idx] = float(m.group(1))
                    else:
                        coerced.loc[idx] = np.nan

        common_unit = None
        if suffixes:
            common_unit = pd.Series(suffixes).value_counts().index[0]

        return coerced, common_unit

    return None, None


def plan_clean(
    df: pd.DataFrame,
    schema: Optional[TableSchema] = None,
    target: Optional[str] = None,
    config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
    target_column: Optional[str] = None,
) -> List[Decision]:
    """
    Generate all cleaning Decisions across dedupe, drops, normalization,
    gap filling, and outlier handling without mutating data.
    """
    target = target_column or target
    decisions: List[Decision] = []

    # 1. Dedupe
    decisions.extend(plan_dedupe(df, config=config))

    # 2. Drops (High missingness, ID, Constant)
    decisions.extend(plan_drops(df, schema=schema, target=target, config=config))

    # 3. Category Normalization & Rare Categories
    decisions.extend(plan_normalization(df, schema=schema, target=target, config=config))

    # 4. Outliers (Winsorization)
    decisions.extend(plan_outliers(df, schema=schema, target=target, config=config))

    # 5. Missing Indicators & Gap Filling
    decisions.extend(plan_impute(df, schema=schema, target=target, config=config))

    return decisions


def apply_clean(
    df: pd.DataFrame,
    decisions: List[Decision],
    schema: Optional[TableSchema] = None,
) -> pd.DataFrame:
    """
    Apply all cleaning decisions in standard pipeline execution order.
    """
    df_clean = df.copy()

    # 1. Dedupe
    df_clean, _ = apply_dedupe(df_clean, decisions)

    # 2. Drops
    df_clean, _ = apply_drops(df_clean, decisions)

    # 3. Normalization & Rare categories
    df_clean = apply_normalization(df_clean, decisions)

    # 4. Outliers
    df_clean = apply_outliers(df_clean, decisions)

    # 5. Missing markers, indicators & Gap filling
    df_clean = apply_impute(df_clean, decisions, schema=schema)

    return df_clean


def run_cleaner(
    df: pd.DataFrame,
    profiles: Any,
    target: str = "",
    drop_threshold: float = 0.6,
    outlier_method: str = "iqr",
) -> Tuple[pd.DataFrame, Any, List[Any], Dict[str, Any]]:
    """
    Backwards-compatible run_cleaner API supporting v1 test suites and transformers.
    """
    config = CleanerConfig(
        drop_missing_threshold=drop_threshold,
        outlier_method=outlier_method,
    )

    # Convert profiles to mock TableSchema or extract column info if ColumnProfile list
    schema = None
    if isinstance(profiles, list) and profiles:
        cols = []
        for idx, p in enumerate(profiles):
            st = getattr(p, "semantic_type", None)
            st_val = getattr(st, "value", str(st)) if st else "text"
            col_name = getattr(p, "name", None) or (p.columns[0] if getattr(p, "columns", None) else f"col_{idx}")
            cols.append(ColumnRawMetadata(name=col_name, position=idx, inferred_type=st_val))
        schema = TableSchema(columns=cols)
    elif isinstance(profiles, TableSchema):
        schema = profiles

    decisions = plan_clean(df, schema=schema, target=target, config=config)
    df_cleaned = apply_clean(df, decisions, schema=schema)

    # Generate v1-compatible specs and surviving profiles
    surviving_cols = set(df_cleaned.columns)
    def _col_of(p: Any) -> str:
        return p.name if hasattr(p, "name") else (p.columns[0] if getattr(p, "columns", None) else str(p))
    surviving_profiles = [p for p in profiles if _col_of(p) in surviving_cols] if isinstance(profiles, list) else profiles

    specs: Dict[str, Any] = {}
    for d in decisions:
        if d.columns:
            col = d.columns[0]
            if col not in specs:
                specs[col] = {}
            if d.kind == "add_missing_indicator":
                specs[col]["add_missing_indicator"] = True
            elif d.kind in ("fill_median", "impute_median"):
                specs[col]["impute_strategy"] = "median"
                specs[col]["impute_value"] = d.evidence.get("fill_value")
            elif d.kind in ("fill_mode", "impute_mode"):
                specs[col]["impute_strategy"] = "mode"
                specs[col]["impute_value"] = d.evidence.get("fill_value")
            elif d.kind == "constant_fill":
                specs[col]["impute_strategy"] = "constant"
                specs[col]["impute_value"] = d.evidence.get("fill_value", "__missing__")
            elif d.kind == "clip_outliers":
                specs[col]["outlier_method"] = d.evidence.get("method", outlier_method)
            elif d.kind == "group_rare_categories":
                specs[col]["rare_categories"] = d.evidence.get("rare_categories", [])

    from preflight.types import ReportEntry
    report_entries = [
        ReportEntry(
            stage="cleaner",
            column=d.columns[0] if d.columns else "__all__",
            action=d.action,
            rationale=d.plain_language,
            severity="info",
        )
        for d in decisions
    ]

    return df_cleaned, surviving_profiles, report_entries, specs
