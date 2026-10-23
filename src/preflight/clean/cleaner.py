"""
Cleaner orchestration and backwards-compatible API for PreFlight v2.0.
"""

import re
from dataclasses import dataclass
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
from preflight.clean.markers import apply_markers, plan_markers
from preflight.signals import ColumnSignals, compute_signals
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


def plan_coerce(
    df: pd.DataFrame,
    schema: Optional[TableSchema] = None,
    target: Optional[str] = None,
) -> List[Decision]:
    """
    Plan conversion of text columns that the schema types as numbers but that
    still hold text with units or stray spaces (v1: Profiler/Cleaner
    `coerce_numeric_string_column`). Typed columns from the reading/typing
    stages never need this.
    """
    decisions: List[Decision] = []
    if not schema or not schema.columns:
        return decisions
    for c in schema.columns:
        col = c.name
        if col == target or col not in df.columns:
            continue
        if (c.effective_type() or "").lower() not in ("int", "float", "numeric"):
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        coerced, unit = coerce_numeric_string_column(series)
        if coerced is None:
            continue
        decisions.append(
            Decision(
                id="",
                stage="clean",
                kind="coerce_numeric",
                columns=[col],
                action="coerce_numeric",
                evidence={"unit": unit},
                confidence=float(min(1.0, max(0.0, c.confidence))),
                default="coerce_numeric",
                needs_attention=False,
                plain_language=(
                    f"Read column '{col}' as numbers"
                    + (f" (removing the unit '{unit}')." if unit else " (removing stray spaces).")
                ),
                status="automatic",
            )
        )
    return decisions


def apply_coerce(df: pd.DataFrame, decisions: List[Decision]) -> pd.DataFrame:
    df_out = df.copy()
    for d in decisions:
        if d.kind == "coerce_numeric" and d.action == "coerce_numeric" and d.columns:
            col = d.columns[0]
            if col in df_out.columns:
                coerced, _ = coerce_numeric_string_column(df_out[col])
                if coerced is not None:
                    df_out[col] = coerced
    return df_out


@dataclass
class CleanPlan:
    """Cleaning Decisions plus the state the engineer stage reads."""
    decisions: List[Decision]
    cleaned: pd.DataFrame
    signals: Dict[str, ColumnSignals]


def plan_clean(
    df: pd.DataFrame,
    schema: Optional[TableSchema] = None,
    target: Optional[str] = None,
    config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
    target_column: Optional[str] = None,
    task: Optional[str] = None,
) -> List[Decision]:
    """Cleaning Decisions only; see `plan_clean_full`."""
    return plan_clean_full(df, schema, target, config, target_column, task).decisions


def plan_clean_full(
    df: pd.DataFrame,
    schema: Optional[TableSchema] = None,
    target: Optional[str] = None,
    config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
    target_column: Optional[str] = None,
    task: Optional[str] = None,
) -> CleanPlan:
    """
    Generate every cleaning Decision without touching `df`.

    The steps are simulated in order on a private copy, so each step sees the
    table as the previous steps leave it (this is what v1 did): markers,
    numeric coercion, column signals, duplicates, drops, text tidying and
    rare values, gap filling, then outlier clipping. Applying the returned
    Decisions with `apply_clean` reproduces that same table.
    """
    target = target_column or target
    decisions: List[Decision] = []
    work = df.copy()

    d_mark = plan_markers(work, schema)
    work = apply_markers(work, d_mark)
    d_coerce = plan_coerce(work, schema, target)
    work = apply_coerce(work, d_coerce)
    decisions.extend(d_mark)
    decisions.extend(d_coerce)

    sigs = compute_signals(
        work,
        schema=schema,
        target=target,
        task=task,
        cardinality_threshold=config.cardinality_threshold,
        rare_threshold=config.rare_category_threshold,
    )

    d_dedupe = plan_dedupe(work, config=config)
    work, _ = apply_dedupe(work, d_dedupe)
    d_drops = plan_drops(work, schema=schema, target=target, config=config, signals=sigs)
    work, _ = apply_drops(work, d_drops)
    d_norm = plan_normalization(work, schema=schema, target=target, config=config, signals=sigs)
    work = apply_normalization(work, d_norm)
    d_impute = plan_impute(work, schema=schema, target=target, config=config, signals=sigs)
    work = apply_impute(work, d_impute)
    d_out = plan_outliers(work, schema=schema, target=target, config=config, signals=sigs)

    for group in (d_dedupe, d_drops, d_norm, d_impute, d_out):
        decisions.extend(group)
    work = apply_outliers(work, d_out)
    return CleanPlan(decisions=decisions, cleaned=work, signals=sigs)


def apply_clean(
    df: pd.DataFrame,
    decisions: List[Decision],
    schema: Optional[TableSchema] = None,
) -> pd.DataFrame:
    """Apply cleaning Decisions in the order `plan_clean` simulated them."""
    out = apply_markers(df.copy(), decisions)
    out = apply_coerce(out, decisions)
    out, _ = apply_dedupe(out, decisions)
    out, _ = apply_drops(out, decisions)
    out = apply_normalization(out, decisions)
    out = apply_impute(out, decisions)
    out = apply_outliers(out, decisions)
    return out
