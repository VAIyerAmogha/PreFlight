"""
Missing value handling, indicators, and gap filling for PreFlight v2.0.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from preflight.clean._common import need_signals, py
from preflight.clean.config import DEFAULT_CLEANER_CONFIG, CleanerConfig
from preflight.clean.markers import apply_missing_markers_to_series  # noqa: F401  (re-export)
from preflight.decisions import Decision
from preflight.schema import TableSchema
from preflight.signals import CATEGORICAL_HIGH, CATEGORICAL_LOW, NUMERIC_FEATURE, ColumnSignals


def add_missing_indicator(series: pd.Series) -> pd.Series:
    """
    Returns a new boolean Series named f"{series.name}_missing",
    True where original was null. Does not mutate input.
    """
    indicator = series.isnull()
    indicator.name = f"{series.name}_missing"
    return indicator


def median_impute(series: pd.Series) -> tuple[pd.Series, float]:
    """
    Fill missing values with median for numeric columns.
    Returns (imputed series, median value used).
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(
            f"We can only fill missing values with the median for numeric columns, but '{series.name}' has type {series.dtype}."
        )

    median_val = float(series.median()) if not series.isnull().all() else 0.0
    imputed = series.fillna(median_val)
    return imputed, median_val


def mode_impute(series: pd.Series) -> tuple[pd.Series, object]:
    """
    Fill missing values with mode for categorical columns.
    Returns (imputed series, mode value used).
    """
    modes = series.mode(dropna=True)
    if modes.empty:
        mode_val = np.nan
    else:
        mode_val = modes.iloc[0]

    imputed = series.fillna(mode_val)
    return imputed, mode_val


def constant_fill(series: pd.Series, fill_value: Any = "__missing__") -> pd.Series:
    """
    Fills nulls with fill_value for high-cardinality or high-missingness columns.
    """
    return series.fillna(fill_value)


def plan_impute(
    df: pd.DataFrame,
    schema: Optional[TableSchema] = None,
    target: Optional[str] = None,
    cardinality_threshold: Optional[int] = None,
    config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
    signals: Optional[dict[str, ColumnSignals]] = None,
) -> list[Decision]:
    """
    Plan blank-flag columns and gap filling, per v1 rules:

    - numeric: flag, then median if missing rate < threshold (otherwise flag only)
    - low-variety categorical: flag, then most common value if rate < threshold, else constant
    - high-variety categorical: constant fill, no flag
    - other kinds (boolean, text, datetime): untouched, as in v1

    `df` is expected to already have schema missing markers converted.
    """
    card = cardinality_threshold if cardinality_threshold is not None else config.cardinality_threshold
    sigs = need_signals(df, signals, schema, target, card)
    decisions: list[Decision] = []

    for col in df.columns:
        if col == target or col not in sigs:
            continue
        sig = sigs[col]
        if sig.kind not in (NUMERIC_FEATURE, CATEGORICAL_LOW, CATEGORICAL_HIGH):
            continue
        series = df[col]
        missing_rate = float(series.isna().mean()) if len(series) else 0.0
        if missing_rate <= 0:
            continue

        pct = f"{missing_rate:.1%}"

        if sig.kind in (NUMERIC_FEATURE, CATEGORICAL_LOW):
            decisions.append(
                Decision(
                    id="",
                    stage="clean",
                    kind="add_missing_indicator",
                    columns=[col],
                    action="add_missing_indicator",
                    evidence={"missing_rate": missing_rate, "indicator_column": f"{col}_missing"},
                    confidence=1.0,
                    default="add_missing_indicator",
                    needs_attention=False,
                    plain_language=f"Add a yes/no column '{col}_missing' marking where '{col}' was blank ({pct} blank).",
                    status="automatic",
                )
            )

        if sig.kind == NUMERIC_FEATURE:
            if missing_rate < config.impute_strategy_threshold:
                median_val = float(series.median()) if not series.isnull().all() else 0.0
                decisions.append(
                    Decision(
                        id="",
                        stage="clean",
                        kind="fill_median",
                        columns=[col],
                        action="fill_median",
                        evidence={
                            "missing_rate": missing_rate,
                            "fill_value": median_val,
                            "threshold": config.impute_strategy_threshold,
                        },
                        confidence=0.95,
                        default="fill_median",
                        needs_attention=False,
                        plain_language=f"Fill blanks in '{col}' with the middle value ({median_val:.4g}).",
                        status="automatic",
                    )
                )
        elif sig.kind == CATEGORICAL_LOW and missing_rate < config.impute_strategy_threshold:
            modes = series.mode(dropna=True)
            if modes.empty:
                continue
            mode_val = py(modes.iloc[0])
            decisions.append(
                Decision(
                    id="",
                    stage="clean",
                    kind="fill_mode",
                    columns=[col],
                    action="fill_mode",
                    evidence={
                        "missing_rate": missing_rate,
                        "fill_value": mode_val,
                        "threshold": config.impute_strategy_threshold,
                    },
                    confidence=0.95,
                    default="fill_mode",
                    needs_attention=False,
                    plain_language=f"Fill blanks in '{col}' with its most common value ('{mode_val}').",
                    status="automatic",
                )
            )
        else:
            decisions.append(
                Decision(
                    id="",
                    stage="clean",
                    kind="constant_fill",
                    columns=[col],
                    action="constant_fill",
                    evidence={
                        "missing_rate": missing_rate,
                        "fill_value": "__missing__",
                        "threshold": config.impute_strategy_threshold,
                    },
                    confidence=0.95,
                    default="constant_fill",
                    needs_attention=False,
                    plain_language=f"Fill blanks in '{col}' with the label '__missing__' so they stay visible.",
                    status="automatic",
                )
            )

    return decisions


def apply_impute(
    df: pd.DataFrame,
    decisions: list[Decision],
) -> pd.DataFrame:
    """Add blank-flag columns and fill gaps as specified by decisions."""
    df_out = df.copy()

    for d in decisions:
        if not d.columns:
            continue
        col = d.columns[0]
        if col not in df_out.columns:
            continue

        if d.kind == "add_missing_indicator" and d.action == "add_missing_indicator":
            indicator = add_missing_indicator(df_out[col])
            df_out[indicator.name] = indicator

        elif d.kind == "fill_median" and d.action == "fill_median":
            df_out[col] = df_out[col].fillna(d.evidence.get("fill_value"))

        elif d.kind == "fill_mode" and d.action == "fill_mode":
            df_out[col] = df_out[col].fillna(d.evidence.get("fill_value"))

        elif d.kind == "constant_fill" and d.action == "constant_fill":
            df_out[col] = constant_fill(df_out[col], fill_value=d.evidence.get("fill_value", "__missing__"))

    return df_out
