"""
Missing value handling, indicators, and gap filling for PreFlight v2.0.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from preflight.decisions import Decision
from preflight.schema import TableSchema
from preflight.clean.config import CleanerConfig, DEFAULT_CLEANER_CONFIG


def add_missing_indicator(series: pd.Series) -> pd.Series:
    """
    Returns a new boolean Series named f"{series.name}_missing",
    True where original was null. Does not mutate input.
    """
    indicator = series.isnull()
    indicator.name = f"{series.name}_missing"
    return indicator


def median_impute(series: pd.Series) -> Tuple[pd.Series, float]:
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


def mode_impute(series: pd.Series) -> Tuple[pd.Series, object]:
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


def apply_missing_markers_to_series(
    series: pd.Series,
    missing_markers: Sequence[Union[str, Dict[str, Any]]],
) -> pd.Series:
    """
    Replace detected missing-value markers with pd.NA / np.nan.
    """
    markers = set()
    for m in missing_markers:
        if isinstance(m, dict):
            markers.add(str(m.get("marker", "")).strip())
        else:
            markers.add(str(m).strip())
    markers.discard("")

    if not markers:
        return series

    def _clean(val: Any) -> Any:
        if pd.isna(val):
            return np.nan
        if str(val).strip() in markers:
            return np.nan
        return val

    return series.apply(_clean)


def plan_impute(
    df: pd.DataFrame,
    schema: Optional[TableSchema] = None,
    target: Optional[str] = None,
    cardinality_threshold: int = 20,
    config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
) -> List[Decision]:
    """
    Plan missing indicator columns and gap filling (median, mode, or constant).
    """
    type_map = {}
    markers_map = {}
    if schema and schema.columns:
        for c in schema.columns:
            eff_type = c.effective_type() or "text"
            type_map[c.name] = eff_type.lower()
            markers_map[c.name] = c.missing_markers or []

    decisions: List[Decision] = []

    for col in df.columns:
        if col == target:
            continue

        series = df[col]
        # Clean any detected missing markers first to compute accurate missing rate
        if col in markers_map and markers_map[col]:
            series = apply_missing_markers_to_series(series, markers_map[col])

        missing_rate = float(series.isna().mean())
        if missing_rate <= 0:
            continue

        col_type = type_map.get(col, "")
        is_numeric = col_type in ("int", "float", "numeric") or (
            pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        )
        is_categorical = col_type in ("categorical", "boolean") or isinstance(series.dtype, pd.CategoricalDtype)
        if not is_categorical and not is_numeric:
            n_unique = int(series.dropna().nunique())
            if n_unique < len(series) * 0.5:
                is_categorical = True

        # 1. Add missing indicator for qualifying columns
        ind_dec = Decision(
            id="",
            stage="clean",
            kind="add_missing_indicator",
            columns=[col],
            action="add_missing_indicator",
            evidence={
                "missing_rate": round(missing_rate, 4),
                "indicator_column": f"{col}_missing",
            },
            confidence=1.0,
            default="add_missing_indicator",
            needs_attention=False,
            plain_language=f"Add missing value flag '{col}_missing' ({missing_rate:.1%} blank entries).",
            status="automatic",
        )
        decisions.append(ind_dec)

        # 2. Imputation Strategy
        if is_numeric:
            if missing_rate < config.impute_strategy_threshold:
                median_val = float(series.median()) if not series.isnull().all() else 0.0
                fill_dec = Decision(
                    id="",
                    stage="clean",
                    kind="fill_median",
                    columns=[col],
                    action="fill_median",
                    evidence={
                        "missing_rate": round(missing_rate, 4),
                        "fill_value": round(median_val, 4),
                        "threshold": config.impute_strategy_threshold,
                    },
                    confidence=0.95,
                    default="fill_median",
                    needs_attention=False,
                    plain_language=f"Fill blank values in numeric column '{col}' with median ({median_val:.2f}).",
                    status="automatic",
                )
                decisions.append(fill_dec)

        elif is_categorical:
            n_unique = int(series.dropna().nunique())
            if n_unique >= cardinality_threshold or missing_rate >= config.impute_strategy_threshold:
                fill_dec = Decision(
                    id="",
                    stage="clean",
                    kind="constant_fill",
                    columns=[col],
                    action="constant_fill",
                    evidence={
                        "missing_rate": round(missing_rate, 4),
                        "fill_value": "__missing__",
                        "threshold": config.impute_strategy_threshold,
                    },
                    confidence=0.95,
                    default="constant_fill",
                    needs_attention=False,
                    plain_language=f"Fill blank values in column '{col}' with a distinct missing marker.",
                    status="automatic",
                )
                decisions.append(fill_dec)
            else:
                modes = series.mode(dropna=True)
                mode_val = modes.iloc[0] if not modes.empty else ""
                fill_dec = Decision(
                    id="",
                    stage="clean",
                    kind="fill_mode",
                    columns=[col],
                    action="fill_mode",
                    evidence={
                        "missing_rate": round(missing_rate, 4),
                        "fill_value": str(mode_val),
                        "threshold": config.impute_strategy_threshold,
                    },
                    confidence=0.95,
                    default="fill_mode",
                    needs_attention=False,
                    plain_language=f"Fill blank values in categorical column '{col}' with most frequent value ('{mode_val}').",
                    status="automatic",
                )
                decisions.append(fill_dec)

    return decisions


def apply_impute(
    df: pd.DataFrame,
    decisions: List[Decision],
    schema: Optional[TableSchema] = None,
) -> pd.DataFrame:
    """
    Apply missing marker cleaning, missing indicators, and gap filling.
    """
    df_out = df.copy()

    # 1. Apply missing value markers from schema if available
    if schema and schema.columns:
        for c in schema.columns:
            if c.name in df_out.columns and c.missing_markers:
                df_out[c.name] = apply_missing_markers_to_series(df_out[c.name], c.missing_markers)

    # 2. Add indicators and fill values based on decisions
    for d in decisions:
        if not d.columns:
            continue
        col = d.columns[0]
        if col not in df_out.columns:
            continue

        if d.kind == "add_missing_indicator" and d.action == "add_missing_indicator":
            indicator = add_missing_indicator(df_out[col])
            df_out[indicator.name] = indicator

        elif d.kind in ("fill_median", "impute_median") and d.action in ("fill_median", "impute_median"):
            val = d.evidence.get("fill_value")
            if val is None:
                _, val = median_impute(df_out[col])
            df_out[col] = df_out[col].fillna(val)

        elif d.kind in ("fill_mode", "impute_mode") and d.action in ("fill_mode", "impute_mode"):
            val = d.evidence.get("fill_value")
            if val is None:
                _, val = mode_impute(df_out[col])
            df_out[col] = df_out[col].fillna(val)

        elif d.kind == "constant_fill" and d.action == "constant_fill":
            val = d.evidence.get("fill_value", "__missing__")
            df_out[col] = constant_fill(df_out[col], fill_value=val)

    return df_out
