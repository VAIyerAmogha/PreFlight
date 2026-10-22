"""
Outlier handling and winsorization planning and execution for PreFlight v2.0.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from preflight.decisions import Decision
from preflight.schema import TableSchema
from preflight.clean.config import CleanerConfig, DEFAULT_CLEANER_CONFIG


def winsorize_outliers(series: pd.Series, method: str = "iqr") -> pd.Series:
    """
    Clips outliers based on IQR or z-score.
    Must not be called if series missingness exceeds 30% or if IQR / std is zero.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(
            f"We can only limit extreme values for numeric columns, but received type {series.dtype}."
        )

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return series
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
    elif method == "zscore":
        mean = series.mean()
        std = series.std()
        if std == 0:
            return series
        lower = mean - 3 * std
        upper = mean + 3 * std
    else:
        raise ValueError(f"Outlier clipping method '{method}' is not supported. Use 'iqr' or 'zscore'.")

    return series.clip(lower=lower, upper=upper)


def plan_outliers(
    df: pd.DataFrame,
    schema: Optional[TableSchema] = None,
    target: Optional[str] = None,
    config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
) -> List[Decision]:
    """
    Plan outlier clipping for numeric columns.
    Destructive check: if fraction of column rows clipped exceeds
    config.outlier_impact_threshold, needs_attention is set to True.
    """
    type_map = {}
    if schema and schema.columns:
        for c in schema.columns:
            eff_type = c.effective_type() or "text"
            type_map[c.name] = eff_type.lower()

    decisions: List[Decision] = []

    for col in df.columns:
        if col == target:
            continue

        series = df[col]
        col_type = type_map.get(col, "")
        is_numeric = col_type in ("int", "float", "numeric") or pd.api.types.is_numeric_dtype(series)
        if not is_numeric or pd.api.types.is_bool_dtype(series):
            continue

        missing_rate = float(series.isna().mean())
        if missing_rate > config.winsorize_missing_limit:
            continue

        non_null = series.dropna()
        if len(non_null) == 0:
            continue

        method = config.outlier_method
        if method == "iqr":
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower_bound = float(q1 - 1.5 * iqr)
            upper_bound = float(q3 + 1.5 * iqr)
        elif method == "zscore":
            mean = non_null.mean()
            std = non_null.std()
            if std == 0:
                continue
            lower_bound = float(mean - 3 * std)
            upper_bound = float(mean + 3 * std)
        else:
            continue

        outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
        outlier_count = len(outliers)
        if outlier_count == 0:
            continue

        impact = outlier_count / len(non_null)
        needs_att = impact >= config.outlier_impact_threshold

        entry_word = "entry" if outlier_count == 1 else "entries"
        plain_lang = (
            f"Limit {outlier_count} extreme {entry_word} in column '{col}' to range [{lower_bound:.2f}, {upper_bound:.2f}]."
            if not needs_att
            else f"Review limiting {outlier_count} extreme {entry_word} in column '{col}' ({impact:.1%} of entries)."
        )

        dec = Decision(
            id="",
            stage="clean",
            kind="clip_outliers",
            columns=[col],
            action="clip_outliers",
            evidence={
                "method": method,
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "outlier_count": outlier_count,
                "total_entries": len(non_null),
                "impact_fraction": round(impact, 4),
                "threshold": config.outlier_impact_threshold,
            },
            confidence=0.95,
            default="keep_outliers",
            needs_attention=needs_att,
            plain_language=plain_lang,
            status="automatic",
        )
        decisions.append(dec)

    return decisions


def apply_outliers(
    df: pd.DataFrame,
    decisions: List[Decision],
) -> pd.DataFrame:
    """
    Apply outlier clipping specified by decisions.
    """
    df_out = df.copy()

    for d in decisions:
        if d.kind == "clip_outliers" and d.action == "clip_outliers" and d.columns:
            col = d.columns[0]
            if col in df_out.columns:
                lower = d.evidence.get("lower_bound")
                upper = d.evidence.get("upper_bound")
                if lower is not None and upper is not None:
                    # Integer dtypes (including pandas nullable Int64) cannot hold
                    # fractional IQR bounds; widen to a float dtype before clipping.
                    if pd.api.types.is_integer_dtype(df_out[col].dtype):
                        is_nullable = pd.api.types.is_extension_array_dtype(df_out[col].dtype)
                        df_out[col] = df_out[col].astype("Float64" if is_nullable else "float64")
                    df_out[col] = df_out[col].clip(lower=lower, upper=upper)

    return df_out
