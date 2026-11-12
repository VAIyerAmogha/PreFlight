"""
Outlier handling and winsorization planning and execution for PreFlight v2.0.
"""

from typing import Optional

import pandas as pd

from preflight.clean._common import need_signals
from preflight.clean.config import DEFAULT_CLEANER_CONFIG, CleanerConfig
from preflight.decisions import Decision
from preflight.schema import TableSchema
from preflight.signals import NUMERIC_FEATURE, ColumnSignals


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
    signals: Optional[dict[str, ColumnSignals]] = None,
) -> list[Decision]:
    """
    Plan outlier clipping (winsorization) for numeric feature columns.

    As in v1, a column is skipped when its (original) missing rate is above
    config.winsorize_missing_limit or when the spread is zero. A Decision is
    emitted even when no value currently lies outside the limits, because the
    limits are also applied to new data. `df` is expected to be the table
    after gap filling (v1 clipped after filling). needs_attention is set when
    the fraction of entries clipped exceeds config.outlier_impact_threshold,
    and the safe default is then to keep the values.
    """
    sigs = need_signals(df, signals, schema, target, config.cardinality_threshold)
    decisions: list[Decision] = []

    for col in df.columns:
        if col == target or col not in sigs:
            continue
        sig = sigs[col]
        if sig.kind != NUMERIC_FEATURE:
            continue
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        if sig.missing_rate > config.winsorize_missing_limit:
            continue

        method = config.outlier_method
        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            lower_bound = float(q1 - 1.5 * iqr)
            upper_bound = float(q3 + 1.5 * iqr)
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            if std == 0 or pd.isna(std):
                continue
            lower_bound = float(mean - 3 * std)
            upper_bound = float(mean + 3 * std)
        else:
            raise ValueError(f"Outlier clipping method '{method}' is not supported. Use 'iqr' or 'zscore'.")

        non_null = series.dropna()
        outlier_count = int(((non_null < lower_bound) | (non_null > upper_bound)).sum())
        total = len(non_null)
        impact = outlier_count / total if total else 0.0
        needs_att = impact > config.outlier_impact_threshold
        entry_word = "entry" if outlier_count == 1 else "entries"

        if outlier_count == 0:
            plain = (
                f"Set limits [{lower_bound:.4g}, {upper_bound:.4g}] for '{col}' "
                f"(no current entry is outside them; they apply to new data)."
            )
        elif needs_att:
            plain = (
                f"Pull {outlier_count} extreme {entry_word} in '{col}' ({impact:.1%} of entries) "
                f"back to the range [{lower_bound:.4g}, {upper_bound:.4g}]? This is a large share, so please confirm."
            )
        else:
            plain = (
                f"Pull {outlier_count} extreme {entry_word} in '{col}' back to the range "
                f"[{lower_bound:.4g}, {upper_bound:.4g}]."
            )

        decisions.append(
            Decision(
                id="",
                stage="clean",
                kind="clip_outliers",
                columns=[col],
                action="clip_outliers",
                evidence={
                    "method": method,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outlier_count": outlier_count,
                    "total_entries": total,
                    "impact_fraction": impact,
                    "threshold": config.outlier_impact_threshold,
                },
                confidence=0.95,
                default="keep_outliers" if needs_att else "clip_outliers",
                needs_attention=needs_att,
                plain_language=plain,
                status="automatic",
            )
        )

    return decisions


def apply_outliers(
    df: pd.DataFrame,
    decisions: list[Decision],
) -> pd.DataFrame:
    """Apply outlier clipping specified by decisions."""
    df_out = df.copy()

    for d in decisions:
        if d.kind == "clip_outliers" and d.action == "clip_outliers" and d.columns:
            col = d.columns[0]
            if col in df_out.columns:
                lower = d.evidence.get("lower_bound")
                upper = d.evidence.get("upper_bound")
                if lower is not None and upper is not None:
                    # Nullable integer dtypes cannot hold fractional limits; widen first.
                    if pd.api.types.is_integer_dtype(df_out[col].dtype) and pd.api.types.is_extension_array_dtype(
                        df_out[col].dtype
                    ):
                        df_out[col] = df_out[col].astype("Float64")
                    df_out[col] = df_out[col].clip(lower=lower, upper=upper)

    return df_out
