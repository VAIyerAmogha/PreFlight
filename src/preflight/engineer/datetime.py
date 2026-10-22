"""
Datetime expansion and cyclical feature generation for PreFlight v2.0.
"""

import itertools
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd

from preflight.clean.cleaner import coerce_string_dates_to_datetime
from preflight.engineer.config import FeatureConfig


def expand_datetime(series: pd.Series) -> pd.DataFrame:
    """
    Expands a datetime column into multiple numeric and boolean features:
    year, month, day, dayofweek, hour, and is_weekend.
    Propagates NaT as NaN rather than raising.
    """
    if not pd.api.types.is_datetime64_any_dtype(series):
        # Attempt date parsing if string/object
        try:
            series = coerce_string_dates_to_datetime(series)
        except Exception:
            raise ValueError(f"The column '{series.name}' must be a datetime format to extract date features.")

    df = pd.DataFrame(index=series.index)
    prefix = series.name

    df[f"{prefix}_year"] = series.dt.year
    df[f"{prefix}_month"] = series.dt.month
    df[f"{prefix}_day"] = series.dt.day
    df[f"{prefix}_dayofweek"] = series.dt.dayofweek
    df[f"{prefix}_hour"] = series.dt.hour

    is_weekend = series.dt.dayofweek >= 5
    is_weekend = is_weekend.where(series.notna(), np.nan)
    df[f"{prefix}_is_weekend"] = is_weekend

    return df


def generate_datetime_cyclical_features(
    df: pd.DataFrame,
    dt_columns: Sequence[str],
    config: FeatureConfig,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Generate cyclical sine/cosine features and pairwise datetime deltas.
    """
    df_out = df.copy()
    reports = []

    for col in dt_columns:
        if col not in df_out.columns:
            continue
        series = df_out[col]
        if not pd.api.types.is_datetime64_any_dtype(series):
            series = coerce_string_dates_to_datetime(series)
            df_out[col] = series

        if config.datetime_cyclical:
            df_out[f"{col}_month_sin"] = np.sin(2 * np.pi * series.dt.month / 12)
            df_out[f"{col}_month_cos"] = np.cos(2 * np.pi * series.dt.month / 12)
            df_out[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * series.dt.dayofweek / 7)
            df_out[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * series.dt.dayofweek / 7)
            is_weekend = series.dt.dayofweek >= 5
            df_out[f"{col}_is_weekend"] = is_weekend.where(series.notna(), np.nan)

            reports.append({
                "column": col,
                "action": "datetime_cyclical",
                "summary": f"Generated cyclical month/day and weekend features for '{col}'.",
            })

    if config.datetime_deltas and len(dt_columns) >= 2:
        for a, b in itertools.combinations(dt_columns, 2):
            if a in df_out.columns and b in df_out.columns:
                col_name = f"{a}_to_{b}_days"
                df_out[col_name] = (df_out[b] - df_out[a]).dt.total_seconds() / (24 * 3600)
                reports.append({
                    "column": col_name,
                    "action": "datetime_deltas",
                    "summary": f"Generated elapsed days between '{a}' and '{b}'.",
                })

    if config.datetime_reference_col and config.datetime_reference_col in dt_columns:
        ref = config.datetime_reference_col
        if ref in df_out.columns:
            for col in dt_columns:
                if col != ref and col in df_out.columns:
                    col_name = f"{col}_days_since_ref"
                    df_out[col_name] = (df_out[col] - df_out[ref]).dt.total_seconds() / (24 * 3600)
                    reports.append({
                        "column": col_name,
                        "action": "datetime_reference",
                        "summary": f"Generated elapsed days from '{ref}' to '{col}'.",
                    })

    return df_out, reports
