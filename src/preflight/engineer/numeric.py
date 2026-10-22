"""
Numeric feature transformations for PreFlight v2.0 (scaling and log1p).
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import scipy.stats


def needs_log_transform(series: pd.Series, skew_threshold: float = 1.0) -> bool:
    """
    Returns True if the absolute skewness of the numeric series exceeds skew_threshold.
    Only evaluates non-negative columns because log1p requires x > -1.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return False
    if clean.min() <= -1:
        return False

    skew_val = scipy.stats.skew(clean)
    return bool(abs(skew_val) > skew_threshold)


def log1p_transform(series: pd.Series) -> pd.Series:
    """
    Applies np.log1p to the series.
    """
    return np.log1p(series)


def standard_scale(series: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Applies z-score scaling: (x - mean) / std.
    Returns (scaled series, {'mean': ..., 'std': ...}) for reuse at inference time.
    If std == 0, returns series unchanged and std=1.0 to avoid division by zero.
    """
    mean_val = float(series.mean()) if not pd.isna(series.mean()) else 0.0
    std_val = float(series.std()) if not pd.isna(series.std()) else 1.0

    if pd.isna(std_val) or std_val == 0.0:
        return series.copy(), {"mean": mean_val, "std": 1.0}

    scaled = (series - mean_val) / std_val
    return scaled, {"mean": mean_val, "std": std_val}
