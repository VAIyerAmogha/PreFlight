"""
Categorical encoding algorithms for PreFlight v2.0 (ordinal, one-hot, cross-fit target).
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def ordinal_encode(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Maps each unique category to an integer via a deterministic sorted mapping.
    Returns (encoded series, mapping dict) for reuse at inference time.
    Unseen categories at inference time map to -1.
    """
    unique_vals = sorted([x for x in series.dropna().unique()])
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    encoded = series.map(mapping).fillna(-1).astype(int)
    return encoded, mapping


def one_hot_encode(series: pd.Series) -> pd.DataFrame:
    """
    Returns a DataFrame with one boolean column per category, named f"{series.name}_{category}".
    Uses pandas.get_dummies internally and ensures column names are deterministically sorted.
    """
    dummies = pd.get_dummies(series, prefix=series.name, prefix_sep="_")
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)
    return dummies.astype(bool)


def target_encode_cross_fit(
    series: pd.Series,
    target: pd.Series,
    n_folds: int = 5,
    smoothing: float = 10.0,
) -> Tuple[pd.Series, Dict[Any, float]]:
    """
    Implements 5-fold cross-fit target encoding to protect against data leakage.

    For each fold, category-mean-of-target is computed using only OTHER folds' data,
    so no row's encoded value is influenced by its own target value.
    Applies smoothing toward global target mean:
    smoothed_mean = (count * cat_mean + smoothing * global_mean) / (count + smoothing).

    Returns (encoded series, full dataset mapping dict).
    """
    target_numeric = target
    if not pd.api.types.is_numeric_dtype(target):
        unique_vals = sorted(target.dropna().unique())
        mapping_labels = {val: idx for idx, val in enumerate(unique_vals)}
        target_numeric = target.map(mapping_labels)

    encoded = pd.Series(index=series.index, dtype=float)
    global_mean = float(target_numeric.mean()) if len(target_numeric) > 0 else 0.0

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    series_np = series.values
    target_np = target_numeric.values

    # If dataset has fewer rows than n_folds, fallback to global smoothed mapping
    if len(series) < n_folds:
        stats_full = target_numeric.groupby(series).agg(["count", "mean"])
        smoothed_full = (
            stats_full["count"] * stats_full["mean"] + smoothing * global_mean
        ) / (stats_full["count"] + smoothing)
        mapping_dict = smoothed_full.to_dict()
        return series.map(mapping_dict).fillna(global_mean), mapping_dict

    for train_idx, val_idx in kf.split(series_np):
        train_series = pd.Series(series_np[train_idx], index=train_idx)
        train_target = pd.Series(target_np[train_idx], index=train_idx)
        val_series = pd.Series(series_np[val_idx], index=val_idx)

        stats = train_target.groupby(train_series).agg(["count", "mean"])
        smoothed = (
            stats["count"] * stats["mean"] + smoothing * global_mean
        ) / (stats["count"] + smoothing)

        mapped = val_series.map(smoothed).fillna(global_mean)
        encoded.iloc[val_idx] = mapped.values

    stats_full = target_numeric.groupby(series).agg(["count", "mean"])
    smoothed_full = (
        stats_full["count"] * stats_full["mean"] + smoothing * global_mean
    ) / (stats_full["count"] + smoothing)
    mapping_dict = smoothed_full.to_dict()

    return encoded, mapping_dict
