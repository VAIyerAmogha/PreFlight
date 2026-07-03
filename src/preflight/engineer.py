"""
ordinal encoding, one-hot encoding, target encoding (cross-fit), StandardScaler, log1p transform, datetime expansion, model_hint branching
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import scipy.stats
from preflight.types import ColumnProfile, ReportEntry, SemanticType

def ordinal_encode(series: pd.Series) -> tuple[pd.Series, dict]:
    """
    Maps each unique category to an integer via a deterministic sorted mapping.
    Returns (encoded series, mapping dict) for reuse at inference time.
    
    Note: Unseen categories at inference time should map to -1.
    """
    unique_vals = sorted([x for x in series.dropna().unique()])
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    encoded = series.map(mapping).fillna(-1).astype(int)
    return encoded, mapping

def one_hot_encode(series: pd.Series) -> pd.DataFrame:
    """
    Returns a DataFrame with one boolean column per category, named f"{series.name}_{category}".
    Uses pandas.get_dummies internally but ensures column names are deterministic (sorted)
    regardless of category appearance order.
    """
    dummies = pd.get_dummies(series, prefix=series.name, prefix_sep='_')
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)
    dummies = dummies.astype(bool)
    return dummies

def target_encode_cross_fit(
    series: pd.Series,
    target: pd.Series,
    n_folds: int = 5,
    smoothing: float = 10.0,
) -> tuple[pd.Series, dict]:
    """
    Implements 5-fold cross-fit target encoding.
    
    Cross-fit protects against leakage in a way naive target encoding doesn't:
    For each fold, we compute category-mean-of-target using only the OTHER folds' data, 
    so no row's encoded value is influenced by its own target. This prevents the target 
    from leaking into features during training.
    
    Applies smoothing toward the global target mean for categories with few observations:
    smoothed_mean = (count * cat_mean + smoothing * global_mean) / (count + smoothing).
    
    Returns (encoded series, a fitted mapping dict computed on the FULL dataset).
    """
    if not pd.api.types.is_numeric_dtype(target):
        raise ValueError("Target must be numeric for target encoding.")
    
    encoded = pd.Series(index=series.index, dtype=float)
    global_mean = target.mean()
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    series_np = series.values
    target_np = target.values
    
    for train_idx, val_idx in kf.split(series_np):
        train_series = pd.Series(series_np[train_idx], index=train_idx)
        train_target = pd.Series(target_np[train_idx], index=train_idx)
        val_series = pd.Series(series_np[val_idx], index=val_idx)
        
        stats = train_target.groupby(train_series).agg(['count', 'mean'])
        smoothed = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        
        mapped = val_series.map(smoothed).fillna(global_mean)
        encoded.iloc[val_idx] = mapped.values
        
    stats_full = target.groupby(series).agg(['count', 'mean'])
    smoothed_full = (stats_full['count'] * stats_full['mean'] + smoothing * global_mean) / (stats_full['count'] + smoothing)
    mapping_dict = smoothed_full.to_dict()
    
    return encoded, mapping_dict

def standard_scale(series: pd.Series) -> tuple[pd.Series, dict]:
    """
    Applies z-score scaling: (x - mean) / std.
    Returns (scaled series, {"mean": ..., "std": ...}) for reuse at inference time.
    
    If std == 0 (e.g., constant column), returns the series unchanged 
    and std=1.0 in the dict to avoid division by zero.
    """
    mean_val = series.mean()
    std_val = series.std()
    
    # Handle the edge case of constant column (std == 0)
    # to avoid division by zero or NaN outputs.
    if pd.isna(std_val) or std_val == 0.0:
        return series.copy(), {"mean": mean_val, "std": 1.0}
        
    scaled = (series - mean_val) / std_val
    return scaled, {"mean": mean_val, "std": std_val}

def needs_log_transform(series: pd.Series, skew_threshold: float = 1.0) -> bool:
    """
    Returns True if the absolute skewness of the numeric series exceeds skew_threshold.
    
    Only evaluates non-negative columns because log1p requires x > -1. 
    If the series contains values <= -1, returns False as log1p would be undefined/unsafe.
    """
    # Guard against undefined/unsafe log1p values
    if series.min() <= -1:
        return False
        
    skew_val = scipy.stats.skew(series.dropna())
    return bool(abs(skew_val) > skew_threshold)

def log1p_transform(series: pd.Series) -> pd.Series:
    """
    Applies np.log1p to the series.
    
    Note: Caller is responsible for calling needs_log_transform first. 
    This function is a pure mechanical transform.
    """
    return np.log1p(series)

def expand_datetime(series: pd.Series) -> pd.DataFrame:
    """
    Expands a datetime column into multiple numeric and boolean features:
    year, month, day, dayofweek, hour, and is_weekend.
    
    Raises ValueError if the input series is not of datetime64 dtype.
    
    Note for tests: Pandas' native .dt accessors automatically propagate NaT 
    as NaN (or NaT/pd.NA depending on the type) in the resulting derived columns. 
    This behavior is intentional so that rows with missing dates do not crash 
    the pipeline and are not silently dropped.
    """
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise ValueError(f"Input series '{series.name}' must be datetime64 dtype.")
        
    df = pd.DataFrame(index=series.index)
    prefix = series.name
    
    df[f"{prefix}_year"] = series.dt.year
    df[f"{prefix}_month"] = series.dt.month
    df[f"{prefix}_day"] = series.dt.day
    df[f"{prefix}_dayofweek"] = series.dt.dayofweek
    df[f"{prefix}_hour"] = series.dt.hour
    
    # dayofweek: Monday=0, Sunday=6. Weekend is >= 5.
    # Note: For NaT, dayofweek is NaN, and (NaN >= 5) evaluates to False. 
    # To keep NaN propagation for the boolean column, we map NaNs back if necessary,
    # or just use nullable boolean dtype. Since pandas .dt.dayofweek is float when there are NaNs,
    # we can do this:
    is_weekend = series.dt.dayofweek >= 5
    # Mask NaNs with np.nan to preserve missingness rather than False
    is_weekend = is_weekend.where(series.notna(), np.nan)
    df[f"{prefix}_is_weekend"] = is_weekend
    
    return df

def run_engineer(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    target: str,
    model_hint: str,
    cardinality_threshold: int = 20,
) -> tuple[pd.DataFrame, list[ReportEntry], dict]:
    """
    Orchestrates the feature engineering phase based on column profiles and model_hint.
    """
    if model_hint not in ("tree", "linear"):
        raise ValueError("model_hint must be 'tree' or 'linear'")
        
    df_out = df.copy()
    report = []
    specs = {}
    
    for profile in profiles:
        col = profile.name
        stype = profile.semantic_type
        
        if col == target:
            continue
            
        if col.endswith("_missing") or stype == SemanticType.BOOLEAN:
            # Passthrough
            specs[col] = {"transform": "passthrough"}
            
        elif stype == SemanticType.CONSTANT:
            # Passthrough
            specs[col] = {"transform": "passthrough"}
            
        elif stype == SemanticType.DATETIME_NATIVE:
            expanded = expand_datetime(df_out[col])
            df_out = df_out.drop(columns=[col]).join(expanded)
            report.append(ReportEntry(
                stage="engineer",
                column=col,
                action="expand_datetime",
                rationale="Expanded datetime into year, month, day, dayofweek, hour, is_weekend.",
                severity="info"
            ))
            specs[col] = {"transform": "expand_datetime"}
            
        elif stype == SemanticType.CATEGORICAL_LOW:
            if model_hint == "tree":
                encoded, mapping = ordinal_encode(df_out[col])
                df_out[col] = encoded
                report.append(ReportEntry(
                    stage="engineer",
                    column=col,
                    action="ordinal_encode",
                    rationale=f"Applied ordinal encoding for {model_hint} model_hint.",
                    severity="info"
                ))
                specs[col] = {"transform": "ordinal_encode", "mapping": mapping}
            else:
                dummies = one_hot_encode(df_out[col])
                df_out = df_out.drop(columns=[col]).join(dummies)
                report.append(ReportEntry(
                    stage="engineer",
                    column=col,
                    action="one_hot_encode",
                    rationale=f"Applied one-hot encoding for {model_hint} model_hint.",
                    severity="info"
                ))
                specs[col] = {"transform": "one_hot_encode", "columns": list(dummies.columns)}
                
        elif stype == SemanticType.CATEGORICAL_HIGH:
            encoded, mapping = target_encode_cross_fit(df_out[col], df_out[target])
            df_out[col] = encoded
            report.append(ReportEntry(
                stage="engineer",
                column=col,
                action="target_encode_cross_fit",
                rationale="Applied 5-fold cross-fit target encoding to prevent leakage for high-cardinality categorical.",
                severity="info"
            ))
            specs[col] = {"transform": "target_encode_cross_fit", "mapping": mapping}
            
        elif stype == SemanticType.NUMERIC_FEATURE:
            if model_hint == "linear":
                log_applied = False
                if needs_log_transform(df_out[col]):
                    df_out[col] = log1p_transform(df_out[col])
                    log_applied = True
                    report.append(ReportEntry(
                        stage="engineer",
                        column=col,
                        action="log1p_transform",
                        rationale=f"Applied log1p transform due to skewness for {model_hint} model_hint.",
                        severity="info"
                    ))
                
                scaled, params = standard_scale(df_out[col])
                df_out[col] = scaled
                report.append(ReportEntry(
                    stage="engineer",
                    column=col,
                    action="standard_scale",
                    rationale=f"Applied z-score standardization for {model_hint} model_hint.",
                    severity="info"
                ))
                specs[col] = {
                    "transform": "linear_numeric",
                    "log1p": log_applied,
                    "scale_params": params
                }
            else:
                # Tree models do not need numeric scaling or log transform
                specs[col] = {"transform": "passthrough"}
                
    return df_out, report, specs
