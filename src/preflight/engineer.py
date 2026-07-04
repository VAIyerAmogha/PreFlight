"""
ordinal encoding, one-hot encoding, target encoding (cross-fit), StandardScaler, log1p transform, datetime expansion, model_hint branching
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import scipy.stats
import itertools
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from preflight.report import Report
from preflight.types import ColumnProfile, ReportEntry, SemanticType, FeatureConfig, PrepResult

class FeatureAugmenterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeatureConfig, profiles: list[ColumnProfile], target: str, cluster_info: dict, skipped_cols: list[str]):
        self.config = config
        self.profiles = profiles
        self.target = target
        self.cluster_info = cluster_info
        self.skipped_cols = skipped_cols

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.config.interactions:
            X_int, _ = generate_interaction_features(X_out, self.profiles, self.target, self.config)
            for col in X_int.columns.difference(X_out.columns):
                if col not in self.skipped_cols:
                    X_out[col] = X_int[col]
                    
        if self.config.datetime_cyclical or self.config.datetime_deltas or self.config.datetime_reference_col:
            X_dt, _ = generate_datetime_cyclical_features(X_out, self.profiles, self.config)
            for col in X_dt.columns.difference(X_out.columns):
                if col not in self.skipped_cols:
                    X_out[col] = X_dt[col]
                    
        if self.config.clustering and self.cluster_info and "model" in self.cluster_info:
            model = self.cluster_info["model"]
            features = self.cluster_info["features"]
            X_clust = X_out[features].fillna(0)
            if "cluster_label" not in self.skipped_cols:
                X_out["cluster_label"] = model.predict(X_clust)
            if "cluster_dist_to_centroid" not in self.skipped_cols:
                dists = model.transform(X_clust)
                X_out["cluster_dist_to_centroid"] = dists.min(axis=1)
                
        return X_out


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

def generate_interaction_features(df: pd.DataFrame, profiles: list[ColumnProfile], target: str, config: FeatureConfig) -> tuple[pd.DataFrame, list[ReportEntry]]:
    df_out = df.copy()
    reports = []
    
    numeric_profiles = [p for p in profiles if p.semantic_type == SemanticType.NUMERIC_FEATURE]
    scored_profiles = []
    for p in numeric_profiles:
        score = 0.0
        if p.correlation_with_target is not None and not pd.isna(p.correlation_with_target):
            score = max(score, abs(p.correlation_with_target))
        if p.mutual_info_with_target is not None and not pd.isna(p.mutual_info_with_target):
            score = max(score, abs(p.mutual_info_with_target))
        scored_profiles.append((p.name, score))
        
    scored_profiles.sort(key=lambda x: x[1], reverse=True)
    top_cols = [x[0] for x in scored_profiles[:config.interaction_top_k] if x[0] in df_out.columns]
    
    for a, b in itertools.combinations(top_cols, 2):
        if "ratio" in config.interaction_types:
            col_name = f"{a}_div_{b}"
            df_out[col_name] = df_out[a] / df_out[b].replace(0, np.nan)
            reports.append(ReportEntry(stage="engineer", column=col_name, action="created_interaction", rationale=f"Ratio of {a} and {b}", severity="info"))
        if "product" in config.interaction_types:
            col_name = f"{a}_times_{b}"
            df_out[col_name] = df_out[a] * df_out[b]
            reports.append(ReportEntry(stage="engineer", column=col_name, action="created_interaction", rationale=f"Product of {a} and {b}", severity="info"))
        if "difference" in config.interaction_types:
            col_name = f"{a}_minus_{b}"
            df_out[col_name] = df_out[a] - df_out[b]
            reports.append(ReportEntry(stage="engineer", column=col_name, action="created_interaction", rationale=f"Difference of {a} and {b}", severity="info"))
            
    return df_out, reports

def generate_datetime_cyclical_features(df: pd.DataFrame, profiles: list[ColumnProfile], config: FeatureConfig) -> tuple[pd.DataFrame, list[ReportEntry]]:
    df_out = df.copy()
    reports = []
    
    dt_cols = [p.name for p in profiles if p.semantic_type in (SemanticType.DATETIME_NATIVE, SemanticType.DATETIME_STRING)]
    
    for col in dt_cols:
        if col not in df_out.columns:
            continue
        series = df_out[col]
        if not pd.api.types.is_datetime64_any_dtype(series):
            try:
                from preflight.cleaner import coerce_string_dates_to_datetime
                series = coerce_string_dates_to_datetime(series)
            except ImportError:
                series = pd.to_datetime(series, errors='coerce')
            df_out[col] = series
                
        if config.datetime_cyclical:
            df_out[f"{col}_month_sin"] = np.sin(2 * np.pi * series.dt.month / 12)
            df_out[f"{col}_month_cos"] = np.cos(2 * np.pi * series.dt.month / 12)
            df_out[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * series.dt.dayofweek / 7)
            df_out[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * series.dt.dayofweek / 7)
            is_weekend = series.dt.dayofweek >= 5
            df_out[f"{col}_is_weekend"] = is_weekend.where(series.notna(), np.nan)
            
            reports.append(ReportEntry(stage="engineer", column=col, action="datetime_cyclical", rationale=f"Cyclical features and is_weekend for {col}", severity="info"))
            
    if config.datetime_deltas and len(dt_cols) >= 2:
        for a, b in itertools.combinations(dt_cols, 2):
            if a in df_out.columns and b in df_out.columns:
                col_name = f"{a}_to_{b}_days"
                df_out[col_name] = (df_out[b] - df_out[a]).dt.total_seconds() / (24 * 3600)
                reports.append(ReportEntry(stage="engineer", column=col_name, action="datetime_deltas", rationale=f"Days between {a} and {b}", severity="info"))
                
    if config.datetime_reference_col and config.datetime_reference_col in dt_cols:
        ref = config.datetime_reference_col
        if ref in df_out.columns:
            for col in dt_cols:
                if col != ref and col in df_out.columns:
                    col_name = f"{col}_days_since_ref"
                    df_out[col_name] = (df_out[col] - df_out[ref]).dt.total_seconds() / (24 * 3600)
                    reports.append(ReportEntry(stage="engineer", column=col_name, action="datetime_reference", rationale=f"Days since reference {ref} for {col}", severity="info"))
                    
    return df_out, reports

def generate_cluster_features(df: pd.DataFrame, profiles: list[ColumnProfile], config: FeatureConfig) -> tuple[pd.DataFrame, list[ReportEntry], dict]:
    df_out = df.copy()
    reports = []
    
    if config.cluster_features == "numeric_only":
        features = [p.name for p in profiles if p.semantic_type == SemanticType.NUMERIC_FEATURE and p.name in df_out.columns]
    else:
        features = [f for f in config.cluster_features if f in df_out.columns]
        
    if not features:
        return df_out, reports, {}
        
    X = df_out[features].fillna(0)
    
    if config.cluster_k == "auto":
        best_k = 2
        best_score = -1
        models = {}
        for k in range(2, min(11, len(X))):
            if k >= len(X):
                break
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                models[k] = (km, score)
                if score > best_score:
                    best_score = score
                    best_k = k
        if best_score == -1:
            best_k = 2
            best_model = KMeans(n_clusters=best_k, random_state=42, n_init="auto").fit(X)
        else:
            best_model = models[best_k][0]
    else:
        best_k = config.cluster_k
        best_model = KMeans(n_clusters=best_k, random_state=42, n_init="auto").fit(X)
        
    df_out["cluster_label"] = best_model.labels_
    dists = best_model.transform(X)
    df_out["cluster_dist_to_centroid"] = dists.min(axis=1)
    
    reports.append(ReportEntry(stage="engineer", column="cluster_label", action="cluster_features", rationale=f"KMeans clustering with k={best_k} on {len(features)} features", severity="info"))
    
    return df_out, reports, {"model": best_model, "features": features}

def run_engineer(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    target: str,
    model_hint: str,
    cardinality_threshold: int = 20,
    feature_config: Optional[FeatureConfig] = None,
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
                
    if feature_config is not None:
        if feature_config.interactions:
            df_out, rep = generate_interaction_features(df_out, profiles, target, feature_config)
            report.extend(rep)
            
        if feature_config.datetime_cyclical or feature_config.datetime_deltas or feature_config.datetime_reference_col:
            dt_df, rep = generate_datetime_cyclical_features(df, profiles, feature_config)
            new_cols = dt_df.columns.difference(df.columns)
            df_out = df_out.join(dt_df[new_cols])
            report.extend(rep)
            
        if feature_config.clustering:
            df_out, rep, fitted = generate_cluster_features(df_out, profiles, feature_config)
            report.extend(rep)
            if fitted:
                specs["_clustering"] = {"transform": "cluster_features", "fitted_info": fitted}
                
    return df_out, report, specs


def add_features(
    result: PrepResult,
    feature_config: FeatureConfig,
    profiles: Optional[list[ColumnProfile]] = None,
    target: Optional[str] = None
) -> PrepResult:
    """
    Applies FeatureConfig-driven feature engineering to an ALREADY-PREPARED PrepResult,
    without rerunning Profiler/Cleaner from scratch.
    """
    if result.pipeline is None:
        raise ValueError("add_features() requires a full prepare() result with a pipeline.")
        
    profiles_to_use = profiles if profiles is not None else getattr(result.report, "profiles", getattr(result.report, "_profiles", None))
    target_to_use = target if target is not None else getattr(result.report, "target", getattr(result.report, "_target", None))
    
    if profiles_to_use is None or target_to_use is None:
        raise ValueError("add_features() requires profiles and target. Pass them explicitly or use a full prepare() result.")
        
    all_off = not (
        feature_config.interactions or
        feature_config.datetime_cyclical or
        feature_config.datetime_deltas or
        feature_config.datetime_reference_col or
        feature_config.clustering
    )
    
    if all_off:
        entry = ReportEntry(
            stage="engineer", column="dataset", action="add_features_skipped", 
            rationale="No features requested in FeatureConfig.", severity="info"
        )
        new_report = Report(
            result.report.entries + [entry], 
            result.report._df, 
            result.report._profiles, 
            result.report._target
        )
        new_pipeline = Pipeline(steps=result.pipeline.steps)
        return PrepResult(df=result.df.copy(), pipeline=new_pipeline, report=new_report)

    df_out = result.df.copy()
    new_entries = []
    skipped_cols = []
    cluster_info = {}
    
    df_raw = getattr(result.report, "_df", None)
    if df_raw is None:
        df_raw = df_out.copy()
    else:
        df_raw = df_raw.copy()
    
    # 1. Interactions
    if feature_config.interactions:
        df_int, reps = generate_interaction_features(df_out, profiles_to_use, target_to_use, feature_config)
        for col in df_int.columns.difference(df_out.columns):
            if col in result.df.columns:
                skipped_cols.append(col)
                new_entries.append(ReportEntry(stage="engineer", column=col, action="skipped_duplicate_feature", rationale=f"Skipped generated interaction feature {col} due to name collision.", severity="warning"))
            else:
                df_out[col] = df_int[col]
        for rep in reps:
            if rep.column not in skipped_cols:
                new_entries.append(rep)
                
    # 2. Datetimes
    if feature_config.datetime_cyclical or feature_config.datetime_deltas or feature_config.datetime_reference_col:
        df_dt, reps = generate_datetime_cyclical_features(df_raw, profiles_to_use, feature_config)
        for col in df_dt.columns.difference(df_raw.columns):
            if col in result.df.columns:
                skipped_cols.append(col)
                new_entries.append(ReportEntry(stage="engineer", column=col, action="skipped_duplicate_feature", rationale=f"Skipped generated datetime feature {col} due to name collision.", severity="warning"))
            else:
                df_out[col] = df_dt[col]
        for rep in reps:
            if rep.column not in skipped_cols:
                new_entries.append(rep)
                
    # 3. Clustering
    if feature_config.clustering:
        df_clust, reps, cluster_info = generate_cluster_features(df_out, profiles_to_use, feature_config)
        for col in ["cluster_label", "cluster_dist_to_centroid"]:
            if col in df_clust.columns:
                if col in result.df.columns:
                    skipped_cols.append(col)
                    new_entries.append(ReportEntry(stage="engineer", column=col, action="skipped_duplicate_feature", rationale=f"Skipped generated cluster feature {col} due to name collision.", severity="warning"))
                else:
                    df_out[col] = df_clust[col]
        for rep in reps:
            if rep.column not in skipped_cols:
                new_entries.append(rep)
                
    new_report = Report(
        result.report.entries + new_entries, 
        result.report._df, 
        result.report._profiles, 
        result.report._target
    )
    
    augmenter = FeatureAugmenterTransformer(feature_config, profiles_to_use, target_to_use, cluster_info, skipped_cols)
    augmenter.is_fitted_ = True
    step = ("augmenter", augmenter)
    new_pipeline = Pipeline(steps=result.pipeline.steps + [step])
    
    return PrepResult(df=df_out, pipeline=new_pipeline, report=new_report)
