import logging
import pandas as pd
from typing import Any, List, Optional, Dict, Tuple
import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder
from dateutil.parser import parse, ParserError

from .types import SemanticType, ColumnProfile, ReportEntry

logger = logging.getLogger(__name__)

def detect_mixed_types(series: pd.Series) -> bool:
    """
    Detect if a pandas Series contains mixed Python types among its non-null values.
    Returns True if more than one type is present.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    # Find the unique python types in the series
    types = non_null.apply(type)
    return types.nunique() > 1

def infer_semantic_type(series: pd.Series, cardinality_threshold: int = 20) -> SemanticType:
    """
    Infer the SemanticType of a pandas Series based on its values, dtype, and name.
    
    Args:
        series: The pandas Series to evaluate.
        cardinality_threshold: The threshold above which a categorical column is considered high cardinality.
        
    Returns:
        SemanticType classification for the column.
    """
    non_null = series.dropna()
    n_unique = non_null.nunique()
    total_len = len(series)
    
    # 1. CONSTANT (single unique non-null value, or all-null)
    if n_unique <= 1:
        return SemanticType.CONSTANT
        
    # 2. BOOLEAN (exactly 2 unique values, or bool dtype)
    if pd.api.types.is_bool_dtype(series) or n_unique == 2:
        return SemanticType.BOOLEAN
        
    # 3. DATETIME_NATIVE (already datetime64 dtype)
    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.DATETIME_NATIVE
        
    # 4. NUMERIC_ID and NUMERIC_FEATURE
    if pd.api.types.is_numeric_dtype(series):
        name_lower = str(series.name).lower() if series.name else ""
        is_high_cardinality = (n_unique / total_len >= 0.95) if total_len > 0 else False
        
        id_patterns = ["id", "_id", "pk", "index"]
        matches_id_pattern = any(pat in name_lower for pat in id_patterns)
        
        is_integer = pd.api.types.is_integer_dtype(series)
        is_monotonic = series.is_monotonic_increasing or series.is_monotonic_decreasing
        
        if is_high_cardinality and (matches_id_pattern or (is_monotonic and is_integer)):
            logger.info(
                f"Column '{series.name}' inferred as NUMERIC_ID vs NUMERIC_FEATURE. "
                f"Rationale: high cardinality ({n_unique}/{total_len}) AND "
                f"(matches ID pattern={matches_id_pattern} OR is monotonic int sequence={is_monotonic and is_integer})."
            )
            return SemanticType.NUMERIC_ID
            
        return SemanticType.NUMERIC_FEATURE
        
    # 5. DATETIME_STRING (object/string dtype parsing to date)
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
        # Sample to avoid slow date parsing on huge datasets
        sample_size = min(100, len(non_null))
        sample = non_null.sample(n=sample_size, random_state=42) if len(non_null) > 100 else non_null
        
        if len(sample) > 0:
            success_count = 0
            for val in sample:
                try:
                    if isinstance(val, str):
                        # Use fuzzy=False to be stricter about what's considered a date
                        parse(val, fuzzy=False)
                        success_count += 1
                except (ValueError, TypeError, OverflowError, ParserError):
                    pass
                    
            if success_count / len(sample) > 0.9:
                return SemanticType.DATETIME_STRING

    # 6. CATEGORICAL (LOW and HIGH)
    if n_unique < cardinality_threshold:
        return SemanticType.CATEGORICAL_LOW
    else:
        return SemanticType.CATEGORICAL_HIGH

def compute_missing_rate(series: pd.Series) -> float:
    """Compute the proportion of missing values in the series."""
    if len(series) == 0:
        return 0.0
    return float(series.isna().mean())

def compute_outlier_rate(series: pd.Series) -> Optional[float]:
    """
    Compute the proportion of outliers using the IQR method.
    Returns None for non-numeric columns, or if IQR is degenerate (Q1 == Q3).
    """
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
        
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
        
    q1 = non_null.quantile(0.25)
    q3 = non_null.quantile(0.75)
    iqr = q3 - q1
    
    if iqr == 0:
        return None
        
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = non_null[(non_null < lower_bound) | (non_null > upper_bound)]
    return float(len(outliers) / len(non_null))

def compute_cardinality(series: pd.Series) -> int:
    """Compute the count of unique non-null values."""
    return int(series.nunique())

def identify_rare_categories(series: pd.Series, threshold: float = 0.01) -> List[str]:
    """
    Identify categories occurring below the threshold frequency of non-null values.
    Returns an empty list for non-categorical distributions.
    Caller is expected to invoke only where relevant (CATEGORICAL_LOW/HIGH).
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return []
        
    freqs = non_null.value_counts(normalize=True)
    rare = freqs[freqs < threshold]
    
    # Return string representations of categories as requested by type hint
    return [str(x) for x in rare.index]

def compute_correlation_with_target(series: pd.Series, target: pd.Series) -> Optional[float]:
    """Compute Pearson correlation with target, numeric columns only."""
    if not pd.api.types.is_numeric_dtype(series) or not pd.api.types.is_numeric_dtype(target):
        return None
    val = series.corr(target)
    if pd.isna(val):
        return None
    return float(val)

def compute_mutual_info_with_target(series: pd.Series, target: pd.Series, task: str) -> Optional[float]:
    """Compute mutual information with target based on task type."""
    valid_idx = series.notna() & target.notna()
    s = series[valid_idx]
    t = target[valid_idx]
    
    if len(s) == 0:
        return None
        
    s_encoded = s
    if not pd.api.types.is_numeric_dtype(s):
        s_encoded = OrdinalEncoder().fit_transform(s.to_frame()).flatten()
        
    X = np.array(s_encoded).reshape(-1, 1)
    y = np.array(t)
    
    if task == "regression":
        if not pd.api.types.is_numeric_dtype(target):
            return None
        mi = mutual_info_regression(X, y)
    elif task == "classification":
        mi = mutual_info_classif(X, y)
    else:
        return None
        
    return float(mi[0])

def flag_leakage_suspect(correlation: Optional[float], threshold: float = 0.95) -> bool:
    """Flag if correlation is above threshold."""
    if correlation is None:
        return False
    return abs(correlation) > threshold

def compute_class_imbalance_ratio(target: pd.Series) -> Optional[float]:
    """Compute ratio of majority class count to minority class count (classification only)."""
    if pd.api.types.is_float_dtype(target):
        return None
    counts = target.dropna().value_counts()
    if len(counts) < 2:
        return None
    return float(counts.max() / counts.min())

def compute_vif_scores(df: pd.DataFrame, numeric_columns: List[str]) -> Tuple[Dict[str, Optional[float]], bool]:
    """
    Compute Variance Inflation Factor across numeric columns.
    Caps at top 50 numeric features by variance.
    Returns (scores_dict, capped_flag).
    """
    result = {col: None for col in numeric_columns}
    if not numeric_columns:
        return result, False
        
    capped = False
    cols_to_compute = numeric_columns
    
    if len(numeric_columns) > 50:
        capped = True
        variances = df[numeric_columns].var()
        cols_to_compute = variances.nlargest(50).index.tolist()
        
    df_clean = df[cols_to_compute].dropna()
    if len(df_clean) < 2 or len(cols_to_compute) < 2:
        return result, capped
        
    corr = df_clean.corr().values
    try:
        inv_corr = np.linalg.inv(corr)
        for i, col in enumerate(cols_to_compute):
            vif = inv_corr[i, i]
            result[col] = float(vif) if vif > 0 else 0.0
    except np.linalg.LinAlgError:
        pass
        
    return result, capped

def run_profiler(
    df: pd.DataFrame,
    target: str,
    task: str,
    cardinality_threshold: int = 20,
) -> Tuple[List[ColumnProfile], List[ReportEntry]]:
    """
    Orchestrate the profiling stage across all columns.
    Computes SemanticType, structural signals, target-dependent signals, and global signals (VIF, Class Imbalance).
    Produces a list of ColumnProfiles and a list of ReportEntries for warnings/criticals.
    """
    profiles: List[ColumnProfile] = []
    reports: List[ReportEntry] = []
    
    target_series = df[target] if target in df.columns else None
    numeric_feature_cols = []
    
    for col in df.columns:
        if col == target:
            continue
            
        series = df[col]
        
        # ------------------------------------------------------------------
        # Do not re-infer SemanticType after Profiler has run
        # This is the only place SemanticType gets computed.
        # ------------------------------------------------------------------
        sem_type = infer_semantic_type(series, cardinality_threshold=cardinality_threshold)
        
        if sem_type == SemanticType.NUMERIC_FEATURE:
            numeric_feature_cols.append(col)
            
        # 1. Structural signals
        has_mixed = detect_mixed_types(series)
        if has_mixed:
            reports.append(ReportEntry(
                stage="profiler", column=col, action="flagged_mixed_types",
                rationale="Column contains multiple Python types among non-null values.",
                severity="warning"
            ))
            
        missing = compute_missing_rate(series)
        if missing > 0.3:
            reports.append(ReportEntry(
                stage="profiler", column=col, action="flagged_high_missingness",
                rationale=f"Missingness rate is {missing:.1%}, > 30%",
                severity="warning", before_stats={"missing_rate": missing}
            ))
            
        outlier = compute_outlier_rate(series)
        cardinality = compute_cardinality(series)
        
        rare = None
        if sem_type in (SemanticType.CATEGORICAL_LOW, SemanticType.CATEGORICAL_HIGH):
            rare = identify_rare_categories(series, threshold=0.01)
            
        if sem_type == SemanticType.CATEGORICAL_HIGH:
            reports.append(ReportEntry(
                stage="profiler", column=col, action="flagged_high_cardinality",
                rationale=f"Cardinality is {cardinality}, >= {cardinality_threshold}",
                severity="warning", before_stats={"cardinality": cardinality}
            ))
            
        # 2. Target-dependent signals
        corr = None
        if sem_type in (SemanticType.NUMERIC_FEATURE, SemanticType.NUMERIC_ID) and target_series is not None:
            corr = compute_correlation_with_target(series, target_series)
            
        is_leak = flag_leakage_suspect(corr)
        if is_leak:
            reports.append(ReportEntry(
                stage="profiler", column=col, action="flagged_leakage_suspect",
                rationale=f"Absolute correlation with target is {abs(corr):.3f} > 0.95",
                severity="critical", before_stats={"correlation": corr}
            ))
            
        mi = None
        if target_series is not None and sem_type not in (SemanticType.CONSTANT, SemanticType.NUMERIC_ID):
            mi = compute_mutual_info_with_target(series, target_series, task=task)
            
        # Build Profile
        profile = ColumnProfile(
            name=col, semantic_type=sem_type, missing_rate=missing,
            dtype=str(series.dtype), outlier_rate=outlier, cardinality=cardinality,
            rare_categories=rare, vif_score=None, correlation_with_target=corr,
            mutual_info_with_target=mi, is_leakage_suspect=is_leak
        )
        profiles.append(profile)
        
    # 3. Global Signals: VIF
    if numeric_feature_cols:
        vif_scores, capped = compute_vif_scores(df, numeric_feature_cols)
        if capped:
            reports.append(ReportEntry(
                stage="profiler", column="dataset", action="capped_vif_computation",
                rationale="VIF computation capped at top 50 numeric features by variance.",
                severity="warning"
            ))
            
        for prof in profiles:
            if prof.name in numeric_feature_cols:
                prof.vif_score = vif_scores.get(prof.name, None)
                
    # 4. Global Signals: Class Imbalance
    if task == "classification" and target_series is not None:
        imbalance = compute_class_imbalance_ratio(target_series)
        if imbalance is not None and imbalance > 10.0:
            reports.append(ReportEntry(
                stage="profiler", column=target, action="flagged_class_imbalance",
                rationale=f"Class imbalance ratio is {imbalance:.1f}:1",
                severity="warning", before_stats={"imbalance_ratio": imbalance}
            ))
            
    return profiles, reports
