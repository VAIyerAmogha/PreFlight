"""
median/mode/constant imputation, missing indicators, column drops, outlier winsorization, duplicate removal, ID column drop, category normalization, rare grouping, dtype coercion
"""
import pandas as pd
import numpy as np
import dateutil.parser
from preflight.types import ColumnProfile, SemanticType, ReportEntry

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
    Numeric only. Returns (imputed series, median value used).
    Raises ValueError if series is non-numeric.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"median_impute requires numeric dtype for series '{series.name}', got {series.dtype}")
    
    # If the column is entirely null, median() returns nan. 
    median_val = float(series.median()) if not series.isnull().all() else 0.0
    imputed = series.fillna(median_val)
    return imputed, median_val

def mode_impute(series: pd.Series) -> tuple[pd.Series, object]:
    """
    Categorical only. Returns (imputed series, mode value used).
    If multiple modes exist, uses the first per pandas' default .mode() ordering.
    """
    modes = series.mode(dropna=True)
    if modes.empty:
        # Fallback if entire series is null
        mode_val = np.nan
    else:
        mode_val = modes.iloc[0]
        
    imputed = series.fillna(mode_val)
    return imputed, mode_val

def constant_fill(series: pd.Series, fill_value: str = "__missing__") -> pd.Series:
    """
    Fills nulls with fill_value. For high-cardinality categoricals where mode 
    imputation would be misleading.
    """
    imputed = series.fillna(fill_value)
    return imputed

def should_drop_column(missing_rate: float, drop_threshold: float = 0.6) -> bool:
    """
    Pure decision function, no side effects. 
    True if missing_rate exceeds drop_threshold.
    """
    return missing_rate > drop_threshold

def drop_high_missingness_columns(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    drop_threshold: float = 0.6,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (df with qualifying columns dropped, list of dropped column names).
    Does not mutate input df.
    """
    cols_to_drop = [
        p.name for p in profiles 
        if should_drop_column(p.missing_rate, drop_threshold) and p.name in df.columns
    ]
    if not cols_to_drop:
        return df.copy(), []
    
    new_df = df.drop(columns=cols_to_drop)
    return new_df, cols_to_drop

def drop_numeric_id_columns(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drops all columns where profile.semantic_type == SemanticType.NUMERIC_ID.
    Returns (new df, list of dropped column names).
    """
    cols_to_drop = [
        p.name for p in profiles 
        if p.semantic_type == SemanticType.NUMERIC_ID and p.name in df.columns
    ]
    if not cols_to_drop:
        return df.copy(), []
    
    new_df = df.drop(columns=cols_to_drop)
    return new_df, cols_to_drop

def drop_constant_columns(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drops all columns where profile.semantic_type == SemanticType.CONSTANT.
    Returns (new df, list of dropped column names).
    """
    cols_to_drop = [
        p.name for p in profiles 
        if p.semantic_type == SemanticType.CONSTANT and p.name in df.columns
    ]
    if not cols_to_drop:
        return df.copy(), []
    
    new_df = df.drop(columns=cols_to_drop)
    return new_df, cols_to_drop

def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Returns (deduplicated df, count of rows removed).
    Uses standard full-row duplicate definition.
    """
    initial_len = len(df)
    new_df = df.drop_duplicates()
    removed_count = initial_len - len(new_df)
    return new_df, removed_count

def winsorize_outliers(series: pd.Series, method: str = "iqr") -> pd.Series:
    """
    Clips outliers based on IQR or z-score.
    
    Note: This function itself does not check missingness — that gate belongs in 
    orchestration, which must NOT call this function for columns with missing_rate > 0.30.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"winsorize_outliers requires numeric dtype, got {series.dtype}")
        
    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
    elif method == "zscore":
        mean = series.mean()
        std = series.std()
        lower = mean - 3 * std
        upper = mean + 3 * std
    else:
        raise ValueError(f"Unsupported winsorize method: {method}")
        
    return series.clip(lower=lower, upper=upper)

def normalize_category_values(series: pd.Series) -> pd.Series:
    """
    Lowercases and strips whitespace from string values. 
    Leaves nulls as null. Non-string values passed through unchanged.
    """
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        normalized = series.apply(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )
        return normalized.replace({np.nan: None})
    return series.copy()

def coerce_string_dates_to_datetime(series: pd.Series) -> pd.Series:
    """
    Uses python-dateutil to parse a DATETIME_STRING column into proper datetime64.
    Unparseable individual values become NaT rather than raising.
    """
    def _parse(val):
        if pd.isnull(val):
            return pd.NaT
        try:
            return dateutil.parser.parse(str(val))
        except (ValueError, TypeError, OverflowError):
            return pd.NaT

    parsed = series.apply(_parse)
    return pd.to_datetime(parsed, errors='coerce')

def group_rare_categories(
    series: pd.Series,
    rare_categories: list[str],
    label: str = "__rare__",
) -> pd.Series:
    """
    Replaces any value in `rare_categories` with `label`. 
    
    Note: This function is reused at both profiling-adjacent and cleaning stages, but this
    implementation itself is stateless and simply applies the mapping given a precomputed 
    rare_categories list (produced by profiler.identify_rare_categories).
    Rare category grouping happens before cardinality is finalized in Profiler.
    """
    rare_set = set(rare_categories)
    return series.apply(lambda x: label if x in rare_set else x)

def run_cleaner(
    df: pd.DataFrame,
    profiles: list[ColumnProfile],
    target: str,
    drop_threshold: float = 0.6,
    outlier_method: str = "iqr",
) -> tuple[pd.DataFrame, list[ColumnProfile], list[ReportEntry], dict]:
    df = df.copy()
    report = []
    specs = {}
    
    # 1. Remove duplicates
    initial_rows = len(df)
    df, removed = remove_duplicate_rows(df)
    if removed > 0:
        report.append(ReportEntry(
            stage="cleaner", column="__all__", action="remove_duplicates",
            rationale=f"Removed {removed} duplicate rows", severity="info",
            before_stats={"rows": initial_rows}, after_stats={"rows": len(df)}
        ))
        
    # 2. Drop high missingness columns
    df, dropped_missing = drop_high_missingness_columns(df, profiles, drop_threshold)
    for col in dropped_missing:
        prof = next((p for p in profiles if p.name == col), None)
        m_rate = prof.missing_rate if prof else 1.0
        report.append(ReportEntry(
            stage="cleaner", column=col, action="drop_column",
            rationale=f"Missing rate {m_rate:.2%} > threshold {drop_threshold:.2%}",
            severity="warning"
        ))
    
    # 3. Drop NUMERIC_ID columns
    df, dropped_id = drop_numeric_id_columns(df, profiles)
    for col in dropped_id:
        report.append(ReportEntry(
            stage="cleaner", column=col, action="drop_column",
            rationale="Column is a NUMERIC_ID", severity="warning"
        ))
        
    # 3.5 Drop CONSTANT columns
    df, dropped_constant = drop_constant_columns(df, profiles)
    for col in dropped_constant:
        report.append(ReportEntry(
            stage="cleaner", column=col, action="drop_column",
            rationale="Column is constant", severity="warning"
        ))
        
    # Prune profiles
    dropped_all = set(dropped_missing + dropped_id + dropped_constant)
    surviving_profiles = [p for p in profiles if p.name not in dropped_all]
    
    # 4. Iterate over surviving columns
    for p in surviving_profiles:
        col = p.name
        if col not in df.columns or col == target:
            continue
            
        stype = p.semantic_type
        # Use missing rate from the profile as instructed by the test explicitly providing it
        current_missing_rate = p.missing_rate
        specs[col] = {}
        
        if stype == SemanticType.CATEGORICAL_LOW:
            df[col] = normalize_category_values(df[col])
            report.append(ReportEntry(
                stage="cleaner", column=col, action="normalize_categories",
                rationale="Lowercased and stripped strings", severity="info"
            ))
            
            if current_missing_rate > 0:
                missing_ind = add_missing_indicator(df[col])
                df[missing_ind.name] = missing_ind
                report.append(ReportEntry(
                    stage="cleaner", column=col, action="add_missing_indicator",
                    rationale=f"Missing rate {current_missing_rate:.2%} > 0", severity="info"
                ))
                
                if current_missing_rate < 0.3:
                    df[col], mode_val = mode_impute(df[col])
                    specs[col]["impute_strategy"] = "mode"
                    specs[col]["impute_value"] = mode_val
                    report.append(ReportEntry(
                        stage="cleaner", column=col, action="mode_impute",
                        rationale=f"Missing rate {current_missing_rate:.2%} < 30%", severity="info"
                    ))
                else:
                    df[col] = constant_fill(df[col])
                    specs[col]["impute_strategy"] = "constant"
                    specs[col]["impute_value"] = "__missing__"
                    report.append(ReportEntry(
                        stage="cleaner", column=col, action="constant_fill",
                        rationale=f"Missing rate {current_missing_rate:.2%} >= 30%", severity="info"
                    ))
                    
        elif stype == SemanticType.CATEGORICAL_HIGH:
            df[col] = normalize_category_values(df[col])
            report.append(ReportEntry(
                stage="cleaner", column=col, action="normalize_categories",
                rationale="Lowercased and stripped strings", severity="info"
            ))
            
            if p.rare_categories:
                df[col] = group_rare_categories(df[col], p.rare_categories)
                specs[col]["rare_categories"] = p.rare_categories
                report.append(ReportEntry(
                    stage="cleaner", column=col, action="group_rare_categories",
                    rationale=f"Grouped {len(p.rare_categories)} rare categories", severity="info"
                ))
                
            if current_missing_rate > 0:
                df[col] = constant_fill(df[col])
                specs[col]["impute_strategy"] = "constant"
                specs[col]["impute_value"] = "__missing__"
                report.append(ReportEntry(
                    stage="cleaner", column=col, action="constant_fill",
                    rationale="High cardinality categorical missing imputation", severity="info"
                ))
                
        elif stype == SemanticType.NUMERIC_FEATURE:
            if current_missing_rate > 0:
                missing_ind = add_missing_indicator(df[col])
                df[missing_ind.name] = missing_ind
                report.append(ReportEntry(
                    stage="cleaner", column=col, action="add_missing_indicator",
                    rationale=f"Missing rate {current_missing_rate:.2%} > 0", severity="info"
                ))
                
                if current_missing_rate < 0.3:
                    df[col], median_val = median_impute(df[col])
                    specs[col]["impute_strategy"] = "median"
                    specs[col]["impute_value"] = median_val
                    report.append(ReportEntry(
                        stage="cleaner", column=col, action="median_impute",
                        rationale=f"Missing rate {current_missing_rate:.2%} < 30%", severity="info"
                    ))
            
            if current_missing_rate <= 0.30:
                before_clip = df[col].copy()
                df[col] = winsorize_outliers(df[col], method=outlier_method)
                specs[col]["outlier_method"] = outlier_method
                changed = (before_clip != df[col]).sum()
                if changed > 0:
                    sev = "warning" if (changed / len(df)) > 0.1 else "info"
                    report.append(ReportEntry(
                        stage="cleaner", column=col, action="winsorize_outliers",
                        rationale=f"Winsorized {changed} outliers via {outlier_method}",
                        severity=sev
                    ))
            else:
                report.append(ReportEntry(
                    stage="cleaner", column=col, action="skip_winsorize",
                    rationale=f"Skipped winsorization: missing rate {current_missing_rate:.2%} > 30%",
                    severity="info"
                ))
                
        elif stype == SemanticType.DATETIME_STRING:
            df[col] = coerce_string_dates_to_datetime(df[col])
            report.append(ReportEntry(
                stage="cleaner", column=col, action="coerce_datetime",
                rationale="Parsed string dates to datetime64", severity="info"
            ))
            
    return df, surviving_profiles, report, specs
