import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

from preflight.types import ColumnProfile, SemanticType, PrepResult
from preflight.profiler import run_profiler
from preflight.report import Report
from preflight.cleaner import (
    run_cleaner,
    add_missing_indicator,
    normalize_category_values,
    coerce_string_dates_to_datetime,
    group_rare_categories,
)
from preflight.engineer import run_engineer, expand_datetime

# =====================================================================
# Architectural Decision: Two-Phase Pipeline Assembly
# ---------------------------------------------------------------------
# EngineerTransformer requires the POST-CLEANING column profiles. 
# However, the surviving profiles are only known AFTER CleanerTransformer 
# computes its dynamic drops during `fit()`.
# 
# To resolve this coupling without breaking standard sklearn Pipeline 
# semantics, the `build_pipeline` function returns an UNFIT pipeline 
# (which would fail or process dropped columns incorrectly if fitted 
# blindly). Instead, `build_pipeline_two_phase` acts as the actual 
# usable constructor: it manually fits the Cleaner stage, reads the 
# resulting `fitted_profiles_`, uses them to construct and fit the 
# Engineer stage, and then returns a fully assembled, pre-fitted 
# Pipeline ready for `transform()`.
# =====================================================================

class CleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        profiles: list[ColumnProfile],
        target: str,
        drop_threshold: float = 0.6,
        outlier_method: str = "iqr",
    ) -> None:
        self.profiles = profiles
        self.target = target
        self.drop_threshold = drop_threshold
        self.outlier_method = outlier_method

    def fit(self, X: pd.DataFrame, y=None) -> "CleanerTransformer":
        # run_cleaner computes the required stats
        df_clean, self.fitted_profiles_, self.report_entries_, self.specs_ = run_cleaner(
            X, self.profiles, self.target, self.drop_threshold, self.outlier_method
        )
        
        surviving_cols = {p.name for p in self.fitted_profiles_}
        self.columns_to_drop_ = [p.name for p in self.profiles if p.name not in surviving_cols]
        
        # Augment specs_ with exact bounds and indicators to avoid recomputing on test data
        for p in self.fitted_profiles_:
            col = p.name
            if col not in self.specs_:
                self.specs_[col] = {}
                
            # Missing indicator logic from cleaner.py
            if p.missing_rate > 0 and p.semantic_type in (SemanticType.CATEGORICAL_LOW, SemanticType.NUMERIC_FEATURE):
                self.specs_[col]["add_missing_indicator"] = True
                
            # Winsorization bounds logic
            if p.semantic_type == SemanticType.NUMERIC_FEATURE and "outlier_method" in self.specs_[col]:
                series = X[col]
                method = self.specs_[col]["outlier_method"]
                if method == "iqr":
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    self.specs_[col]["lower_bound"] = q1 - 1.5 * iqr
                    self.specs_[col]["upper_bound"] = q3 + 1.5 * iqr
                elif method == "zscore":
                    mean = series.mean()
                    std = series.std()
                    self.specs_[col]["lower_bound"] = mean - 3 * std
                    self.specs_[col]["upper_bound"] = mean + 3 * std

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "specs_"):
            raise NotFittedError("CleanerTransformer is not fitted yet. Call 'fit' before 'transform'.")
            
        X_out = X.copy()
        
        # 1. Drop columns (by name, not by recomputing missingness)
        cols_to_drop = [c for c in self.columns_to_drop_ if c in X_out.columns]
        X_out = X_out.drop(columns=cols_to_drop)
        
        # 2. Duplicate row removal is explicitly skipped here.
        # Removing rows during transform (inference) would silently drop rows the caller expects back.
        
        for p in self.fitted_profiles_:
            col = p.name
            if col not in X_out.columns or col == self.target:
                continue
                
            spec = self.specs_.get(col, {})
            stype = p.semantic_type
            
            # Category normalization applies to both CATEGORICAL_LOW and CATEGORICAL_HIGH
            if stype in (SemanticType.CATEGORICAL_LOW, SemanticType.CATEGORICAL_HIGH):
                X_out[col] = normalize_category_values(X_out[col])
                
            # Date coercion
            if stype == SemanticType.DATETIME_STRING:
                X_out[col] = coerce_string_dates_to_datetime(X_out[col])
                
            # Group rare categories
            if stype == SemanticType.CATEGORICAL_HIGH and "rare_categories" in spec:
                X_out[col] = group_rare_categories(X_out[col], spec["rare_categories"])
                
            # Add missing indicator based on frozen fit-time decision
            if spec.get("add_missing_indicator"):
                missing_ind = add_missing_indicator(X_out[col])
                X_out[missing_ind.name] = missing_ind
                
            # Impute values
            if "impute_strategy" in spec:
                X_out[col] = X_out[col].fillna(spec["impute_value"])
                
            # Winsorize using bounds computed at fit-time
            if stype == SemanticType.NUMERIC_FEATURE and "lower_bound" in spec and "upper_bound" in spec:
                X_out[col] = X_out[col].clip(
                    lower=spec["lower_bound"], 
                    upper=spec["upper_bound"]
                )
                
        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "fitted_profiles_"):
            raise NotFittedError("CleanerTransformer is not fitted yet.")
        cols = []
        missing_cols = []
        for p in self.fitted_profiles_:
            cols.append(p.name)
            if self.specs_.get(p.name, {}).get("add_missing_indicator"):
                missing_cols.append(f"{p.name}_missing")
        return np.array(cols + missing_cols)

class EngineerTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        profiles: list[ColumnProfile],
        target: str,
        model_hint: str,
        cardinality_threshold: int = 20,
    ) -> None:
        self.profiles = profiles
        self.target = target
        self.model_hint = model_hint
        self.cardinality_threshold = cardinality_threshold

    def fit(self, X: pd.DataFrame, y=None) -> "EngineerTransformer":
        if y is None:
            has_high_card = any(p.semantic_type == SemanticType.CATEGORICAL_HIGH for p in self.profiles)
            if has_high_card:
                raise ValueError("Target y must be provided for target encoding high-cardinality columns.")
                
        df_for_engineer = X.copy()
        if y is not None:
            df_for_engineer[self.target] = y

        df_eng, self.report_entries_, self.specs_ = run_engineer(
            df_for_engineer, self.profiles, self.target, self.model_hint, self.cardinality_threshold
        )
        
        cols = list(df_eng.columns)
        if self.target in cols and (y is not None and self.target not in X.columns):
            cols.remove(self.target)
        self.output_columns_ = cols
        
        if y is not None:
            if isinstance(y, pd.Series):
                global_mean = y.mean()
            else:
                global_mean = pd.Series(y).mean()
            for p in self.profiles:
                if p.semantic_type == SemanticType.CATEGORICAL_HIGH:
                    if p.name in self.specs_ and isinstance(self.specs_[p.name], dict):
                        self.specs_[p.name]["global_mean"] = global_mean

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "specs_"):
            raise NotFittedError("EngineerTransformer is not fitted yet.")
            
        X_out = X.copy()
        
        for p in self.profiles:
            col = p.name
            stype = p.semantic_type
            
            if col not in X_out.columns or col == self.target:
                continue
                
            spec = self.specs_.get(col, {})
            trans_type = spec.get("transform")
            
            if trans_type == "passthrough":
                continue
                
            elif trans_type == "expand_datetime":
                expanded = expand_datetime(X_out[col])
                X_out = X_out.drop(columns=[col]).join(expanded)
                
            elif trans_type == "ordinal_encode":
                mapping = spec["mapping"]
                X_out[col] = X_out[col].map(mapping).fillna(-1).astype(int)
                
            elif trans_type == "one_hot_encode":
                dummies = pd.get_dummies(X_out[col], prefix=col, prefix_sep='_')
                X_out = X_out.drop(columns=[col]).join(dummies)
                
            elif trans_type == "target_encode_cross_fit":
                mapping = spec["mapping"]
                global_mean = spec.get("global_mean", 0.0)
                X_out[col] = X_out[col].map(mapping).fillna(global_mean)
                
            elif trans_type == "linear_numeric":
                if spec.get("log1p"):
                    X_out[col] = np.log1p(X_out[col])
                
                params = spec["scale_params"]
                mean_val = params["mean"]
                std_val = params["std"]
                X_out[col] = (X_out[col] - mean_val) / std_val
                
        for expected_col in self.output_columns_:
            if expected_col not in X_out.columns:
                X_out[expected_col] = False
                
        X_out = X_out.reindex(columns=self.output_columns_)
        
        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "output_columns_"):
            raise NotFittedError("EngineerTransformer is not fitted yet.")
        return np.array(self.output_columns_)

def build_pipeline(
    profiles: list[ColumnProfile],
    target: str,
    model_hint: str,
    drop_threshold: float = 0.6,
    outlier_method: str = "iqr",
    cardinality_threshold: int = 20,
) -> Pipeline:
    cleaner = CleanerTransformer(
        profiles=profiles,
        target=target,
        drop_threshold=drop_threshold,
        outlier_method=outlier_method,
    )
    engineer = EngineerTransformer(
        profiles=profiles,  # Will be overridden in two-phase fit
        target=target,
        model_hint=model_hint,
        cardinality_threshold=cardinality_threshold,
    )
    
    pipeline = Pipeline([
        ("cleaner", cleaner),
        ("engineer", engineer),
    ])
    pipeline.set_output(transform="pandas")
    return pipeline

def build_pipeline_two_phase(
    profiles: list[ColumnProfile],
    target: str,
    model_hint: str,
    X: pd.DataFrame,
    y: pd.Series,
    drop_threshold: float = 0.6,
    outlier_method: str = "iqr",
    cardinality_threshold: int = 20,
) -> Pipeline:
    cleaner = CleanerTransformer(
        profiles=profiles,
        target=target,
        drop_threshold=drop_threshold,
        outlier_method=outlier_method,
    )
    
    cleaner.fit(X, y)
    X_cleaned = cleaner.transform(X)
    
    engineer = EngineerTransformer(
        profiles=cleaner.fitted_profiles_,
        target=target,
        model_hint=model_hint,
        cardinality_threshold=cardinality_threshold,
    )
    engineer.fit(X_cleaned, y)
    
    pipeline = Pipeline([
        ("cleaner", cleaner),
        ("engineer", engineer),
    ])
    pipeline.set_output(transform="pandas")
    return pipeline

def run_assembler(
    df: pd.DataFrame,
    target: str,
    task: str,
    model_hint: str,
    drop_threshold: float = 0.6,
    outlier_method: str = "iqr",
    cardinality_threshold: int = 20,
) -> PrepResult:
    # 1. Run profiler
    profiles, profiler_report = run_profiler(
        df, target, task, cardinality_threshold
    )
    
    # 2. Split df into X and y
    X = df.drop(columns=[target])
    y = df[target]
    
    # 3. Call build_pipeline_two_phase
    pipeline = build_pipeline_two_phase(
        profiles=profiles,
        target=target,
        model_hint=model_hint,
        X=X,
        y=y,
        drop_threshold=drop_threshold,
        outlier_method=outlier_method,
        cardinality_threshold=cardinality_threshold,
    )
    
    # 4. Transform X and reattach target
    X_final = pipeline.transform(X)
    final_df = X_final.copy()
    final_df[target] = y
    
    # 5. Collect all ReportEntries
    cleaner_step = pipeline.named_steps["cleaner"]
    engineer_step = pipeline.named_steps["engineer"]
    all_entries = profiler_report + cleaner_step.report_entries_ + engineer_step.report_entries_
    
    # 6. Construct Report
    report = Report(all_entries)
    
    # 7. Return PrepResult
    return PrepResult(
        df=final_df,
        pipeline=pipeline,
        report=report
    )

def transform_new_data(
    pipeline: Pipeline, 
    new_df: pd.DataFrame, 
    target: str | None = None
) -> pd.DataFrame:
    df_in = new_df.copy()
    target_series = None
    
    if target is not None and target in df_in.columns:
        target_series = df_in.pop(target)
        
    df_out = pipeline.transform(df_in)
    
    if target_series is not None:
        df_out[target] = target_series
        
    return df_out
