"""
Internal, test-only adapter bridging v1.0.1 public-API test calls
(prepare, profile, clean, engineer, compare, add_features, run_profiler,
save_compare_pdf) onto v2.0 components.

This module intentionally lives under tests/ rather than src/preflight/ so
that the v2 public package surface (preflight/__init__.py) stays clean of
v1 compatibility shims. See docs/port_contract.md and docs/port_regressions.md
for the rationale behind each bridging decision.

The profiling logic below (`run_profiler` and its helpers) is a direct port
of v1 `preflight/profiler.py`: v2 deliberately does not have a monolithic
profiler (see port_contract.md section 5), so the EDA signals it used to
compute (missing_rate, cardinality, rare_categories, correlation/mutual-info
with target, VIF, leakage suspicion, text stats) are reconstructed here
purely from pandas/numpy/sklearn, unchanged from v1 behavior.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.parser import ParserError, parse
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder

from preflight.assemble import build_pipeline_two_phase
from preflight.clean.cleaner import coerce_numeric_string_column, run_cleaner
from preflight.engineer.engineer import add_features, run_engineer
from preflight.report.report import (
    SEVERITY_COLORS,
    Report,
    _truncate_label,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_missingness_heatmap,
    plot_mutual_info_bar_chart,
)
from preflight.types import PRESETS, ColumnProfile, FeatureConfig, PrepResult, ReportEntry, SemanticType, _UNSET
from preflight.validation import _validate_inputs

logger = logging.getLogger(__name__)

__all__ = [
    "prepare", "profile", "clean", "engineer", "compare",
    "add_features", "run_profiler", "save_compare_pdf",
]


# ===========================================================================
# Profiler port (v1 preflight/profiler.py -> v2 bridge; no v2 equivalent)
# ===========================================================================

def detect_mixed_types(series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    return non_null.apply(type).nunique() > 1


def infer_semantic_type(series: pd.Series, cardinality_threshold: int = 20) -> SemanticType:
    non_null = series.dropna()
    n_unique = non_null.nunique()
    total_len = len(series)

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample_size = min(100, len(non_null))
        sample = non_null.sample(n=sample_size, random_state=42) if len(non_null) > 100 else non_null
        sample_str = sample.astype(str)
        avg_len = sample_str.str.len().mean()
        avg_words = sample_str.str.split().str.len().mean()
        if avg_len > 50 or (avg_len > 30 and avg_words >= 5):
            return SemanticType.TEXT

    if n_unique <= 1:
        return SemanticType.CONSTANT

    if pd.api.types.is_bool_dtype(series) or n_unique == 2:
        return SemanticType.BOOLEAN

    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.DATETIME_NATIVE

    if pd.api.types.is_numeric_dtype(series):
        name_lower = str(series.name).lower() if series.name else ""
        cat_patterns = ["zip", "postal", "postcode"]
        if any(pat in name_lower for pat in cat_patterns):
            return SemanticType.CATEGORICAL_LOW if n_unique < cardinality_threshold else SemanticType.CATEGORICAL_HIGH

        is_high_cardinality = (n_unique / total_len >= 0.95) if total_len > 0 else False
        id_patterns = ["id", "_id", "pk", "index"]
        matches_id_pattern = any(pat in name_lower for pat in id_patterns)
        is_integer = pd.api.types.is_integer_dtype(series)
        is_monotonic = series.is_monotonic_increasing or series.is_monotonic_decreasing

        if is_high_cardinality and (matches_id_pattern or (is_monotonic and is_integer)):
            return SemanticType.NUMERIC_ID
        return SemanticType.NUMERIC_FEATURE

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        sample_size = min(100, len(non_null))
        sample = non_null.sample(n=sample_size, random_state=42) if len(non_null) > 100 else non_null
        if len(sample) > 0:
            success_count = 0
            for val in sample:
                try:
                    if isinstance(val, str):
                        parse(val, fuzzy=False)
                        success_count += 1
                except (ValueError, TypeError, OverflowError, ParserError):
                    pass
            if success_count / len(sample) > 0.9:
                return SemanticType.DATETIME_STRING

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        uniqueness_ratio = n_unique / len(non_null) if len(non_null) > 0 else 0
        if n_unique >= cardinality_threshold or uniqueness_ratio >= 0.1:
            sample_size = min(100, len(non_null))
            sample = non_null.sample(n=sample_size, random_state=42) if len(non_null) > 100 else non_null
            sample_str = sample.astype(str)
            avg_len = sample_str.str.len().mean()
            avg_words = sample_str.str.split().str.len().mean()
            if avg_len > 50 or (avg_len > 30 and avg_words >= 5):
                return SemanticType.TEXT

    return SemanticType.CATEGORICAL_LOW if n_unique < cardinality_threshold else SemanticType.CATEGORICAL_HIGH


def compute_missing_rate(series: pd.Series) -> float:
    return 0.0 if len(series) == 0 else float(series.isna().mean())


def compute_outlier_rate(series: pd.Series) -> Optional[float]:
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = non_null[(non_null < lower) | (non_null > upper)]
    return float(len(outliers) / len(non_null))


def compute_cardinality(series: pd.Series) -> int:
    return int(series.nunique())


def identify_rare_categories(series: pd.Series, threshold: float = 0.01) -> List[str]:
    non_null = series.dropna()
    if len(non_null) == 0:
        return []
    freqs = non_null.value_counts(normalize=True)
    return [str(x) for x in freqs[freqs < threshold].index]


def compute_correlation_with_target(series: pd.Series, target: pd.Series) -> Optional[float]:
    if not pd.api.types.is_numeric_dtype(series) or not pd.api.types.is_numeric_dtype(target):
        return None
    val = series.corr(target)
    return None if pd.isna(val) else float(val)


def compute_mutual_info_with_target(series: pd.Series, target: pd.Series, task: str) -> Optional[float]:
    valid_idx = series.notna() & target.notna()
    s, t = series[valid_idx], target[valid_idx]
    if len(s) == 0:
        return None
    s_encoded = s if pd.api.types.is_numeric_dtype(s) else OrdinalEncoder().fit_transform(s.to_frame()).flatten()
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
    return False if correlation is None else abs(correlation) > threshold


def compute_class_imbalance_ratio(target: pd.Series) -> Optional[float]:
    if pd.api.types.is_float_dtype(target):
        return None
    counts = target.dropna().value_counts()
    if len(counts) < 2:
        return None
    return float(counts.max() / counts.min())


def compute_vif_scores(df: pd.DataFrame, numeric_columns: List[str]) -> Tuple[Dict[str, Optional[float]], bool]:
    result: Dict[str, Optional[float]] = {col: None for col in numeric_columns}
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
    column_types: Optional[Dict[str, SemanticType]] = None,
) -> Tuple[List[ColumnProfile], List[ReportEntry]]:
    profiles: List[ColumnProfile] = []
    reports: List[ReportEntry] = []

    df = df.copy()
    target_series = df[target] if target in df.columns else None
    numeric_feature_cols: List[str] = []

    for col in df.columns:
        if col == target:
            continue

        series = df[col]
        name_lower = str(col).lower()
        matches_cat_pattern = any(pat in name_lower for pat in ("zip", "postal", "postcode"))

        coerced_series, unit = (None, None)
        if not matches_cat_pattern:
            coerced_series, unit = coerce_numeric_string_column(series)

        if coerced_series is not None:
            total_len = len(series)
            n_unique = series.nunique()
            is_high_cardinality = (n_unique / total_len >= 0.95) if total_len > 0 else False

            if is_high_cardinality:
                id_patterns = ["id", "_id", "pk", "index"]
                matches_id_pattern = any(pat in name_lower for pat in id_patterns)
                try:
                    coerced_non_null = coerced_series.dropna()
                    is_integer = (coerced_non_null % 1 == 0).all() if len(coerced_non_null) > 0 else False
                except Exception:
                    is_integer = False
                is_monotonic = coerced_series.is_monotonic_increasing or coerced_series.is_monotonic_decreasing

                if matches_id_pattern or (is_monotonic and is_integer):
                    series = coerced_series
                    df[col] = coerced_series
                    reports.append(ReportEntry(
                        stage="profiler", column=col, action="coerce_numeric",
                        rationale=f"Inferred string column as numeric ID. Cleaned unit/whitespace suffix: '{unit}'." if unit else "Inferred string column as numeric ID by stripping whitespace.",
                        severity="info",
                    ))
                elif is_integer and not unit:
                    reports.append(ReportEntry(
                        stage="profiler", column=col, action="skip_coerce",
                        rationale="Classified as high-cardinality categorical / ID-like despite looking numeric.",
                        severity="info",
                    ))
                else:
                    series = coerced_series
                    df[col] = coerced_series
                    reports.append(ReportEntry(
                        stage="profiler", column=col, action="coerce_numeric",
                        rationale=f"Inferred string column as numeric. Cleaned unit/whitespace suffix: '{unit}'." if unit else "Inferred string column as numeric by stripping whitespace.",
                        severity="info",
                    ))
            else:
                series = coerced_series
                df[col] = coerced_series
                reports.append(ReportEntry(
                    stage="profiler", column=col, action="coerce_numeric",
                    rationale=f"Inferred string column as numeric. Cleaned unit/whitespace suffix: '{unit}'." if unit else "Inferred string column as numeric by stripping whitespace.",
                    severity="info",
                ))

        sem_type = infer_semantic_type(series, cardinality_threshold=cardinality_threshold)

        if sem_type == SemanticType.NUMERIC_FEATURE:
            numeric_feature_cols.append(col)

        has_mixed = detect_mixed_types(series)
        if has_mixed:
            reports.append(ReportEntry(
                stage="profiler", column=col, action="flagged_mixed_types",
                rationale="Column contains multiple Python types among non-null values.",
                severity="warning",
            ))

        missing = compute_missing_rate(series)
        if missing > 0.3:
            reports.append(ReportEntry(
                stage="profiler", column=col, action="flagged_high_missingness",
                rationale=f"Missingness rate is {missing:.1%}, > 30%",
                severity="warning", before_stats={"missing_rate": missing},
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
                severity="warning", before_stats={"cardinality": cardinality},
            ))

        corr = None
        if sem_type in (SemanticType.NUMERIC_FEATURE, SemanticType.NUMERIC_ID) and target_series is not None:
            corr = compute_correlation_with_target(series, target_series)

        is_leak = flag_leakage_suspect(corr)
        if is_leak:
            reports.append(ReportEntry(
                stage="profiler", column=col, action="flagged_leakage_suspect",
                rationale=f"Absolute correlation with target is {abs(corr):.3f} > 0.95",
                severity="critical", before_stats={"correlation": corr},
            ))

        mi = None
        if target_series is not None and sem_type not in (SemanticType.CONSTANT, SemanticType.NUMERIC_ID):
            mi = compute_mutual_info_with_target(series, target_series, task=task)

        prof = ColumnProfile(
            name=col, semantic_type=sem_type, missing_rate=missing,
            dtype=str(series.dtype), outlier_rate=outlier, cardinality=cardinality,
            rare_categories=rare, vif_score=None, correlation_with_target=corr,
            mutual_info_with_target=mi, is_leakage_suspect=is_leak,
        )

        if sem_type == SemanticType.TEXT:
            non_null_text = series.dropna().astype(str)
            text_avg_length = float(non_null_text.str.len().mean()) if len(non_null_text) > 0 else 0.0
            text_avg_word_count = float(non_null_text.str.split().str.len().mean()) if len(non_null_text) > 0 else 0.0
            prof.text_avg_length = text_avg_length
            prof.text_avg_word_count = text_avg_word_count
            prof.text_missing_rate = missing
            reports.append(ReportEntry(
                stage="profiler", column=col, action="flagged_text_column",
                rationale="Column detected as text (high cardinality, high average length). Only basic stats computed in this phase, no feature generation.",
                severity="info",
                before_stats={
                    "text_avg_length": text_avg_length,
                    "text_avg_word_count": text_avg_word_count,
                    "text_missing_rate": missing,
                },
            ))

        profiles.append(prof)

    if column_types:
        for p in profiles:
            if p.name in column_types:
                orig_type = p.semantic_type
                p.semantic_type = column_types[p.name]
                reports.append(ReportEntry(
                    stage="profiler", column=p.name, action="manual_semantic_type_override",
                    rationale=f"column '{p.name}' forced to {p.semantic_type.name} (auto-inferred was {orig_type.name})",
                    severity="info",
                ))
                if p.semantic_type == SemanticType.NUMERIC_FEATURE and p.name not in numeric_feature_cols:
                    numeric_feature_cols.append(p.name)
                elif p.semantic_type != SemanticType.NUMERIC_FEATURE and p.name in numeric_feature_cols:
                    numeric_feature_cols.remove(p.name)

    if numeric_feature_cols:
        vif_scores, capped = compute_vif_scores(df, numeric_feature_cols)
        if capped:
            reports.append(ReportEntry(
                stage="profiler", column="dataset", action="capped_vif_computation",
                rationale="VIF computation capped at top 50 numeric features by variance.",
                severity="warning",
            ))
        for prof in profiles:
            if prof.name in numeric_feature_cols:
                prof.vif_score = vif_scores.get(prof.name, None)

    if task == "classification" and target_series is not None:
        imbalance = compute_class_imbalance_ratio(target_series)
        if imbalance is not None and imbalance > 10.0:
            reports.append(ReportEntry(
                stage="profiler", column=target, action="flagged_class_imbalance",
                rationale=f"Class imbalance ratio is {imbalance:.1f}:1",
                severity="warning", before_stats={"imbalance_ratio": imbalance},
            ))

    return profiles, reports


# ===========================================================================
# v1 public API bridge: prepare / profile / clean / engineer / compare
# ===========================================================================

def _col_name(p: Any) -> str:
    return p.name if hasattr(p, "name") else (p.columns[0] if getattr(p, "columns", None) else str(p))


def prepare(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    model_hint: str = "tree",
    drop_threshold: float = _UNSET,
    outlier_method: str = _UNSET,
    cardinality_threshold: int = _UNSET,
    feature_config: Optional[FeatureConfig] = _UNSET,
    column_types: Optional[Dict[str, SemanticType]] = _UNSET,
    preset: Optional[str] = None,
    dry_run: bool = False,
) -> PrepResult:
    if not isinstance(dry_run, bool):
        raise TypeError("The 'dry_run' argument must be True or False.")

    actual_drop_threshold = 0.6
    actual_outlier_method = "iqr"
    actual_cardinality_threshold = 20
    actual_feature_config = None
    actual_column_types = None

    if preset is not None:
        if preset not in PRESETS:
            valid = ", ".join(PRESETS.keys())
            raise ValueError(f"We don't recognize the preset name '{preset}'. Please choose one of these instead: {valid}.")
        p_dict = PRESETS[preset]
        actual_drop_threshold = p_dict.get("drop_threshold", actual_drop_threshold)
        actual_outlier_method = p_dict.get("outlier_method", actual_outlier_method)
        actual_cardinality_threshold = p_dict.get("cardinality_threshold", actual_cardinality_threshold)
        actual_feature_config = p_dict.get("feature_config", actual_feature_config)

    if drop_threshold is not _UNSET:
        actual_drop_threshold = drop_threshold
    if outlier_method is not _UNSET:
        actual_outlier_method = outlier_method
    if cardinality_threshold is not _UNSET:
        actual_cardinality_threshold = cardinality_threshold
    if feature_config is not _UNSET:
        actual_feature_config = feature_config
    if column_types is not _UNSET:
        actual_column_types = column_types

    warning = _validate_inputs(df, target, task, model_hint=model_hint, column_types=actual_column_types)

    if dry_run:
        profiles, profiler_entries = run_profiler(
            df=df, target=target, task=task, cardinality_threshold=actual_cardinality_threshold, column_types=actual_column_types,
        )
        _df_clean, surviving_profiles, cleaner_entries, _specs = run_cleaner(
            df=df, profiles=profiles, target=target, drop_threshold=actual_drop_threshold, outlier_method=actual_outlier_method,
        )
        _, engineer_entries, _ = run_engineer(
            df=_df_clean, profiles=surviving_profiles, target=target, model_hint=model_hint,
            cardinality_threshold=actual_cardinality_threshold, feature_config=actual_feature_config,
        )

        all_entries = profiler_entries + cleaner_entries + engineer_entries
        all_entries.insert(0, ReportEntry(
            stage="profiler", column="dataset", action="dry_run",
            rationale="dry run mode enabled: no data was transformed and no pipeline was fitted.",
            severity="info",
        ))

        report = Report(all_entries, df=df, profiles=surviving_profiles, target=target)
        result = PrepResult(df=df, pipeline=None, report=report)
    else:
        profiles, profiler_entries = run_profiler(
            df=df, target=target, task=task, cardinality_threshold=actual_cardinality_threshold, column_types=actual_column_types,
        )
        X = df.drop(columns=[target]) if target in df.columns else df.copy()
        y = df[target] if target in df.columns else None

        pipeline = build_pipeline_two_phase(
            profiles=profiles, target=target, model_hint=model_hint, X=X, y=y,
            drop_threshold=actual_drop_threshold, outlier_method=actual_outlier_method,
            cardinality_threshold=actual_cardinality_threshold, feature_config=actual_feature_config,
        )

        X_final = pipeline.transform(X)
        final_df = X_final.copy()
        if y is not None:
            final_df[target] = y

        cleaner_step = pipeline.named_steps["cleaner"]
        engineer_step = pipeline.named_steps["engineer"]
        all_entries = profiler_entries + cleaner_step.report_entries_ + engineer_step.report_entries_
        report = Report(entries=all_entries, df=df, profiles=cleaner_step.fitted_profiles_, target=target)

        result = PrepResult(df=final_df, pipeline=pipeline, report=report)

    if preset is not None and result.report is not None:
        preset_expanded = {
            "drop_threshold": actual_drop_threshold,
            "outlier_method": actual_outlier_method,
            "cardinality_threshold": actual_cardinality_threshold,
            "feature_config": str(actual_feature_config) if actual_feature_config else None,
        }
        result.report._entries.insert(0, ReportEntry(
            stage="profiler", column="__all__", action="apply_preset",
            rationale=f"Applied preset '{preset}'. Expanded parameters: {preset_expanded}",
            severity="info",
        ))

    if warning is not None and result.report is not None:
        result.report._entries.insert(0, ReportEntry(
            stage="profiler", column=target, action="task_target_mismatch_warning",
            rationale=warning, severity="warning",
        ))

    if not dry_run and actual_feature_config is not None:
        result = add_features(result, actual_feature_config, target=target)

    return result


def profile(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    model_hint: str = "tree",
    cardinality_threshold: int = 20,
    column_types: Optional[Dict[str, SemanticType]] = None,
) -> PrepResult:
    warning = _validate_inputs(df, target, task, model_hint=model_hint, column_types=column_types)
    profiles, profiler_entries = run_profiler(
        df=df, target=target, task=task, cardinality_threshold=cardinality_threshold, column_types=column_types,
    )
    if warning is not None:
        profiler_entries = [ReportEntry(
            stage="profiler", column=target, action="task_target_mismatch_warning",
            rationale=warning, severity="warning",
        )] + profiler_entries

    return PrepResult(df=df, pipeline=None, report=Report(profiler_entries, df=df, profiles=profiles, target=target))


def clean(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    model_hint: str = "tree",
    drop_threshold: float = 0.6,
    outlier_method: str = "iqr",
    cardinality_threshold: int = 20,
    column_types: Optional[Dict[str, SemanticType]] = None,
) -> PrepResult:
    warning = _validate_inputs(df, target, task, model_hint=model_hint, column_types=column_types)
    profiles, profiler_entries = run_profiler(
        df=df, target=target, task=task, cardinality_threshold=cardinality_threshold, column_types=column_types,
    )
    df_clean, surviving_profiles, cleaner_entries, _specs = run_cleaner(
        df=df, profiles=profiles, target=target, drop_threshold=drop_threshold, outlier_method=outlier_method,
    )

    all_entries = profiler_entries + cleaner_entries
    if warning is not None:
        all_entries = [ReportEntry(
            stage="profiler", column=target, action="task_target_mismatch_warning",
            rationale=warning, severity="warning",
        )] + all_entries

    return PrepResult(df=df_clean, pipeline=None, report=Report(all_entries, df=df_clean, profiles=surviving_profiles, target=target))


def engineer(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    model_hint: str = "tree",
    drop_threshold: float = 0.6,
    outlier_method: str = "iqr",
    cardinality_threshold: int = 20,
    column_types: Optional[Dict[str, SemanticType]] = None,
) -> PrepResult:
    warning = _validate_inputs(df, target, task, model_hint=model_hint, column_types=column_types)
    profiles, profiler_entries = run_profiler(
        df=df, target=target, task=task, cardinality_threshold=cardinality_threshold, column_types=column_types,
    )
    df_clean, surviving_profiles, cleaner_entries, _specs_c = run_cleaner(
        df=df, profiles=profiles, target=target, drop_threshold=drop_threshold, outlier_method=outlier_method,
    )
    df_eng, engineer_entries, _specs_e = run_engineer(
        df=df_clean, profiles=surviving_profiles, target=target, model_hint=model_hint,
        cardinality_threshold=cardinality_threshold,
    )

    all_entries = profiler_entries + cleaner_entries + engineer_entries
    if warning is not None:
        all_entries = [ReportEntry(
            stage="profiler", column=target, action="task_target_mismatch_warning",
            rationale=warning, severity="warning",
        )] + all_entries

    return PrepResult(df=df_eng, pipeline=None, report=Report(all_entries, df=df_eng, profiles=surviving_profiles, target=target))


def _extract_actions_per_column(report: Optional[Report]) -> Dict[str, set]:
    if report is None:
        return {}
    actions: Dict[str, set] = {}
    for entry in report.entries:
        actions.setdefault(entry.column, set()).add(entry.action)
    return actions


def _compute_decision_diff(actions_a: Dict[str, set], actions_b: Dict[str, set]) -> List[str]:
    diff = []
    for col in sorted(set(actions_a.keys()) & set(actions_b.keys())):
        if actions_a[col] != actions_b[col]:
            diff.append(col)
    return diff


def compare(a: PrepResult, b: PrepResult) -> dict:
    if a is None or b is None:
        raise ValueError("You must provide two preparation results to compare them. Check that neither result is empty.")
    if a.df is None or b.df is None:
        raise ValueError("We cannot compare these results because they are missing their datasets.")

    cols_a, cols_b = set(a.df.columns), set(b.df.columns)
    columns_only_in_a = sorted(cols_a - cols_b)
    columns_only_in_b = sorted(cols_b - cols_a)
    columns_in_both = sorted(cols_a & cols_b)

    report_entry_counts_a = a.report.summary_counts() if a.report is not None else None
    report_entry_counts_b = b.report.summary_counts() if b.report is not None else None

    actions_a = _extract_actions_per_column(a.report)
    actions_b = _extract_actions_per_column(b.report)
    decision_diff = [] if (a.report is None or b.report is None) else _compute_decision_diff(actions_a, actions_b)

    diff = {
        "shape_a": a.df.shape,
        "shape_b": b.df.shape,
        "columns_only_in_a": columns_only_in_a,
        "columns_only_in_b": columns_only_in_b,
        "columns_in_both": columns_in_both,
        "report_entry_counts_a": report_entry_counts_a,
        "report_entry_counts_b": report_entry_counts_b,
        "decision_diff": decision_diff,
    }

    print("=== PreFlight Compare ===")
    print(f"Shape A: {diff['shape_a']}")
    print(f"Shape B: {diff['shape_b']}")
    if columns_only_in_a:
        print(f"Columns only in A ({len(columns_only_in_a)}): {', '.join(columns_only_in_a[:5])}{'...' if len(columns_only_in_a) > 5 else ''}")
    if columns_only_in_b:
        print(f"Columns only in B ({len(columns_only_in_b)}): {', '.join(columns_only_in_b[:5])}{'...' if len(columns_only_in_b) > 5 else ''}")
    if decision_diff:
        print("Differing decisions in shared columns:")
        for col in decision_diff:
            print(f"  {col}: A={actions_a.get(col, set())} | B={actions_b.get(col, set())}")
    print("=========================")

    return diff


def save_compare_pdf(result_a: PrepResult, result_b: PrepResult, path: str) -> None:
    import matplotlib.patches as patches
    from matplotlib.backends.backend_pdf import PdfPages

    if result_a is not None and not isinstance(result_a, PrepResult):
        raise TypeError("The first argument must be a PrepResult object. Please provide a valid PrepResult from prepare().")
    if result_b is not None and not isinstance(result_b, PrepResult):
        raise TypeError("The second argument must be a PrepResult object. Please provide a valid PrepResult from prepare().")
    if result_a is None or result_b is None:
        raise ValueError("You must provide two preparation results to compare them. Check that neither result is empty.")
    if result_a.df is None or result_b.df is None:
        raise ValueError("We cannot compare these results because they are missing their datasets.")
    if result_a.report is None or result_b.report is None:
        raise ValueError("We cannot generate a comparison PDF because one or both of the results are missing their report.")

    diff = compare(result_a, result_b)
    counts_a = diff.get("report_entry_counts_a", {}) or {}
    counts_b = diff.get("report_entry_counts_b", {}) or {}

    with PdfPages(path) as pdf:
        fig = plt_figure_summary(diff, counts_a, counts_b, result_a, result_b)
        pdf.savefig(fig)
        plt_close(fig)


def plt_figure_summary(diff, counts_a, counts_b, result_a, result_b):
    """Minimal single-page comparison summary figure (text-only)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    lines = [
        "PreFlight Compare Report",
        "",
        f"Shape A: {diff['shape_a']}    Shape B: {diff['shape_b']}",
        f"Columns only in A: {len(diff['columns_only_in_a'])}",
        f"Columns only in B: {len(diff['columns_only_in_b'])}",
        f"Columns in both: {len(diff['columns_in_both'])}",
        "",
        f"Report entry counts A: {counts_a}",
        f"Report entry counts B: {counts_b}",
        "",
        f"Differing decisions in shared columns: {diff['decision_diff']}",
    ]
    ax.text(0.02, 0.98, "\n".join(str(line) for line in lines), va="top", ha="left", fontsize=10, wrap=True)
    return fig


def plt_close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)
