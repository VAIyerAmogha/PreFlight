"""
exposes prepare(), profile(), clean(), engineer(), compare(), add_features() as public API
"""

__all__ = ["prepare", "profile", "clean", "engineer", "compare", "add_features"]

import pandas as pd
from preflight import assembler
from preflight.types import PrepResult, ReportEntry
from preflight.report import Report
from preflight.profiler import run_profiler
from preflight.cleaner import run_cleaner
from preflight.engineer import run_engineer, add_features
from preflight.validation import _validate_inputs
from typing import Optional
from preflight.types import FeatureConfig, SemanticType, PRESETS, _UNSET


def prepare(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    model_hint: str = "tree",
    drop_threshold: float = _UNSET,
    outlier_method: str = _UNSET,
    cardinality_threshold: int = _UNSET,
    feature_config: Optional[FeatureConfig] = _UNSET,
    column_types: Optional[dict[str, SemanticType]] = _UNSET,
    preset: Optional[str] = None,
    dry_run: bool = False,
) -> PrepResult:
    if not isinstance(dry_run, bool):
        raise TypeError(f"dry_run must be a boolean, got {type(dry_run).__name__}")

    actual_drop_threshold = 0.6
    actual_outlier_method = "iqr"
    actual_cardinality_threshold = 20
    actual_feature_config = None
    actual_column_types = None

    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Invalid preset '{preset}'. Valid presets are: {list(PRESETS.keys())}")
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
            df=df, target=target, task=task, cardinality_threshold=actual_cardinality_threshold, column_types=actual_column_types
        )
        df_clean, surviving_profiles, cleaner_entries, _ = run_cleaner(
            df=df, profiles=profiles, target=target, drop_threshold=actual_drop_threshold, outlier_method=actual_outlier_method
        )
        _, engineer_entries, _ = run_engineer(
            df=df_clean, profiles=surviving_profiles, target=target, model_hint=model_hint, 
            cardinality_threshold=actual_cardinality_threshold, feature_config=actual_feature_config
        )
        
        all_entries = profiler_entries + cleaner_entries + engineer_entries
        info_entry = ReportEntry(
            stage="profiler",
            column="dataset",
            action="dry_run",
            rationale="dry run mode enabled: no data was transformed and no pipeline was fitted.",
            severity="info"
        )
        all_entries.insert(0, info_entry)
        
        report = Report(all_entries, df=df, profiles=surviving_profiles, target=target)
        result = PrepResult(df=df, pipeline=None, report=report)
    else:
        # warning is handled inside assembler via the profiler entries; we surface it
        # by injecting a ReportEntry into the returned Report below if needed.
        # assembler.run_assembler owns Report construction, so we pass the warning text
        # through as an optional annotation appended after assembly.
        result = assembler.run_assembler(
            df=df,
            target=target,
            task=task,
            model_hint=model_hint,
            drop_threshold=actual_drop_threshold,
            outlier_method=actual_outlier_method,
            cardinality_threshold=actual_cardinality_threshold,
            column_types=actual_column_types,
        )
        
    if preset is not None and result.report is not None:
        preset_expanded = {
            "drop_threshold": actual_drop_threshold,
            "outlier_method": actual_outlier_method,
            "cardinality_threshold": actual_cardinality_threshold,
            "feature_config": str(actual_feature_config) if actual_feature_config else None,
        }
        preset_entry = ReportEntry(
            stage="profiler",
            column="__all__",
            action="apply_preset",
            rationale=f"Applied preset '{preset}'. Expanded parameters: {preset_expanded}",
            severity="info",
        )
        result.report._entries.insert(0, preset_entry)
        
    if warning is not None and result.report is not None:
        entry = ReportEntry(
            stage="profiler",
            column=target,
            action="task_target_mismatch_warning",
            rationale=warning,
            severity="warning",
        )
        # Report.entries is a read-only property (returns a copy); mutate _entries directly.
        result.report._entries.insert(0, entry)
        
    if not dry_run and actual_feature_config is not None:
        # Applies feature engineering post-hoc and returns a new PrepResult
        result = add_features(result, actual_feature_config, target=target)
        
    return result


def profile(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    cardinality_threshold: int = 20,
    column_types: Optional[dict[str, SemanticType]] = None,
) -> PrepResult:
    warning = _validate_inputs(df, target, task, column_types=column_types)

    profiles, profiler_entries = run_profiler(
        df=df, target=target, task=task, cardinality_threshold=cardinality_threshold, column_types=column_types
    )

    if warning is not None:
        profiler_entries = [
            ReportEntry(
                stage="profiler",
                column=target,
                action="task_target_mismatch_warning",
                rationale=warning,
                severity="warning",
            )
        ] + profiler_entries

    return PrepResult(
        df=df,
        pipeline=None,
        report=Report(profiler_entries, df=df, profiles=profiles, target=target),
    )


def clean(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    drop_threshold: float = 0.6,
    outlier_method: str = "iqr",
    cardinality_threshold: int = 20,
    column_types: Optional[dict[str, SemanticType]] = None,
) -> PrepResult:
    warning = _validate_inputs(df, target, task, column_types=column_types)

    profiles, profiler_entries = run_profiler(
        df=df, target=target, task=task, cardinality_threshold=cardinality_threshold, column_types=column_types
    )
    df_clean, surviving_profiles, cleaner_entries, specs = run_cleaner(
        df=df, profiles=profiles, target=target,
        drop_threshold=drop_threshold, outlier_method=outlier_method,
    )

    all_entries = profiler_entries + cleaner_entries
    if warning is not None:
        all_entries = [
            ReportEntry(
                stage="profiler",
                column=target,
                action="task_target_mismatch_warning",
                rationale=warning,
                severity="warning",
            )
        ] + all_entries

    return PrepResult(
        df=df_clean,
        pipeline=None,
        report=Report(all_entries, df=df_clean, profiles=surviving_profiles, target=target),
    )


def engineer(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    model_hint: str = "tree",
    drop_threshold: float = 0.6,
    outlier_method: str = "iqr",
    cardinality_threshold: int = 20,
    column_types: Optional[dict[str, SemanticType]] = None,
) -> PrepResult:
    warning = _validate_inputs(df, target, task, model_hint=model_hint, column_types=column_types)

    profiles, profiler_entries = run_profiler(
        df=df, target=target, task=task, cardinality_threshold=cardinality_threshold, column_types=column_types
    )
    df_clean, surviving_profiles, cleaner_entries, specs_c = run_cleaner(
        df=df, profiles=profiles, target=target,
        drop_threshold=drop_threshold, outlier_method=outlier_method,
    )
    df_eng, engineer_entries, specs_e = run_engineer(
        df=df_clean, profiles=surviving_profiles, target=target,
        model_hint=model_hint, cardinality_threshold=cardinality_threshold,
    )

    all_entries = profiler_entries + cleaner_entries + engineer_entries
    if warning is not None:
        all_entries = [
            ReportEntry(
                stage="profiler",
                column=target,
                action="task_target_mismatch_warning",
                rationale=warning,
                severity="warning",
            )
        ] + all_entries

    return PrepResult(
        df=df_eng,
        pipeline=None,
        report=Report(all_entries, df=df_eng, profiles=surviving_profiles, target=target),
    )


def _extract_actions_per_column(report: Report | None) -> dict[str, set[str]]:
    if report is None:
        return {}

    actions = {}
    for entry in report.entries:
        if entry.column not in actions:
            actions[entry.column] = set()
        actions[entry.column].add(entry.action)
    return actions


def _compute_decision_diff(actions_a: dict[str, set[str]], actions_b: dict[str, set[str]]) -> list[str]:
    diff = []
    shared_cols = set(actions_a.keys()).intersection(actions_b.keys())
    for col in sorted(shared_cols):
        if actions_a[col] != actions_b[col]:
            diff.append(col)
    return diff


def compare(a: PrepResult, b: PrepResult) -> dict:
    cols_a = set(a.df.columns)
    cols_b = set(b.df.columns)

    columns_only_in_a = sorted(cols_a - cols_b)
    columns_only_in_b = sorted(cols_b - cols_a)
    columns_in_both = sorted(cols_a.intersection(cols_b))

    report_entry_counts_a = a.report.summary_counts() if a.report is not None else None
    report_entry_counts_b = b.report.summary_counts() if b.report is not None else None

    actions_a = _extract_actions_per_column(a.report)
    actions_b = _extract_actions_per_column(b.report)

    if a.report is None or b.report is None:
        decision_diff = []
    else:
        decision_diff = _compute_decision_diff(actions_a, actions_b)

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

    # Print summary
    print("=== PreFlight Compare ===")
    print(f"Shape A: {diff['shape_a']}")
    print(f"Shape B: {diff['shape_b']}")

    if columns_only_in_a:
        print(f"Columns only in A ({len(columns_only_in_a)}): {', '.join(columns_only_in_a[:5])}{'...' if len(columns_only_in_a) > 5 else ''}")

    if columns_only_in_b:
        print(f"Columns only in B ({len(columns_only_in_b)}): {', '.join(columns_only_in_b[:5])}{'...' if len(columns_only_in_b) > 5 else ''}")

    if decision_diff:
        print(f"Differing decisions in shared columns:")
        for col in decision_diff:
            print(f"  {col}: A={actions_a.get(col, set())} | B={actions_b.get(col, set())}")

    print("=========================")

    return diff
