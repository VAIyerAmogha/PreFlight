"""
exposes prepare(), profile(), clean(), engineer(), compare() as public API
"""

import pandas as pd
from preflight import assembler
from preflight.types import PrepResult
from preflight.report import Report
from preflight.profiler import run_profiler
from preflight.cleaner import run_cleaner
from preflight.engineer import run_engineer
def prepare(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    model_hint: str = "tree",
    drop_threshold: float = 0.6,
    outlier_method: str = "iqr",
    cardinality_threshold: int = 20,
) -> PrepResult:
    if len(df) == 0:
        raise ValueError("DataFrame cannot be empty")
        
    if target not in df.columns:
        raise ValueError(f"target '{target}' not found in DataFrame columns: {list(df.columns)}")
        
    if len(df.columns) < 2:
        raise ValueError("DataFrame must contain at least one feature column besides the target")
        
    if df.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names")
        
    if task not in ["classification", "regression"]:
        raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")
        
    if model_hint not in ["tree", "linear"]:
        raise ValueError(f"model_hint must be 'tree' or 'linear', got '{model_hint}'")

    return assembler.run_assembler(
        df=df,
        target=target,
        task=task,
        model_hint=model_hint,
        drop_threshold=drop_threshold,
        outlier_method=outlier_method,
        cardinality_threshold=cardinality_threshold,
    )

def profile(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    cardinality_threshold: int = 20,
) -> PrepResult:
    if len(df) == 0:
        raise ValueError("DataFrame cannot be empty")
    if target not in df.columns:
        raise ValueError(f"target '{target}' not found in DataFrame columns: {list(df.columns)}")
    if len(df.columns) < 2:
        raise ValueError("DataFrame must contain at least one feature column besides the target")
    if df.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names")
    if task not in ["classification", "regression"]:
        raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")
        
    profiles, profiler_entries = run_profiler(
        df=df, target=target, task=task, cardinality_threshold=cardinality_threshold
    )
    return PrepResult(
        df=df,
        pipeline=None,
        report=Report(profiler_entries)
    )

def clean(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    drop_threshold: float = 0.6,
    outlier_method: str = "iqr",
    cardinality_threshold: int = 20,
) -> PrepResult:
    if len(df) == 0:
        raise ValueError("DataFrame cannot be empty")
    if target not in df.columns:
        raise ValueError(f"target '{target}' not found in DataFrame columns: {list(df.columns)}")
    if len(df.columns) < 2:
        raise ValueError("DataFrame must contain at least one feature column besides the target")
    if df.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names")
    if task not in ["classification", "regression"]:
        raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")
        
    profiles, profiler_entries = run_profiler(
        df=df, target=target, task=task, cardinality_threshold=cardinality_threshold
    )
    df_clean, surviving_profiles, cleaner_entries, specs = run_cleaner(
        df=df, profiles=profiles, target=target, drop_threshold=drop_threshold, outlier_method=outlier_method
    )
    return PrepResult(
        df=df_clean,
        pipeline=None,
        report=Report(profiler_entries + cleaner_entries)
    )

def engineer(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    model_hint: str = "tree",
    drop_threshold: float = 0.6,
    outlier_method: str = "iqr",
    cardinality_threshold: int = 20,
) -> PrepResult:
    if len(df) == 0:
        raise ValueError("DataFrame cannot be empty")
    if target not in df.columns:
        raise ValueError(f"target '{target}' not found in DataFrame columns: {list(df.columns)}")
    if len(df.columns) < 2:
        raise ValueError("DataFrame must contain at least one feature column besides the target")
    if df.columns.duplicated().any():
        raise ValueError("DataFrame contains duplicate column names")
    if task not in ["classification", "regression"]:
        raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")
    if model_hint not in ["tree", "linear"]:
        raise ValueError(f"model_hint must be 'tree' or 'linear', got '{model_hint}'")
        
    profiles, profiler_entries = run_profiler(
        df=df, target=target, task=task, cardinality_threshold=cardinality_threshold
    )
    df_clean, surviving_profiles, cleaner_entries, specs_c = run_cleaner(
        df=df, profiles=profiles, target=target, drop_threshold=drop_threshold, outlier_method=outlier_method
    )
    df_eng, engineer_entries, specs_e = run_engineer(
        df=df_clean, profiles=surviving_profiles, target=target, model_hint=model_hint, cardinality_threshold=cardinality_threshold
    )
    return PrepResult(
        df=df_eng,
        pipeline=None,
        report=Report(profiler_entries + cleaner_entries + engineer_entries)
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
