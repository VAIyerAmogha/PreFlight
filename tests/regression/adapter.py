"""
Internal, test-only adapter: runs v1.0.1-style test calls through the v2 CSV path.

v1 tests pass DataFrames to `prepare(df, ...)` etc. v2 deliberately has no public
DataFrame API (see docs/port_contract.md), so every call here:

  1. writes the DataFrame to a CSV in a temporary directory,
  2. runs it through v2: `Planner.plan` (read, infer types, plan cleaning and
     engineering), `apply(mode="defaults")` (safe defaults for open questions),
  3. wraps the outcome in the v1 result shapes the old assertions expect
     (PrepResult, Report, ColumnProfile, SemanticType).

Nothing here is exported from `preflight`. No v1 profiler logic lives here: column
kinds and numbers come from `preflight.signals` (schema-driven). The v1 `Report`,
`PrepResult` and `ColumnProfile` containers come from the compat modules in `src`.
`compare` and `save_compare_pdf` are v1 test helpers carried over unchanged.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from sklearn.base import clone

from preflight.assemble import build_pipeline_two_phase
from preflight.clean.cleaner import apply_clean
from preflight.clean.config import CleanerConfig
from preflight.decisions import Decision
from preflight.engineer.engineer import apply_engineer, plan_engineer
from preflight.interact.resolve import resolve
from preflight.plan import (
    ApplyContext,
    Planner,
    ReadingStage,
    TypingStage,
    _schema_with_actions,
    _working_frame,
    replan_prep,
)
from preflight.plan import (
    apply as v2_apply,
)
from preflight.report.report import Report
from preflight.signals import compute_signals
from preflight.types import (
    _UNSET,
    PRESETS,
    ColumnProfile,
    FeatureConfig,
    PrepResult,
    ReportEntry,
    SemanticType,
)
from preflight.validation import _validate_inputs

__all__ = [
    "prepare", "profile", "clean", "engineer", "compare",
    "add_features", "run_profiler", "save_compare_pdf",
]

_TYPE_FOR_KIND = {
    SemanticType.NUMERIC_FEATURE: "float",
    SemanticType.NUMERIC_ID: "ID",
    SemanticType.CATEGORICAL_LOW: "categorical",
    SemanticType.CATEGORICAL_HIGH: "categorical",
    SemanticType.BOOLEAN: "boolean",
    SemanticType.CONSTANT: "constant",
    SemanticType.TEXT: "text",
    SemanticType.DATETIME_NATIVE: "datetime",
    SemanticType.DATETIME_STRING: "datetime",
}


class _Run:
    """One DataFrame pushed through the v2 CSV path."""

    def __init__(self, df, target, task, model_hint, cleaner_config, feature_config, column_types):
        self.df, self.target, self.task, self.model_hint = df, target, task, model_hint
        self.cleaner_config, self.feature_config = cleaner_config, feature_config
        self.tmp = Path(tempfile.mkdtemp(prefix="pf_adapter_"))
        try:
            self.csv = self.tmp / "data.csv"
            df.to_csv(self.csv, index=False)
            self.plan = Planner().plan(
                self.csv, target=target, task=task, model_hint=model_hint,
                feature_config=feature_config, cleaner_config=cleaner_config,
            )
            if column_types:
                for c in self.plan.schema.columns:
                    if c.name in column_types:
                        c.type_override = _TYPE_FOR_KIND[column_types[c.name]]
                replan_prep(self.plan, cleaner_config)
            # defaults mode: every open question takes its safe default (v1 had no questions)
            resolve(self.plan, mode="defaults", out_dir=self.tmp / "out")
        except BaseException:
            self.close()
            raise

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def typed(self) -> pd.DataFrame:
        """Table as v2 reads and types it (working dtypes), nothing cleaned yet."""
        raw = ReadingStage().apply(ApplyContext(source_path=self.csv), self.plan.decisions)
        ctx = ApplyContext(
            source_path=self.csv, target_column=self.target, task_type=self.task,
            raw_data=raw, schema=self.plan.schema, decisions=self.plan.decisions,
        )
        return _working_frame(TypingStage().apply(ctx, self.plan.decisions))

    def profiles(self, typed: pd.DataFrame) -> list[ColumnProfile]:
        schema = _schema_with_actions(self.plan.schema, [d for d in self.plan.decisions if d.stage == "infer"])
        sigs = compute_signals(
            typed, schema, self.target, task=self.task,
            cardinality_threshold=self.cleaner_config.cardinality_threshold,
        )
        out = []
        for name, s in sigs.items():
            out.append(ColumnProfile(
                name=name, semantic_type=SemanticType[s.kind], missing_rate=s.missing_rate,
                dtype=typed[name].dtype, outlier_rate=s.outlier_rate, cardinality=s.cardinality,
                rare_categories=s.rare_categories, vif_score=s.vif_score,
                correlation_with_target=s.correlation_with_target,
                mutual_info_with_target=s.mutual_info_with_target,
                is_leakage_suspect=s.is_leakage_suspect,
                text_avg_length=s.text_avg_length, text_avg_word_count=s.text_avg_word_count,
            ))
        return out


def _entries(decisions: list[Decision], stages: tuple[str, ...]) -> list[ReportEntry]:
    out = []
    for d in decisions:
        stage = "profiler" if d.stage in ("ingest", "infer") else d.stage
        if stage not in stages:
            continue
        out.append(ReportEntry(
            stage=stage, column=(d.columns[0] if d.columns else "__all__"), action=d.action,
            rationale=d.plain_language or d.action,
            severity="warning" if d.needs_attention else "info",
        ))
    return out


class _CsvPipeline:
    """Fitted pipeline whose `transform` reads new rows through the same CSV path."""

    def __init__(self, pipeline, run: _Run):
        self._pipeline, self._run = pipeline, run
        self.named_steps = pipeline.named_steps

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        tmp = Path(tempfile.mkdtemp(prefix="pf_adapter_new_"))
        try:
            csv = tmp / "new.csv"
            X.to_csv(csv, index=False)
            raw = ReadingStage().apply(ApplyContext(source_path=csv), self._run.plan.decisions)
            schema = self._run.plan.schema
            sub = type(schema).from_dict(schema.to_dict())
            sub.columns = [c for c in sub.columns if c.name in raw.columns]
            ctx = ApplyContext(source_path=csv, raw_data=raw, schema=sub, decisions=self._run.plan.decisions)
            typed = _working_frame(TypingStage().apply(ctx, self._run.plan.decisions))
            return self._pipeline.transform(typed)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def __getattr__(self, name):
        return getattr(self._pipeline, name)


def _cleaner_config(drop_threshold, outlier_method, cardinality_threshold) -> CleanerConfig:
    return CleanerConfig(
        drop_missing_threshold=drop_threshold, outlier_method=outlier_method,
        cardinality_threshold=cardinality_threshold,
    )


def run_profiler(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    cardinality_threshold: int = 20,
    column_types: dict[str, SemanticType] | None = None,
) -> tuple[list[ColumnProfile], list[ReportEntry]]:
    run = _Run(df, target, task, "tree", _cleaner_config(0.6, "iqr", cardinality_threshold), None, column_types)
    try:
        return run.profiles(run.typed()), _entries(run.plan.decisions, ("profiler",))
    finally:
        run.close()


def _v1_full_table(res, target: str) -> pd.DataFrame:
    prepared = res.prepared
    if res.pipeline is None or target not in prepared.columns:
        return prepared.copy()
    X, y = prepared.drop(columns=[target]), prepared[target]
    out = clone(res.pipeline).fit_transform(X, y)
    out[target] = y.to_numpy()
    return out


def prepare(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    model_hint: str = "tree",
    drop_threshold: float = _UNSET,
    outlier_method: str = _UNSET,
    cardinality_threshold: int = _UNSET,
    feature_config: FeatureConfig | None = _UNSET,
    column_types: dict[str, SemanticType] | None = _UNSET,
    preset: str | None = None,
    dry_run: bool = False,
) -> PrepResult:
    if not isinstance(dry_run, bool):
        raise TypeError("The 'dry_run' argument must be True or False.")

    drop_t, outlier_m, card_t, fcfg, ctypes = 0.6, "iqr", 20, None, None
    if preset is not None:
        if preset not in PRESETS:
            valid = ", ".join(PRESETS.keys())
            raise ValueError(f"We don't recognize the preset name '{preset}'. Please choose one of these instead: {valid}.")
        p = PRESETS[preset]
        drop_t, outlier_m = p.get("drop_threshold", drop_t), p.get("outlier_method", outlier_m)
        card_t, fcfg = p.get("cardinality_threshold", card_t), p.get("feature_config", fcfg)
    if drop_threshold is not _UNSET:
        drop_t = drop_threshold
    if outlier_method is not _UNSET:
        outlier_m = outlier_method
    if cardinality_threshold is not _UNSET:
        card_t = cardinality_threshold
    if feature_config is not _UNSET:
        fcfg = feature_config
    if column_types is not _UNSET:
        ctypes = column_types

    warning = _validate_inputs(df, target, task, model_hint=model_hint, column_types=ctypes)
    run = _Run(df, target, task, model_hint, _cleaner_config(drop_t, outlier_m, card_t), fcfg, ctypes)
    try:
        typed = run.typed()
        profiles = run.profiles(typed)
        prefix: list[ReportEntry] = []
        if warning is not None:
            prefix.append(ReportEntry(stage="profiler", column=target, action="task_target_mismatch_warning",
                                      rationale=warning, severity="warning"))
        if preset is not None:
            prefix.append(ReportEntry(
                stage="profiler", column="__all__", action="apply_preset",
                rationale=f"Applied preset '{preset}'. Expanded parameters: "
                          f"{ {'drop_threshold': drop_t, 'outlier_method': outlier_m, 'cardinality_threshold': card_t} }",
                severity="info"))
        entries = prefix + _entries(run.plan.decisions, ("profiler", "clean", "engineer"))

        if dry_run:
            entries.insert(0, ReportEntry(
                stage="profiler", column="dataset", action="dry_run",
                rationale="dry run mode enabled: no data was transformed and no pipeline was fitted.",
                severity="info"))
            return PrepResult(df=df, pipeline=None, report=Report(entries, df=df, profiles=profiles, target=target))

        res = v2_apply(run.plan, mode="defaults", out_dir=run.tmp / "out2")
        # v1's prepare(df) returned the table with every step fitted on all rows. v2's
        # apply() leaves the data-learning steps to an unfitted pipeline (fit hygiene),
        # so the v1 shape is rebuilt here by fitting that pipeline on every row.
        final = _v1_full_table(res, target)
        X = typed.drop(columns=[target]) if target in typed.columns else typed
        y = typed[target] if target in typed.columns else None
        pipe = build_pipeline_two_phase(
            X, y, target, model_hint, res.decisions, run.plan.schema, fcfg,
            run.cleaner_config, card_t, task,
        )
        return PrepResult(
            df=final, pipeline=_CsvPipeline(pipe, run),
            report=Report(entries, df=df, profiles=profiles, target=target),
        )
    finally:
        # the pipeline keeps the plan (decisions, schema) it needs; only files are removed
        run.close()


def profile(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    model_hint: str = "tree",
    cardinality_threshold: int = 20,
    column_types: dict[str, SemanticType] | None = None,
) -> PrepResult:
    warning = _validate_inputs(df, target, task, model_hint=model_hint, column_types=column_types)
    profiles, entries = run_profiler(df, target, task, cardinality_threshold, column_types)
    if warning is not None:
        entries = [ReportEntry(stage="profiler", column=target, action="task_target_mismatch_warning",
                               rationale=warning, severity="warning")] + entries
    return PrepResult(df=df, pipeline=None, report=Report(entries, df=df, profiles=profiles, target=target))


def _run_steps(df, target, task, model_hint, drop_threshold, outlier_method, cardinality_threshold,
               column_types, with_engineer):
    warning = _validate_inputs(df, target, task, model_hint=model_hint, column_types=column_types)
    run = _Run(df, target, task, model_hint, _cleaner_config(drop_threshold, outlier_method, cardinality_threshold),
               None, column_types)
    try:
        typed = run.typed()
        profiles = run.profiles(typed)
        decs = run.plan.decisions
        cleaned = apply_clean(typed, [d for d in decs if d.stage == "clean"], run.plan.schema)
        stages = ("profiler", "clean")
        if with_engineer:
            cleaned, _ = apply_engineer(cleaned, [d for d in decs if d.stage == "engineer"], target=target)
            stages = stages + ("engineer",)
        entries = _entries(decs, stages)
        if warning is not None:
            entries = [ReportEntry(stage="profiler", column=target, action="task_target_mismatch_warning",
                                   rationale=warning, severity="warning")] + entries
        return PrepResult(df=cleaned, pipeline=None, report=Report(entries, df=cleaned, profiles=profiles, target=target))
    finally:
        run.close()


def clean(df, target, task="classification", model_hint="tree", drop_threshold=0.6,
          outlier_method="iqr", cardinality_threshold=20, column_types=None) -> PrepResult:
    return _run_steps(df, target, task, model_hint, drop_threshold, outlier_method,
                      cardinality_threshold, column_types, with_engineer=False)


def engineer(df, target, task="classification", model_hint="tree", drop_threshold=0.6,
             outlier_method="iqr", cardinality_threshold=20, column_types=None) -> PrepResult:
    return _run_steps(df, target, task, model_hint, drop_threshold, outlier_method,
                      cardinality_threshold, column_types, with_engineer=True)


def add_features(result: PrepResult, config: FeatureConfig, target: str | None = None) -> PrepResult:
    """Add generated columns (interactions, clusters, ...) to an already prepared table."""
    if result is None or result.df is None:
        raise ValueError("You must provide a preparation result that has a dataset.")
    task = "classification"
    decs = plan_engineer(result.df, None, target, "tree", 20, config, task)
    created = [d for d in decs if d.kind.startswith("create_")]
    df2, _ = apply_engineer(result.df, created, target=target)
    entries = list(result.report.entries) if result.report is not None else []
    entries += _entries(created, ("engineer",))
    return PrepResult(df=df2, pipeline=result.pipeline,
                      report=Report(entries, df=df2, profiles=getattr(result.report, "profiles", None), target=target))


# ---------------------------------------------------------------------------
# v1 test helpers carried over unchanged
# ---------------------------------------------------------------------------

def _extract_actions_per_column(report: Report | None) -> dict[str, set]:
    if report is None:
        return {}
    actions: dict[str, set] = {}
    for entry in report.entries:
        actions.setdefault(entry.column, set()).add(entry.action)
    return actions


def _compute_decision_diff(actions_a: dict[str, set], actions_b: dict[str, set]) -> list[str]:
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
