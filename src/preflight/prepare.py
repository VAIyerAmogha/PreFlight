"""
preflight.prepare(): the public real-run entry point.

Runs the existing plan -> resolve -> apply flow (reusing `plan()` and `apply()`
unchanged) and then, as the unconditional last step, generates the compulsory
Report PDF. There is no argument to turn the Report PDF off.

Fit hygiene (docs/fit_hygiene.md): the cleaned data written to out_dir has only
the steps that learn nothing from data. Every step that learns from data is in
the returned UNFITTED `pipeline`; fit it on training rows only (for example
inside each CV fold, after sklearn.base.clone). `pipeline_fitted_on_all_rows`
is the same pipeline fitted on every row, for transforming new data after model
selection is finished: it has seen every row, so it must NOT be used to evaluate
a model.

Failure handling: the cleaned data, schema.json, decisions.json, and both
pipelines are written to out_dir BEFORE the PDF is rendered (see
preflight.report.report_pdf.write_report_artifacts). If rendering fails, a
ReportGenerationError is raised naming all of those already-saved paths.
"""

from pathlib import Path
from typing import Any, Optional, Union

from sklearn.base import clone

from preflight.engineer.config import FeatureConfig
from preflight.plan import ApplyResult, default_out_dir
from preflight.plan import apply as _apply
from preflight.plan import plan as _plan
from preflight.report.report_pdf import write_report_artifacts


def _fit_on_all_rows(pipeline: Any, result: ApplyResult, target: str) -> Any:
    """
    Convenience only: a clone of the unfitted pipeline fitted on EVERY row.
    Never use it to score a model (it has seen the rows you would test on).
    """
    df = result.prepared
    if df is None or target not in df.columns:
        return None
    fitted = clone(pipeline)
    fitted.fit(df.drop(columns=[target]), df[target])
    return fitted


def prepare(
    path: Union[str, Path],
    target: str,
    task: Optional[str] = None,
    mode: str = "auto",
    channel: Any = None,
    out_dir: Optional[Union[str, Path]] = None,
    model_hint: str = "tree",
    feature_config: Optional[FeatureConfig] = None,
) -> dict[str, Any]:
    """
    Run the full real pipeline on `path` and write every real-run artifact to
    out_dir, ending unconditionally with the Report PDF.

    Args:
        path: source data file. Never modified.
        target: target column for the pipeline (required: the fitted
            pipeline and task inference both need it).
        task: 'classification' | 'regression' | None (infer).
        mode: 'auto' | 'ask' | 'defaults' | 'strict' -- how open questions are
            resolved before the real run proceeds (see preflight.resolve). 'auto'
            (default) asks in an interactive session (terminal or notebook) and is
            strict otherwise: a non-interactive run with open questions stops with an
            error that says how to proceed (mode='defaults', or answer the review file).
        channel: resolution channel, forwarded to preflight.resolve when mode='ask'.
        out_dir: destination directory for every artifact (plan.json, plan.pdf,
            cleaned_data.csv, schema.json, decisions.json, pipeline.joblib,
            report.pdf). Defaults to preflight_out/ in the current working
            directory; nothing is ever written next to the source file. The output
            folder is printed at the end of the run.
        model_hint: 'tree' or 'linear', shapes the fitted pipeline's engineering step.
        feature_config: optional engineering feature configuration.

    Returns:
        A dict with keys:
        'apply_result'; 'pipeline' (UNFITTED: every step that learns from data;
        fit it on training rows only, e.g. inside CV folds; its input is
        apply_result.prepared without the target); 'pipeline_fitted_on_all_rows'
        (the same pipeline fitted on every row, for new data once model choice is
        done; NOT for evaluation); 'plan'; and 'paths' (artifact paths from
        write_report_artifacts, plus 'plan_json' and 'plan_pdf').

    Raises:
        preflight.report.report_pdf.ReportGenerationError if the Report PDF
        fails to render; its message names the cleaned data, schema,
        decisions, and pipeline paths, which are saved before rendering is attempted.
    """
    p = Path(path).resolve()
    out_d = Path(out_dir).resolve() if out_dir is not None else default_out_dir()
    out_d.mkdir(parents=True, exist_ok=True)

    plan_obj = _plan(p, target=target, task=task, model_hint=model_hint, feature_config=feature_config)

    # apply() with mode= resolves every open question first (writing the
    # resolved plan.json / plan.pdf into out_dir), then runs the pipeline
    # stages with the final decisions in force.
    result = _apply(plan_obj, mode=mode, channel=channel, out_dir=out_d)

    pipeline = result.pipeline
    fitted_all = _fit_on_all_rows(pipeline, result, target)

    artifact_paths = write_report_artifacts(plan_obj, result, pipeline, out_d, fitted_pipeline=fitted_all)
    artifact_paths["plan_json"] = out_d / "plan.json"
    artifact_paths["plan_pdf"] = out_d / "plan.pdf"

    print(f"PreFlight: output written to {out_d}")
    return {
        "apply_result": result,
        "pipeline": pipeline,
        "pipeline_fitted_on_all_rows": fitted_all,
        "plan": plan_obj,
        "paths": artifact_paths,
    }


__all__ = ["prepare"]
