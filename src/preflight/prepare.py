"""
preflight.prepare(): the public real-run entry point.

Runs the existing plan -> resolve -> apply flow (reusing `plan()` and `apply()`
unchanged) and then, as the unconditional last step, builds a fitted
scikit-learn pipeline and generates the compulsory Report PDF. There is no
argument to turn the Report PDF off.

Failure handling: the cleaned data, schema.json, decisions.json, and the
fitted pipeline are written to out_dir BEFORE the PDF is rendered (see
preflight.report.report_pdf.write_report_artifacts). If rendering fails, a
ReportGenerationError is raised naming all of those already-saved paths.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from preflight.assemble import build_pipeline_two_phase
from preflight.engineer.config import FeatureConfig
from preflight.plan import ApplyResult, Plan, apply as _apply, plan as _plan
from preflight.report.report_pdf import write_report_artifacts


def _resolved_task(plan_obj: Plan) -> str:
    if plan_obj.task_type and plan_obj.task_type.lower() in ("classification", "regression"):
        return plan_obj.task_type.lower()
    for d in plan_obj.decisions:
        if d.kind in ("task_inference", "ambiguous_task") and d.action in ("classification", "regression"):
            return d.action
    return "classification"


def _fit_pipeline(plan_obj: Plan, result: ApplyResult, model_hint: str, feature_config: Optional[FeatureConfig]):
    """Fit the real sklearn Pipeline from the Decisions actually in force for this run."""
    target = plan_obj.target_column
    df = result.data
    if target is None or df is None or target not in df.columns:
        # No target column to fit against: an empty, unfitted cleaner/engineer
        # pair would be misleading, so build the pipeline on the full table
        # with no target (cleaning/engineering Decisions still apply).
        X, y = df, None
    else:
        X = df.drop(columns=[target])
        y = df[target]

    return build_pipeline_two_phase(
        X,
        y,
        target,
        model_hint=model_hint,
        decisions=result.decisions,
        schema=plan_obj.schema,
        feature_config=feature_config,
        task=_resolved_task(plan_obj),
    )


def prepare(
    path: Union[str, Path],
    target: str,
    task: Optional[str] = None,
    mode: str = "defaults",
    channel: Any = None,
    out_dir: Optional[Union[str, Path]] = None,
    model_hint: str = "tree",
    feature_config: Optional[FeatureConfig] = None,
) -> Dict[str, Any]:
    """
    Run the full real pipeline on `path` and write every real-run artifact to
    out_dir, ending unconditionally with the Report PDF.

    Args:
        path: source data file. Never modified.
        target: target column for the pipeline (required: the fitted
            pipeline and task inference both need it).
        task: 'classification' | 'regression' | None (infer).
        mode: 'ask' | 'defaults' | 'strict' -- how open questions are resolved
            before the real run proceeds (see preflight.resolve). Defaults to
            'defaults' (safe defaults applied automatically) since this is the
            non-interactive, real-run entry point.
        channel: resolution channel, forwarded to preflight.resolve when mode='ask'.
        out_dir: destination directory for every artifact (plan.json, plan.pdf,
            cleaned_data.csv, schema.json, decisions.json, pipeline.joblib,
            report.pdf). Defaults to <source file's folder>/preflight_out.
        model_hint: 'tree' or 'linear', shapes the fitted pipeline's engineering step.
        feature_config: optional engineering feature configuration.

    Returns:
        A dict with keys 'apply_result', 'pipeline', and 'paths' (the dict of
        artifact paths returned by write_report_artifacts, plus 'plan_json'
        and 'plan_pdf').

    Raises:
        preflight.report.report_pdf.ReportGenerationError if the Report PDF
        fails to render; its message names the cleaned data, schema,
        decisions, and pipeline paths, which are saved before rendering is attempted.
    """
    p = Path(path).resolve()
    out_d = Path(out_dir).resolve() if out_dir is not None else p.parent / "preflight_out"
    out_d.mkdir(parents=True, exist_ok=True)

    plan_obj = _plan(p, target=target, task=task, model_hint=model_hint, feature_config=feature_config)

    # apply() with mode= resolves every open question first (writing the
    # resolved plan.json / plan.pdf into out_dir), then runs the pipeline
    # stages with the final decisions in force.
    result = _apply(plan_obj, mode=mode, channel=channel, out_dir=out_d)

    pipeline = _fit_pipeline(plan_obj, result, model_hint, feature_config)

    artifact_paths = write_report_artifacts(plan_obj, result, pipeline, out_d)
    artifact_paths["plan_json"] = out_d / "plan.json"
    artifact_paths["plan_pdf"] = out_d / "plan.pdf"

    return {
        "apply_result": result,
        "pipeline": pipeline,
        "plan": plan_obj,
        "paths": artifact_paths,
    }


__all__ = ["prepare"]
