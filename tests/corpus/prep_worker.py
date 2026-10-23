"""
Worker for test_pipeline_corpus.py: runs ONE corpus file through the whole pipeline
(plan -> dry_run under defaults -> apply under defaults) and prints one JSON object.
Run as a subprocess so the parent can enforce a hard timeout.

    python prep_worker.py <csv path>

Target choice (the corpus labels carry no target): the LAST column of the table as read,
usable when the table has >= 20 rows, the column has >= 2 distinct values, and (if it is
text) at most 50 distinct values. Task: classification when the column is text or has at
most 10 distinct values, else regression. Decided before looking at any result.
"""

import json
import sys
import tempfile
import time
import traceback
from dataclasses import fields
from pathlib import Path


def choose_target(path: Path):
    import pandas as pd
    from preflight.plan import ApplyContext, PlanContext, ReadingStage

    stage = ReadingStage()
    decs = stage.plan(PlanContext(source_path=path))
    raw = stage.apply(ApplyContext(source_path=path), decs)
    info = {"rows": int(len(raw)), "columns": int(raw.shape[1])}
    if raw.shape[1] < 2:
        return None, None, {**info, "reason": "fewer than 2 columns"}
    if len(raw) < 20:
        return None, None, {**info, "reason": "fewer than 20 rows"}
    col = raw.columns[-1]
    s = raw[col].dropna()
    n = int(s.nunique())
    if n < 2:
        return None, None, {**info, "reason": "last column has fewer than 2 distinct values"}
    numeric = pd.api.types.is_numeric_dtype(s)
    if not numeric and n > 50:
        return None, None, {**info, "reason": "last column is text with more than 50 distinct values"}
    task = "classification" if (not numeric or n <= 10) else "regression"
    return str(col), task, info


def comparable(d):
    skip = {"status", "timestamp", "history"}
    return {f.name: getattr(d, f.name) for f in fields(d) if f.name not in skip}


def main(path: str) -> dict:
    from preflight.plan import apply, dry_run, plan

    p = Path(path)
    res = {"file": p.name, "eligible": False, "ok": False}
    t0 = time.time()
    try:
        target, task, info = choose_target(p)
    except Exception as exc:
        res.update(stage="read", error=f"{type(exc).__name__}: {exc}"[:300])
        return res
    res.update(info)
    if target is None:
        return res
    res.update(eligible=True, target=target, task=task)
    try:
        with tempfile.TemporaryDirectory() as td:
            stage = "dry_run"
            dry, pdf = dry_run(p, target=target, task=task, out_dir=Path(td) / "dry", mode="defaults")
            stage = "plan"
            base = plan(p, target=target, task=task)
            res["needs_attention"] = sum(d.needs_attention for d in base.decisions)
            res["decisions"] = len(base.decisions)
            res["prep_decisions"] = sum(d.stage in ("clean", "engineer") for d in base.decisions)
            stage = "apply"
            run = apply(base, mode="defaults", out_dir=Path(td) / "run")
            res["prepared_shape"] = list(run.prepared.shape) if run.prepared is not None else None
            res["typed_rows"] = int(len(run.data))
            a, b = [comparable(d) for d in dry.decisions], [comparable(d) for d in run.decisions]
            res["plan_run_agree"] = a == b
            if a != b:
                res["disagree_detail"] = (
                    f"{len(a)} vs {len(b)} decisions; first difference at index "
                    f"{next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))}"
                )
        res["ok"] = True
    except Exception as exc:
        res.update(stage=stage, error=f"{type(exc).__name__}: {exc}"[:300], trace=traceback.format_exc()[-600:])
    res["seconds"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    print("@@RESULT@@" + json.dumps(main(sys.argv[1]), default=str))
