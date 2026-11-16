"""Browser-side bridge between the demo page and the preflight package.

Runs inside Pyodide. app.js writes the uploaded CSV to the in-memory filesystem
(/data/upload.csv) and calls the two functions below. Both return JSON strings
so the JS side never has to touch Python objects directly.

Phase 1  analyze(path, target)          -> plan summary + open questions
Phase 2  run(path, target, answers_json) -> cleaned preview + decisions + file paths

STATUS: scaffold. The call shapes match the current public API (preflight.plan,
preflight.prepare, the callback channel), but this has not been run under Pyodide
yet. See site/DEMO_DEV.md.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

import preflight
from preflight.interact.question import Question, create_review_session

OUT_DIR = Path("/data/preflight_out")
PREVIEW_ROWS = 50


def _decision_summary(d: Any) -> dict[str, Any]:
    return {
        "id": d.id,
        "stage": d.stage,
        "kind": d.kind,
        "columns": list(d.columns),
        "action": d.action,
        "confidence": round(float(d.confidence), 3),
        "needs_attention": bool(d.needs_attention),
        "status": d.status,
        "plain_language": d.plain_language,
    }


def _question_summary(q: Question) -> dict[str, Any]:
    return {
        "question_id": q.question_id,
        "column": q.column,
        "text": q.text,
        "explanation": q.explanation,
        "evidence_summary": q.evidence_summary,
        "options": list(q.options),
        "recommended_default": q.recommended_default,
        "preview": q.preview,
    }


def _frame_preview(df: pd.DataFrame) -> dict[str, Any]:
    head = df.head(PREVIEW_ROWS)
    return {
        "columns": [str(c) for c in head.columns],
        "rows": json.loads(head.to_json(orient="values", date_format="iso")),
        "shape": list(df.shape),
    }


def columns(path: str) -> str:
    """Column names exactly as preflight will read them (headerless files get col_0, col_1, ...).

    The page uses this to fill the target dropdown, so the names always match what
    plan()/prepare() expect.
    """
    plan_obj = preflight.plan(path)
    names = [c.name for c in plan_obj.schema.columns] if plan_obj.schema else []
    return json.dumps(names)


def analyze(path: str, target: str) -> str:
    """Plan only. Nothing is written; returns decisions and the open questions."""
    plan_obj = preflight.plan(path, target=target or None)
    try:
        raw = pd.read_csv(path, nrows=PREVIEW_ROWS, dtype=str, keep_default_na=False)
        raw_preview = _frame_preview(raw)
    except Exception:  # The raw preview is cosmetic; the plan already read the file properly.
        raw_preview = None
    session = create_review_session(plan_obj.decisions, path, schema=plan_obj.schema)
    return json.dumps(
        {
            "rows": plan_obj.row_count,
            "columns": plan_obj.column_count,
            "task": plan_obj.task_type,
            "decisions": [_decision_summary(d) for d in plan_obj.decisions],
            "questions": [_question_summary(q) for q in session.questions],
            "raw_preview": raw_preview,
        },
        default=str,
    )


def run(path: str, target: str, answers_json: str) -> str:
    """Full prepare() with the page's answers fed through the callback channel."""
    answers: dict[str, str] = json.loads(answers_json or "{}")

    def handler(questions: list[Question]) -> dict[str, str]:
        # Anything the page did not answer falls back to the recommended default.
        return {q.question_id: answers.get(q.question_id, q.recommended_default) for q in questions}

    out = preflight.prepare(path, target=target, mode="ask", channel=handler, out_dir=OUT_DIR)
    result = out["apply_result"]
    cleaned = result.prepared if result.prepared is not None else result.data
    return json.dumps(
        {
            "cleaned_preview": _frame_preview(cleaned),
            "decisions": [_decision_summary(d) for d in result.decisions],
            "files": {k: str(v) for k, v in out["paths"].items()},
        },
        default=str,
    )
