"""Bridge between the demo site and the preflight package.

This one file runs in two places:
  * in the browser, inside Pyodide (site/js/worker.js imports it);
  * under CPython, from site/scripts/build_fixtures.py, which pre-computes the
    JSON behind the static figures in the docs.

Rule (dev-docs/demo-site-plan.md): every value the page shows comes from the
package. This module only *calls* preflight and serialises what it returns. The
one piece of comparison logic here is `_cell_changes`, which lines up the raw
table and the prepared table by row index so the page can highlight cells that
changed. It decides nothing about the data. Everything crosses into JS as a JSON
string.
"""

import json
import math
from pathlib import Path
from typing import Any, Optional

import pandas as pd

import preflight
from preflight.interact.question import Question, create_review_session
from preflight.interact.resolve import _load_raw  # raw all-text table, as preflight read it

OUT_DIR = Path("/data/preflight_out")
PREVIEW_ROWS = 100
RAW_LINES = 14


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _cell(v: Any) -> Any:
    """One cell -> JSON value. Missing becomes None, so the page can style it."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):  # numpy scalar
        v = v.item()
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (bool, int, str)):
        return v
    return str(v)


def _frame(df: Optional[pd.DataFrame], rows: int = PREVIEW_ROWS) -> Optional[dict[str, Any]]:
    if df is None:
        return None
    head = df.head(rows)
    return {
        "columns": [str(c) for c in head.columns],
        "dtypes": [str(t) for t in head.dtypes],
        "index": [_cell(i) for i in head.index],
        "rows": [[_cell(v) for v in row] for row in head.itertuples(index=False, name=None)],
        "shape": [int(df.shape[0]), int(df.shape[1])],
    }


def _decision(d: Any) -> dict[str, Any]:
    return {
        "id": d.id,
        "stage": d.stage,
        "kind": d.kind,
        "columns": list(d.columns),
        "action": d.action,
        "default": d.default,
        "confidence": round(float(d.confidence), 3),
        "needs_attention": bool(d.needs_attention),
        "status": d.status,
        "plain_language": d.plain_language,
        "evidence": d.evidence,
        "history": d.history,
    }


def _question(q: Question) -> dict[str, Any]:
    return {
        "question_id": q.question_id,
        "decision_id": q.decision_id,
        "category": q.category,
        "column": q.column,
        "text": q.text,
        "explanation": q.explanation,
        "evidence_summary": q.evidence_summary,
        "options": list(q.options),
        "option_actions": dict(q.option_actions),
        "recommended_default": q.recommended_default,
        "preview": q.preview,
    }


def _column(c: Any) -> dict[str, Any]:
    ev = c.evidence or {}
    return {
        "name": c.effective_name(),
        "position": c.position,
        "type": c.effective_type(),
        "confidence": round(float(c.confidence), 3),
        "runner_up": c.runner_up,
        "abstained": bool(c.abstained),
        "missing_markers": c.missing_markers,
        "anomalies": list(c.anomalies),
        "candidate_scores": ev.get("candidate_scores"),
        "margin": ev.get("margin"),
        "non_missing": ev.get("non_missing_count"),
        "total_rows": ev.get("total_rows"),
    }


def _reading(plan_obj: Any) -> dict[str, Any]:
    s = plan_obj.schema
    return {
        "encoding": {"name": s.encoding.effective_encoding(), "confidence": s.encoding.confidence,
                     "has_bom": s.encoding.has_bom},
        "delimiter": {"char": s.dialect.effective_delimiter(), "confidence": s.dialect.confidence,
                      "margin": s.dialect.margin},
        "header": {"row": s.header.effective_header_row(), "confidence": s.header.confidence,
                   "title_rows": list(s.header.title_rows), "blank_rows": list(s.header.blank_rows)},
    }


def _raw_lines(path: str, encoding: str) -> list[str]:
    """First lines of the file as text, for the 'what the file looks like' panel."""
    with open(path, "rb") as fh:
        blob = fh.read(64 * 1024)
    text = blob.decode(encoding or "utf-8", errors="replace")
    return text.splitlines()[:RAW_LINES]


def _naive_pandas(path: str) -> dict[str, Any]:
    """What plain pandas.read_csv(path) gives, for the comparison figure."""
    try:
        return {"ok": True, "frame": _frame(pd.read_csv(path), rows=8)}
    except Exception as exc:  # The point of the figure is to show this failing.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _cell_changes(raw: Optional[pd.DataFrame], prepared: pd.DataFrame) -> dict[str, Any]:
    """Line up raw and prepared by row index and report what differs.

    'changed' = the value's text form differs; 'blanked' = it became missing.
    A value cast from "39.0" to 39.0 is not counted as changed.
    """
    if raw is None:
        return {"available": False, "reason": "The raw table could not be re-read."}
    raw_cols = [str(c) for c in raw.columns]
    prep_cols = [str(c) for c in prepared.columns]
    # If rows were removed AND the index was renumbered 0..n-1, labels no longer
    # point at the same rows (preflight's missing-target step resets the index).
    # Lining up by label would highlight the wrong cells, so report counts only.
    renumbered = len(prepared) < len(raw) and prepared.index.equals(pd.RangeIndex(len(prepared)))
    if renumbered:
        return {
            "available": False,
            "reason": (
                f"{len(raw) - len(prepared)} row(s) were removed and the remaining rows were renumbered, "
                "so cells cannot be matched to their original rows. Cell highlighting is off for this file."
            ),
            "dropped_columns": [c for c in raw_cols if c not in prep_cols],
            "added_columns": [c for c in prep_cols if c not in raw_cols],
            "dropped_rows": [],
            "dropped_rows_total": len(raw) - len(prepared),
            "changed": {},
            "blanked": {},
            "raw_rows": int(raw.shape[0]),
        }
    kept_rows = set(prepared.index.tolist())
    dropped_rows = [_cell(i) for i in raw.index if i not in kept_rows]
    changed: dict[str, list[Any]] = {}
    blanked: dict[str, list[Any]] = {}
    shown = prepared.head(PREVIEW_ROWS)
    for col in prep_cols:
        if col not in raw_cols:
            continue
        for idx, after in shown[col].items():
            if idx not in raw.index:
                continue
            before = raw.at[idx, col]
            a = _cell(after)
            b = _cell(before)
            if a is None and b not in (None, ""):
                blanked.setdefault(col, []).append(_cell(idx))
            elif a is not None and str(b).strip() != _plain(a):
                changed.setdefault(col, []).append(_cell(idx))
    return {
        "available": True,
        "dropped_columns": [c for c in raw_cols if c not in prep_cols],
        "added_columns": [c for c in prep_cols if c not in raw_cols],
        "dropped_rows": dropped_rows[:500],
        "dropped_rows_total": len(dropped_rows),
        "changed": changed,
        "blanked": blanked,
        "raw_rows": int(raw.shape[0]),
    }


def _plain(v: Any) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _pipeline_steps(pipe: Any) -> list[dict[str, Any]]:
    if pipe is None or not hasattr(pipe, "steps"):
        return []
    return [{"name": n, "class": type(s).__name__, "module": type(s).__module__} for n, s in pipe.steps]


def _code(path: str, target: str, task: Optional[str], model_hint: str, answers: dict[str, str]) -> str:
    """The Python that the page actually ran, shown in the 'Show code' panel."""
    name = Path(path).name
    task_arg = f", task={task!r}" if task else ""
    hint_arg = f", model_hint={model_hint!r}" if model_hint != "tree" else ""
    lines = [
        "import preflight",
        "",
        f"plan = preflight.plan({name!r}, target={target!r}{task_arg}{hint_arg})",
        "",
    ]
    if answers:
        lines += ["answers = {"] + [f"    {k!r}: {v!r}," for k, v in answers.items()] + ["}", ""]
        lines += [
            "def from_page(questions):",
            "    return {q.question_id: answers.get(q.question_id, q.recommended_default)",
            "            for q in questions}",
            "",
            f"out = preflight.prepare({name!r}, target={target!r}{task_arg}{hint_arg},",
            "                        mode=\"ask\", channel=from_page)",
        ]
    else:
        lines.append(f"out = preflight.prepare({name!r}, target={target!r}{task_arg}{hint_arg}, mode=\"defaults\")")
    lines += [
        "",
        "prepared = out[\"apply_result\"].prepared   # cleaned table",
        "pipeline = out[\"pipeline\"]                 # unfitted: fit it inside each CV fold",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry points (JSON string in, JSON string out)
# ---------------------------------------------------------------------------

def version() -> str:
    return json.dumps({"preflight": preflight.__version__, "pandas": pd.__version__})


def inspect(path: str) -> str:
    """Column names exactly as preflight will read them (headerless files get col_0 ...)."""
    plan_obj = preflight.plan(path)
    names = [c.effective_name() for c in plan_obj.schema.columns] if plan_obj.schema else []
    return json.dumps({"columns": names, "rows": plan_obj.row_count})


def analyze(path: str, target: str, task: str = "", model_hint: str = "tree") -> str:
    """Stages 1-4: read, type, plan, questions. Writes nothing."""
    plan_obj = preflight.plan(path, target=target or None, task=task or None, model_hint=model_hint)
    raw = _load_raw(plan_obj)
    session = create_review_session(plan_obj.decisions, path, raw_df=raw, schema=plan_obj.schema)
    reading = _reading(plan_obj)
    return json.dumps(
        {
            "file": Path(path).name,
            "rows": plan_obj.row_count,
            "columns": plan_obj.column_count,
            "task": plan_obj.task_type,
            "reading": reading,
            "raw_lines": _raw_lines(path, reading["encoding"]["name"]),
            "naive_pandas": _naive_pandas(path),
            "raw_table": _frame(raw),
            "schema": [_column(c) for c in plan_obj.schema.columns],
            "decisions": [_decision(d) for d in plan_obj.decisions],
            "questions": [_question(q) for q in session.questions],
        },
        default=str,
    )


def run(path: str, target: str, answers_json: str = "{}", task: str = "", model_hint: str = "tree",
        out_dir: str = "") -> str:
    """Stage 5+: full prepare() with the page's answers fed through the callback channel."""
    answers: dict[str, str] = json.loads(answers_json or "{}")

    def from_page(questions: list[Question]) -> dict[str, str]:
        return {q.question_id: answers.get(q.question_id, q.recommended_default) for q in questions}

    out = preflight.prepare(
        path, target=target, task=task or None, model_hint=model_hint,
        mode="ask", channel=from_page, out_dir=Path(out_dir) if out_dir else OUT_DIR,
    )
    result = out["apply_result"]
    prepared = result.prepared if result.prepared is not None else result.data
    raw = _load_raw(out["plan"])

    model_ready = None
    fitted = out.get("pipeline_fitted_on_all_rows")
    if fitted is not None and target in prepared.columns:
        try:
            transformed = fitted.transform(prepared.drop(columns=[target]))
            if not isinstance(transformed, pd.DataFrame):
                transformed = pd.DataFrame(transformed)
            model_ready = _frame(transformed)
        except Exception as exc:  # Shown on the page; the run itself already succeeded.
            model_ready = {"error": f"{type(exc).__name__}: {exc}"}

    return json.dumps(
        {
            "task": out["plan"].task_type,
            "raw_table": _frame(raw),
            "prepared": _frame(prepared),
            "model_ready": model_ready,
            "changes": _cell_changes(raw, prepared),
            "pipeline": _pipeline_steps(out["pipeline"]),
            "decisions": [_decision(d) for d in result.decisions],
            "anomalies": result.anomalies,
            "files": {k: str(v) for k, v in out["paths"].items()},
            "code": _code(path, target, task or None, model_hint, answers),
        },
        default=str,
    )
