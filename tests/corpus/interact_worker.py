"""
Worker for test_interact_corpus.py: runs ONE corpus file through the interaction layer
and prints one JSON object. Run as a subprocess so the parent can enforce a hard timeout.

    python interact_worker.py <csv path>
"""

import json
import sys
import tempfile
import time
import traceback
from copy import deepcopy
from pathlib import Path


def main(path: str) -> dict:
    from preflight import apply, plan, resolve
    from preflight.interact.modes import StrictResolutionError
    from preflight.interact.question import MAX_OPTIONS, MIN_OPTIONS, create_review_session
    from preflight.interact.resolve import _load_raw

    res = {"file": Path(path).name, "ok": False}
    t0 = time.time()
    try:
        base = plan(path)
    except Exception as exc:  # planning is Phase 4; recorded, not hidden
        res.update(stage="plan", error=f"{type(exc).__name__}: {exc}"[:300])
        return res
    res["plan_seconds"] = round(time.time() - t0, 2)

    open_decs = [d for d in base.decisions if d.needs_attention]
    session = create_review_session(open_decs, path, raw_df=_load_raw(base), schema=base.schema)
    qs = session.questions
    res["questions"] = len(qs)
    res["categories"] = {}
    for q in qs:
        res["categories"][q.category] = res["categories"].get(q.category, 0) + 1
    res["bounds_ok"] = all(MIN_OPTIONS <= len(q.options) <= MAX_OPTIONS and q.recommended_default in q.options for q in qs)
    res["column_questions_without_values"] = sum(1 for q in qs if q.column and not q.preview["before"])
    res["column_questions"] = sum(1 for q in qs if q.column)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
    import re
    from test_jargon_lint import STRICT_BANNED_PATTERNS

    texts = []
    for q in qs:
        texts += [q.text, q.explanation, q.evidence_summary, q.recommended_default, *q.options]
        if q.preview:
            texts += [q.preview.get("note", ""), *q.preview["after"].keys()]
    res["jargon_hits"] = sorted({m.group(0) for t in texts for pat in STRICT_BANNED_PATTERNS for m in [re.search(pat, t.lower())] if m})

    with tempfile.TemporaryDirectory() as td:
        # strict
        try:
            resolve(deepcopy(base), mode="strict", out_dir=Path(td) / "strict")
            res["strict_raised"] = False
        except StrictResolutionError as exc:
            res["strict_raised"] = True
            res["strict_listed"] = len(exc.unresolved_questions)
        # defaults, then the real run
        try:
            run = apply(deepcopy(base), mode="defaults", out_dir=Path(td) / "defaults")
            res["defaults_completed"] = True
            res["rows"] = int(len(run.data))
            res["still_open"] = sum(d.needs_attention for d in run.decisions)
        except Exception as exc:
            res["defaults_completed"] = False
            res["defaults_error"] = f"{type(exc).__name__}: {exc}"[:300]
    res["ok"] = True
    res["seconds"] = round(time.time() - t0, 2)
    return res


if __name__ == "__main__":
    try:
        out = main(sys.argv[1])
    except Exception:
        out = {"file": Path(sys.argv[1]).name, "ok": False, "stage": "worker", "error": traceback.format_exc()[-600:]}
    print("RESULT " + json.dumps(out))
