#!/usr/bin/env python3
"""Summarize one corpus test session's per-file records.

The corpus suites (pytest -m corpus) save one JSON record per file per suite under
.pytest_cache/corpus_results/run_<time>/<suite>/. This script:

1. Checks the corpus-wide thresholds that the suites asserted before they were split per
   file. The numbers and comparisons are unchanged. A suite is checked only when it has a
   record for every expected file; a partial run (-k, a subset) prints "partial" instead.
2. Builds the targets table (unhandled exceptions, dialect, column type, wrong type with no
   flag, evidence+confidence, needs_attention, timeouts) from the accuracy records.
   By default only the tuning set is scored. Held-out and combined columns appear only with
   --final (Phase 7b rule: held-out is scored once, at the end).

Usage:
    python scripts/corpus_summary.py [RUN_DIR] [--final] [--out FILE]
RUN_DIR defaults to the newest run under .pytest_cache/corpus_results/.
Prints PASS/FAIL lines only; the full table goes to --out (default RUN_DIR/summary.md).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "corpus" / "labels"
RESULTS_ROOT = ROOT / ".pytest_cache" / "corpus_results"
EXPECTED_FILES = 254
EXPECTED_COLUMNS = 2850


def _load(run: Path, suite: str) -> list[dict[str, Any]]:
    d = run / suite
    if not d.is_dir():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))]


def _pct(n: float, d: float) -> float:
    return 100.0 * n / d if d else 0.0


class _Checks:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failed = False

    def check(self, suite: str, name: str, ok: bool, detail: str) -> None:
        self.lines.append(f"{'PASS' if ok else 'FAIL'} [{suite}] {name}: {detail}")
        self.failed = self.failed or not ok

    def note(self, suite: str, text: str) -> None:
        self.lines.append(f"INFO [{suite}] {text}")


def _complete(c: _Checks, suite: str, recs: list[dict[str, Any]], expected: int) -> bool:
    if not recs:
        return False
    if len(recs) < expected:
        c.note(suite, f"partial run: {len(recs)}/{expected} files, corpus-wide checks skipped")
        return False
    return True


def _timeouts(c: _Checks, suite: str, recs: list[dict[str, Any]]) -> None:
    t = sorted(r["file"] for r in recs if r.get("status") == "TIMEOUT")
    if recs:
        c.check(suite, "timeouts", not t, f"{len(t)} {t[:10]}")


def _ingest(c: _Checks, run: Path) -> None:
    recs = _load(run, "ingest")
    _timeouts(c, "ingest", recs)
    if not _complete(c, "ingest", recs, EXPECTED_FILES):
        return
    n = len(recs)
    exc = [r["file"] for r in recs if r.get("exception")]
    c.check("ingest", "unhandled exceptions", not exc, f"{len(exc)} {exc[:10]}")
    acc = {k: _pct(sum(bool(r.get(k)) for r in recs), n) for k in ("delim_ok", "enc_ok", "header_ok")}
    dia = _pct(sum(bool(r.get("delim_ok") and r.get("enc_ok") and r.get("header_ok")) for r in recs), n)
    c.check("ingest", "delimiter accuracy >= 99.0%", acc["delim_ok"] >= 99.0, f"{acc['delim_ok']:.2f}%")
    c.check("ingest", "encoding accuracy >= 98.0%", acc["enc_ok"] >= 98.0, f"{acc['enc_ok']:.2f}%")
    c.check("ingest", "header accuracy == 100.0%", acc["header_ok"] == 100.0, f"{acc['header_ok']:.2f}%")
    c.check("ingest", "dialect accuracy >= 98.0%", dia >= 98.0, f"{dia:.2f}%")


def _infer(c: _Checks, run: Path) -> None:
    recs = _load(run, "infer")
    _timeouts(c, "infer", recs)
    if not _complete(c, "infer", recs, EXPECTED_FILES):
        return
    exc = [r["file"] for r in recs if r.get("exception")]
    c.check("infer", "unhandled exceptions", not exc, f"{len(exc)} {exc[:10]}")
    if exc:
        return
    tot = sum(r["total"] for r in recs)
    c.check("infer", f"labelled columns == {EXPECTED_COLUMNS}", tot == EXPECTED_COLUMNS, str(tot))
    acc = _pct(sum(r["correct"] for r in recs), tot)
    wu = _pct(sum(r["wrong_unflagged"] for r in recs), tot)
    c.check("infer", "type accuracy >= 85.0%", acc >= 85.0, f"{acc:.2f}%")
    c.check("infer", "wrong type, no flag <= 2.0%", wu <= 2.0, f"{wu:.2f}%")
    c.note("infer", f"abstained {_pct(sum(r['abstained'] for r in recs), tot):.2f}%")


def _plan(c: _Checks, run: Path) -> None:
    recs = _load(run, "plan")
    _timeouts(c, "plan", recs)
    if not _complete(c, "plan", recs, EXPECTED_FILES):
        return
    exc = [r["file"] for r in recs if r.get("exception")]
    done = [r for r in recs if r.get("pages") is not None]
    c.check("plan", "unhandled exceptions", not exc, f"{len(exc)} {exc[:10]}")
    c.check("plan", f"completions == {EXPECTED_FILES}", len(done) == EXPECTED_FILES, str(len(done)))
    if done:
        att = [r["needs_attention"] for r in done]
        pages = [r["pages"] for r in done]
        c.note("plan", f"needs_attention median {statistics.median(att)}, max {max(att)}, "
                       f"files > 10: {sum(a > 10 for a in att)}; PDF pages median "
                       f"{statistics.median(pages)}, max {max(pages)}")


def _interact(c: _Checks, run: Path) -> None:
    recs = _load(run, "interact")
    _timeouts(c, "interact", recs)
    if not recs:
        return
    hung = [r["file"] for r in recs if r.get("hang")]
    bad = [r["file"] for r in recs if not r.get("ok") and not r.get("hang") and r.get("status") != "TIMEOUT"]
    c.check("interact", "hung files", not hung, f"{len(hung)} {hung[:10]}")
    c.check("interact", "worker failures", not bad, f"{len(bad)} {bad[:10]}")
    done = [r for r in recs if r.get("ok")]
    if done:
        q = [r["questions"] for r in done]
        c.note("interact", f"{len(done)}/{len(recs)} files; questions median {statistics.median(q)}, "
                           f"max {max(q)}, files > 10: {sum(x > 10 for x in q)}, "
                           f"strict raised: {sum(1 for r in done if r.get('strict_raised'))}")


def _pipeline(c: _Checks, run: Path) -> None:
    recs = _load(run, "pipeline")
    _timeouts(c, "pipeline", recs)
    if not recs:
        return
    eligible = [r for r in recs if r.get("eligible") or r.get("status") == "TIMEOUT"]
    failed = [r["file"] for r in eligible if not r.get("ok")]
    disagree = [r["file"] for r in eligible if r.get("ok") and not r.get("plan_run_agree")]
    if _complete(c, "pipeline", recs, EXPECTED_FILES):
        c.check("pipeline", "some file has a usable target", bool(eligible), str(len(eligible)))
    c.check("pipeline", "unhandled exceptions or hangs", not failed, f"{len(failed)} {failed[:10]}")
    c.check("pipeline", "plan and run agree", not disagree, f"{len(disagree)} {disagree[:10]}")
    c.note("pipeline", f"{len(eligible)} files with a usable target, "
                       f"{sum(1 for r in eligible if r.get('ok'))} completed")


# ---- targets table (from the accuracy records) ------------------------------------------


def _metrics(rs: list[dict[str, Any]], label_cols: int) -> dict[str, Any]:
    t = len(rs)
    ok = [r for r in rs if r.get("loaded_ok")]
    cols = [col for r in rs for col in r.get("column_results", [])]
    na = [r["needs_attention_count"] for r in ok]
    dtot = sum(r.get("decisions_total", 0) for r in rs)
    return {
        "timeouts": sum(r.get("status") == "TIMEOUT" for r in rs),
        "unhandled": (t - len(ok), t),
        "dialect": (sum(bool(r.get("dialect_correct")) for r in rs), t),
        "coltype": (sum(col["correct"] for col in cols), label_cols),
        "wrong_noflag": (sum(bool(col["wrong_no_flag"]) for col in cols), label_cols),
        "evidence": (dtot - sum(r.get("decisions_missing_evidence_or_conf", 0) for r in rs), dtot),
        "na_median": statistics.median(na) if na else None,
        "na_max": max(na) if na else None,
        "na_gt10": sum(x > 10 for x in na),
    }


def _frac(p: tuple[int, int]) -> str:
    return f"{p[0]}/{p[1]} = {100 * p[0] / p[1]:.2f}%" if p[1] else "n/a"


def _ratio_ok(p: tuple[int, int], test: Callable[[float], bool]) -> bool:
    return bool(p[1]) and test(p[0] / p[1])


TARGET_ROWS: list[tuple[str, str, Callable[[dict[str, Any]], bool], Callable[[dict[str, Any]], str]]] = [
    ("Files without unhandled exceptions", "100%",
     lambda m: m["unhandled"][0] == 0, lambda m: f"{m['unhandled'][1] - m['unhandled'][0]}/{m['unhandled'][1]}"),
    ("Dialect accuracy", "≥95%", lambda m: _ratio_ok(m["dialect"], lambda x: x >= 0.95), lambda m: _frac(m["dialect"])),
    ("Column type accuracy", "≥95%", lambda m: _ratio_ok(m["coltype"], lambda x: x >= 0.95), lambda m: _frac(m["coltype"])),
    ("Wrong type, no flag", "≤1%",
     lambda m: _ratio_ok(m["wrong_noflag"], lambda x: x <= 0.01), lambda m: _frac(m["wrong_noflag"])),
    ("Decisions with evidence+confidence", "100%",
     lambda m: m["evidence"][0] == m["evidence"][1], lambda m: _frac(m["evidence"])),
    ("needs_attention median", "≤10", lambda m: m["na_median"] is not None and m["na_median"] <= 10,
     lambda m: f"median {m['na_median']}, max {m['na_max']}, files>10: {m['na_gt10']}"),
    ("Timeouts", "0", lambda m: m["timeouts"] == 0, lambda m: str(m["timeouts"])),
]


def targets_table(run: Path, final: bool) -> tuple[list[str], dict[str, Any]]:
    recs = _load(run, "accuracy")
    if not recs:
        return [], {}
    split = json.loads((LABELS_DIR / "split.json").read_text(encoding="utf-8"))["assignments"]
    labels = {}
    for p in LABELS_DIR.glob("*.json"):
        if p.name != "split.json":
            lab = json.loads(p.read_text(encoding="utf-8"))
            labels[lab["filename"]] = lab
    # A TIMEOUT record has only "file" (the raw file name); accuracy records have "filename".
    for r in recs:
        lab = labels.get(r.get("filename") or r.get("file", ""))
        r["group"] = split.get(lab["file_id"]) if lab else None
        r["_label_cols"] = len(lab["columns"]) if lab else 0
    sets = {
        "tuning": [r for r in recs if r["group"] == "tuning"],
        "held_out": [r for r in recs if r["group"] == "held_out"],
        "combined": recs,
    }
    shown = ("tuning", "held_out", "combined") if final else ("tuning",)
    metrics = {k: _metrics(sets[k], sum(r["_label_cols"] for r in sets[k])) for k in shown}
    head = " | ".join({"tuning": "Tuning", "held_out": "Held-out", "combined": "Combined"}[k] for k in shown)
    lines = [
        f"| Target | Goal | {head} |",
        "|---|---|" + "---|" * len(shown),
    ]
    for name, goal, ok, fmt in TARGET_ROWS:
        cells = " | ".join(f"{'PASS' if ok(metrics[k]) else 'FAIL'} {fmt(metrics[k])}" for k in shown)
        lines.append(f"| {name} | {goal} | {cells} |")
    counts = {k: len(sets[k]) for k in shown}
    return lines, {"metrics": metrics, "files": counts}


def summarize(run: Path, final: bool = False) -> dict[str, Any]:
    c = _Checks()
    for fn in (_ingest, _infer, _plan, _interact, _pipeline):
        fn(c, run)
    acc = _load(run, "accuracy")
    if acc:
        _timeouts(c, "accuracy", acc)
        exc = [r["filename"] for r in acc if r.get("status") != "TIMEOUT" and not r.get("loaded_ok")]
        c.check("accuracy", "unhandled exceptions", not exc, f"{len(exc)} {exc[:10]}")
    table, data = targets_table(run, final)
    lines = [f"# Corpus summary for {run.name}", "", "## Checks", ""] + [f"- {x}" for x in c.lines]
    if table:
        scope = "tuning, held-out and combined (--final)" if final else "tuning set only (held-out is scored once, with --final)"
        lines += ["", f"## Targets ({scope})", "", *table]
    return {"lines": lines, "check_lines": c.lines, "failed": c.failed, "targets": data, "table": table}


def _latest_run() -> Path:
    runs = sorted(p for p in RESULTS_ROOT.glob("run_*") if p.is_dir())
    if not runs:
        sys.exit(f"no corpus runs under {RESULTS_ROOT}")
    return runs[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--final", action="store_true", help="also score held-out and combined")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    run = args.run_dir or _latest_run()
    s = summarize(run, final=args.final)
    out = args.out or run / ("summary_final.md" if args.final else "summary.md")
    out.write_text("\n".join(s["lines"]) + "\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(s["targets"], indent=1, default=str), encoding="utf-8")
    for line in s["check_lines"]:
        print(line)
    for row in s["table"][2:]:
        print(row)
    print(f"wrote {out}")
    return 1 if s["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
