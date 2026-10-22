"""
Corpus run of the interaction layer (Phase 5 / M4).

Every file in corpus/raw/ goes through plan -> resolve(strict) and plan -> apply(defaults)
in its own subprocess with a hard per-file timeout. Recorded (and written to
.pytest_cache/interact_corpus_results.json):
  files that complete, files that hang (must be zero), questions per file
  (median, max), files above 10 questions, files where strict raises.
"""

import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "corpus" / "raw"
WORKER = Path(__file__).with_name("interact_worker.py")
PER_FILE_TIMEOUT = 180  # seconds
WORKERS = 4


def run_one(path: Path) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER), str(path)],
            capture_output=True, text=True, timeout=PER_FILE_TIMEOUT, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {"file": path.name, "ok": False, "hang": True}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return {"file": path.name, "ok": False, "stage": "no result", "error": (proc.stderr or "")[-500:]}


@pytest.mark.skipif(not RAW.is_dir(), reason="corpus/raw not present")
def test_interact_over_corpus():
    files = sorted(p for p in RAW.iterdir() if p.is_file())
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(run_one, files))

    hung = [r["file"] for r in results if r.get("hang")]
    done = [r for r in results if r.get("ok")]
    not_ok = [r for r in results if not r.get("ok") and not r.get("hang")]
    qcounts = [r["questions"] for r in done]
    summary = {
        "files": len(files),
        "completed_worker": len(done),
        "hung": hung,
        "worker_failures": [{"file": r["file"], "stage": r.get("stage"), "error": r.get("error")} for r in not_ok],
        "defaults_completed": sum(1 for r in done if r.get("defaults_completed")),
        "defaults_failed": [{"file": r["file"], "error": r.get("defaults_error")} for r in done if not r.get("defaults_completed")],
        "questions_median": statistics.median(qcounts) if qcounts else None,
        "questions_max": max(qcounts) if qcounts else None,
        "questions_total": sum(qcounts),
        "files_with_zero_questions": sum(1 for q in qcounts if q == 0),
        "files_above_10_questions": sorted((r["file"], r["questions"]) for r in done if r["questions"] > 10),
        "strict_raised": sum(1 for r in done if r.get("strict_raised")),
        "categories": {},
        "column_questions": sum(r["column_questions"] for r in done),
        "column_questions_without_values": sum(r["column_questions_without_values"] for r in done),
        "jargon_hits": sorted({h for r in done for h in r["jargon_hits"]}),
    }
    for r in done:
        for k, v in r["categories"].items():
            summary["categories"][k] = summary["categories"].get(k, 0) + v

    out = ROOT / ".pytest_cache" / "interact_corpus_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "files": results}, indent=1), encoding="utf-8")
    print("\nINTERACT CORPUS SUMMARY\n" + json.dumps(summary, indent=1))

    assert not hung, f"files hung: {hung}"
    assert not not_ok, f"worker failures: {summary['worker_failures']}"
    assert all(r["bounds_ok"] for r in done), "option bounds / default membership violated"
    assert not summary["jargon_hits"], summary["jargon_hits"]
    # strict raises exactly when there is something to ask
    assert all(r["strict_raised"] == (r["questions"] > 0) for r in done)
    assert all(r.get("strict_listed", 0) == r["questions"] for r in done if r["strict_raised"])
    # defaults mode resolves everything it touches
    assert all(r.get("still_open", 0) == 0 for r in done if r.get("defaults_completed"))
