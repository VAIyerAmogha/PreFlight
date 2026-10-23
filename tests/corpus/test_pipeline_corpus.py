"""
Corpus pipeline validation for PreFlight v2.0 Phase 6 (M5): cleaning,
engineering, and assembly wired end-to-end through the Plan/Apply registry.

Where tests/corpus/test_plan_corpus.py exercises dry_run() structurally
across all 254 messy-format corpus files (Phase 4 / M3 scope: ingest + typing
only, no target column), this suite picks 10 real datasets from corpus/raw/
that have an unambiguous target column, and verifies:

1. dry_run() succeeds and produces a Plan PDF with all 5 content sections
   populated (no "Nothing to do at this stage" placeholder across every
   engineering subsection — i.e. clean/engineer decisions are actually
   reaching the Plan PDF, not just the registry).
2. apply(plan, mode="defaults") succeeds end-to-end and returns non-empty,
   fully-typed data with no unhandled exception.
"""

import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile

import pandas as pd
import pypdf
import pytest

from preflight.plan import apply, dry_run

CORPUS_ROOT = Path(__file__).resolve().parent.parent.parent / "corpus"
RAW_DIR = CORPUS_ROOT / "raw"

# (filename, target_column, task) — hand-picked for an unambiguous, modelable
# target column (see docs/v2_port_report.md for the selection rationale).
DATASETS = [
    ("AER_CreditCard.csv", "card", "classification"),
    ("AER_HousePrices.csv", "price", "regression"),
    ("AER_CPS1985.csv", "wage", "regression"),
    ("AER_Affairs.csv", "affairs", "regression"),
    ("AER_Guns.csv", "murder", "regression"),
    ("AER_Mortgage.csv", "rate", "regression"),
    ("AER_BankWages.csv", "job", "classification"),
    ("AER_CollegeDistance.csv", "score", "regression"),
    ("AER_MurderRates.csv", "rate", "regression"),
    ("AER_NMES1988.csv", "visits", "regression"),
]

PLACEHOLDER = "Nothing to do at this stage."


@pytest.mark.parametrize("filename,target,task", DATASETS)
class TestPipelineCorpusValidation:

    def test_dry_run_produces_populated_plan_pdf(self, filename, target, task):
        raw_path = RAW_DIR / filename
        assert raw_path.exists(), f"Missing corpus fixture: {raw_path}"

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            plan_obj, pdf_path = dry_run(raw_path, target=target, task=task, out_dir=out_dir)

            assert pdf_path.exists()
            assert pdf_path.stat().st_size > 0
            assert plan_obj.target_column == target
            assert len(plan_obj.decisions) > 0

            reader = pypdf.PdfReader(str(pdf_path))
            assert len(reader.pages) >= 1
            full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

            # The 5 content sections must all render (not crash / not be absent).
            for heading in (
                "Needs your decision",
                "What we will do",
                "What we won't touch",
                "Sign-off checklist",
            ):
                assert heading in full_text, f"{filename}: missing Plan PDF section '{heading}'"

            # "What we will do" has 5 subgroups, in this fixed order:
            # Reading / Fixing / Filling gaps / Removing / Creating, followed
            # by the "What we won't touch" section. Slice the text strictly
            # between each heading and the next one that actually follows it,
            # so a placeholder belonging to the *next* subsection never bleeds
            # into this one's check.
            subsection_order = ["Reading", "Fixing", "Filling gaps", "Removing", "Creating", "What we won't touch"]
            positions = [(h, full_text.index(h)) for h in subsection_order if h in full_text]
            positions.sort(key=lambda hp: hp[1])

            engineering_subsections = {"Fixing", "Filling gaps", "Removing", "Creating"}
            placeholder_count = 0
            checked = 0
            for i, (heading, pos) in enumerate(positions):
                if heading not in engineering_subsections:
                    continue
                end = positions[i + 1][1] if i + 1 < len(positions) else len(full_text)
                section_body = full_text[pos:end]
                checked += 1
                if PLACEHOLDER in section_body:
                    placeholder_count += 1

            assert placeholder_count < checked, (
                f"{filename}: every clean/engineer PDF subsection is an empty placeholder"
            )

    def test_apply_defaults_executes_full_pipeline(self, filename, target, task):
        raw_path = RAW_DIR / filename
        plan_obj = None
        with tempfile.TemporaryDirectory() as td:
            plan_obj, _pdf_path = dry_run(raw_path, target=target, task=task, out_dir=Path(td))

        result = apply(plan_obj, mode="defaults")

        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) > 0
        assert len(result.data.columns) > 0
        assert isinstance(result.anomalies, list)



# ---------------------------------------------------------------------------
# Every corpus file with a usable target (see prep_worker.py for the rule)
# ---------------------------------------------------------------------------

ROOT = CORPUS_ROOT.parent
WORKER = Path(__file__).with_name("prep_worker.py")
PER_FILE_TIMEOUT = 1500  # seconds; the largest file needs three type-inference passes
WORKERS = 4


def _run_worker(path: Path) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER), str(path)],
            capture_output=True, text=True, timeout=PER_FILE_TIMEOUT, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {"file": path.name, "eligible": True, "ok": False, "hang": True}
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {"file": path.name, "eligible": True, "ok": False, "stage": "no result", "error": (proc.stderr or "")[-500:]}


@pytest.mark.skipif(not RAW_DIR.is_dir(), reason="corpus/raw not present")
def test_full_pipeline_over_every_corpus_file_with_a_target():
    files = sorted(p for p in RAW_DIR.iterdir() if p.is_file())
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(_run_worker, files))

    eligible = [r for r in results if r.get("eligible")]
    done = [r for r in eligible if r.get("ok")]
    failed = [r for r in eligible if not r.get("ok")]
    disagree = [r["file"] for r in done if not r["plan_run_agree"]]
    attn = [r["needs_attention"] for r in done]
    summary = {
        "files_in_corpus": len(files),
        "files_with_usable_target": len(eligible),
        "files_not_usable": len(files) - len(eligible),
        "completed": len(done),
        "unhandled_exceptions": [{"file": r["file"], "stage": r.get("stage"), "error": r.get("error")}
                                 for r in failed if not r.get("hang")],
        "hung": [r["file"] for r in failed if r.get("hang")],
        "plan_and_run_disagree": disagree,
        "needs_attention_median": statistics.median(attn) if attn else None,
        "needs_attention_max": max(attn) if attn else None,
        "files_above_10_needs_attention": sorted((r["file"], r["needs_attention"]) for r in done if r["needs_attention"] > 10),
        "files_with_cleaning_or_engineering_decisions": sum(1 for r in done if r["prep_decisions"] > 0),
        "total_seconds": round(sum(r.get("seconds", 0) for r in eligible), 1),
        "slowest": sorted(((r.get("seconds", 0), r["file"]) for r in eligible), reverse=True)[:3],
    }
    out = ROOT / ".pytest_cache" / "prep_corpus_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "files": results}, indent=1), encoding="utf-8")
    print("\nPREP CORPUS SUMMARY\n" + json.dumps(summary, indent=1))

    assert eligible, "no corpus file had a usable target"
    assert not failed, f"unhandled exceptions or hangs: {summary['unhandled_exceptions']} {summary['hung']}"
    assert not disagree, f"plan and run disagree on: {disagree}"
