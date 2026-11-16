"""
Corpus-wide validation suite for PreFlight v2.0 Plan, Dry-Run, and Plan PDF (Phase 4 / M3).

Executes dry_run on every file in corpus/raw/:
1. Validates 0 unhandled exceptions across the corpus.
2. Records per file: Plan PDF page count, needs_attention count, columns and rows.

Phase 7b: one test per file. The records are summarized (completions == 254, pages and
needs_attention median/max, files above 10 items) by scripts/corpus_summary.py at the
end of the session (see tests/corpus/conftest.py). The summary replaces the old
/tmp/preflight_corpus_plan_metrics.json.
"""

from pathlib import Path

import pypdf
import pytest
from corpus_support import EXPECTED_FILES, raw_files, raw_params, record

from preflight.plan import dry_run

SUITE = "plan"
pytestmark = [pytest.mark.corpus, pytest.mark.timeout(600)]


def test_corpus_has_expected_file_count():
    assert len(raw_files()) == EXPECTED_FILES, f"Expected {EXPECTED_FILES} corpus files"


@pytest.mark.parametrize("corpus_file", raw_params())
def test_dry_run_and_plan_pdf(corpus_file: Path, tmp_path: Path):
    fn = corpus_file.name
    try:
        plan_obj, pdf_path = dry_run(corpus_file, out_dir=tmp_path, mode=None)
        num_pages = len(pypdf.PdfReader(str(pdf_path)).pages)
    except Exception as exc:
        record(SUITE, fn, {"file": fn, "exception": f"{type(exc).__name__}: {exc}"})
        raise

    record(SUITE, fn, {
        "file": fn,
        "pages": num_pages,
        "needs_attention": sum(1 for d in plan_obj.decisions if d.needs_attention),
        "columns": plan_obj.column_count,
        "rows": plan_obj.row_count,
    })
    assert num_pages > 0
