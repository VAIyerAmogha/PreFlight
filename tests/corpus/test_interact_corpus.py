"""
Corpus run of the interaction layer (Phase 5 / M4).

Every file in corpus/raw/ goes through plan -> resolve(strict) and plan -> apply(defaults)
in its own subprocess with a hard per-file timeout. Recorded per file: completes or hangs
(hangs must be zero), questions, whether strict raises, and question categories.

Phase 7b: one test per file (the per-file assertions are unchanged). The corpus-wide
summary (questions median/max, files above 10 questions, strict raises) comes from
scripts/corpus_summary.py at the end of the session. It replaces
.pytest_cache/interact_corpus_results.json.
"""

from pathlib import Path

import pytest
from corpus_support import RAW_DIR, raw_params, record, run_worker

SUITE = "interact"
WORKER = Path(__file__).with_name("interact_worker.py")
PER_FILE_TIMEOUT = 180  # seconds, enforced on the worker process
pytestmark = [
    pytest.mark.corpus,
    pytest.mark.timeout(PER_FILE_TIMEOUT + 60),
    pytest.mark.skipif(not RAW_DIR.is_dir(), reason="corpus/raw not present"),
]


@pytest.mark.parametrize("corpus_file", raw_params())
def test_interact_on_file(corpus_file: Path):
    r = run_worker(WORKER, corpus_file, PER_FILE_TIMEOUT, "RESULT ")
    record(SUITE, corpus_file.name, r)

    assert not r.get("hang"), f"{corpus_file.name} hung"
    assert r.get("ok"), f"worker failure: stage={r.get('stage')} error={r.get('error')}"
    assert r["bounds_ok"], "option bounds / default membership violated"
    assert not r["jargon_hits"], r["jargon_hits"]
    # strict raises exactly when there is something to ask
    assert r["strict_raised"] == (r["questions"] > 0)
    if r["strict_raised"]:
        assert r.get("strict_listed", 0) == r["questions"]
    # defaults mode resolves everything it touches
    if r.get("defaults_completed"):
        assert r.get("still_open", 0) == 0
