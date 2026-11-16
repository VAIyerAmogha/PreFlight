"""
Corpus accuracy measurement, one test per labelled file (Phase 7b).

Each test runs scripts/corpus_accuracy.py's `measure_file` (full plan() on the file,
compared with its label) and saves the record. scripts/corpus_summary.py builds the
targets table from these records: tuning set by default, held-out and combined only
with --final. The per-file timeout is 120 s, the limit the Timeouts target is defined with.
A file that times out is saved as a TIMEOUT record by tests/corpus/conftest.py.
"""

import importlib.util
from pathlib import Path

import pytest
from corpus_support import ROOT, label_params, load_label, record

SUITE = "accuracy"
pytestmark = [pytest.mark.corpus, pytest.mark.timeout(120)]

_spec = importlib.util.spec_from_file_location("corpus_accuracy", ROOT / "scripts" / "corpus_accuracy.py")
assert _spec is not None and _spec.loader is not None
corpus_accuracy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(corpus_accuracy)
SPLIT = corpus_accuracy.load_split()


@pytest.mark.parametrize("corpus_file", label_params())
def test_plan_measured_against_label(corpus_file: Path):
    label = load_label(corpus_file)
    rec = corpus_accuracy.measure_file(label, SPLIT.get(label["file_id"]))
    record(SUITE, label["filename"], rec)
    assert rec["loaded_ok"], rec["error"]
