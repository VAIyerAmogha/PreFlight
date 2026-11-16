"""
Corpus-wide validation suite for PreFlight v2.0 type inference subsystem (Phase 3 / M2).

Evaluates type inference, anomaly capture, abstain logic, and Decision records
against all 254 real-world messy CSV files in corpus/raw/ using the ground-truth
labels in corpus/labels/ (2,850 columns).

Computes and validates:
1. Column-type accuracy (% of columns where inferred canonical type matches ground truth)
2. Silent failure rate: % wrong-type-with-no-flag (must be near zero, compared to 9.40% in v1)
3. Abstain rate (% of columns where inference safely abstains with needs_attention=True)
4. Decision emission and lossless schema round-trip across the corpus

Phase 7b: one test per file. Each file's counts are saved, and the corpus-wide thresholds
(2,850 labelled columns, accuracy >= 85%, wrong-type-with-no-flag <= 2%, no unhandled
exceptions) are checked over all 254 records by scripts/corpus_summary.py at the end of
the session (see tests/corpus/conftest.py).
"""

from pathlib import Path

import pytest
from corpus_support import RAW_DIR, cached_infer, label_params, load_label, record

from preflight.infer import to_canonical_type

SUITE = "infer"
pytestmark = [pytest.mark.corpus, pytest.mark.timeout(300)]


@pytest.mark.parametrize("corpus_file", label_params())
def test_type_inference_matches_label(corpus_file: Path):
    label_data = load_label(corpus_file)
    fn = label_data["filename"]
    raw_path = RAW_DIR / fn
    true_cols = label_data["columns"]
    assert raw_path.exists(), f"Raw file {fn} missing from {RAW_DIR}"

    total = correct = wrong = wrong_unflagged = abstained_n = 0
    columns = []
    try:
        schema = cached_infer(raw_path)
        col_by_name = {c.name: c for c in schema.columns}
        col_by_idx = {c.position: c for c in schema.columns}

        for c_true in true_cols:
            total += 1
            true_type = c_true["semantic_type"]
            c_inferred = col_by_name.get(c_true["column_name"]) or col_by_idx.get(c_true["column_index"])
            if c_inferred is None:
                wrong += 1
                wrong_unflagged += 1
                columns.append({"column": c_true["column_name"], "true": true_type, "got": None,
                                "flagged": False})
                continue

            canon_type = to_canonical_type(c_inferred.inferred_type or "text")
            abstained = c_inferred.abstained
            if abstained:
                abstained_n += 1

            # A column is flagged if it abstained, or has low confidence, anomalies,
            # missing markers, or a low margin
            is_flagged = (
                abstained
                or (c_inferred.confidence < 0.90)
                or len(c_inferred.anomalies) > 0
                or len(c_inferred.missing_markers) > 0
                or (c_inferred.evidence.get("margin", 1.0) < 0.25)
            )
            if canon_type == true_type:
                correct += 1
            else:
                wrong += 1
                if not is_flagged:
                    wrong_unflagged += 1
            columns.append({"column": c_true["column_name"], "true": true_type, "got": canon_type,
                            "flagged": bool(is_flagged)})
    except Exception as e:
        record(SUITE, fn, {"file": fn, "exception": f"{type(e).__name__}: {e}"})
        raise

    record(SUITE, fn, {
        "file": fn, "total": total, "correct": correct, "wrong": wrong,
        "wrong_unflagged": wrong_unflagged, "abstained": abstained_n, "columns": columns,
    })


@pytest.mark.parametrize(
    "fn", ["tips.csv", "iris.data", "table-with-accents.csv", "classic-rock-raw-data.csv"]
)
def test_corpus_decisions_structure_and_fingerprints(fn: str):
    """Decision records are emitted for every column with valid deterministic fingerprints."""
    raw_path = RAW_DIR / fn
    if not raw_path.exists():
        pytest.skip(f"{fn} not in corpus/raw")

    schema = cached_infer(raw_path)
    assert len(schema.decisions) == len(schema.columns)
    for dec in schema.decisions:
        assert dec.stage == "infer"
        assert dec.kind in ("type_inference", "type_abstain")
        assert len(dec.columns) == 1
        assert dec.fingerprint is not None
        assert len(dec.fingerprint) == 16
        assert dec.status == "automatic"
        # Plain text, unless the decision is flagged but its type is clear (placeholders
        # for missing data only, or a single-unit column): then the proposal is the default.
        assert dec.default in ("keep_as_string", dec.action)
        if dec.default != "keep_as_string":
            assert dec.needs_attention and not dec.evidence.get("unresolved_trap")
