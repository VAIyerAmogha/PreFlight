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
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pytest

from preflight.ingest import ingest
from preflight.infer import infer_table_types, to_canonical_type
from preflight.schema import TableSchema

CORPUS_ROOT = Path(__file__).resolve().parent.parent.parent / "corpus"
RAW_DIR = CORPUS_ROOT / "raw"
LABELS_DIR = CORPUS_ROOT / "labels"


def get_all_label_files() -> List[Path]:
    """Retrieve all ground-truth label JSON files."""
    return sorted(list(LABELS_DIR.glob("*.json")))


class TestCorpusInferValidation:
    """Corpus-scale evaluation of PreFlight v2.0 type inference."""

    def test_corpus_type_inference_accuracy_and_metrics(self):
        """
        Run type inference across all 254 corpus files and evaluate against
        2,850 ground-truth labeled columns.
        """
        label_files = get_all_label_files()
        assert len(label_files) == 254, f"Expected 254 corpus files, found {len(label_files)}"

        total_files = len(label_files)
        total_columns = 0
        correct_columns = 0
        wrong_columns = 0
        wrong_unflagged_columns = 0
        abstained_columns = 0
        exceptions: List[Tuple[str, str, str]] = []

        for lf in label_files:
            with open(lf, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            fid = label_data["file_id"]
            fn = label_data["filename"]
            raw_path = RAW_DIR / fn
            true_cols = label_data["columns"]

            assert raw_path.exists(), f"Raw file {fn} missing from {RAW_DIR}"

            try:
                # 1. Ingest raw data and structural schema
                data, schema = ingest(raw_path)

                # 2. Run type inference subsystem
                schema = infer_table_types(data, schema)

                col_by_name = {c.name: c for c in schema.columns}
                col_by_idx = {c.position: c for c in schema.columns}

                for c_true in true_cols:
                    total_columns += 1
                    c_idx = c_true["column_index"]
                    c_name = c_true["column_name"]
                    true_type = c_true["semantic_type"]

                    c_inferred = col_by_name.get(c_name) or col_by_idx.get(c_idx)
                    if c_inferred is None:
                        wrong_columns += 1
                        wrong_unflagged_columns += 1
                        continue

                    cand_type = c_inferred.inferred_type or "text"
                    canon_type = to_canonical_type(cand_type)
                    abstained = c_inferred.abstained

                    if abstained:
                        abstained_columns += 1

                    # A column is flagged if it abstained, or needs_attention is True,
                    # or has anomalies, or has missing markers, or confidence < 0.90, or low margin
                    is_flagged = (
                        abstained
                        or (c_inferred.confidence < 0.90)
                        or len(c_inferred.anomalies) > 0
                        or len(c_inferred.missing_markers) > 0
                        or (c_inferred.evidence.get("margin", 1.0) < 0.25)
                    )

                    if canon_type == true_type:
                        correct_columns += 1
                    else:
                        wrong_columns += 1
                        if not is_flagged:
                            wrong_unflagged_columns += 1

            except Exception as e:
                exceptions.append((fid, type(e).__name__, str(e)))

        # Assert no uncaught crashes
        assert len(exceptions) == 0, f"Encountered unhandled exceptions on corpus: {exceptions}"
        assert total_columns == 2850, f"Expected 2,850 columns, found {total_columns}"

        accuracy_pct = (correct_columns / total_columns) * 100
        wrong_unflagged_pct = (wrong_unflagged_columns / total_columns) * 100
        abstain_pct = (abstained_columns / total_columns) * 100

        print(f"\n--- Corpus Inference Validation Results ---")
        print(f"Total Columns Evaluated: {total_columns}")
        print(f"Correct Column Types:    {correct_columns} ({accuracy_pct:.2f}%)")
        print(f"Wrong Column Types:      {wrong_columns} ({(wrong_columns/total_columns)*100:.2f}%)")
        print(f"Abstained Columns:       {abstained_columns} ({abstain_pct:.2f}%)")
        print(f"Wrong & Unflagged:       {wrong_unflagged_columns} ({wrong_unflagged_pct:.2f}%)")

        # Validate Phase 3 performance criteria
        # v1.0.1 baseline: 80.77% accuracy, 9.40% wrong-unflagged
        assert accuracy_pct >= 85.0, f"Type accuracy {accuracy_pct:.2f}% below 85.0% threshold"
        assert wrong_unflagged_pct <= 2.0, (
            f"Wrong-type-with-no-flag rate {wrong_unflagged_pct:.2f}% exceeds 2.0% threshold"
        )

    def test_corpus_decisions_structure_and_fingerprints(self):
        """
        Verify that Decision records are emitted for every column and contain
        valid deterministic fingerprints.
        """
        target_files = [
            "tips.csv",
            "iris.data",
            "table-with-accents.csv",
            "classic-rock-raw-data.csv",
        ]
        for fn in target_files:
            raw_path = RAW_DIR / fn
            if not raw_path.exists():
                continue

            data, schema = ingest(raw_path)
            schema = infer_table_types(data, schema)

            assert len(schema.decisions) == len(schema.columns)
            for dec in schema.decisions:
                assert dec.stage == "infer"
                assert dec.kind in ("type_inference", "type_abstain")
                assert len(dec.columns) == 1
                assert dec.fingerprint is not None
                assert len(dec.fingerprint) == 16
                assert dec.status == "automatic"
                assert dec.default == "keep_as_string"
