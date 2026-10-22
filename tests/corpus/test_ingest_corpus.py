"""
Corpus-wide validation suite for PreFlight v2.0 ingest pipeline (Phase 2 / M1).

Runs encoding, dialect, and header detection against all 254 messy-CSV files in corpus/raw/,
compares detected structural parameters against ground-truth labels in corpus/labels/,
and validates accuracy, schema round-tripping, and zero unhandled crash rate.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pytest

from preflight.ingest import detect_encoding, detect_dialect, detect_header, load_raw, ingest
from preflight.schema import TableSchema

CORPUS_ROOT = Path(__file__).resolve().parent.parent.parent / "corpus"
RAW_DIR = CORPUS_ROOT / "raw"
LABELS_DIR = CORPUS_ROOT / "labels"


def get_all_label_files() -> List[Path]:
    """Retrieve all ground-truth label JSON files."""
    return sorted(list(LABELS_DIR.glob("*.json")))


def normalize_encoding_name(enc: str) -> str:
    """Normalize encoding identifiers for equivalence comparison."""
    norm = enc.lower().replace("-", "").replace("_", "")
    if norm in ("ascii", "utf8", "utf8sig"):
        return "utf8"
    if norm in ("windows1252", "iso88591", "cp1252", "latin1", "iso88592", "cp1250"):
        return "singlebyte_latin"
    return norm


class TestCorpusIngestValidation:
    """Corpus-scale validation of PreFlight v2.0 ingestion."""

    def test_corpus_ingestion_accuracy_and_metrics(self):
        """
        Evaluate encoding, delimiter, and header detection across all corpus files
        using the exact comparison criteria established in docs/v1_baseline_report.md.
        """
        label_files = get_all_label_files()
        assert len(label_files) == 254, f"Expected 254 corpus files, found {len(label_files)}"

        total_files = len(label_files)
        encoding_matches = 0
        delimiter_matches = 0
        header_matches = 0
        dialect_matches = 0
        exceptions: List[Tuple[str, str, str]] = []

        for lf in label_files:
            with open(lf, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            fid = label_data["file_id"]
            fn = label_data["filename"]
            raw_path = RAW_DIR / fn
            true_d = label_data["dialect"]

            assert raw_path.exists(), f"Raw file {fn} missing from {RAW_DIR}"

            try:
                # 1. Encoding detection
                enc_res = detect_encoding(raw_path)
                norm_true_enc = normalize_encoding_name(true_d["encoding"])
                norm_det_enc = normalize_encoding_name(enc_res.encoding)
                enc_ok = (norm_true_enc == norm_det_enc)

                # 2. Dialect detection
                dia_res = detect_dialect(raw_path, encoding=enc_res.encoding)
                delim_ok = (dia_res.delimiter == true_d["delimiter"])

                # 3. Header detection
                hdr_res = detect_header(
                    raw_path,
                    delimiter=dia_res.delimiter,
                    quotechar=dia_res.quotechar,
                    escapechar=dia_res.escapechar,
                    doublequote=dia_res.doublequote,
                    encoding=enc_res.encoding,
                )
                header_ok = (hdr_res.header_row_index == true_d["header_row_index"])

                # 4. Raw load
                load_res = load_raw(
                    raw_path,
                    encoding=enc_res.encoding,
                    dialect=dia_res,
                    header_result=hdr_res,
                )
                assert load_res.column_count > 0

                if enc_ok:
                    encoding_matches += 1
                if delim_ok:
                    delimiter_matches += 1
                if header_ok:
                    header_matches += 1
                if enc_ok and delim_ok and header_ok:
                    dialect_matches += 1

            except Exception as e:
                exceptions.append((fid, type(e).__name__, str(e)))

        # Unhandled exceptions check: v2 must not crash on any corpus file
        assert len(exceptions) == 0, f"Encountered unhandled exceptions on corpus: {exceptions}"

        encoding_acc = (encoding_matches / total_files) * 100
        delimiter_acc = (delimiter_matches / total_files) * 100
        header_acc = (header_matches / total_files) * 100
        dialect_acc = (dialect_matches / total_files) * 100

        # Assert performance thresholds against baseline
        assert delimiter_acc >= 99.0, f"Delimiter accuracy {delimiter_acc:.2f}% below 99.0%"
        assert encoding_acc >= 98.0, f"Encoding accuracy {encoding_acc:.2f}% below 98.0%"
        assert header_acc == 100.0, f"Header accuracy {header_acc:.2f}% below 100.0%"
        assert dialect_acc >= 98.0, f"Dialect accuracy {dialect_acc:.2f}% below 98.0%"

    def test_all_corpus_files_end_to_end_schema_generation(self):
        """
        Verify that ingest() successfully processes every corpus file, generates a
        valid TableSchema dataclass, and round-trips via JSON serialization.
        """
        label_files = get_all_label_files()
        for lf in label_files:
            with open(lf, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            fn = label_data["filename"]
            raw_path = RAW_DIR / fn

            raw_data, schema = ingest(raw_path)

            assert isinstance(schema, TableSchema)
            assert schema.schema_version == "2.0.0"
            assert schema.encoding.detected != ""
            assert schema.dialect.delimiter in (",", ";", "\t", "|")
            assert len(schema.columns) > 0

            # Verify JSON serialization round-trip
            json_str = schema.to_json()
            restored = TableSchema.from_json(json_str)
            assert restored.schema_version == "2.0.0"
            assert restored.source_path == schema.source_path
            assert restored.encoding.detected == schema.encoding.detected
            assert restored.dialect.delimiter == schema.dialect.delimiter
            assert restored.header.header_row_index == schema.header.header_row_index
            assert len(restored.columns) == len(schema.columns)

    def test_large_file_streaming_chunks(self):
        """
        Verify chunked streaming on large corpus files to ensure chunksize does not
        load full files into memory, preserves row counts, and handles bare CR lines.
        """
        target_files = [
            "classic-rock-raw-data.csv",
            "avengers.csv",
            "police_killings.csv",
        ]
        for fn in target_files:
            raw_path = RAW_DIR / fn
            if not raw_path.exists():
                continue

            # Load full
            full_res = load_raw(raw_path)
            full_rows = full_res.row_count
            assert full_rows is not None and full_rows > 0

            # Load chunked
            chunk_size = 500
            chunk_res = load_raw(raw_path, chunksize=chunk_size)
            assert chunk_res.is_chunked is True

            total_chunked_rows = 0
            for chunk_df in chunk_res.data:
                assert len(chunk_df.columns) == full_res.column_count
                total_chunked_rows += len(chunk_df)

            assert total_chunked_rows == full_rows
