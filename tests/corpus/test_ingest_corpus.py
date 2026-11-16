"""
Corpus-wide validation suite for PreFlight v2.0 ingest pipeline (Phase 2 / M1).

Runs encoding, dialect, and header detection against all 254 messy-CSV files in corpus/raw/,
compares detected structural parameters against ground-truth labels in corpus/labels/,
and validates accuracy, schema round-tripping, and zero unhandled crash rate.

Phase 7b: one test per file. Each file's match results are saved, and the corpus-wide
thresholds (delimiter >= 99%, encoding >= 98%, header == 100%, dialect >= 98%, no
unhandled exceptions) are checked over all 254 records by scripts/corpus_summary.py
at the end of the session (see tests/corpus/conftest.py).
"""

from pathlib import Path

import pytest
from corpus_support import (
    EXPECTED_FILES,
    RAW_DIR,
    cached_ingest,
    label_files,
    label_params,
    load_label,
    record,
)

from preflight.ingest import detect_dialect, detect_encoding, detect_header, load_raw
from preflight.schema import TableSchema

SUITE = "ingest"
pytestmark = [pytest.mark.corpus, pytest.mark.timeout(300)]


def normalize_encoding_name(enc: str) -> str:
    """Normalize encoding identifiers for equivalence comparison."""
    norm = enc.lower().replace("-", "").replace("_", "")
    if norm in ("ascii", "utf8", "utf8sig"):
        return "utf8"
    if norm in ("windows1252", "iso88591", "cp1252", "latin1", "iso88592", "cp1250"):
        return "singlebyte_latin"
    return norm


def test_corpus_has_expected_file_count():
    assert len(label_files()) == EXPECTED_FILES, f"Expected {EXPECTED_FILES} corpus files"


@pytest.mark.parametrize("corpus_file", label_params())
def test_ingestion_matches_label(corpus_file: Path):
    """Encoding, delimiter and header detection for one file, compared with its label using
    the criteria in docs/v1_baseline_report.md."""
    label_data = load_label(corpus_file)
    fn = label_data["filename"]
    raw_path = RAW_DIR / fn
    true_d = label_data["dialect"]
    assert raw_path.exists(), f"Raw file {fn} missing from {RAW_DIR}"

    try:
        enc_res = detect_encoding(raw_path)
        enc_ok = normalize_encoding_name(true_d["encoding"]) == normalize_encoding_name(enc_res.encoding)

        dia_res = detect_dialect(raw_path, encoding=enc_res.encoding)
        delim_ok = dia_res.delimiter == true_d["delimiter"]

        hdr_res = detect_header(
            raw_path,
            delimiter=dia_res.delimiter,
            quotechar=dia_res.quotechar,
            escapechar=dia_res.escapechar,
            doublequote=dia_res.doublequote,
            encoding=enc_res.encoding,
        )
        header_ok = hdr_res.header_row_index == true_d["header_row_index"]

        load_res = load_raw(raw_path, encoding=enc_res.encoding, dialect=dia_res, header_result=hdr_res)
        assert load_res.column_count > 0
    except Exception as e:
        record(SUITE, fn, {"file": fn, "exception": f"{type(e).__name__}: {e}"})
        raise

    record(SUITE, fn, {
        "file": fn, "enc_ok": enc_ok, "delim_ok": delim_ok, "header_ok": header_ok,
        "got": {"encoding": enc_res.encoding, "delimiter": dia_res.delimiter,
                "header_row_index": hdr_res.header_row_index},
    })


@pytest.mark.parametrize("corpus_file", label_params())
def test_schema_generation_and_round_trip(corpus_file: Path):
    """ingest() gives a valid TableSchema for the file, and it round-trips through JSON."""
    raw_path = RAW_DIR / load_label(corpus_file)["filename"]
    _, schema = cached_ingest(raw_path)

    assert isinstance(schema, TableSchema)
    assert schema.schema_version == "2.0.0"
    assert schema.encoding.detected != ""
    # " " is the whitespace sentinel (runs of spaces/tabs, UCI-style .data files)
    assert schema.dialect.delimiter in (",", ";", "\t", "|", " ")
    assert len(schema.columns) > 0

    restored = TableSchema.from_json(schema.to_json())
    assert restored.schema_version == "2.0.0"
    assert restored.source_path == schema.source_path
    assert restored.encoding.detected == schema.encoding.detected
    assert restored.dialect.delimiter == schema.dialect.delimiter
    assert restored.header.header_row_index == schema.header.header_row_index
    assert len(restored.columns) == len(schema.columns)


@pytest.mark.parametrize("fn", ["classic-rock-raw-data.csv", "avengers.csv", "police_killings.csv"])
def test_large_file_streaming_chunks(fn: str):
    """Chunked streaming preserves row counts and column counts, including bare-CR files."""
    raw_path = RAW_DIR / fn
    if not raw_path.exists():
        pytest.skip(f"{fn} not in corpus/raw")

    full_res = load_raw(raw_path)
    full_rows = full_res.row_count
    assert full_rows is not None and full_rows > 0

    chunk_res = load_raw(raw_path, chunksize=500)
    assert chunk_res.is_chunked is True

    total_chunked_rows = 0
    for chunk_df in chunk_res.data:
        assert len(chunk_df.columns) == full_res.column_count
        total_chunked_rows += len(chunk_df)
    assert total_chunked_rows == full_rows
