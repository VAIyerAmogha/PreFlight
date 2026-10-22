"""
Corpus and Label Integrity Test Suite for Phase 1 (M0).

Verifies:
1. Every raw file in corpus/raw/ has a matching label file in corpus/labels/.
2. Every label file is valid against the documented schema in corpus/labels/GUIDE.md.
3. No duplicate file_ids exist across labels or sources.csv.
4. Column counts in label records match actual column counts parsed from the raw files.
5. All datasets recorded in sources.csv possess verified, non-empty permissive open licenses.
"""

import csv
import json
from pathlib import Path
import pytest

CORPUS_ROOT = Path(__file__).resolve().parent.parent.parent / "corpus"
RAW_DIR = CORPUS_ROOT / "raw"
LABELS_DIR = CORPUS_ROOT / "labels"
SOURCES_CSV = LABELS_DIR / "sources.csv"

VALID_SEMANTIC_TYPES = {
    "numeric",
    "categorical",
    "datetime",
    "text",
    "boolean",
    "constant",
    "ID",
}


def get_label_files():
    """Retrieve all label json files."""
    return sorted(list(LABELS_DIR.glob("*.json")))


def test_no_duplicate_file_ids():
    """Ensure no duplicate file_ids exist across label files or sources manifest."""
    assert SOURCES_CSV.exists(), f"Missing {SOURCES_CSV}"
    with open(SOURCES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sources = list(reader)

    source_ids = [row["file_id"] for row in sources]
    assert len(source_ids) == len(set(source_ids)), "Duplicate file_ids detected in sources.csv"

    label_files = get_label_files()
    assert len(label_files) > 0, "No label files found"
    label_ids = [f.stem for f in label_files]
    assert len(label_ids) == len(set(label_ids)), "Duplicate label filenames detected in corpus/labels"
    assert set(source_ids) == set(label_ids), "Mismatch between sources.csv file_ids and label files"


def test_every_raw_file_has_matching_label():
    """Ensure every raw file on disk has an associated label JSON file."""
    assert RAW_DIR.exists(), f"Raw directory {RAW_DIR} does not exist"
    raw_files = [f for f in RAW_DIR.glob("*") if f.is_file()]
    assert len(raw_files) >= 200, f"Expected at least 200 raw files, found {len(raw_files)}"

    label_files = get_label_files()
    labeled_raw_filenames = set()
    for lf in label_files:
        with open(lf, "r", encoding="utf-8") as f:
            data = json.load(f)
            labeled_raw_filenames.add(data["filename"])

    for rf in raw_files:
        assert rf.name in labeled_raw_filenames, f"Raw file {rf.name} has no matching label record"


def test_every_label_has_matching_raw_file():
    """Ensure every label JSON references a raw file that physically exists."""
    label_files = get_label_files()
    assert len(label_files) >= 200, f"Expected at least 200 label files, found {len(label_files)}"

    for lf in label_files:
        with open(lf, "r", encoding="utf-8") as f:
            data = json.load(f)
        fn = data.get("filename")
        assert fn is not None, f"Label {lf.name} missing 'filename' field"
        raw_path = RAW_DIR / fn
        assert raw_path.exists(), f"Raw file {fn} referenced by {lf.name} does not exist"
        assert raw_path.stat().st_size > 0, f"Raw file {fn} referenced by {lf.name} is empty (0 bytes)"


@pytest.mark.parametrize("label_path", get_label_files(), ids=lambda p: p.stem)
def test_label_schema_validity(label_path: Path):
    """Validate each label record against documented schema."""
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "file_id" in data and isinstance(data["file_id"], str) and data["file_id"]
    assert data["file_id"] == label_path.stem, f"file_id {data['file_id']} does not match stem {label_path.stem}"
    assert "filename" in data and isinstance(data["filename"], str) and data["filename"]

    # Validate dialect
    assert "dialect" in data and isinstance(data["dialect"], dict)
    dialect = data["dialect"]
    assert "delimiter" in dialect and isinstance(dialect["delimiter"], str) and len(dialect["delimiter"]) >= 1
    assert "encoding" in dialect and isinstance(dialect["encoding"], str) and dialect["encoding"]
    assert "header_row_index" in dialect
    assert dialect["header_row_index"] is None or (
        isinstance(dialect["header_row_index"], int) and dialect["header_row_index"] >= 0
    )

    # Validate columns
    assert "columns" in data and isinstance(data["columns"], list)
    columns = data["columns"]
    assert len(columns) > 0, f"Dataset {data['file_id']} has 0 labeled columns"

    for idx, col in enumerate(columns):
        assert "column_name" in col and isinstance(col["column_name"], str)
        assert "column_index" in col and isinstance(col["column_index"], int)
        assert col["column_index"] == idx, f"Column index {col['column_index']} out of sequence at {idx}"
        assert "semantic_type" in col
        assert col["semantic_type"] in VALID_SEMANTIC_TYPES, (
            f"Invalid semantic_type '{col['semantic_type']}' in column '{col['column_name']}' "
            f"of {data['file_id']}. Expected one of {VALID_SEMANTIC_TYPES}"
        )
        assert "anomalies" in col and isinstance(col["anomalies"], list)
        for anomaly in col["anomalies"]:
            assert isinstance(anomaly, str) and anomaly, f"Invalid anomaly tag in {data['file_id']}"


@pytest.mark.parametrize("label_path", get_label_files(), ids=lambda p: p.stem)
def test_column_counts_match_raw_file(label_path: Path):
    """
    Verify that parsing the raw file with the labeled dialect yields
    the exact same number of columns as defined in the label record.
    """
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_path = RAW_DIR / data["filename"]
    dialect = data["dialect"]
    delim = dialect["delimiter"]
    enc = dialect["encoding"]
    header_idx = dialect["header_row_index"]

    with open(raw_path, "rb") as f:
        raw_bytes = f.read()

    try:
        text = raw_bytes.decode(enc)
    except UnicodeDecodeError:
        text = raw_bytes.decode("iso-8859-1")

    lines = text.splitlines()
    if header_idx is not None and header_idx > 0:
        lines = lines[header_idx:]

    reader = csv.reader(lines, delimiter=delim)
    rows = list(reader)
    assert len(rows) > 0, f"No rows parsed from {raw_path.name}"

    first_row_cols = len(rows[0])
    label_cols = len(data["columns"])
    assert label_cols == first_row_cols, (
        f"Column count mismatch for {data['file_id']}: "
        f"raw file has {first_row_cols} columns, but label specifies {label_cols} columns"
    )


def test_sources_manifest_licensing():
    """Verify all files recorded in sources.csv have clear permissive licenses."""
    with open(SOURCES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sources = list(reader)

    assert len(sources) >= 200, f"Expected at least 200 sources in manifest, got {len(sources)}"
    for row in sources:
        assert row["file_id"], "Missing file_id in sources row"
        assert row["filename"], "Missing filename in sources row"
        assert row["source_name"], "Missing source_name in sources row"
        assert row["source_url"], "Missing source_url in sources row"
        assert row["license"], f"Missing license for {row['file_id']}"
        lic = row["license"].lower()
        # Ensure license is permissible and clearly stated
        assert any(
            term in lic for term in ["mit", "bsd", "cc by", "cc0", "gpl", "public", "open"]
        ), f"Unclear or restrictive license for {row['file_id']}: '{row['license']}'"
