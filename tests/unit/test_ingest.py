"""
Unit test suite for PreFlight v2.0 ingest pipeline.

Tests hand-built edge cases:
- BOM present vs absent (UTF-8-SIG, UTF-16)
- Character encoding detection and low-confidence flagging
- Candidate delimiters (comma, semicolon, tab, pipe)
- Dialect row-length regularity scoring and low-margin flagging
- Quoting style detection (doubled quotes vs backslash escaped)
- Header vs data distinction using type consistency
- Excluded lines: title/preamble comments, footer notes, blank rows, total lines
- Ragged row detection and flagging
- Raw loading: preservation of string values without automatic NA conversion
- Chunked reading for large files
- TableSchema versioned serialization and round-trip fidelity
"""

import json
from pathlib import Path
import tempfile
import pytest
import pandas as pd

from preflight.ingest.encoding import detect_encoding, EncodingResult
from preflight.ingest.dialect import detect_dialect, DialectResult
from preflight.ingest.header import detect_header, HeaderResult
from preflight.ingest.loader import load_raw, RawLoadResult
from preflight.ingest import ingest
from preflight.schema import (
    TableSchema,
    EncodingSchema,
    DialectSchema,
    HeaderSchema,
    ColumnRawMetadata,
)


# =====================================================================
# 1. Encoding Detection Unit Tests
# =====================================================================

def test_encoding_bom_utf8_sig():
    """Verify UTF-8 BOM detection."""
    content = b"\xef\xbb\xbfcol1,col2\nval1,val2\n"
    res = detect_encoding(content)
    assert isinstance(res, EncodingResult)
    assert res.encoding == "utf-8-sig"
    assert res.has_bom is True
    assert res.confidence == 1.0
    assert res.low_confidence is False


def test_encoding_bom_utf16_le():
    """Verify UTF-16-LE BOM detection."""
    text = "col1,col2\nval1,val2\n"
    content = b"\xff\xfe" + text.encode("utf-16-le")
    res = detect_encoding(content)
    assert res.encoding == "utf-16-le"
    assert res.has_bom is True
    assert res.confidence == 1.0


def test_encoding_bom_absent_ascii():
    """Verify pure ASCII file has no BOM and high confidence UTF-8."""
    content = b"name,age,city\nAlice,30,NYC\nBob,25,LA\n"
    res = detect_encoding(content)
    assert res.encoding == "utf-8"
    assert res.has_bom is False
    assert res.confidence == 1.0
    assert res.low_confidence is False


def test_encoding_non_utf8_latin1():
    """Verify Latin-1 byte stream is identified and decoded without crash."""
    # Byte 0xe9 is 'é' and 0xa9 is '©' in ISO-8859-1 (invalid UTF-8)
    content = "id,name\n1,Caf\xe9\n2,Copyright \xa9\n".encode("iso-8859-1")
    res = detect_encoding(content)
    assert res.has_bom is False
    assert res.encoding in ("iso-8859-1", "cp1252")


def test_encoding_low_confidence_flagged():
    """Verify low-confidence ambiguous inputs set low_confidence=True."""
    # Single obscure non-ASCII byte without linguistic context
    content = b"\x81\x82"
    res = detect_encoding(content, confidence_threshold=0.99)
    assert res.low_confidence is True


# =====================================================================
# 2. Dialect Detection Unit Tests
# =====================================================================

def test_dialect_comma():
    """Verify comma delimiter detection."""
    content = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n"
    res = detect_dialect(content.encode("utf-8"))
    assert isinstance(res, DialectResult)
    assert res.delimiter == ","
    assert res.confidence > 0.8
    assert res.field_count_mode == 3
    assert res.flagged is False


def test_dialect_semicolon():
    """Verify semicolon delimiter detection."""
    content = "col_a;col_b;col_c\nval1;val2;val3\nval4;val5;val6\nval7;val8;val9\n"
    res = detect_dialect(content.encode("utf-8"))
    assert res.delimiter == ";"
    assert res.field_count_mode == 3
    assert res.flagged is False


def test_dialect_tab():
    """Verify tab delimiter detection."""
    content = "col_a\tcol_b\tcol_c\n1\t2\t3\n4\t5\t6\n"
    res = detect_dialect(content.encode("utf-8"))
    assert res.delimiter == "\t"
    assert res.field_count_mode == 3


def test_dialect_pipe():
    """Verify pipe delimiter detection."""
    content = "id|name|score\n1|Alice|95.5\n2|Bob|88.0\n3|Charlie|92.3\n"
    res = detect_dialect(content.encode("utf-8"))
    assert res.delimiter == "|"
    assert res.field_count_mode == 3


def test_dialect_semicolon_with_commas_in_text():
    """Verify semicolon delimiter is preferred when commas appear in decimal or text."""
    content = (
        "id;name;ratio;amount\n"
        "1;Smith, John;0,75;1,200\n"
        "2;Doe, Jane;0,85;3,400\n"
        "3;Brown, Bob;0,90;2,100\n"
    )
    res = detect_dialect(content.encode("utf-8"))
    assert res.delimiter == ";"
    assert res.field_count_mode == 4


def test_dialect_low_margin_flagging():
    """Verify ambiguous files with competing delimiters trigger flagged=True."""
    # Equal occurrences of comma and pipe
    content = "a,b|c,d\n1,2|3,4\n5,6|7,8\n"
    res = detect_dialect(content.encode("utf-8"), margin_threshold=0.20)
    assert res.flagged is True


def test_dialect_quoting_doubled_vs_escaped():
    """Verify doubled quote detection vs backslash escape detection."""
    # Doubled quotes
    doubled = 'name,desc\n"Alice","She said ""hello"""\n"Bob","He said ""hi"""\n'
    res_doubled = detect_dialect(doubled.encode("utf-8"))
    assert res_doubled.quotechar == '"'
    assert res_doubled.doublequote is True
    assert res_doubled.escapechar is None

    # Backslash escaped
    escaped = 'name,desc\n"Alice","She said \\"hello\\""\n"Bob","He said \\"hi\\""\n'
    res_escaped = detect_dialect(escaped.encode("utf-8"))
    assert res_escaped.quotechar == '"'
    assert res_escaped.escapechar == "\\"


# =====================================================================
# 3. Header & Table Detection Unit Tests
# =====================================================================

def test_header_standard():
    """Verify standard header with type disparity from numeric data."""
    lines = [
        "sepal_length,sepal_width,species",
        "5.1,3.5,setosa",
        "4.9,3.0,setosa",
        "4.7,3.2,setosa",
    ]
    res = detect_header(lines, delimiter=",")
    assert isinstance(res, HeaderResult)
    assert res.header_row_index == 0
    assert res.column_names == ["sepal_length", "sepal_width", "species"]
    assert res.title_rows == []


def test_header_headerless_dataset():
    """Verify headerless dataset (e.g. iris.data) is detected as header_row_index=None."""
    lines = [
        "5.1,3.5,1.4,0.2,Iris-setosa",
        "4.9,3.0,1.4,0.2,Iris-setosa",
        "4.7,3.2,1.3,0.2,Iris-setosa",
        "4.6,3.1,1.5,0.2,Iris-setosa",
    ]
    res = detect_header(lines, delimiter=",")
    assert res.header_row_index is None
    assert res.column_names == ["col_0", "col_1", "col_2", "col_3", "col_4"]


def test_header_preamble_comment_and_blank_rows():
    """Verify comment lines and blank lines above header are excluded."""
    lines = [
        "# Table Metadata: Experiment Results",
        "// Generated on 2026-09-29",
        "",
        "id,treatment,response",
        "1,control,12.5",
        "2,active,18.9",
        "3,active,21.0",
    ]
    res = detect_header(lines, delimiter=",")
    assert res.header_row_index == 3
    assert res.title_rows == [0, 1]
    assert 2 in res.blank_rows
    assert res.column_names == ["id", "treatment", "response"]


def test_header_detects_footer_and_total_rows():
    """Verify total/summary rows and footer notes at table bottom are captured."""
    lines = [
        "department,budget,staff",
        "Engineering,500000,20",
        "Sales,300000,15",
        "Marketing,200000,8",
        "Total,1000000,43",
        "Source: Internal HR Database 2026",
    ]
    res = detect_header(lines, delimiter=",")
    assert res.header_row_index == 0
    assert 4 in res.total_rows
    assert 5 in res.footer_rows


def test_header_flags_ragged_rows():
    """Verify inconsistent row field counts are flagged as ragged."""
    lines = [
        "col1,col2,col3",
        "val1,val2,val3",
        "val4,val5",         # 2 fields instead of 3
        "val6,val7,val8,val9", # 4 fields instead of 3
        "val10,val11,val12",
    ]
    res = detect_header(lines, delimiter=",")
    assert res.is_ragged is True
    assert len(res.ragged_rows) == 2
    assert res.flagged is True


# =====================================================================
# 4. Raw Loading Unit Tests
# =====================================================================

def test_raw_load_preserves_strings_and_sentinels():
    """Verify raw load does not convert NA, null, or ? to NaN."""
    csv_content = (
        "id,val_missing,leading_zero,status\n"
        "1,NA,01234,active\n"
        "2,null,00567,NA\n"
        "3,?,00890,pending\n"
        "4,,00012,-\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    try:
        res = load_raw(path)
        assert isinstance(res, RawLoadResult)
        assert res.is_chunked is False
        df = res.data
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

        # Verify no NaN values exist anywhere
        assert df.isna().sum().sum() == 0

        # Exact sentinel string preservation
        assert df.loc[0, "val_missing"] == "NA"
        assert df.loc[1, "val_missing"] == "null"
        assert df.loc[2, "val_missing"] == "?"
        assert df.loc[3, "val_missing"] == ""

        # Leading zero preservation
        assert df.loc[0, "leading_zero"] == "01234"
        assert df.loc[1, "leading_zero"] == "00567"
        assert df.loc[3, "leading_zero"] == "00012"
    finally:
        Path(path).unlink(missing_ok=True)


def test_raw_load_chunked():
    """Verify chunked reading returns generator of chunks without loading full file."""
    csv_content = (
        "id,name\n"
        "1,Alice\n"
        "2,Bob\n"
        "3,Charlie\n"
        "4,Diana\n"
        "5,Evan\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        path = f.name

    try:
        res = load_raw(path, chunksize=2)
        assert res.is_chunked is True
        assert res.row_count is None

        chunks = list(res.data)
        assert len(chunks) == 3
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1
        assert chunks[0].loc[0, "name"] == "Alice"
    finally:
        Path(path).unlink(missing_ok=True)


# =====================================================================
# 5. Schema Serialization & Round-Trip Tests
# =====================================================================

def test_schema_round_trip():
    """Verify TableSchema serializes to versioned JSON and round-trips identically."""
    schema = TableSchema(
        schema_version="2.0.0",
        source_path="/path/to/data.csv",
        row_count=100,
        column_count=3,
        encoding=EncodingSchema(
            detected="utf-8", confidence=1.0, has_bom=False, low_confidence=False
        ),
        dialect=DialectSchema(
            delimiter=",", quotechar='"', escapechar=None, doublequote=True,
            confidence=0.98, flagged=False, margin=0.5
        ),
        header=HeaderSchema(
            header_row_index=0, confidence=0.95, title_rows=[0], ragged_rows=[],
            is_ragged=False, flagged=False
        ),
        columns=[
            ColumnRawMetadata(name="id", position=0),
            ColumnRawMetadata(name="age", position=1),
            ColumnRawMetadata(name="income", position=2, name_override="annual_income"),
        ],
        user_overrides={"delimiter": ","},
    )

    json_str = schema.to_json()
    parsed_json = json.loads(json_str)
    assert parsed_json["schema_version"] == "2.0.0"

    # Reconstruct from JSON
    reconstructed = TableSchema.from_json(json_str)
    assert reconstructed.schema_version == schema.schema_version
    assert reconstructed.source_path == schema.source_path
    assert reconstructed.row_count == schema.row_count
    assert reconstructed.column_count == schema.column_count
    assert reconstructed.encoding.detected == schema.encoding.detected
    assert reconstructed.dialect.delimiter == schema.dialect.delimiter
    assert reconstructed.header.header_row_index == schema.header.header_row_index
    assert len(reconstructed.columns) == 3
    assert reconstructed.columns[2].name_override == "annual_income"
    assert reconstructed.to_dict() == schema.to_dict()


def test_schema_user_override_effects():
    """Verify user overrides take precedence when effective properties are queried."""
    enc = EncodingSchema(detected="iso-8859-1", confidence=0.6, has_bom=False, low_confidence=True, override="utf-8")
    assert enc.effective_encoding() == "utf-8"

    dialect = DialectSchema(
        delimiter=";", quotechar='"', escapechar=None, doublequote=True,
        confidence=0.8, flagged=False, margin=0.2, delimiter_override=","
    )
    assert dialect.effective_delimiter() == ","

    header = HeaderSchema(header_row_index=0, confidence=0.9, header_row_override=1)
    assert header.effective_header_row() == 1


# =====================================================================
# 6. Integrated Ingest Function Test
# =====================================================================

def test_full_ingest_orchestration():
    """Verify ingest() helper returns DataFrame and populated TableSchema."""
    content = "col_x;col_y\n10;20\n30;40\n"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(content)
        path = f.name

    try:
        df, schema = ingest(path)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(schema, TableSchema)
        assert schema.dialect.delimiter == ";"
        assert schema.header.header_row_index == 0
        assert schema.column_count == 2
        assert len(df) == 2
        assert list(df.columns) == ["col_x", "col_y"]
    finally:
        Path(path).unlink(missing_ok=True)
