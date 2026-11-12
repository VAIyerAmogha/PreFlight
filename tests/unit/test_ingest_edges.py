"""Edge paths of the ingest layer: source types, overrides, empty input, footers and fallbacks."""

import io

import pandas as pd
import pytest

from preflight.ingest import ingest
from preflight.ingest.dialect import _score_delimiter, detect_dialect
from preflight.ingest.encoding import detect_encoding, normalize_encoding_name
from preflight.ingest.header import (
    _classify_token_type,
    _is_float_or_negative,
    _is_numeric,
    detect_header,
)
from preflight.ingest.loader import load_raw

ROWS = "id,name,score\n1,ann,3.5\n2,bob,4.0\n3,cy,2.5\n"


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(ROWS, encoding="utf-8")
    return p


# --- missing files and unsupported source types --------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda p: ingest(p),
        lambda p: load_raw(p),
        lambda p: detect_dialect(p),
        lambda p: detect_header(p),
        lambda p: detect_encoding(p),
    ],
)
def test_missing_file_raises_file_not_found(tmp_path, call):
    with pytest.raises(FileNotFoundError):
        call(tmp_path / "absent.csv")


@pytest.mark.parametrize("fn", [detect_dialect, detect_header, detect_encoding])
def test_unsupported_source_type_raises_type_error(fn):
    with pytest.raises(TypeError):
        fn(12345)


# --- in-memory sources give the same answer as the file ------------------------------


def test_dialect_from_list_and_bytes_matches_file(csv_file):
    from_file = detect_dialect(csv_file)
    from_list = detect_dialect(ROWS.splitlines())
    from_bytes = detect_dialect(ROWS.encode("utf-8"))
    assert from_file.delimiter == from_list.delimiter == from_bytes.delimiter == ","


def test_header_from_bytes_matches_file(csv_file):
    assert detect_header(ROWS.encode("utf-8")).column_names == detect_header(csv_file).column_names
    assert detect_header(csv_file).column_names == ["id", "name", "score"]


def test_encoding_from_file_object_and_empty_input():
    res = detect_encoding(io.BytesIO(ROWS.encode("utf-8")))
    assert res.encoding in ("utf-8", "ascii")
    empty = detect_encoding(b"")
    assert empty.encoding == "utf-8" and empty.low_confidence


def test_normalize_encoding_name_empty_defaults_to_utf8():
    assert normalize_encoding_name("") == "utf-8"


# --- empty and blank input ------------------------------------------------------------


def test_empty_and_blank_inputs_do_not_crash():
    assert detect_dialect([]).delimiter == ","
    assert detect_dialect(["   ", ""]).delimiter == ","
    assert detect_header([]).column_names == []
    assert detect_header(["", ""]).column_names == []


def test_score_delimiter_on_empty_or_blank_lines_is_zero():
    assert _score_delimiter([], ",") == (0.0, 0, 0.0)
    assert _score_delimiter(["  ", ""], ",") == (0.0, 0, 0.0)


def test_score_delimiter_scores_zero_when_the_sample_cannot_be_parsed():
    # A two-character delimiter is rejected by the csv module.
    assert _score_delimiter(["a,b", "c,d"], ",,") == (0.0, 0, 0.0)


# --- token classification ----------------------------------------------------------------


def test_token_helpers_on_empty_and_year_tokens():
    assert not _is_numeric("")
    assert not _is_float_or_negative("")
    assert _classify_token_type("  ") == "empty"
    assert _classify_token_type("1999") == "year"


# --- quoting ---------------------------------------------------------------------------------


def test_single_quote_quotechar_detected():
    lines = ["a,b,c"] + [f"{i},'x, y',{i}" for i in range(20)]
    res = detect_dialect(lines)
    assert res.quotechar == "'"


# --- header detection ----------------------------------------------------------------------------


def test_numeric_first_row_is_not_a_header():
    lines = [f"{i},{i * 2},{i * 3}" for i in range(1, 20)]
    res = detect_header(lines)
    assert res.header_row_index is None


def test_duplicate_header_names_are_made_unique():
    lines = ["a,a,b"] + [f"x{i},y{i},z{i}" for i in range(10)]
    names = detect_header(lines).column_names
    assert len(set(names)) == 3 and names[0] == "a" and names[1].startswith("a_")


def test_total_and_footer_rows_are_detected_and_dropped(tmp_path):
    body = "\n".join(f"r{i},{i},{i * 2}" for i in range(10))
    p = tmp_path / "f.csv"
    p.write_text(f"name,a,b\n{body}\nTotal,45,90\nSource: survey office\n", encoding="utf-8")
    h = detect_header(p)
    assert h.total_rows and h.footer_rows
    df, _ = ingest(p)
    assert len(df) == 10
    assert not df["name"].str.lower().isin(["total", "source: survey office"]).any()


# --- loader fallbacks -------------------------------------------------------------------------------


def test_bare_carriage_return_line_endings_load(tmp_path):
    p = tmp_path / "cr.csv"
    p.write_bytes(ROWS.replace("\n", "\r").encode("utf-8"))
    df = load_raw(p).data
    assert isinstance(df, pd.DataFrame) and len(df) == 3


def test_parser_error_falls_back_to_python_engine(tmp_path, monkeypatch):
    p = tmp_path / "t.csv"
    p.write_text(ROWS, encoding="utf-8")
    real = pd.read_csv
    calls = []

    def flaky(*args, **kwargs):
        calls.append(kwargs.get("engine"))
        if kwargs.get("engine") != "python":
            raise pd.errors.ParserError("simulated C-engine failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", flaky)
    df = load_raw(p).data
    assert len(df) == 3 and calls[-1] == "python"


def test_chunked_reading_falls_back_to_python_engine(tmp_path, monkeypatch):
    p = tmp_path / "t.csv"
    p.write_text(ROWS, encoding="utf-8")
    real = pd.read_csv

    def flaky(*args, **kwargs):
        if kwargs.get("engine") == "python":
            return real(*args, **kwargs)

        def broken():
            raise pd.errors.ParserError("simulated failure mid-stream")
            yield  # makes this function a generator

        return broken()

    monkeypatch.setattr(pd, "read_csv", flaky)
    res = load_raw(p, chunksize=2)
    assert res.is_chunked
    assert sum(len(c) for c in res.data) == 3


# --- user overrides ----------------------------------------------------------------------------------


def test_ingest_honours_encoding_delimiter_and_header_overrides(tmp_path):
    p = tmp_path / "semi.csv"
    p.write_text("title line\nx;y\n1;2\n3;4\n", encoding="utf-8")
    df, schema = ingest(
        p, user_overrides={"encoding": "utf-8", "delimiter": ";", "header_row_index": 1}
    )
    assert [c.name for c in schema.columns] == ["x", "y"]
    assert len(df) == 2
