"""
Raw tabular data loading module for PreFlight v2.0 ingest pipeline.

Loads every cell as a raw string without automatic NA casting or dtype coercion.
Supports chunked reading for large datasets without loading the full file into memory.
"""

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd

from .dialect import DialectResult, detect_dialect
from .encoding import EncodingResult, detect_encoding
from .header import HeaderResult, detect_header


@dataclass(frozen=True)
class RawLoadResult:
    """Typed result object for raw tabular data loading."""
    data: Union[pd.DataFrame, Iterator[pd.DataFrame]]
    row_count: Optional[int]
    column_count: int
    column_names: List[str]
    is_chunked: bool
    ragged_rows_found: int


def _clean_chunk(
    chunk: pd.DataFrame,
    column_names: List[str],
) -> pd.DataFrame:
    """Ensure chunk DataFrame strictly contains string data without NaN."""
    # Ensure columns match expected names
    if len(chunk.columns) == len(column_names):
        chunk.columns = column_names
    # Convert all object or string columns to pure python str without NaN
    chunk = chunk.fillna("")
    return chunk.astype(str)


def load_raw(
    source: Union[str, Path],
    encoding: Optional[str] = None,
    dialect: Optional[Union[DialectResult, Dict[str, Any]]] = None,
    header_result: Optional[HeaderResult] = None,
    chunksize: Optional[int] = None,
) -> RawLoadResult:
    """
    Load tabular file into memory or as chunked iterator of raw string DataFrames.

    No automatic NA conversion is performed; all values remain raw strings
    (equivalent to keep_default_na=False, dtype=str).

    Args:
        source: Path to tabular file.
        encoding: Text encoding name. If None, auto-detected.
        dialect: DialectResult or dict of dialect parameters. If None, auto-detected.
        header_result: HeaderResult object. If None, auto-detected.
        chunksize: Number of rows per chunk for streaming. If None, loads full table.

    Returns:
        RawLoadResult containing DataFrame (or chunk generator) and structural metadata.
    """
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    # 1. Resolve encoding if omitted
    enc_name: str
    if encoding is None:
        enc_res = detect_encoding(p)
        enc_name = enc_res.encoding
    else:
        enc_name = encoding

    # 2. Resolve dialect if omitted
    delim = ","
    qchar = '"'
    esc_char = None
    double_q = True

    if dialect is None:
        d_res = detect_dialect(p, encoding=enc_name)
        delim = d_res.delimiter
        qchar = d_res.quotechar
        esc_char = d_res.escapechar
        double_q = d_res.doublequote
    elif isinstance(dialect, DialectResult):
        delim = dialect.delimiter
        qchar = dialect.quotechar
        esc_char = dialect.escapechar
        double_q = dialect.doublequote
    elif isinstance(dialect, dict):
        delim = dialect.get("delimiter", ",")
        qchar = dialect.get("quotechar", '"')
        esc_char = dialect.get("escapechar", None)
        double_q = dialect.get("doublequote", True)

    # 3. Resolve header and table bounds if omitted
    if header_result is None:
        h_res = detect_header(
            p,
            delimiter=delim,
            quotechar=qchar,
            escapechar=esc_char,
            doublequote=double_q,
            encoding=enc_name,
        )
    else:
        h_res = header_result

    # Configure pandas read_csv arguments
    skiprows = h_res.title_rows if h_res.title_rows else None
    col_names = h_res.column_names

    read_csv_kwargs: Dict[str, Any] = {
        "filepath_or_buffer": p,
        "sep": delim,
        "encoding": enc_name,
        "quotechar": qchar,
        "doublequote": double_q,
        "escapechar": esc_char,
        "dtype": str,
        "keep_default_na": False,
        "na_values": [],
        "skip_blank_lines": True,
        "skiprows": skiprows,
        "on_bad_lines": "warn",
    }

    if h_res.header_row_index is not None:
        # Preamble rows were skipped via skiprows, so header row is at row 0 of remaining data
        read_csv_kwargs["header"] = 0
    else:
        # Headerless dataset
        read_csv_kwargs["header"] = None
        read_csv_kwargs["names"] = col_names

    # Check for bare carriage return line endings (Mac OS 9 style \r without \n)
    # Pandas C parser fails with Buffer Overflow on bare \r, so python engine is required.
    try:
        with open(p, "rb") as bf:
            sample_bytes = bf.read(4096)
            if b"\r" in sample_bytes and b"\n" not in sample_bytes:
                read_csv_kwargs["engine"] = "python"
    except Exception:
        pass

    def _safe_read_csv(kwargs: Dict[str, Any]) -> Any:
        try:
            return pd.read_csv(**kwargs)
        except pd.errors.ParserError:
            fallback = dict(kwargs)
            fallback["engine"] = "python"
            return pd.read_csv(**fallback)

    # 4. Handle chunked reading
    if chunksize is not None and chunksize > 0:
        read_csv_kwargs["chunksize"] = chunksize
        reader_iter = _safe_read_csv(read_csv_kwargs)

        def chunk_generator() -> Iterator[pd.DataFrame]:
            try:
                for chunk in reader_iter:
                    yield _clean_chunk(chunk, col_names)
            except pd.errors.ParserError:
                fallback_kwargs = dict(read_csv_kwargs)
                fallback_kwargs["engine"] = "python"
                for chunk in pd.read_csv(**fallback_kwargs):
                    yield _clean_chunk(chunk, col_names)

        return RawLoadResult(
            data=chunk_generator(),
            row_count=None,
            column_count=len(col_names),
            column_names=col_names,
            is_chunked=True,
            ragged_rows_found=len(h_res.ragged_rows),
        )

    # 5. Full file reading
    df = _safe_read_csv(read_csv_kwargs)
    df = _clean_chunk(df, col_names)

    # Exclude footer and total rows if detected and present in DataFrame
    if h_res.total_rows or h_res.footer_rows:
        rows_to_drop = set(h_res.total_rows + h_res.footer_rows)
        # Adjust indices if title rows were skipped
        skipped_offset = len(h_res.title_rows) + (1 if h_res.header_row_index is not None else 0)
        df_drop_indices = [
            r - skipped_offset for r in rows_to_drop if 0 <= r - skipped_offset < len(df)
        ]
        if df_drop_indices:
            df = df.drop(index=df_drop_indices).reset_index(drop=True)

    return RawLoadResult(
        data=df,
        row_count=len(df),
        column_count=len(df.columns),
        column_names=list(df.columns),
        is_chunked=False,
        ragged_rows_found=len(h_res.ragged_rows),
    )
