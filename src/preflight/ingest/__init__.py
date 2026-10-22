"""
PreFlight v2.0 Ingest Pipeline Package.

Exposes:
- detect_encoding: Heuristic and BOM character encoding detection.
- detect_dialect: Delimiter and quoting regularity detection.
- detect_header: Header and table boundary detection.
- load_raw: Raw string tabular data loading (with chunking).
- ingest: High-level orchestration function returning raw data and TableSchema.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd

from preflight.schema import (
    ColumnRawMetadata,
    DialectSchema,
    EncodingSchema,
    HeaderSchema,
    TableSchema,
)

from .dialect import DialectResult, detect_dialect
from .encoding import EncodingResult, detect_encoding
from .header import HeaderResult, detect_header
from .loader import RawLoadResult, load_raw

__all__ = [
    "detect_encoding",
    "EncodingResult",
    "detect_dialect",
    "DialectResult",
    "detect_header",
    "HeaderResult",
    "load_raw",
    "RawLoadResult",
    "ingest",
]


def ingest(
    source: Union[str, Path],
    chunksize: Optional[int] = None,
    user_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Union[pd.DataFrame, Iterator[pd.DataFrame]], TableSchema]:
    """
    Run full ingestion pipeline on a tabular data file.

    1. Detects character encoding (with BOM priority).
    2. Detects dialect (delimiter and quoting regularity).
    3. Detects header and table bounds (excluding preambles/footers).
    4. Loads raw string data without NA conversion or dtype coercion.
    5. Builds and returns versioned TableSchema and raw data (or chunk generator).

    Args:
        source: Path to file.
        chunksize: Row chunksize for streaming large datasets.
        user_overrides: Optional dictionary of user overrides.

    Returns:
        Tuple of (DataFrame or chunk generator, TableSchema).
    """
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    overrides = user_overrides or {}

    # 1. Encoding detection (or user override)
    enc_override = overrides.get("encoding")
    if enc_override:
        enc_res = EncodingResult(
            encoding=enc_override,
            confidence=1.0,
            has_bom=False,
            low_confidence=False,
            raw_detected=None,
        )
    else:
        enc_res = detect_encoding(p)

    # 2. Dialect detection (or user override)
    delim_override = overrides.get("delimiter")
    if delim_override:
        dialect_res = DialectResult(
            delimiter=delim_override,
            quotechar=overrides.get("quotechar", '"'),
            escapechar=overrides.get("escapechar", None),
            doublequote=overrides.get("doublequote", True),
            confidence=1.0,
            flagged=False,
            margin=1.0,
            field_count_mode=0,
            row_regularity=1.0,
        )
    else:
        dialect_res = detect_dialect(p, encoding=enc_res.encoding)

    # 3. Header detection (or user override)
    header_res = detect_header(
        p,
        delimiter=dialect_res.delimiter,
        quotechar=dialect_res.quotechar,
        escapechar=dialect_res.escapechar,
        doublequote=dialect_res.doublequote,
        encoding=enc_res.encoding,
    )

    if "header_row_index" in overrides:
        # Override header row
        h_idx = overrides["header_row_index"]
        header_res = HeaderResult(
            header_row_index=h_idx,
            column_names=header_res.column_names,
            confidence=1.0,
            title_rows=header_res.title_rows,
            footer_rows=header_res.footer_rows,
            blank_rows=header_res.blank_rows,
            total_rows=header_res.total_rows,
            ragged_rows=header_res.ragged_rows,
            is_ragged=header_res.is_ragged,
            flagged=False,
        )

    # 4. Load raw data
    load_res = load_raw(
        p,
        encoding=enc_res.encoding,
        dialect=dialect_res,
        header_result=header_res,
        chunksize=chunksize,
    )

    # 5. Build TableSchema
    cols_meta = [
        ColumnRawMetadata(
            name=name,
            position=idx,
            name_override=overrides.get("column_names", {}).get(name),
        )
        for idx, name in enumerate(load_res.column_names)
    ]

    schema = TableSchema(
        source_path=str(p.resolve()),
        row_count=load_res.row_count,
        column_count=load_res.column_count,
        encoding=EncodingSchema(
            detected=enc_res.encoding,
            confidence=enc_res.confidence,
            has_bom=enc_res.has_bom,
            low_confidence=enc_res.low_confidence,
            raw_detected=enc_res.raw_detected,
            override=enc_override,
        ),
        dialect=DialectSchema(
            delimiter=dialect_res.delimiter,
            quotechar=dialect_res.quotechar,
            escapechar=dialect_res.escapechar,
            doublequote=dialect_res.doublequote,
            confidence=dialect_res.confidence,
            flagged=dialect_res.flagged,
            margin=dialect_res.margin,
            delimiter_override=delim_override,
        ),
        header=HeaderSchema(
            header_row_index=header_res.header_row_index,
            confidence=header_res.confidence,
            title_rows=header_res.title_rows,
            footer_rows=header_res.footer_rows,
            blank_rows=header_res.blank_rows,
            total_rows=header_res.total_rows,
            ragged_rows=header_res.ragged_rows,
            is_ragged=header_res.is_ragged,
            flagged=header_res.flagged,
            header_row_override=overrides.get("header_row_index"),
        ),
        columns=cols_meta,
        user_overrides=overrides,
    )

    return load_res.data, schema
