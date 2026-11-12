"""
Schema definition module for PreFlight v2.0.

Provides versioned schema dataclasses representing structural ingestion metadata:
- Detected encoding
- Detected dialect
- Detected header and table boundaries
- Per-column raw metadata (name, position)
- User-override placeholders for future pipeline phases (Phase 3 / Phase 5).

Supports lossless JSON round-trip serialization and deserialization.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from preflight.decisions import Decision

SCHEMA_VERSION = "2.0.0"


@dataclass
class EncodingSchema:
    """Encoding detection metadata and override shape."""
    detected: str = "utf-8"
    confidence: float = 1.0
    has_bom: bool = False
    low_confidence: bool = False
    raw_detected: Optional[str] = None
    override: Optional[str] = None

    def effective_encoding(self) -> str:
        """Return user override if specified, otherwise detected encoding."""
        return self.override if self.override is not None else self.detected


@dataclass
class DialectSchema:
    """Dialect detection metadata and override shape."""
    delimiter: str = ","
    quotechar: str = '"'
    escapechar: Optional[str] = None
    doublequote: bool = True
    confidence: float = 1.0
    flagged: bool = False
    margin: float = 1.0
    delimiter_override: Optional[str] = None
    quotechar_override: Optional[str] = None
    escapechar_override: Optional[str] = None

    def effective_delimiter(self) -> str:
        """Return user delimiter override if specified, otherwise detected delimiter."""
        return self.delimiter_override if self.delimiter_override is not None else self.delimiter


@dataclass
class HeaderSchema:
    """Header and table boundary metadata and override shape."""
    header_row_index: Optional[int] = 0
    confidence: float = 1.0
    title_rows: list[int] = field(default_factory=list)
    footer_rows: list[int] = field(default_factory=list)
    blank_rows: list[int] = field(default_factory=list)
    total_rows: list[int] = field(default_factory=list)
    ragged_rows: list[int] = field(default_factory=list)
    is_ragged: bool = False
    flagged: bool = False
    header_row_override: Optional[int] = None

    def effective_header_row(self) -> Optional[int]:
        """Return header row override if specified, otherwise detected header row."""
        return self.header_row_override if self.header_row_override is not None else self.header_row_index


@dataclass
class ColumnRawMetadata:
    """
    Column metadata supporting structural ingest, type inference, and user overrides.
    """
    name: str
    position: int
    name_override: Optional[str] = None
    type_override: Optional[str] = None

    # Phase 3: Type inference, confidence, missing markers, anomalies, and samples
    inferred_type: Optional[str] = None
    confidence: float = 1.0
    missing_markers: list[dict[str, Any]] = field(default_factory=list)
    runner_up: Optional[str] = None
    abstained: bool = False
    anomalies: list[str] = field(default_factory=list)
    failed_samples: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def effective_name(self) -> str:
        """Return user name override if specified, otherwise raw column name."""
        return self.name_override if self.name_override is not None else self.name

    def effective_type(self) -> Optional[str]:
        """Return user type override if specified, otherwise inferred type."""
        return self.type_override if self.type_override is not None else self.inferred_type


# Alias ColumnSchema to ColumnRawMetadata for clarity and backwards compatibility
ColumnSchema = ColumnRawMetadata


@dataclass
class TableSchema:
    """
    Versioned table structural and semantic schema for PreFlight v2.0.

    Encapsulates ingest structural properties, type inferences, decisions, and
    provides lossless JSON round-tripping.
    """
    schema_version: str = SCHEMA_VERSION
    source_path: Optional[str] = None
    row_count: Optional[int] = None
    column_count: int = 0
    encoding: EncodingSchema = field(
        default_factory=lambda: EncodingSchema("utf-8", 1.0, False, False)
    )
    dialect: DialectSchema = field(
        default_factory=lambda: DialectSchema(",", '"', None, True, 1.0, False, 1.0)
    )
    header: HeaderSchema = field(
        default_factory=lambda: HeaderSchema(0, 1.0)
    )
    columns: list[ColumnRawMetadata] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    user_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert schema to dictionary representation."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize schema to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableSchema":
        """Reconstruct TableSchema from a dictionary."""
        enc_data = data.get("encoding", {})
        encoding = EncodingSchema(**enc_data) if isinstance(enc_data, dict) else enc_data

        dial_data = data.get("dialect", {})
        dialect = DialectSchema(**dial_data) if isinstance(dial_data, dict) else dial_data

        hdr_data = data.get("header", {})
        header = HeaderSchema(**hdr_data) if isinstance(hdr_data, dict) else hdr_data

        cols_raw = data.get("columns", [])
        columns = [
            ColumnRawMetadata(**c) if isinstance(c, dict) else c for c in cols_raw
        ]

        decs_raw = data.get("decisions", [])
        decisions = [
            Decision.from_dict(d) if isinstance(d, dict) else d for d in decs_raw
        ]

        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            source_path=data.get("source_path"),
            row_count=data.get("row_count"),
            column_count=data.get("column_count", len(columns)),
            encoding=encoding,
            dialect=dialect,
            header=header,
            columns=columns,
            decisions=decisions,
            user_overrides=data.get("user_overrides", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "TableSchema":
        """Deserialize TableSchema from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

