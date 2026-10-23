"""
Hypothesis property test: TableSchema JSON serialization round-trips.

A randomly generated TableSchema (with nested Encoding/Dialect/Header schemas,
column metadata, and Decisions) must survive `to_json()` -> `from_json()`
and compare equal field-for-field.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from preflight.decisions import Decision
from preflight.schema import (
    ColumnRawMetadata,
    DialectSchema,
    EncodingSchema,
    HeaderSchema,
    TableSchema,
)

short_text = st.text(min_size=0, max_size=12, alphabet=st.characters(min_codepoint=32, max_codepoint=126))
col_name = st.text(min_size=1, max_size=10, alphabet="abcdefghij_")


@st.composite
def decision_strategy(draw):
    columns = draw(st.lists(col_name, min_size=0, max_size=3, unique=True))
    return Decision(
        id="",
        stage=draw(st.sampled_from(["reading", "infer", "clean", "engineer"])),
        kind=draw(st.sampled_from(["type_inference", "encoding", "dialect", "header", "ambiguous_task"])),
        columns=columns,
        action=draw(short_text),
        evidence={"k": draw(st.integers(min_value=-100, max_value=100))},
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        default=draw(short_text),
        needs_attention=draw(st.booleans()),
        plain_language=draw(short_text),
    )


@st.composite
def column_meta_strategy(draw, position):
    return ColumnRawMetadata(
        name=draw(col_name),
        position=position,
        name_override=draw(st.one_of(st.none(), col_name)),
        type_override=draw(st.one_of(st.none(), st.sampled_from(["int", "float", "text", "categorical"]))),
        inferred_type=draw(st.one_of(st.none(), st.sampled_from(["int", "float", "text"]))),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        abstained=draw(st.booleans()),
    )


@st.composite
def table_schema_strategy(draw):
    n_cols = draw(st.integers(min_value=0, max_value=5))
    columns = [draw(column_meta_strategy(i)) for i in range(n_cols)]
    decisions = draw(st.lists(decision_strategy(), min_size=0, max_size=4))
    return TableSchema(
        source_path=draw(st.one_of(st.none(), short_text)),
        row_count=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=10_000))),
        column_count=n_cols,
        encoding=EncodingSchema(
            detected=draw(st.sampled_from(["utf-8", "iso-8859-1", "cp1252"])),
            confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            has_bom=draw(st.booleans()),
            low_confidence=draw(st.booleans()),
        ),
        dialect=DialectSchema(
            delimiter=draw(st.sampled_from([",", ";", "\t", "|"])),
            confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            flagged=draw(st.booleans()),
        ),
        header=HeaderSchema(
            header_row_index=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=5))),
            confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            is_ragged=draw(st.booleans()),
            flagged=draw(st.booleans()),
        ),
        columns=columns,
        decisions=decisions,
    )


@given(schema=table_schema_strategy())
@settings(max_examples=50, deadline=1000, suppress_health_check=[HealthCheck.too_slow])
def test_table_schema_json_roundtrip(schema):
    json_str = schema.to_json()
    restored = TableSchema.from_json(json_str)
    assert restored.to_dict() == schema.to_dict()
