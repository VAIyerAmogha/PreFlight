"""
Unit test suite for PreFlight v2.0 type inference subsystem (Phase 3 / M2).

Tests hand-built edge cases and trap scenarios:
- Leading zeros: postal codes and IDs (don't coerce to int or float, preserve zeroes, flag anomaly)
- Decimal comma: European floats (e.g. '12,5', '1.234,56') correctly parsed as float
- Ambiguous dates: dd/mm vs mm/dd ambiguity flagged and abstained
- Non-ambiguous dates: clear dd/mm or mm/dd resolved cleanly
- Embedded units: text/numeric mix (e.g. '15 kmpl', '$12.50') flagged lossy-if-coerced and abstained
- Disguised missing values: standard sentinels ('NA', '?', '-', 'null') and context-dependent '-1'/'999'
- Contextual '-1': detected in non-negative column, ignored in negative-spanning column
- Margin scoring and abstain: close candidate scores force abstention with needs_attention=True
- Decision record structure and deterministic fingerprint stability across runs
- TableSchema versioned serialization and lossless JSON roundtrip
"""

import json
import time
import pytest
import pandas as pd

from preflight.decisions import Decision, compute_decision_fingerprint
from preflight.schema import TableSchema, ColumnRawMetadata
from preflight.infer.missing import detect_missing_markers, MissingMarker
from preflight.infer.types import (
    infer_column_type,
    infer_table_types,
    to_canonical_type,
    TypeInferenceResult,
)


# =====================================================================
# 1. Missing-Value Marker Detection Tests
# =====================================================================

def test_missing_markers_standard_sentinels():
    """Verify standard string sentinels are detected with high confidence."""
    col = ["10", "20", "NA", "30", "null", "?", "-"]
    markers = detect_missing_markers(col, "sample_col")
    marker_strs = {m.marker for m in markers}
    assert "NA" in marker_strs
    assert "null" in marker_strs
    assert "?" in marker_strs
    assert "-" in marker_strs

    for m in markers:
        assert m.confidence >= 0.90
        assert m.count >= 1
        assert m.frequency > 0.0


def test_missing_markers_contextual_negative_one():
    """Verify '-1' is detected in strictly non-negative column but not in negative distribution."""
    # Strictly non-negative column (e.g. age, counts, years)
    pos_col = ["10", "25", "30", "-1", "40", "-1"]
    pos_markers = detect_missing_markers(pos_col, "age")
    assert any(m.marker == "-1" for m in pos_markers)
    pos_m1 = next(m for m in pos_markers if m.marker == "-1")
    assert pos_m1.confidence >= 0.80
    assert "non-negative" in pos_m1.reason

    # Column with legitimate negative values (e.g. temperatures, profit deltas)
    neg_col = ["-15", "-10", "-5", "-1", "0", "5", "10"]
    neg_markers = detect_missing_markers(neg_col, "temperature_delta")
    assert not any(m.marker == "-1" for m in neg_markers)


def test_missing_markers_contextual_extreme_positive():
    """Verify '999' is detected when rest of distribution is bounded far below."""
    col = ["18", "25", "34", "999", "45", "999"]
    markers = detect_missing_markers(col, "age")
    assert any(m.marker == "999" for m in markers)
    m999 = next(m for m in markers if m.marker == "999")
    assert m999.confidence >= 0.85
    assert "far above" in m999.reason


def test_missing_markers_column_specific_outlier():
    """Verify isolated non-numeric token in predominantly numeric column is flagged."""
    col = ["100", "105", "110", "ERR", "120", "125", "130", "135", "140", "145"]
    markers = detect_missing_markers(col, "sensor_reading")
    assert any(m.marker == "ERR" for m in markers)
    err_marker = next(m for m in markers if m.marker == "ERR")
    assert err_marker.confidence >= 0.80
    assert "column_specific_outlier" in err_marker.reason


# =====================================================================
# 2. Known Trap Scenarios & Type Scoring Tests
# =====================================================================

def test_trap_leading_zero_ids():
    """Verify postal codes and IDs with leading zeros reject integer/float coercion and infer ID."""
    zip_codes = ["02138", "02139", "02140", "02141", "02142"]
    res = infer_column_type(zip_codes, "zip_code")
    assert res.inferred_type == "ID"
    assert res.abstained is False
    assert "leading_zero" in res.anomalies
    assert res.confidence >= 0.90
    assert res.evidence["candidate_scores"]["int"] == 0.0
    assert res.evidence["candidate_scores"]["float"] == 0.0


def test_trap_decimal_comma_floats():
    """Verify European decimal comma numbers (e.g. '12,5') are parsed as float."""
    european_floats = ["12,5", "14,8", "9,2", "100,05", "0,6115"]
    res = infer_column_type(european_floats, "revpar")
    assert res.inferred_type == "float"
    assert res.abstained is False
    assert res.confidence == 1.0
    assert res.evidence["decimal_comma_detected"] is True


def test_trap_ambiguous_dates_abstains():
    """Verify dd/mm vs mm/dd ambiguity flags 'ambiguous_date_format' and abstains."""
    ambig_dates = ["01/02/2020", "03/04/2020", "05/06/2020", "07/08/2020", "09/10/2020"]
    res = infer_column_type(ambig_dates, "event_date")
    assert res.abstained is True
    assert res.needs_attention is True
    assert res.inferred_type == "text"
    assert "ambiguous_date_format" in res.anomalies
    assert res.decision.kind == "type_abstain"
    assert res.decision.default == "keep_as_string"


def test_resolved_dates_does_not_abstain():
    """Verify dates with clear dd/mm (day > 12) infer datetime cleanly."""
    clear_dates = ["15/01/2020", "28/02/2020", "31/03/2020", "04/05/2020"]
    res = infer_column_type(clear_dates, "invoice_date")
    assert res.inferred_type == "datetime"
    assert res.abstained is False
    assert "ambiguous_date_format" not in res.anomalies


def test_trap_embedded_units_abstains():
    """Verify embedded units (e.g. '15 kmpl', '$12.50') flag lossy-if-coerced and abstain."""
    unit_values = ["15 kmpl", "18 kmpl", "22 kmpl", "14 kmpl"]
    res = infer_column_type(unit_values, "fuel_efficiency")
    assert res.abstained is True
    assert res.needs_attention is True
    assert res.inferred_type == "text"
    assert any("embedded_unit" in a for a in res.anomalies)
    assert res.decision.kind == "type_abstain"
    assert res.decision.default == "keep_as_string"


def test_margin_scoring_mixed_column_abstains():
    """Verify column with close competitor scores (e.g. 50% numbers, 50% words) abstains."""
    mixed_values = ["10", "20", "apple", "banana"]
    res = infer_column_type(mixed_values, "mixed_data")
    assert res.abstained is True
    assert res.needs_attention is True
    assert res.inferred_type == "text"
    assert res.decision.kind == "type_abstain"
    assert res.decision.default == "keep_as_string"


def test_boolean_inference():
    """Verify boolean detection on binary true/false, yes/no, and 1/0."""
    tf_col = ["True", "False", "True", "True"]
    assert infer_column_type(tf_col, "is_active").inferred_type == "boolean"

    yn_col = ["yes", "no", "yes", "yes"]
    assert infer_column_type(yn_col, "confirmed").inferred_type == "boolean"

    binary_col = ["1", "0", "1", "1", "0"]
    res = infer_column_type(binary_col, "flag")
    assert res.inferred_type == "boolean"
    assert res.abstained is False


def test_constant_column_inference():
    """Verify zero-variance or all-missing columns infer as constant."""
    const_col = ["Active", "Active", "Active"]
    res = infer_column_type(const_col, "status")
    assert res.inferred_type == "constant"
    assert res.abstained is False

    empty_col = ["", "NA", "null", ""]
    res_empty = infer_column_type(empty_col, "blank")
    assert res_empty.inferred_type == "constant"


# =====================================================================
# 3. Decision Record & Fingerprint Stability Tests
# =====================================================================

def test_decision_record_fields_and_shape():
    """Verify Decision record conforms to spec with all required fields."""
    col = ["10", "20", "30", "40"]
    res = infer_column_type(col, "age")
    dec = res.decision

    assert isinstance(dec, Decision)
    assert dec.id.startswith("dec_infer_age_")
    assert dec.stage == "infer"
    assert dec.kind == "type_inference"
    assert dec.columns == ["age"]
    assert dec.action == "cast_to_int"
    assert isinstance(dec.evidence, dict)
    assert dec.confidence == 1.0
    assert dec.default == "keep_as_string"
    assert dec.needs_attention is False
    assert dec.status == "automatic"
    assert dec.timestamp is not None
    assert dec.fingerprint is not None
    assert len(dec.fingerprint) == 16


def test_decision_fingerprint_deterministic_stability():
    """Verify fingerprint is stable across re-runs despite different execution times."""
    col = ["10", "20", "30", "40"]
    res1 = infer_column_type(col, "age")
    time.sleep(0.01)
    res2 = infer_column_type(col, "age")

    assert res1.decision.fingerprint == res2.decision.fingerprint
    assert res1.decision.id == res2.decision.id
    assert res1.decision.timestamp != res2.decision.timestamp


# =====================================================================
# 4. Schema Extension & Round-Trip Serialization Tests
# =====================================================================

def test_schema_extension_and_json_roundtrip():
    """Verify TableSchema with inferred column types and decisions roundtrips losslessly."""
    df = pd.DataFrame({
        "age": ["25", "30", "NA", "45"],
        "zip": ["02138", "02139", "02140", "02141"],
        "price": ["12,5", "14,8", "9,2", "100,0"],
        "unit": ["15 kmpl", "18 kmpl", "20 kmpl", "12 kmpl"],
    })

    schema = infer_table_types(df)
    assert len(schema.columns) == 4
    assert len(schema.decisions) == 4

    # Convert to JSON and back
    json_str = schema.to_json()
    reconstructed = TableSchema.from_json(json_str)

    assert reconstructed.schema_version == schema.schema_version
    assert len(reconstructed.columns) == len(schema.columns)
    assert len(reconstructed.decisions) == len(schema.decisions)

    # Check column metadata integrity
    col_map = {c.name: c for c in reconstructed.columns}
    assert col_map["age"].inferred_type == "int"
    assert col_map["zip"].inferred_type == "ID"
    assert col_map["price"].inferred_type == "float"
    assert col_map["unit"].abstained is True

    # Exact dictionary match
    assert reconstructed.to_dict() == schema.to_dict()


def test_to_canonical_type_mapping():
    """Verify mapping of candidate types to 7 ground truth canonical types."""
    assert to_canonical_type("int") == "numeric"
    assert to_canonical_type("float") == "numeric"
    assert to_canonical_type("numeric") == "numeric"
    assert to_canonical_type("ID") == "ID"
    assert to_canonical_type("id") == "ID"
    assert to_canonical_type("boolean") == "boolean"
    assert to_canonical_type("datetime") == "datetime"
    assert to_canonical_type("categorical") == "categorical"
    assert to_canonical_type("text") == "text"
    assert to_canonical_type("constant") == "constant"
