# PreFlight v2.0 Type Inference Validation Report

## 1. Executive Summary

This report measures the empirical performance of the **PreFlight v2.0 Type Inference Subsystem** (Phase 3 / M2) evaluated against the 254-file messy-CSV benchmark corpus (`corpus/raw/`) and 2,850 ground-truth labeled columns (`corpus/labels/`).

The v2 type inference subsystem replaces v1's rigid pandas dtype casting with heuristic candidate parse-success-rate scoring across 8 candidate types (`int`, `float`, `boolean`, `datetime`, `categorical`, `text`, `ID`, `constant`), explicit trap detection (leading-zero retention, decimal commas, ambiguous dates, embedded units), disguised missing-value detection, margin-based abstention logic, and deterministic Decision records.

---

## 2. Inferred Type Performance Metrics: v1.0.1 Baseline vs. v2.0

All measurements use the identical 254 corpus files, 2,850 ground-truth labeled columns, and metric definitions established in Phase 1 (M0).

| Metric | v1.0.1 Baseline | v2.0 Type Inference | Count (v1) | Count (v2) | Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Corpus Size (Files)** | 254 | 254 | 254 / 254 | 254 / 254 | — |
| **Corpus Size (Columns)** | 2,850 | 2,850 | 2,850 / 2,850 | 2,850 / 2,850 | — |
| **Column-Type Accuracy** | 80.77% | **91.12%** | 2,302 / 2,850 | 2,597 / 2,850 | **+10.35%** |
| **Wrong-Types-With-No-Flag (% of all cols)** | 9.40% | **0.77%** | 268 / 2,850 | 22 / 2,850 | **-8.63%** |
| **Wrong-Types-With-No-Flag (% of wrong cols)**| 48.91% | **8.70%** | 268 / 548 | 22 / 253 | **-40.21%** |
| **Abstain Rate** | 0.00% | **1.12%** | 0 / 2,850 | 32 / 2,850 | +1.12% |
| **Unhandled Exception Rate** | 3.94% | **0.00%** | 10 / 254 | 0 / 254 | **-3.94%** |

> [!NOTE]
> **Pre-Split Notice**: These metrics are reported against the entire 254-file corpus as-is without parameter tuning against the dataset. The wrong-type-with-no-flag rate of **0.77%** meets the $\le 1.0\%$ project reliability objective prior to any held-out split partitioning.

---

## 3. Trap Handling & Anomaly Analysis

### 3.1 Leading-Zero Retention (IDs, FIPS, ZIP Codes)
- **Problem in v1**: Numeric strings with leading zeros (e.g. `"02138"`, `"01001"`) were coerced to integers (`2138`), stripping the zero and corrupting the identifier without generating any flag.
- **v2 Resolution**: `_parse_int` and `_parse_float` explicitly reject digit strings with leading zeros unless representing zero itself (`"0"`). Anomaly `leading_zero` is recorded, and the column is classified as `ID` with 100% string preservation.

### 3.2 European Decimal Comma vs. Thousands Separators
- **Problem in v1**: Numbers formatted with decimal commas (e.g. `"12,5"`, `"0,61155"`) failed pandas numeric parsing and defaulted to `object`/categorical.
- **v2 Resolution**: Regularity checks detect decimal commas and European thousands separators (`1.234,56`), parsing them accurately as `float` while recording `decimal_comma_detected: True` in the evidence dictionary.

### 3.3 Ambiguous Date Formats (dd/mm vs. mm/dd)
- **Problem in v1**: Dates formatted as `01/02/2020` were silently parsed using Python's default locale, producing erroneous calendar timestamps when European format was intended.
- **v2 Resolution**: If across all slash/dash dates in a column both parts are $\le 12$, v2 flags `ambiguous_date_format` and **abstains** (`abstained=True`, `needs_attention=True`), defaulting safely to string preservation rather than making an unverified guess.

### 3.4 Embedded Units (Currency and Physical Quantities)
- **Problem in v1**: Values like `"15 kmpl"` or `"$12.50"` caused columns to fall back to `object` or had prefixes stripped unpredictably.
- **v2 Resolution**: Pattern detection captures appended/prepended units (`$`, `€`, `kmpl`, `kg`, `%`), records anomaly `embedded_unit: <unit>`, flags `lossy_if_coerced`, and abstains to safe string default until clean/transform rules are confirmed.

### 3.5 Disguised Missing Values
- **Resolution**: `detect_missing_markers` identifies sentinel strings (`""`, `"NA"`, `"N/A"`, `"null"`, `"-"`, `"?"`) and contextually suspicious numeric markers (e.g. `"-1"` in non-negative distributions, `"999"`/`"-999"` isolated far outside distribution bounds). These are recorded in column metadata and excluded from candidate type parse scoring without auto-mutating raw data.

---

## 4. The Decision Record Architecture

PreFlight v2.0 introduces the `Decision` dataclass (`src/preflight/decisions.py`) as the standard communication record between pipeline phases.

### 4.1 Schema
- `id`: Deterministic identifier prefixed with stage and column (`dec_infer_age_3a4b5c6d`).
- `stage`: Pipeline stage (`infer`).
- `kind`: `type_inference` or `type_abstain`.
- `columns`: List of affected column names.
- `action`: Proposed operation (`cast_to_int`, `cast_to_float`, `keep_as_string`).
- `evidence`: Diagnostic candidate scores, parse failure samples, margins, and anomaly flags.
- `confidence`: Numerical score between 0.0 and 1.0.
- `default`: Conservative fallback action (`keep_as_string`).
- `needs_attention`: Boolean flag signaling whether human review is recommended.
- `plain_language`: Human-readable rationale.
- `status`: Lifecycle state (`automatic`, `confirmed`, `changed`).
- `timestamp`: UTC ISO 8601 creation timestamp.
- `fingerprint`: 16-character SHA-256 hash derived strictly from semantic inputs (stage, kind, sorted columns, action, default, normalized evidence).

### 4.2 Fingerprint Determinism
Fingerprints remain 100% identical across pipeline re-runs of the same file regardless of execution time or random process seeds, enabling reliable state caching and user override tracking across pipeline phases.

---

## 5. Verification and Test Results

### 5.1 Unit Test Suite (`tests/unit/test_infer.py`)
- **Status**: 16 passed in 0.37s
- **Coverage**: Hand-built edge cases covering leading-zero IDs, European decimal comma floats, ambiguous date abstention, non-ambiguous date resolution, embedded unit lossy abstention, disguised missing value detection, contextual `-1`/`999` sentinels, close-margin abstention, Decision dataclass shape, deterministic fingerprint stability, and TableSchema JSON roundtrip fidelity.

### 5.2 Corpus Validation Suite (`tests/corpus/test_infer_corpus.py`)
- **Status**: 2 passed in 430.73s
- **Coverage**:
  1. `test_corpus_type_inference_accuracy_and_metrics`: Evaluates all 254 files and 2,850 columns, validating $\ge 85.0\%$ column-type accuracy (achieved **91.12%**) and $\le 2.0\%$ wrong-type-with-no-flag rate (achieved **0.77%**) with 0 crashes.
  2. `test_corpus_decisions_structure_and_fingerprints`: Verifies Decision emissions, 16-character deterministic fingerprints, and lossless schema round-trip serialization on real corpus tables.

### 5.3 Combined Unit & Ingestion Test Execution
```
pytest tests/unit/
============================== 38 passed in 0.39s ==============================
```
- `tests/unit/test_infer.py`: 16 passed
- `tests/unit/test_ingest.py`: 22 passed
