# PreFlight v2.0 Ingest Pipeline Validation Report

## 1. Executive Summary

This report measures the empirical performance of the **PreFlight v2.0 Ingestion Subsystem** (Phase 2 / M1) evaluated against the 254-file messy-CSV benchmark corpus (`corpus/raw/`) and ground-truth labels (`corpus/labels/`).

The v2 ingestion architecture replaces hardcoded assumptions with heuristic candidate scoring based on row-length regularity, BOM and charset-normalizer analysis, structural type-consistency header detection, and strict raw-string tabular loading with chunked streaming support.

## 2. Ingest Performance Metrics: v1.0.1 Baseline vs. v2.0

All measurements use the identical 254 corpus files and ground-truth label definitions established in Phase 1 (M0).

| Metric | v1.0.1 Baseline | v2.0 Ingest | Count (v1) | Count (v2) | Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Dialect Accuracy** | 91.73% | **98.82%** | 233 / 254 | 251 / 254 | +7.09% |
| **Delimiter Detection** | 97.24% | **99.61%** | 247 / 254 | 253 / 254 | +2.37% |
| **Encoding Detection** | 96.06% | **98.82%** | 244 / 254 | 251 / 254 | +2.76% |
| **Header Row Detection** | 98.43% | **100.00%** | 250 / 254 | 254 / 254 | +1.57% |
| **Unhandled Exception Rate** | 3.94% | **0.00%** | 10 / 254 | 0 / 254 | -3.94% |

### Unhandled Exception Breakdown

| Exception Class | v1.0.1 Crashes | v2.0 Crashes | Resolution in v2.0 |
| :--- | :---: | :---: | :--- |
| `UnicodeDecodeError` | 9 | 0 | BOM-first check followed by `charset-normalizer` with fallback to `utf-8` / `iso-8859-1` |
| `ParserError` | 1 | 0 | Bare carriage-return (`\r`) detection + automatic fallback to python parsing engine |
| **Total Crashes** | **10** | **0** | **100% crash-free tabular ingestion across all 254 corpus files** |

---

## 3. Discrepancies and Boundary Caveats

### 3.1 Delimiter Detection Boundary: `fl_accents` (`table-with-accents.csv`)
- **Corpus Label**: `,` (Comma)
- **v2 Detected**: `;` (Semicolon)
- **Analysis**:
  - The raw file header is: `Classement;Quartier;Date;Année;Occupancy rate;Average Price;RevPAR;Source`.
  - Data records contain French decimal numbers formatted with commas: `0,611553068;81,20416129;49,66065393;Mkg`.
  - In Phase 1 labeling, `csv.Sniffer` chose `,` because of the high frequency of decimal commas, resulting in a single composite column label.
  - v2.0 regularity scoring correctly identified `;` as the structural field delimiter (regularity score 1.0, 8 columns).
  - v2 flagged the close margin between `,` and `;` (`margin=0.152`), satisfying the requirement to flag low-margin candidate winners.

### 3.2 Single-Byte Encoding Variants
Across the 254 files, `charset-normalizer` identified specific single-byte variants for 3 legacy files labeled as generic `iso-8859-1`:
- `fl_accents`: detected as `mac-latin2` (decoded cleanly without exception).
- `fte_biopics_biopics`: detected as `hp-roman8` (decoded cleanly without exception).
- `fte_food_world_cup_food_world_cup_data`: detected as `hp-roman8` (decoded cleanly without exception).

All bytes decode with zero replacement characters or decode errors, resolving the fatal `UnicodeDecodeError` crashes that occurred under v1.0.1.

### 3.3 Mac OS 9 Carriage Return Line Endings: `fte_classic_rock_classic_rock_raw_data`
- **File**: `corpus/raw/classic-rock-raw-data.csv` (37,673 rows).
- **Issue in v1**: Line endings consist of bare `\r` (CR) without `\n` (LF). Pandas C engine parser buffers bytes seeking `\n`, overflowing its buffer and crashing with `ParserError: Buffer overflow caught`.
- **v2 Handling**: `loader.py` inspects the initial byte stream for bare `\r` and falls back to `engine="python"`, streaming and parsing all 37,673 rows and 9 columns without error in both full-load and chunked modes.

---

## 4. Verification and Test Results

### 4.1 Unit Test Suite (`tests/unit/test_ingest.py`)
- **Status**: 22 passed in 0.39s
- **Coverage**: Hand-built edge cases covering BOM encodings (UTF-8, UTF-16, UTF-32), delimiter regularity scoring, escape and quote parsing, comment/title preamble skips, ragged row detection, total/summary row exclusion, raw string preservation (no NaN conversions), chunked iteration, and versioned schema JSON round-trip fidelity.

### 4.2 Corpus Validation Suite (`tests/corpus/test_ingest_corpus.py`)
- **Status**: 3 passed in 9.62s
- **Coverage**:
  1. `test_corpus_ingestion_accuracy_and_metrics`: Verifies delimiter (>=99%), encoding (>=98%), header (100%), and dialect (>=98%) accuracy across all 254 files with 0 unhandled exceptions.
  2. `test_all_corpus_files_end_to_end_schema_generation`: Runs `ingest()` on all 254 files, verifying valid `TableSchema` creation and JSON round-trip serialization.
  3. `test_large_file_streaming_chunks`: Verifies chunked reading on large corpus datasets (`classic-rock-raw-data.csv`, `avengers.csv`, `police_killings.csv`) without in-memory full buffering.

### 4.3 Full Repository Test Suite
```
pytest tests/
============================= 537 passed in 13.56s =============================
```
- `tests/corpus/test_ingest_corpus.py`: 3 passed
- `tests/corpus/test_label_integrity.py`: 512 passed
- `tests/unit/test_ingest.py`: 22 passed
