# PreFlight v1.0.1 Baseline Measurement Report

## 1. Executive Summary

This report establishes the empirical performance baseline of **PreFlight v1.0.1** evaluated against the newly constructed messy-CSV benchmark corpus (Phase 1 / M0). v1.0.1 was executed without modification directly from the verified install path.

> [!NOTE]
> This phase is strictly measurement-focused. No v2 pipeline code or optimizations have been implemented or claimed.

### Primary Baseline Metrics

| Metric | v1.0.1 Baseline | Count / Total | Definition |
| :--- | :--- | :--- | :--- |
| **Corpus Size (Files)** | **254** | 254 / 200 target | Number of real-world datasets in corpus |
| **Corpus Size (Columns)** | **2,850** | 2850 / 1,000 target | Total ground-truth labeled columns |
| **Dialect Accuracy** | **91.73%** | 233 / 254 | Exact match on delimiter, encoding, header row |
| **Column-Type Accuracy** | **80.77%** | 2,302 / 2,850 | Correct inferral of canonical semantic type |
| **Unhandled Exception Rate** | **3.94%** | 10 / 254 | Crashes during ingestion or profiling |
| **Wrong-Types-With-No-Flag (% of wrong)** | **48.91%** | 268 / 548 | Misclassifications that emitted zero alerts/flags |
| **Wrong-Types-With-No-Flag (% of all cols)** | **9.40%** | 268 / 2,850 | Silent error rate across entire corpus |

### Dialect Sub-Component Breakdown

| Dialect Property | Accuracy | Match Count | Failure Mode in v1.0.1 |
| :--- | :--- | :--- | :--- |
| **Delimiter Detection** | 97.24% | 247 / 254 | Hardcoded to `,`; collapses `;` and `\t` tables into single text column |
| **Encoding Detection** | 96.06% | 244 / 254 | Hardcoded to UTF-8; raises `UnicodeDecodeError` on Latin-1 / ISO-8859-1 |
| **Header Row Detection** | 98.43% | 250 / 254 | Assumes row 0; corrupts headerless datasets and comment preambles |

## 2. Unhandled Ingestion & Profiling Exceptions

PreFlight v1.0.1 crashed on **10 out of 254 files** (3.94%), raising uncaught exceptions before completing profile generation:

| Exception Class | Count | Typical Root Cause |
| :--- | :--- | :--- |
| `UnicodeDecodeError` | 9 | Non-UTF8 byte stream encountered |
| `ParserError` | 1 | Ragged rows or malformed field structure |

## 3. Per-File Failure Analysis

The table below details all files where v1.0.1 encountered fatal crashes or failed dialect detection:

| File ID | Filename | Dialect Correct? | Exception? | Exception Details / Failure Mode |
| :--- | :--- | :---: | :---: | :--- |
| `fl_accents` | `table-with-accents.csv` | No | `UnicodeDecodeError` | 'utf-8' codec can't decode byte 0xe9 in position 28: invalid continuation byte |
| `fl_bom` | `bom.csv` | No | None | Dialect mismatch (e.g. non-comma delimiter or non-standard header) |
| `fl_delimiter_custom` | `table-delimiter.csv` | No | None | Dialect mismatch (e.g. non-comma delimiter or non-standard header) |
| `fl_latin1` | `latin1.csv` | No | `UnicodeDecodeError` | 'utf-8' codec can't decode byte 0xa9 in position 20: invalid start byte |
| `fl_semicolon` | `issue-1009-semicolon.csv` | No | None | Dialect mismatch (e.g. non-comma delimiter or non-standard header) |
| `fl_skip_rows` | `skip-rows-before-headers.csv` | No | None | Dialect mismatch (e.g. non-comma delimiter or non-standard header) |
| `fl_tsv` | `table.tsv` | No | None | Dialect mismatch (e.g. non-comma delimiter or non-standard header) |
| `fte_avengers_avengers` | `avengers.csv` | No | `UnicodeDecodeError` | 'utf-8' codec can't decode byte 0xe6 in position 5764: invalid continuation byte |
| `fte_biopics_biopics` | `biopics.csv` | No | `UnicodeDecodeError` | 'utf-8' codec can't decode byte 0xcc in position 2960: invalid continuation byte |
| `fte_classic_rock_classic_rock_raw_data` | `classic-rock-raw-data.csv` | Yes | `ParserError` | Error tokenizing data. C error: Buffer overflow caught - possible malformed input file.  |
| `fte_food_world_cup_food_world_cup_data` | `food-world-cup-data.csv` | No | `UnicodeDecodeError` | 'utf-8' codec can't decode byte 0xca in position 76: invalid continuation byte |
| `fte_police_killings_police_killings` | `police_killings.csv` | No | `UnicodeDecodeError` | 'utf-8' codec can't decode byte 0xcc in position 39773: invalid continuation byte |
| `fte_star_wars_survey_starwars` | `StarWars.csv` | No | `UnicodeDecodeError` | 'utf-8' codec can't decode byte 0x8c in position 641: invalid start byte |
| `tt_2018_2018_04_23_week4_australian_salary` | `week4_australian_salary.csv` | No | `UnicodeDecodeError` | 'utf-8' codec can't decode byte 0x96 in position 145: invalid start byte |
| `tt_2018_2018_05_14_week7_starwars` | `week7_starwars.csv` | No | `UnicodeDecodeError` | 'utf-8' codec can't decode byte 0x8c in position 641: invalid start byte |
| `uci_adult` | `adult.data` | No | None | Dialect mismatch (e.g. non-comma delimiter or non-standard header) |
| `uci_automobile` | `imports-85.data` | No | None | Dialect mismatch (e.g. non-comma delimiter or non-standard header) |
| `uci_bank_marketing` | `bank.csv` | No | None | Dialect mismatch (e.g. non-comma delimiter or non-standard header) |
| `uci_bank_marketing_full` | `bank-full.csv` | No | None | Dialect mismatch (e.g. non-comma delimiter or non-standard header) |
| `uci_iris` | `iris.data` | No | None | Dialect mismatch (e.g. non-comma delimiter or non-standard header) |
| `uci_wine_red` | `winequality-red.csv` | No | None | Dialect mismatch (e.g. non-comma delimiter or non-standard header) |
| `uci_wine_white` | `winequality-white.csv` | No | None | Dialect mismatch (e.g. non-comma delimiter or non-standard header) |

## 4. Column Classification Failure Modes

Analysis of the misclassified columns identifies the primary structural weaknesses in v1.0.1:

1. **Silent Semicolon / TSV Corruption**: Files with `;` or `\t` delimiters are parsed by `pandas.read_csv` into a single giant column. v1.0.1 profiles this single string as `TEXT` without flagging that delimiter parsing failed.
2. **Disguised Missing Value Marker Confusion**: Sentinel markers such as `?`, `NA`, or `-` in numeric columns cause pandas to infer `object` dtype. v1.0.1 then categorizes these continuous features as `CATEGORICAL_LOW` or `CATEGORICAL_HIGH`.
3. **Leading-Zero ID Loss**: Identifiers such as US ZIP codes (`02138`) or FIPS codes (`01001`) with leading zeros are parsed as integers by pandas and inferred as `NUMERIC_FEATURE` without maintaining string zero preservation.
4. **Headerless Data Truncation**: For datasets without headers, row 0 data is consumed as column header labels, leading to lost data records and corrupted column names.

---
*Report generated automatically by `scripts/measure_v1_baseline.py` in Phase 1 (M0).*
