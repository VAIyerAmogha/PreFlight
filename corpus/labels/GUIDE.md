# PreFlight Corpus Labeling Guide (Phase 1 / M0)

## Overview
This document defines the ground-truth labeling standards for the PreFlight messy-CSV benchmark corpus. The corpus is designed to evaluate tabular data ingest and profiling systems, specifically establishing the baseline accuracy of PreFlight v1.0.1 and setting the evaluation benchmark for v2.0.

All corpus metadata and ground-truth labels are stored under `corpus/labels/`:
- `corpus/labels/sources.csv`: Manifest of all acquired datasets, recording file ID, filename, source portal/repository, exact source URL, and license.
- `corpus/labels/<file_id>.json`: Machine-readable ground-truth label for each dataset covering dialect parameters, structural properties, and per-column semantic classifications and anomalies.

---

## Dialect & Structural Schema

Each dataset's label JSON specifies the true physical dialect required to correctly parse the file without data loss:

```json
{
  "file_id": "<string>",
  "filename": "<string>",
  "dialect": {
    "delimiter": "<string>",       // e.g. ",", ";", "\t", "|", " "
    "encoding": "<string>",        // e.g. "utf-8", "utf-8-sig", "iso-8859-1", "cp1252", "ascii"
    "header_row_index": <int|null> // 0 for line 1, >0 if preamble lines exist, null if headerless
  },
  "columns": [
    {
      "column_name": "<string>",
      "column_index": <int>,
      "semantic_type": "<SemanticType>",
      "anomalies": ["<anomaly_tag>", ...]
    }
  ]
}
```

### Dialect Fields
1. **`delimiter`**: The true character separating fields in each record.
   - `","`: Comma (standard RFC 4180 CSV).
   - `";"`: Semicolon (standard European CSV format where comma is the decimal point).
   - `"\t"`: Tab (Tab-Separated Values / TSV).
   - `"|"`: Pipe delimiter.
   - `" "`: Whitespace delimiter.
2. **`encoding`**: The true character encoding of the raw byte stream.
   - `"utf-8"`: UTF-8 without byte-order mark.
   - `"utf-8-sig"`: UTF-8 with leading Byte Order Mark (`\xef\xbb\xbf`).
   - `"iso-8859-1"` / `"latin-1"`: Western European single-byte encoding.
   - `"cp1252"` / `"windows-1252"`: Windows Western European encoding.
   - `"ascii"`: 7-bit US-ASCII subset.
3. **`header_row_index`**: The 0-based row index containing the column names.
   - `0`: Header is on the very first row (standard).
   - `1`, `2`, ...: File begins with metadata/preamble comments or title rows before the header.
   - `null`: Dataset has no header row; row 0 is already data (e.g. classic UCI `.data` files).

---

## Semantic Type Definitions

Every column is assigned exactly one primary ground-truth semantic type from the canonical 7 types:

| Semantic Type | Definition | Typical Examples |
| :--- | :--- | :--- |
| **`numeric`** | Continuous or discrete mathematical quantities, measurements, counts, currency amounts, percentages, rates. Mathematical operations (mean, standard deviation, scaling) are meaningful. | Age, income, price, temperature, counts, duration, petal length, percentage |
| **`categorical`** | Discrete variables taking values from a fixed or bounded set of categories or qualitative states. May be low or high cardinality; may be text labels or encoded integer codes. | Gender, blood type, species, country, department, marital status, grade |
| **`datetime`** | Temporal values representing calendar dates, times of day, or timestamps. | `2021-04-15`, `04/15/2021`, `15-Apr-2021`, `2020-01-01 14:30:00`, `1994-06` |
| **`text`** | Unstructured, free-form natural language text with high vocabulary entropy and variable length. | User reviews, survey comments, descriptions, book/movie titles, abstracts |
| **`boolean`** | Binary two-state flags indicating truth, presence, or membership. | `True`/`False`, `1`/`0`, `Yes`/`No`, `Y`/`N`, `T`/`F` |
| **`constant`** | Zero-variance columns where all non-null values are identical, or the column is entirely null. Contains no discriminatory information. | All-zero flags, constant country in single-nation survey, all-empty fields |
| **`ID`** | Keys, surrogate tokens, or entity identifiers whose purpose is identification or linking rather than mathematical measurement. | Database primary keys, auto-increment `rownames`, UUIDs, postal/ZIP codes, FIPS codes |

### Disambiguation Rules
- **Numeric ID vs. Numeric Feature**: If a numeric column represents an identity or serial order where computing an average or standard deviation is semantically meaningless (e.g. employee ID, row index `rownames`, postal code `02138`, FIPS code `01001`), it is labeled **`ID`**, not `numeric`.
- **Boolean vs. Categorical**: If a column has strictly two states representing affirmative/negative or binary switch (`Yes`/`No`, `True`/`False`, `1`/`0`), it is labeled **`boolean`**. If it has two states representing distinct classes (e.g. `Male`/`Female`), it is labeled **`categorical`**.
- **Categorical vs. Text**: Short fixed titles or categorical strings with repetition across rows are **`categorical`**. Sentences, paragraphs, or long descriptive fields with average word count $\ge 5$ and average character length $> 30$ are **`text`**.
- **Date vs. Numeric Year**: A column with full calendar dates (`YYYY-MM-DD`, `MM/DD/YYYY`) is **`datetime`**. An isolated 4-digit calendar year (e.g. `1994`, `2020`) used as a numerical timeline variable or grouped integer may be labeled **`numeric`** or **`datetime`** depending on usage context.

---

## Anomaly Tags

Each column record lists zero or more anomaly tags indicating real-world messiness present in that column:

- `disguised_missing`: Missingness represented by sentinel strings rather than standard empty fields. Includes:
  - `disguised_missing: ?`
  - `disguised_missing: NA` / `disguised_missing: N/A`
  - `disguised_missing: null` / `disguised_missing: NULL` / `disguised_missing: None`
  - `disguised_missing: -` / `disguised_missing: --`
  - `disguised_missing: 999` / `disguised_missing: -999`
  - `disguised_missing: unknown`
- `embedded_unit`: Numbers containing appended/prepended units or formatting symbols:
  - `embedded_unit: $` (currency prefix)
  - `embedded_unit: €` / `embedded_unit: £`
  - `embedded_unit: %` (percentage suffix)
  - `embedded_unit: kg` / `embedded_unit: km` / `embedded_unit: mph` / `embedded_unit: years`
- `leading_zero`: Numeric digit string where leading zeroes are semantically significant (e.g., FIPS codes `"01001"`, postal codes `"02138"`).
- `ambiguous_date_format`: Dates where day and month order cannot be determined from values alone without locale (e.g. `03/04/2020`).
- `mixed_date_format`: A single column containing mixed date representations across records.
- `quoted_newline`: Field contains line breaks enclosed in quotes.
- `escaped_quote`: Field contains embedded or escaped quote characters.
- `ragged_row`: File records exhibit varying field counts.
- `semicolon_delimited`: Dataset uses `;` delimiter instead of standard comma.
- `tab_delimited`: Dataset uses `\t` delimiter instead of standard comma.
- `non_utf8_encoding`: Dataset byte stream is encoded in Latin-1 / ISO-8859-1 or CP1252.
- `headerless`: Dataset has no column header row.
- `preamble_header`: Dataset contains comment/metadata rows preceding the header row.

---

## Mapping v1.0.1 `SemanticType` to Canonical Ground-Truth Types

PreFlight v1.0.1 uses an internal enum with 9 values in `preflight.types.SemanticType`. For baseline evaluation, v1.0.1's outputs are mapped to the 7 canonical ground-truth types:

| v1.0.1 `SemanticType` | Canonical Ground-Truth Type |
| :--- | :--- |
| `NUMERIC_FEATURE` | `numeric` |
| `NUMERIC_ID` | `ID` |
| `CATEGORICAL_LOW` | `categorical` |
| `CATEGORICAL_HIGH` | `categorical` |
| `DATETIME_NATIVE` | `datetime` |
| `DATETIME_STRING` | `datetime` |
| `BOOLEAN` | `boolean` |
| `CONSTANT` | `constant` |
| `TEXT` | `text` |

### Measurement Criteria
- **Dialect Accuracy**: Evaluates whether v1.0.1 correctly handled the file's delimiter, encoding, and header index. Since v1.0.1 relies exclusively on `pandas.read_csv()` default arguments (`sep=','`, `encoding='utf-8'`, `header=0`), non-comma, non-UTF-8, or non-standard header files fail this check.
- **Column-Type Accuracy**: Evaluates whether the mapped v1.0.1 inferred type matches the ground-truth semantic type for each column. If v1.0.1 crashed on the file, all columns in that file are scored as incorrect.
- **Wrong-Type-With-No-Flag**: Evaluates cases where v1.0.1 misclassified a column without generating any diagnostic report entry or warning flag (e.g. treating an ID with leading zeroes as numeric, or collapsing semicolon-separated rows into a single text column).
- **Unhandled Exceptions**: Evaluates the proportion of corpus files on which v1.0.1 raised an uncaught runtime exception (e.g. `UnicodeDecodeError`, `ParserError`, `EmptyDataError`).
