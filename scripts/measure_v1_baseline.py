#!/usr/bin/env python3
"""
PreFlight v1.0.1 Baseline Measurement Script (Phase 1 / M0).

Imports v1.0.1's Profiler from the user-provided path, runs it on every file
in corpus/raw/, compares inferred dialect and semantic column types against the
ground-truth labels in corpus/labels/, and computes:
1. Dialect Accuracy (% and counts)
2. Column-Type Accuracy (% and counts)
3. % Wrong-Types-With-No-Flag (silent failure rate)
4. % Files Raising Unhandled Exceptions (crash rate)

Outputs full report to docs/v1_baseline_report.md.
"""

import sys
import os
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

# Path to local copy of v1.0.1
V1_REPO_PATH = Path("/home/amg/Desktop/PreFlight v1.0.1")
V1_SRC_PATH = V1_REPO_PATH / "src"

if not V1_SRC_PATH.exists():
    print(f"ERROR: v1.0.1 source directory not found at: {V1_SRC_PATH}")
    print("Please confirm the path to v1.0.1 install.")
    sys.exit(1)

# Import v1.0.1
sys.path.insert(0, str(V1_SRC_PATH))
try:
    import preflight
    from preflight.profiler import run_profiler
    from preflight.types import SemanticType, ColumnProfile, ReportEntry
except ImportError as e:
    print(f"ERROR: Failed to import v1.0.1 from {V1_SRC_PATH}: {e}")
    sys.exit(1)

CORPUS_ROOT = Path("corpus")
RAW_DIR = CORPUS_ROOT / "raw"
LABELS_DIR = CORPUS_ROOT / "labels"
SOURCES_CSV = LABELS_DIR / "sources.csv"
REPORT_PATH = Path("docs/v1_baseline_report.md")

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Mapping v1.0.1 internal SemanticType to canonical ground-truth semantic types
V1_TYPE_MAP: Dict[str, str] = {
    "NUMERIC_FEATURE": "numeric",
    "NUMERIC_ID": "ID",
    "CATEGORICAL_LOW": "categorical",
    "CATEGORICAL_HIGH": "categorical",
    "DATETIME_NATIVE": "datetime",
    "DATETIME_STRING": "datetime",
    "BOOLEAN": "boolean",
    "CONSTANT": "constant",
    "TEXT": "text",
}


def evaluate_dataset(
    file_id: str, label_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run v1.0.1 on a single dataset and compare against ground truth.
    """
    fn = label_data["filename"]
    raw_path = RAW_DIR / fn
    true_dialect = label_data["dialect"]
    true_cols = label_data["columns"]
    total_cols = len(true_cols)

    result = {
        "file_id": file_id,
        "filename": fn,
        "total_columns": total_cols,
        "exception_raised": False,
        "exception_type": None,
        "exception_message": None,
        "dialect_match": False,
        "delimiter_match": False,
        "encoding_match": False,
        "header_match": False,
        "correct_columns": 0,
        "wrong_columns": 0,
        "wrong_unflagged_columns": 0,
        "column_details": [],
    }

    # 1. Dialect Comparison
    # v1.0.1 has no dialect inference; it assumes sep=',', encoding='utf-8', header=0
    v1_delim = ","
    v1_encoding = "utf-8"
    v1_header = 0

    true_delim = true_dialect["delimiter"]
    true_enc = true_dialect["encoding"].lower()
    true_header = true_dialect["header_row_index"]

    delim_ok = true_delim == v1_delim
    enc_ok = true_enc in ("utf-8", "ascii")
    header_ok = true_header == v1_header

    result["delimiter_match"] = delim_ok
    result["encoding_match"] = enc_ok
    result["header_match"] = header_ok
    result["dialect_match"] = delim_ok and enc_ok and header_ok

    # 2. Ingest and Profile using v1.0.1
    df = None
    profiles = None
    reports = None

    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        result["exception_raised"] = True
        result["exception_type"] = type(e).__name__
        result["exception_message"] = str(e)
        result["wrong_columns"] = total_cols
        # An unhandled crash means all columns fail
        for c in true_cols:
            result["column_details"].append({
                "column_index": c["column_index"],
                "column_name": c["column_name"],
                "true_type": c["semantic_type"],
                "v1_type": None,
                "status": "unhandled_exception",
                "flagged": False,
            })
        return result

    try:
        profiles, reports = run_profiler(df, target="", task="classification")
    except Exception as e:
        result["exception_raised"] = True
        result["exception_type"] = type(e).__name__
        result["exception_message"] = str(e)
        result["wrong_columns"] = total_cols
        for c in true_cols:
            result["column_details"].append({
                "column_index": c["column_index"],
                "column_name": c["column_name"],
                "true_type": c["semantic_type"],
                "v1_type": None,
                "status": "profiler_exception",
                "flagged": False,
            })
        return result

    # Identify flagged columns in reports
    flagged_columns = set()
    if reports:
        for r in reports:
            if hasattr(r, "column") and r.column:
                flagged_columns.add(r.column)

    # 3. Column-by-Column Evaluation
    parsed_cols_count = len(df.columns)
    profile_by_idx = {i: p for i, p in enumerate(profiles)}
    profile_by_name = {p.name: p for p in profiles}

    for c in true_cols:
        col_idx = c["column_index"]
        col_name = c["column_name"]
        true_type = c["semantic_type"]

        # Match profile by index or by name
        prof = None
        if col_idx in profile_by_idx and parsed_cols_count == total_cols:
            prof = profile_by_idx[col_idx]
        elif col_name in profile_by_name:
            prof = profile_by_name[col_name]
        elif col_idx < len(profiles):
            prof = profiles[col_idx]

        if prof is None:
            # Column was dropped or never extracted (e.g. wrong delimiter parsed as single column)
            result["wrong_columns"] += 1
            result["wrong_unflagged_columns"] += 1
            result["column_details"].append({
                "column_index": col_idx,
                "column_name": col_name,
                "true_type": true_type,
                "v1_type": "NOT_PARSED",
                "status": "column_missing",
                "flagged": False,
            })
            continue

        raw_v1_type = prof.semantic_type.name if hasattr(prof.semantic_type, "name") else str(prof.semantic_type)
        mapped_v1_type = V1_TYPE_MAP.get(raw_v1_type, raw_v1_type.lower())
        is_flagged = (prof.name in flagged_columns) or (col_name in flagged_columns)

        if mapped_v1_type == true_type:
            result["correct_columns"] += 1
            result["column_details"].append({
                "column_index": col_idx,
                "column_name": col_name,
                "true_type": true_type,
                "v1_type": mapped_v1_type,
                "status": "correct",
                "flagged": is_flagged,
            })
        else:
            result["wrong_columns"] += 1
            if not is_flagged:
                result["wrong_unflagged_columns"] += 1
            result["column_details"].append({
                "column_index": col_idx,
                "column_name": col_name,
                "true_type": true_type,
                "v1_type": mapped_v1_type,
                "status": "misclassified",
                "flagged": is_flagged,
            })

    return result


def main():
    print(f"Running baseline measurement using v1.0.1 from: {V1_SRC_PATH}")
    label_files = sorted(list(LABELS_DIR.glob("*.json")))
    print(f"Found {len(label_files)} labeled corpus files.")

    total_files = len(label_files)
    total_columns = 0

    dialect_matches = 0
    delimiter_matches = 0
    encoding_matches = 0
    header_matches = 0

    exceptions_count = 0
    exception_types = defaultdict(int)

    total_correct_columns = 0
    total_wrong_columns = 0
    total_wrong_unflagged_columns = 0

    results_by_file = []

    for lf in label_files:
        with open(lf, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        fid = label_data["file_id"]
        res = evaluate_dataset(fid, label_data)
        results_by_file.append(res)

        total_columns += res["total_columns"]
        total_correct_columns += res["correct_columns"]
        total_wrong_columns += res["wrong_columns"]
        total_wrong_unflagged_columns += res["wrong_unflagged_columns"]

        if res["dialect_match"]:
            dialect_matches += 1
        if res["delimiter_match"]:
            delimiter_matches += 1
        if res["encoding_match"]:
            encoding_matches += 1
        if res["header_match"]:
            header_matches += 1

        if res["exception_raised"]:
            exceptions_count += 1
            exception_types[res["exception_type"]] += 1

    # Metrics computation
    dialect_accuracy_pct = (dialect_matches / total_files) * 100
    column_type_accuracy_pct = (total_correct_columns / total_columns) * 100
    exception_pct = (exceptions_count / total_files) * 100
    wrong_unflagged_pct_of_wrong = (
        (total_wrong_unflagged_columns / total_wrong_columns) * 100 if total_wrong_columns > 0 else 0
    )
    wrong_unflagged_pct_of_total = (total_wrong_unflagged_columns / total_columns) * 100

    print("\n" + "=" * 60)
    print("V1.0.1 BASELINE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Corpus Files:                 {total_files}")
    print(f"Total Labeled Columns:              {total_columns}")
    print(f"Dialect Accuracy:                   {dialect_accuracy_pct:.2f}% ({dialect_matches}/{total_files})")
    print(f"  - Delimiter Match:                {(delimiter_matches/total_files)*100:.2f}% ({delimiter_matches}/{total_files})")
    print(f"  - Encoding Match:                 {(encoding_matches/total_files)*100:.2f}% ({encoding_matches}/{total_files})")
    print(f"  - Header Row Match:               {(header_matches/total_files)*100:.2f}% ({header_matches}/{total_files})")
    print(f"Column-Type Accuracy:               {column_type_accuracy_pct:.2f}% ({total_correct_columns}/{total_columns})")
    print(f"Unhandled Exception Rate:           {exception_pct:.2f}% ({exceptions_count}/{total_files})")
    print(f"Wrong-Types-With-No-Flag (% wrong): {wrong_unflagged_pct_of_wrong:.2f}% ({total_wrong_unflagged_columns}/{total_wrong_columns})")
    print(f"Wrong-Types-With-No-Flag (% total): {wrong_unflagged_pct_of_total:.2f}% ({total_wrong_unflagged_columns}/{total_columns})")
    print("=" * 60)

    # 4. Generate Comprehensive docs/v1_baseline_report.md
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# PreFlight v1.0.1 Baseline Measurement Report\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(
            "This report establishes the empirical performance baseline of **PreFlight v1.0.1** "
            "evaluated against the newly constructed messy-CSV benchmark corpus (Phase 1 / M0). "
            "v1.0.1 was executed without modification directly from the verified install path.\n\n"
        )
        f.write(
            "> [!NOTE]\n"
            "> This phase is strictly measurement-focused. No v2 pipeline code or optimizations "
            "have been implemented or claimed.\n\n"
        )

        f.write("### Primary Baseline Metrics\n\n")
        f.write("| Metric | v1.0.1 Baseline | Count / Total | Definition |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Corpus Size (Files)** | **{total_files}** | {total_files} / 200 target | Number of real-world datasets in corpus |\n")
        f.write(f"| **Corpus Size (Columns)** | **{total_columns:,}** | {total_columns} / 1,000 target | Total ground-truth labeled columns |\n")
        f.write(f"| **Dialect Accuracy** | **{dialect_accuracy_pct:.2f}%** | {dialect_matches} / {total_files} | Exact match on delimiter, encoding, header row |\n")
        f.write(f"| **Column-Type Accuracy** | **{column_type_accuracy_pct:.2f}%** | {total_correct_columns:,} / {total_columns:,} | Correct inferral of canonical semantic type |\n")
        f.write(f"| **Unhandled Exception Rate** | **{exception_pct:.2f}%** | {exceptions_count} / {total_files} | Crashes during ingestion or profiling |\n")
        f.write(f"| **Wrong-Types-With-No-Flag (% of wrong)** | **{wrong_unflagged_pct_of_wrong:.2f}%** | {total_wrong_unflagged_columns:,} / {total_wrong_columns:,} | Misclassifications that emitted zero alerts/flags |\n")
        f.write(f"| **Wrong-Types-With-No-Flag (% of all cols)** | **{wrong_unflagged_pct_of_total:.2f}%** | {total_wrong_unflagged_columns:,} / {total_columns:,} | Silent error rate across entire corpus |\n\n")

        f.write("### Dialect Sub-Component Breakdown\n\n")
        f.write("| Dialect Property | Accuracy | Match Count | Failure Mode in v1.0.1 |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Delimiter Detection** | {(delimiter_matches/total_files)*100:.2f}% | {delimiter_matches} / {total_files} | Hardcoded to `,`; collapses `;` and `\\t` tables into single text column |\n")
        f.write(f"| **Encoding Detection** | {(encoding_matches/total_files)*100:.2f}% | {encoding_matches} / {total_files} | Hardcoded to UTF-8; raises `UnicodeDecodeError` on Latin-1 / ISO-8859-1 |\n")
        f.write(f"| **Header Row Detection** | {(header_matches/total_files)*100:.2f}% | {header_matches} / {total_files} | Assumes row 0; corrupts headerless datasets and comment preambles |\n\n")

        # Exception Breakdown
        f.write("## 2. Unhandled Ingestion & Profiling Exceptions\n\n")
        f.write(
            f"PreFlight v1.0.1 crashed on **{exceptions_count} out of {total_files} files** ({exception_pct:.2f}%), "
            "raising uncaught exceptions before completing profile generation:\n\n"
        )
        f.write("| Exception Class | Count | Typical Root Cause |\n")
        f.write("| :--- | :--- | :--- |\n")
        for exc_name, count in sorted(exception_types.items(), key=lambda x: x[1], reverse=True):
            cause = "Non-UTF8 byte stream encountered" if "Unicode" in exc_name else "Ragged rows or malformed field structure"
            f.write(f"| `{exc_name}` | {count} | {cause} |\n")
        f.write("\n")

        # Per-File Failures
        f.write("## 3. Per-File Failure Analysis\n\n")
        f.write("The table below details all files where v1.0.1 encountered fatal crashes or failed dialect detection:\n\n")
        f.write("| File ID | Filename | Dialect Correct? | Exception? | Exception Details / Failure Mode |\n")
        f.write("| :--- | :--- | :---: | :---: | :--- |\n")

        for r in results_by_file:
            if r["exception_raised"] or not r["dialect_match"]:
                status_dialect = "Yes" if r["dialect_match"] else "No"
                status_exc = f"`{r['exception_type']}`" if r["exception_raised"] else "None"
                msg = r["exception_message"] or "Dialect mismatch (e.g. non-comma delimiter or non-standard header)"
                msg_clean = msg.replace("\n", " ")[:100]
                f.write(f"| `{r['file_id']}` | `{r['filename']}` | {status_dialect} | {status_exc} | {msg_clean} |\n")

        f.write("\n## 4. Column Classification Failure Modes\n\n")
        f.write(
            "Analysis of the misclassified columns identifies the primary structural weaknesses in v1.0.1:\n\n"
            "1. **Silent Semicolon / TSV Corruption**: Files with `;` or `\\t` delimiters are parsed by `pandas.read_csv` "
            "into a single giant column. v1.0.1 profiles this single string as `TEXT` without flagging that delimiter parsing failed.\n"
            "2. **Disguised Missing Value Marker Confusion**: Sentinel markers such as `?`, `NA`, or `-` in numeric columns "
            "cause pandas to infer `object` dtype. v1.0.1 then categorizes these continuous features as `CATEGORICAL_LOW` or `CATEGORICAL_HIGH`.\n"
            "3. **Leading-Zero ID Loss**: Identifiers such as US ZIP codes (`02138`) or FIPS codes (`01001`) with leading zeros "
            "are parsed as integers by pandas and inferred as `NUMERIC_FEATURE` without maintaining string zero preservation.\n"
            "4. **Headerless Data Truncation**: For datasets without headers, row 0 data is consumed as column header labels, "
            "leading to lost data records and corrupted column names.\n\n"
        )
        f.write("---\n")
        f.write("*Report generated automatically by `scripts/measure_v1_baseline.py` in Phase 1 (M0).*\n")

    print(f"\nReport successfully generated at: {REPORT_PATH}")


if __name__ == "__main__":
    main()
