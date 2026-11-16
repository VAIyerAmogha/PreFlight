#!/usr/bin/env python3
"""
Measurement-only corpus accuracy harness for PreFlight v2.

Runs ingest + infer + full plan() over every corpus file, compares results
against the frozen per-file ground-truth labels in corpus/labels/, and
reports dialect accuracy, column-type accuracy, wrong-type-with-no-flag
rate, decision evidence/confidence completeness, and needs_attention load,
split by the frozen tuning/held_out groups in corpus/labels/split.json.

This script does NOT tune, fix, or adjust any library behavior. It only
measures and reports.

`measure_file` is the per-file measurement. The corpus test
tests/corpus/test_accuracy_corpus.py calls it once per file (parallel with
pytest-xdist, per-file timeout with pytest-timeout), and scripts/corpus_summary.py
turns the saved records into the targets table. Running this file directly
measures every file in one process: python scripts/corpus_accuracy.py
"""
import json
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from preflight.infer.types import to_canonical_type
from preflight.plan import plan

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "corpus" / "labels"
RAW_DIR = ROOT / "corpus" / "raw"
SPLIT_PATH = LABELS_DIR / "split.json"

ENCODING_EQUIV = {
    "latin-1": "iso-8859-1",
    "iso-8859-1": "iso-8859-1",
    "cp1252": "cp1252",
    "windows-1252": "cp1252",
    "utf-8": "utf-8",
    "utf-8-sig": "utf-8-sig",
    "ascii": "ascii",
}


def norm_enc(e: Optional[str]) -> Optional[str]:
    if e is None:
        return None
    e = e.lower()
    return ENCODING_EQUIV.get(e, e)


def load_labels() -> dict[str, dict[str, Any]]:
    labels = {}
    for p in sorted(LABELS_DIR.glob("*.json")):
        if p.name in ("split.json",):
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        labels[data["file_id"]] = data
    return labels


def load_split() -> dict[str, str]:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["assignments"]


def measure_file(
    label: dict[str, Any], group: Optional[str], plan_fn: Callable[..., Any] = plan
) -> dict[str, Any]:
    """Plan one corpus file and compare it with its label. Never raises for a library error:
    the error is recorded in the returned record (`loaded_ok` False)."""
    raw_path = RAW_DIR / label["filename"]
    rec: dict[str, Any] = {
        "file_id": label["file_id"],
        "group": group,
        "filename": label["filename"],
        "loaded_ok": False,
        "error": None,
        "dialect_correct": None,
        "column_results": [],
        "needs_attention_count": 0,
        "decisions_missing_evidence_or_conf": 0,
        "decisions_total": 0,
    }

    if not raw_path.exists():
        rec["error"] = f"raw file missing: {raw_path}"
        return rec

    try:
        plan_obj = plan_fn(raw_path)
        rec["loaded_ok"] = True
    except Exception as exc:  # Swallowed: recorded in the result; the caller decides what fails
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
        return rec

    schema = plan_obj.schema

    # ---- Dialect accuracy ----
    true_dialect = label["dialect"]
    true_delim = true_dialect.get("delimiter")
    true_enc = norm_enc(true_dialect.get("encoding"))
    true_hdr = true_dialect.get("header_row_index")

    got_delim = schema.dialect.effective_delimiter() if schema else None
    got_enc = norm_enc(schema.encoding.effective_encoding()) if schema else None
    got_hdr = schema.header.effective_header_row() if schema else None

    rec["dialect_correct"] = got_delim == true_delim and got_enc == true_enc and got_hdr == true_hdr
    rec["dialect_detail"] = {
        "true": {"delimiter": true_delim, "encoding": true_enc, "header_row_index": true_hdr},
        "got": {"delimiter": got_delim, "encoding": got_enc, "header_row_index": got_hdr},
    }

    # ---- Decisions: evidence + confidence completeness ----
    all_decisions = plan_obj.decisions
    rec["decisions_total"] = len(all_decisions)
    missing = 0
    for d in all_decisions:
        has_evidence = bool(d.evidence) if isinstance(d.evidence, dict) else d.evidence is not None
        has_confidence = d.confidence is not None
        if not has_evidence or not has_confidence:
            missing += 1
    rec["decisions_missing_evidence_or_conf"] = missing

    # ---- needs_attention count (whole-plan, "per messy file") ----
    rec["needs_attention_count"] = sum(1 for d in all_decisions if d.needs_attention)

    # ---- Column type accuracy + wrong-type-no-flag ----
    # Column-level needs_attention comes from the type_inference/type_abstain decisions
    col_needs_attention = {}
    col_inferred_type = {}
    if schema:
        for c in schema.columns:
            col_inferred_type[c.name] = c.inferred_type
    for d in all_decisions:
        if d.stage == "infer" and d.kind in ("type_inference", "type_abstain") and d.columns:
            col_needs_attention[d.columns[0]] = d.needs_attention

    for col_label in label["columns"]:
        col_name = col_label["column_name"]
        true_type = col_label["semantic_type"]
        inferred_raw = col_inferred_type.get(col_name)
        col_name_lookup = col_name
        if inferred_raw is None:
            # Try matching by position if name differs (e.g. headerless files)
            idx = col_label.get("column_index")
            if schema and idx is not None and 0 <= idx < len(schema.columns):
                inferred_raw = schema.columns[idx].inferred_type
                col_name_lookup = schema.columns[idx].name

        if inferred_raw is None:
            correct = False
            got_canonical = None
        else:
            got_canonical = to_canonical_type(inferred_raw)
            correct = got_canonical == true_type

        flagged = col_needs_attention.get(col_name_lookup, False)
        rec["column_results"].append(
            {
                "column_name": col_name,
                "true_type": true_type,
                "got_type": got_canonical,
                "correct": correct,
                "flagged": flagged,
                "wrong_no_flag": (not correct) and (not flagged),
            }
        )

    return rec


def main() -> tuple[list[dict[str, Any]], int, dict[str, str]]:
    assignments = load_split()
    labels = load_labels()
    total_label_columns = sum(len(lab["columns"]) for lab in labels.values())
    per_file_results = [
        measure_file(label, assignments.get(file_id)) for file_id, label in labels.items()
    ]
    return per_file_results, total_label_columns, assignments


if __name__ == "__main__":
    results, total_cols, _ = main()
    out_path = ROOT / "scripts" / "_corpus_accuracy_raw.json"
    # Strip traceback for size
    serializable = []
    for r in results:
        rr = dict(r)
        rr.pop("traceback", None)
        serializable.append(rr)
    out_path.write_text(json.dumps({"results": serializable, "total_label_columns": total_cols}, indent=2))
    print(f"Wrote raw results to {out_path}")
    print(f"Total files: {len(results)}, total label columns: {total_cols}")
    n_loaded = sum(1 for r in results if r["loaded_ok"])
    print(f"Loaded OK: {n_loaded}/{len(results)}")
    for r in results:
        if not r["loaded_ok"]:
            print(f"  FAILED {r['file_id']}: {r['error']}")
