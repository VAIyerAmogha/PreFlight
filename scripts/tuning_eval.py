#!/usr/bin/env python3
"""Score the TUNING set only (Phase 7c Stage B). Held-out files are never loaded.

Runs scripts/corpus_accuracy.py's measure_file on every tuning file in parallel and
writes the per-column misses to OUT (JSON). Prints one line of counts per bucket:
correct, wrong-but-flagged, wrong-no-flag (column type), plus dialect and errors.

Usage: python scripts/tuning_eval.py OUT.json [--workers 4]
"""

import argparse
import importlib.util
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("corpus_accuracy", ROOT / "scripts" / "corpus_accuracy.py")
assert _spec is not None and _spec.loader is not None
ca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ca)


def _one(label: dict[str, Any]) -> dict[str, Any]:
    warnings.filterwarnings("ignore")
    return ca.measure_file(label, "tuning")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    split = ca.load_split()
    labels = [lab for fid, lab in ca.load_labels().items() if split.get(fid) == "tuning"]
    with ProcessPoolExecutor(a.workers) as ex:
        recs = list(ex.map(_one, labels))
    cols = [c for r in recs for c in r["column_results"]]
    correct = sum(c["correct"] for c in cols)
    wnf = [dict(c, filename=r["filename"]) for r in recs for c in r["column_results"] if c["wrong_no_flag"]]
    wf = sum((not c["correct"]) and c["flagged"] for c in cols)
    dialect = sum(bool(r.get("dialect_correct")) for r in recs)
    errors = sum(not r["loaded_ok"] for r in recs)
    misses = [dict(c, filename=r["filename"]) for r in recs for c in r["column_results"] if not c["correct"]]
    a.out.write_text(json.dumps({"files": len(recs), "columns": len(cols), "misses": misses,
                                 "wrong_no_flag": wnf}, indent=1))
    n = len(cols)
    print(f"tuning files={len(recs)} cols={n} correct={correct} ({100*correct/n:.2f}%) "
          f"wrong_flagged={wf} wrong_no_flag={len(wnf)} ({100*len(wnf)/n:.2f}%) "
          f"correct_or_flagged={100*(correct+wf)/n:.2f}% dialect={dialect}/{len(recs)} errors={errors}")


if __name__ == "__main__":
    main()
