#!/usr/bin/env python3
"""Column-type accuracy per group from one corpus run: strict (correct) and
"correct or flagged" (correct, or wrong but flagged for review). Prints counts only.

Usage: python scripts/type_accuracy_buckets.py [RUN_DIR]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / ".pytest_cache" / "corpus_results").glob("run_*"))[-1]
recs = [json.loads(p.read_text()) for p in (run / "accuracy").glob("*.json")]
for group in ("tuning", "held_out", None):
    cols = [c for r in recs if group is None or r.get("group") == group for c in r["column_results"]]
    n = len(cols)
    ok = sum(c["correct"] for c in cols)
    fl = sum((not c["correct"]) and c["flagged"] for c in cols)
    print(f"{group or 'combined'}: cols={n} strict={ok} ({100*ok/n:.2f}%) "
          f"correct_or_flagged={ok+fl} ({100*(ok+fl)/n:.2f}%) wrong_flagged={fl} wrong_no_flag={n-ok-fl}")
