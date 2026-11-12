#!/usr/bin/env python3
"""Error analysis on the TUNING set only, from a corpus run's accuracy records.

Usage: python scripts/error_analysis.py [RUN_DIR] [--out FILE]
Groups column-type misses by (labelled type -> detected type) and dialect misses by which
part was wrong (encoding, separator, column-names row). Held-out files are never read.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / ".pytest_cache" / "corpus_results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "review_notes" / "phase7b" / "error_analysis.md")
    a = ap.parse_args()
    run = a.run_dir or sorted(RESULTS_ROOT.glob("run_*"))[-1]
    recs = [json.loads(p.read_text()) for p in (run / "accuracy").glob("*.json")]
    recs = [r for r in recs if r.get("group") == "tuning"]
    pairs: Counter = Counter()
    files: dict = {}
    flagged: Counter = Counter()
    for r in recs:
        for c in r["column_results"]:
            if not c["correct"]:
                k = (c["true_type"], c["got_type"])
                pairs[k] += 1
                flagged[k] += c["flagged"]
                files.setdefault(k, Counter())[r["filename"]] += 1
    parts: Counter = Counter()
    dfiles = []
    for r in recs:
        if r.get("dialect_correct") is False:
            t, g = r["dialect_detail"]["true"], r["dialect_detail"]["got"]
            wrong = [k for k in ("encoding", "delimiter", "header_row_index") if t[k] != g[k]]
            parts.update(wrong)
            dfiles.append((r["filename"], ", ".join(f"{k}: {t[k]!r} -> {g[k]!r}" for k in wrong)))
    n_miss = sum(pairs.values())
    out = [f"# Error analysis (tuning set, {len(recs)} files, run {run.name})", "",
           f"## Column type misses: {n_miss}", "",
           "| Labelled | Detected | Misses | Flagged for review | Top files |", "|---|---|---|---|---|"]
    for (t, g), n in pairs.most_common():
        top = ", ".join(f"{f} ({c})" for f, c in files[(t, g)].most_common(3))
        out.append(f"| {t} | {g} | {n} | {flagged[(t, g)]} | {top} |")
    out += ["", f"## Dialect misses: {len(dfiles)} files", "",
            "Wrong part counts: " + ", ".join(f"{k} {v}" for k, v in parts.most_common()), ""]
    out += [f"- {f}: {d}" for f, d in sorted(dfiles)]
    a.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"tuning files {len(recs)}; type misses {n_miss}; dialect misses {len(dfiles)}; wrote {a.out}")


if __name__ == "__main__":
    main()
