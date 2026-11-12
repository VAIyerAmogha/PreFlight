#!/usr/bin/env python3
"""Turn scripts/compare_v1_v2.py output into a markdown table (TIE rule: |diff| <= v1_std + v2_std; equal means with zero spread are a tie).

Usage: python scripts/compare_table.py IN.json OUT.md   (prints verdict counts only)
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def _ms(s: list[float]) -> tuple[float, float]:
    return statistics.mean(s), statistics.stdev(s)


def main() -> None:
    rows = json.loads(Path(sys.argv[1]).read_text())
    out = ["| File | Model | v1 mean ± std | v2 mean ± std | Verdict | Note |", "|---|---|---|---|---|---|"]
    verdicts: Counter = Counter()
    for r in rows:
        metric = "acc" if r["task"] == "classification" else "R²"
        v1, v2 = r["v1"], r["v2"]
        note = "subsampled to 4,000 rows" if r["subsampled"] else ""
        cell = {}
        for side, d in (("v1", v1), ("v2", v2)):
            if d["error"]:
                cell[side] = "failed"
                note = (note + "; " if note else "") + f"{side}: {d['error'][:110]}"
            else:
                m, s = _ms(d["scores"])
                cell[side] = f"{m:.4f} ± {s:.4f}"
        if v1["error"] and v2["error"]:
            verdict = "BOTH FAILED"
        elif v1["error"]:
            verdict = "v1 FAILED"
        elif v2["error"]:
            verdict = "v2 FAILED"
        else:
            (m1, s1), (m2, s2) = _ms(v1["scores"]), _ms(v2["scores"])
            verdict = "TIE" if abs(m1 - m2) <= s1 + s2 else ("v2 better" if m2 > m1 else "v1 better")
        verdicts[verdict] += 1
        out.append(f"| {r['file']} | {r['model']} ({metric}) | {cell['v1']} | {cell['v2']} | {verdict} | {note} |")
    Path(sys.argv[2]).write_text("\n".join(out) + "\n")
    print(dict(verdicts))


if __name__ == "__main__":
    main()
