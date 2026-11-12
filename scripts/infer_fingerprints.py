"""Record per-column type inference output for every corpus file (one worker per file).

Usage: python scripts/infer_fingerprints.py OUT.json [JOBS]
Writes {filename: {column: [type, confidence, needs_attention, anomalies, action, n_non_missing]}}.
Used to check that sampling leaves small columns unchanged and to list what changes on large ones.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _one(path: str):
    from preflight.infer.types import infer_column_type
    from preflight.ingest import ingest
    try:
        df, _ = ingest(path)
    except Exception as e:  # report, do not hide: recorded in the output
        return Path(path).name, {"__error__": f"{type(e).__name__}: {e}"[:200]}
    out = {}
    for c in df.columns:
        r = infer_column_type(df[c], column_name=str(c))
        out[str(c)] = [r.inferred_type, r.confidence, r.needs_attention, r.anomalies,
                       r.decision.action, r.evidence.get("non_missing_count")]
    return Path(path).name, out


def main() -> None:
    files = sorted(str(p) for p in (ROOT / "corpus/raw").iterdir() if p.is_file())
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with ProcessPoolExecutor(jobs) as ex:
        res = dict(ex.map(_one, files))
    Path(sys.argv[1]).write_text(json.dumps(res, indent=0, default=str))
    print("files", len(res), "errors", sum("__error__" in v for v in res.values()))


if __name__ == "__main__":
    main()
