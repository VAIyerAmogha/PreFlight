"""Time plan() on named corpus files, one fresh process per file, sequentially.

Usage: python scripts/time_plan.py OUT.json FILE [FILE ...]
FILE is a filename under corpus/raw. Writes {file: {"wall_s", "fingerprint"}} to OUT.json.
The fingerprint is a hash of every Decision (stage, kind, columns, action, confidence,
needs_attention) so before/after runs can be compared for identical output.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHILD = r"""
import hashlib, json, sys, time
sys.path.insert(0, %(src)r)
from preflight.plan import plan
t = time.perf_counter()
p = plan(%(path)r)
wall = time.perf_counter() - t
rows = [[d.stage, d.kind, list(d.columns), d.action, d.confidence, d.needs_attention] for d in p.decisions]
print(json.dumps({"wall_s": round(wall, 2), "fingerprint": hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()[:16]}))
"""


def main() -> None:
    out = Path(sys.argv[1])
    res = {}
    for name in sys.argv[2:]:
        code = CHILD % {"src": str(ROOT / "src"), "path": str(ROOT / "corpus/raw" / name)}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
        res[name] = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.stderr[-300:]}
        print(name, res[name].get("wall_s", "ERROR"), res[name].get("fingerprint", ""))
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
