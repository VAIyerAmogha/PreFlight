"""Rerun named corpus files one at a time via plan() with a hard timeout; writes docs/review_notes/phase7/timeouts_solo.md."""
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preflight.plan import plan  # noqa: E402

ids = sys.argv[2:]
limit = int(sys.argv[1])
labels = {json.loads(p.read_text())["file_id"]: json.loads(p.read_text())["filename"] for p in (ROOT / "corpus/labels").glob("*.json") if p.name != "split.json"}


def _a(*_):
    raise TimeoutError


signal.signal(signal.SIGALRM, _a)
rows = ["| file_id | size MB | wall s | result |", "|---|---|---|---|"]
for fid in ids:
    p = ROOT / "corpus/raw" / labels[fid]
    signal.alarm(limit)
    t = time.time()
    try:
        plan(p)
        res = "OK"
    except TimeoutError:
        res = f"TIMEOUT>{limit}s"
    except Exception as e:  # noqa: BLE001
        res = f"ERROR {type(e).__name__}: {str(e)[:60]}"
    signal.alarm(0)
    rows.append(f"| {fid} | {p.stat().st_size/1e6:.1f} | {time.time()-t:.1f} | {res} |")
(ROOT / "docs/review_notes/phase7/timeouts_solo.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
