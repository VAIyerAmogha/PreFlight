"""Pre-compute the JSON behind the static figures on the demo site.

Runs public/py/runner.py (the same code the browser runs) under CPython against
the bundled samples and writes data/fixtures/<sample>.json, which the Next.js
pages import at build time. Re-run after any
package change that alters decisions or questions:

    python site/scripts/build_fixtures.py      # from the repo root, with the repo venv
"""

import json
import sys
import tempfile
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE / "public" / "py"))

import runner  # noqa: E402

# Samples the docs figures are drawn from (sample file -> target column).
# The playground runs everything live, so only these two need fixtures.
SAMPLES = {
    "disguised_missing_units.csv": "outcome",
    "messy_noheader_semicolon.csv": "col_3",
}


def main() -> None:
    out = SITE / "data" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(runner.version())
    for name, target in SAMPLES.items():
        path = str(SITE / "public" / "samples" / name)
        analysis = json.loads(runner.analyze(path, target))
        with tempfile.TemporaryDirectory() as tmp:
            result = json.loads(runner.run(path, target, "{}", out_dir=tmp))
        result.pop("files", None)  # temp paths; the browser run produces real downloads
        doc = {"meta": meta, "sample": name, "target": target, "analysis": analysis, "result": result}
        dest = out / (Path(name).stem + ".json")
        dest.write_text(json.dumps(doc, default=str, indent=1))
        print(f"{dest.relative_to(SITE)}: {dest.stat().st_size // 1024} KB, "
              f"{len(analysis['decisions'])} decisions, {len(analysis['questions'])} questions")


if __name__ == "__main__":
    main()
