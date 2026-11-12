"""Shared helpers for the corpus suites.

Each corpus test runs on one file (`corpus_file`) and saves a small JSON record with
`record()`. After the session, tests/corpus/conftest.py hands the records to
scripts/corpus_summary.py, which checks the corpus-wide thresholds the suites assert.

Run the corpus suites with:  pytest -m corpus -n 4 --dist loadgroup
`--dist loadgroup` sends every test for one file to the same worker, so the cached
ingest/infer results below are shared between the ingest, infer and schema tests.
"""

import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pytest

from preflight.infer import infer_table_types
from preflight.ingest import ingest
from preflight.schema import TableSchema

ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = ROOT / "corpus"
RAW_DIR = CORPUS_ROOT / "raw"
LABELS_DIR = CORPUS_ROOT / "labels"
RESULTS_ROOT = ROOT / ".pytest_cache" / "corpus_results"
RUN_DIR_ENV = "PREFLIGHT_CORPUS_RUN_DIR"

EXPECTED_FILES = 254


def label_files() -> list[Path]:
    """Ground-truth label JSON files (split.json is the split manifest, not a label)."""
    return sorted(f for f in LABELS_DIR.glob("*.json") if f.name != "split.json")


def load_label(label_path: Path) -> dict[str, Any]:
    with open(label_path, encoding="utf-8") as f:
        return json.load(f)


def raw_files() -> list[Path]:
    if not RAW_DIR.is_dir():
        return []
    return sorted(f for f in RAW_DIR.iterdir() if f.is_file())


def _param(path: Path, raw_name: str) -> Any:
    return pytest.param(path, id=raw_name, marks=pytest.mark.xdist_group(raw_name))


def label_params() -> list[Any]:
    """One param per label file; the id and xdist group are the raw file name."""
    return [_param(lf, load_label(lf)["filename"]) for lf in label_files()]


def raw_params(files: Optional[list[Path]] = None) -> list[Any]:
    return [_param(p, p.name) for p in (raw_files() if files is None else files)]


def run_dir() -> Path:
    """Where this session's records go (set by conftest; falls back to a 'manual' dir)."""
    return Path(os.environ.get(RUN_DIR_ENV) or RESULTS_ROOT / "manual")


def record(suite: str, name: str, data: dict[str, Any]) -> None:
    out = run_dir() / suite
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(data, default=str), encoding="utf-8")


@lru_cache(maxsize=4)
def _ingested(path: str) -> tuple[Any, str]:
    data, schema = ingest(Path(path))
    return data, schema.to_json()


def cached_ingest(path: Path) -> tuple[Any, TableSchema]:
    """ingest() once per file per worker. Callers get their own schema copy and data copy."""
    data, schema_json = _ingested(str(path))
    return data.copy(), TableSchema.from_json(schema_json)


@lru_cache(maxsize=4)
def _inferred(path: str) -> str:
    data, schema = cached_ingest(Path(path))
    return infer_table_types(data, schema).to_json()


def cached_infer(path: Path) -> TableSchema:
    """ingest() + infer_table_types() once per file per worker; returns a fresh schema copy."""
    return TableSchema.from_json(_inferred(str(path)))


def run_worker(worker: Path, path: Path, timeout: int, prefix: str) -> dict[str, Any]:
    """Run a worker script on one file in its own process with a hard timeout."""
    try:
        proc = subprocess.run(
            [sys.executable, str(worker), str(path)],
            capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {"file": path.name, "ok": False, "hang": True}
    for line in proc.stdout.splitlines():
        if line.startswith(prefix):
            result: dict[str, Any] = json.loads(line[len(prefix):])
            return result
    return {"file": path.name, "ok": False, "stage": "no result", "error": (proc.stderr or "")[-500:]}
