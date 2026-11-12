#!/usr/bin/env python3
"""v2 vs v1.0.1 feature-building comparison on 10 corpus files (Phase 7c re-run).

Same method as docs/v2_vs_v1_scores.md:
- Targets are assumed (standard outcome of each named public dataset), not corpus labels.
- v1 path: plain pandas.read_csv(path) -> per fold: v1 run_profiler + build_pipeline_two_phase
  fitted on the training fold only.
- v2 path: preflight.plan(path, target, task, model_hint) -> apply(mode="defaults") ->
  per fold: clone(result.pipeline) fitted on the training fold only (input: result.prepared).
- Models: linear (LogisticRegression / LinearRegression, standardized) and tree
  (RandomForest, 100 trees), random_state=42. Output columns a version left non-numeric are
  one-hot encoded per fold (training-fold categories) for both versions alike. CV: Repeated(Stratified)KFold, 3 splits x 3
  repeats, random_state=42. A median imputer guards any blanks left in either side's output.
- Files over 4,000 data rows are subsampled to 4,000 rows (seed 42) at the file level, so
  both versions read the same rows.
- Tie rule: |v1_mean - v2_mean| < v1_std + v2_std -> TIE.

v1 and v2 are both importable as `preflight`, so each side runs in its own subprocess.
Usage: python scripts/compare_v1_v2.py OUT.json   (prints one line per file)
"""

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "corpus" / "raw"
V1_SRC = Path("/home/amg/Desktop/PreFlight v1.0.1/src")
SUBSAMPLE = 4000
SEED = 42

# (file, target name, target column index for headerless files, task)
FILES = [
    ("adult.data", "income", 14, "classification"),
    ("iris.data", "species", 4, "classification"),
    ("winequality-red.csv", "quality", None, "regression"),
    ("winequality-white.csv", "quality", None, "regression"),
    ("titanic.csv", "survived", None, "classification"),
    ("tips.csv", "tip", None, "regression"),
    ("AER_Affairs.csv", "affairs", None, "regression"),
    ("diamonds.csv", "price", None, "regression"),
    ("carData_TitanicSurvival.csv", "survived", None, "classification"),
    ("COUNT_titanic.csv", "survived", None, "classification"),
]

WORKER = r'''
import json, sys, warnings, traceback
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

side, path, target, tidx, task, model_kind = sys.argv[1:7]
tidx = None if tidx == "None" else int(tidx)
SEED = 42

def model():
    if model_kind == "linear":
        est = LogisticRegression(max_iter=2000, random_state=SEED) if task == "classification" else LinearRegression()
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    est = (RandomForestClassifier if task == "classification" else RandomForestRegressor)(n_estimators=100, random_state=SEED, n_jobs=1)
    return make_pipeline(SimpleImputer(strategy="median"), est)

def numeric(df):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype("int64", errors="ignore")
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.astype(float)

def encode(Xtr, Xte):
    """Same for v1 and v2: any output column that is still not numeric (a version left it
    unencoded) is one-hot encoded with the categories of the training fold."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    left = [c for c in Xtr.columns if not (pd.api.types.is_numeric_dtype(Xtr[c]) or pd.api.types.is_bool_dtype(Xtr[c])
                                           or pd.api.types.is_datetime64_any_dtype(Xtr[c]))]
    if left:
        dtr = pd.get_dummies(Xtr[left].astype(str), dtype=float)
        dte = pd.get_dummies(Xte[left].astype(str), dtype=float).reindex(columns=dtr.columns, fill_value=0.0)
        Xtr = pd.concat([Xtr.drop(columns=left), dtr], axis=1)
        Xte = pd.concat([Xte.drop(columns=left), dte], axis=1)
    Xtr = numeric(Xtr)
    return Xtr, numeric(Xte).reindex(columns=Xtr.columns)

def cv():
    if task == "classification":
        return RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=SEED)
    return RepeatedKFold(n_splits=3, n_repeats=3, random_state=SEED)

def score(est, X, y):
    from sklearn.metrics import accuracy_score, r2_score
    p = est.predict(X)
    return float(accuracy_score(y, p) if task == "classification" else r2_score(y, p))

res = {"scores": [], "error": None}
try:
    if side == "v2":
        import tempfile, preflight
        plan = preflight.plan(path, task=task, model_hint=model_kind)
        names = [c.name for c in plan.schema.columns]
        tname = names[tidx] if tidx is not None else target
        plan = preflight.plan(path, target=tname, task=task, model_hint=model_kind)
        r = preflight.apply(plan, mode="defaults", out_dir=tempfile.mkdtemp())
        df = r.prepared
        df = df[df[tname].notna()]
        X = df.drop(columns=[tname]); y = df[tname]
        # The feature pipelines get the target as read; only the model sees string labels.
        y = y if task == "classification" else y.astype(float)
        ym = y.astype(str) if task == "classification" else y
        for tr, te in cv().split(X, ym):
            pipe = clone(r.pipeline)
            Xtr, Xte = encode(pipe.fit_transform(X.iloc[tr], y.iloc[tr]), pipe.transform(X.iloc[te]))
            m = model().fit(Xtr, ym.iloc[tr]); res["scores"].append(score(m, Xte, ym.iloc[te]))
    else:
        from preflight.profiler import run_profiler
        from preflight.assembler import build_pipeline_two_phase
        df = pd.read_csv(path)
        if target not in df.columns:
            raise KeyError(f"target {target!r} not in columns after plain read_csv: {list(df.columns)[:6]}")
        df = df[df[target].notna()].reset_index(drop=True)
        X = df.drop(columns=[target]); y = df[target]
        # The feature pipelines get the target as read; only the model sees string labels.
        y = y if task == "classification" else y.astype(float)
        ym = y.astype(str) if task == "classification" else y
        for tr, te in cv().split(X, ym):
            train = df.iloc[tr]
            profiles, _ = run_profiler(train, target=target, task=task)
            pipe = build_pipeline_two_phase(profiles, target, model_kind, X.iloc[tr], y.iloc[tr])
            Xtr, Xte = encode(pipe.transform(X.iloc[tr]), pipe.transform(X.iloc[te]))
            m = model().fit(Xtr, ym.iloc[tr]); res["scores"].append(score(m, Xte, ym.iloc[te]))
except Exception as e:
    res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    res["traceback"] = traceback.format_exc()[-1500:]
print("@@RESULT@@" + json.dumps(res))
'''


def _subsample(path: Path, tmp: Path) -> Path:
    lines = path.read_bytes().splitlines(keepends=True)
    has_header = path.suffix != ".data"
    head, body = (lines[:1], lines[1:]) if has_header else ([], lines)
    body = [b for b in body if b.strip()]
    if len(body) <= SUBSAMPLE:
        return path
    keep = sorted(random.Random(SEED).sample(range(len(body)), SUBSAMPLE))
    out = tmp / path.name
    out.write_bytes(b"".join(head + [body[i] for i in keep]))
    return out


def _run(side: str, path: Path, target: str, tidx: object, task: str, kind: str) -> dict:
    env_path = str(ROOT / "src") if side == "v2" else str(V1_SRC)
    proc = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {env_path!r}); exec(sys.stdin.read())",
         side, str(path), target, str(tidx), task, kind],
        input=WORKER, capture_output=True, text=True, cwd=tempfile.gettempdir(), timeout=3600,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {"scores": [], "error": f"worker crashed: {proc.stderr[-500:]}"}


def main() -> None:
    out = Path(sys.argv[1])
    rows = []
    with tempfile.TemporaryDirectory() as t:
        for fname, target, tidx, task in FILES:
            path = _subsample(RAW / fname, Path(t))
            for kind in ("linear", "tree"):
                r = {"file": fname, "target": target, "task": task, "model": kind,
                     "subsampled": path != RAW / fname}
                for side in ("v1", "v2"):
                    r[side] = _run(side, path, target, tidx, task, kind)
                rows.append(r)
                s = {k: (f"{len(r[k]['scores'])} folds" if not r[k]["error"] else "ERROR") for k in ("v1", "v2")}
                print(fname, kind, s, flush=True)
    out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
