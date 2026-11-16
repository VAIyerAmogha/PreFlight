"""
Fit hygiene (docs/fit_hygiene.md): steps that learn from data live in an UNFITTED,
cloneable pipeline that is fitted inside CV folds; the full table only gets steps
that learn nothing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

from preflight.assemble import FoldCleaner
from preflight.engineer.encoding import target_encode_cross_fit
from preflight.plan import apply, plan
from preflight.prepare import prepare


def _write(tmp_path: Path, df: pd.DataFrame, name: str = "d.csv") -> Path:
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    p = d / name
    df.to_csv(p, index=False)
    return p


def _leaky_toy(n: int = 600, n_cats: int = 40, seed: int = 0) -> pd.DataFrame:
    # The target is pure noise: no feature carries real signal. Any score above chance
    # comes from a statistic that was computed with the rows being scored.
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "shop": [f"s{int(i)}" for i in r.integers(0, n_cats, n)],
        "noise": r.normal(0, 1, n).round(4),
        "target": r.integers(0, 2, n),
    })


def _run(tmp_path: Path, df: pd.DataFrame):
    res = apply(plan(_write(tmp_path, df), target="target"), mode="defaults", out_dir=tmp_path / "out")
    X, y = res.prepared.drop(columns=["target"]), res.prepared["target"].astype(int)
    return res, X, y


def test_in_fold_pipeline_scores_lower_than_a_full_data_fit_on_a_leaky_toy(tmp_path):
    res, X, y = _run(tmp_path, _leaky_toy())
    assert any(d.kind == "target_encode" for d in res.decisions)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    in_fold = cross_val_score(make_pipeline(clone(res.pipeline), LogisticRegression()), X, y, cv=cv, scoring="roc_auc")
    # The wrong way: the target encoding is learned once from every row, then CV only fits the model
    leaked_X = clone(res.pipeline).fit(X, y).transform(X)
    full_fit = cross_val_score(LogisticRegression(), leaked_X, y, cv=cv, scoring="roc_auc")

    assert in_fold.mean() < 0.60  # chance level, as it should be for a noise target
    assert full_fit.mean() > in_fold.mean() + 0.10  # optimistic, because the encoding saw the test rows


def test_pipeline_is_unfitted_and_clone_round_trips(tmp_path):
    res, X, y = _run(tmp_path, _leaky_toy(n=200))
    pipe = res.pipeline
    for _, step in pipe.steps:
        with pytest.raises(NotFittedError):
            check_is_fitted(step)
    twin = clone(pipe)
    assert twin is not pipe
    assert [n for n, _ in twin.steps] == [n for n, _ in pipe.steps] == ["cleaner", "engineer"]
    assert twin.get_params().keys() == pipe.get_params().keys()
    out_a = clone(pipe).fit(X, y).transform(X)
    out_b = twin.fit(X, y).transform(X)
    pd.testing.assert_frame_equal(out_a, out_b)
    for _, step in pipe.steps:  # fitting a clone never fits the original
        with pytest.raises(NotFittedError):
            check_is_fitted(step)


def _gappy(n: int = 300, seed: int = 1) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    x = r.normal(50, 5, n)
    x[r.choice(n, 30, replace=False)] = np.nan
    x[-60:] = x[-60:] + 40  # the last rows are shifted: their statistics differ from the rest
    cats = r.choice([f"c{i}" for i in range(30)], n)
    return pd.DataFrame({"x": x.round(3), "city": cats, "target": r.integers(0, 2, n)})


def test_fitting_inside_a_fold_uses_only_the_training_rows(tmp_path):
    res, X, y = _run(tmp_path, _gappy())
    train_idx, test_idx = np.arange(0, 240), np.arange(240, len(X))
    Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
    fitted = clone(res.pipeline).fit(Xtr, ytr)

    cleaner = fitted.named_steps["cleaner"]
    fill = next(d for d in cleaner.decisions_ if d.kind == "fill_median" and d.columns == ["x"])
    assert fill.evidence["fill_value"] == pytest.approx(Xtr["x"].median())
    assert fill.evidence["fill_value"] != pytest.approx(X["x"].median())

    clip = next(d for d in cleaner.decisions_ if d.kind == "clip_outliers" and d.columns == ["x"])
    filled = Xtr["x"].fillna(Xtr["x"].median())
    q1, q3 = filled.quantile(0.25), filled.quantile(0.75)
    assert clip.evidence["upper_bound"] == pytest.approx(q3 + 1.5 * (q3 - q1))

    eng = fitted.named_steps["engineer"]
    mapping = eng.specs_["city"]["mapping"]
    _, train_only = target_encode_cross_fit(Xtr["city"], ytr)
    assert mapping == pytest.approx(train_only)

    # new rows are transformed with what was learned from the training rows
    out = fitted.transform(X.iloc[test_idx])
    blank = X.iloc[test_idx]["x"].isna().to_numpy()
    assert blank.any()
    assert np.allclose(out["x"].to_numpy()[blank], min(fill.evidence["fill_value"], clip.evidence["upper_bound"]))


def test_rare_values_are_learned_from_the_training_rows_only(tmp_path):
    r = np.random.default_rng(3)
    common = [f"c{i}" for i in range(25)]  # high variety, so rare values are grouped
    head = list(r.choice(common, 398)) + ["q", "w"]  # training rows: 'q' is rare here (0.25%)
    tail = list(r.choice(common, 160)) + ["q"] * 40  # later rows: 'q' is common
    df = pd.DataFrame({"city": head + tail, "target": r.integers(0, 2, 600)})
    res, X, y = _run(tmp_path, df)
    planned = next(d for d in res.decisions if d.kind == "group_rare_categories")
    assert "q" not in planned.evidence["rare_categories"]  # whole table: 'q' is 41/600

    fc = clone(res.pipeline.named_steps["cleaner"]).fit(X.iloc[:400])
    learned = next(d for d in fc.decisions_ if d.kind == "group_rare_categories")
    assert "q" in learned.evidence["rare_categories"] and "w" in learned.evidence["rare_categories"]
    assert isinstance(fc, FoldCleaner)


def test_target_encoding_is_cross_fit_for_the_training_rows(tmp_path):
    res, X, y = _run(tmp_path, _leaky_toy(n=300))
    pipe = clone(res.pipeline)
    train_out = pipe.fit_transform(X, y)
    oof, _ = target_encode_cross_fit(pipe.named_steps["cleaner"].transform(X)["shop"], y)
    assert np.allclose(train_out["shop"].to_numpy(), oof.to_numpy())
    # new data (here the same rows, as if they were new) gets the mapping, which differs
    assert not np.allclose(pipe.transform(X)["shop"].to_numpy(), oof.to_numpy())


def test_full_table_gets_only_steps_that_learn_nothing(tmp_path):
    res, X, _ = _run(tmp_path, _gappy())
    assert X["x"].isna().sum() == 30  # blanks are not filled on the full table
    assert X["city"].dtype == object  # nor encoded
    kinds_in_force = {d.kind for d in res.decisions}
    assert {"fill_median", "clip_outliers", "target_encode"} <= kinds_in_force


def test_prepare_returns_unfitted_pipeline_and_a_named_all_rows_fit(tmp_path, monkeypatch):
    src = _write(tmp_path, _gappy())
    out = prepare(src, target="target", mode="defaults", out_dir=tmp_path / "out")
    for _, step in out["pipeline"].steps:
        with pytest.raises(NotFittedError):
            check_is_fitted(step)
    all_rows = out["pipeline_fitted_on_all_rows"]
    for _, step in all_rows.steps:
        check_is_fitted(step)
    assert out["paths"]["pipeline_fitted_on_all_rows"].exists()
    assert out["paths"]["pipeline"].exists()
    kf = KFold(3)
    X = out["apply_result"].prepared.drop(columns=["target"])
    y = out["apply_result"].prepared["target"]
    for tr, _ in kf.split(X):  # the unfitted pipeline is usable inside folds as returned
        clone(out["pipeline"]).fit(X.iloc[tr], y.iloc[tr])
