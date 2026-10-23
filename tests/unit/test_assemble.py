"""
Unit tests for the Assembler (Phase 6 / M5): a fitted scikit-learn pipeline built from
Decisions that reproduces what `apply` produces, learns only from training rows, and
survives a save and reload.
"""

import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from preflight.assemble import build_pipeline_two_phase, overlay_actions, transform_new_data
from preflight.clean.cleaner import apply_clean, plan_clean_full
from preflight.engineer.engineer import apply_engineer, plan_engineer


def make_frame(n=160, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(50, 10, n)
    x[rng.choice(n, 12, replace=False)] = np.nan
    return pd.DataFrame({
        "x": x,
        "z": rng.normal(0, 1, n),
        "size": rng.choice(["small", "medium", "large"], n),
        "when": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
        "target": rng.integers(0, 2, n),
    })


def fitted(df, model_hint="tree"):
    cp = plan_clean_full(df, None, "target", task="classification")
    eng = plan_engineer(cp.cleaned, None, "target", model_hint, task="classification", signals=cp.signals)
    X, y = df.drop(columns=["target"]), df["target"]
    pipe = build_pipeline_two_phase(
        X, y, "target", model_hint, cp.decisions + eng, task="classification",
    )
    return cp, eng, pipe, X, y


@pytest.mark.parametrize("hint", ["tree", "linear"])
def test_pipeline_built_from_decisions_reproduces_apply(hint):
    df = make_frame()
    cp, eng, pipe, X, _ = fitted(df, hint)
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["cleaner", "engineer"]

    cleaned = apply_clean(df, cp.decisions)
    expected, _ = apply_engineer(cleaned, eng, target="target")
    expected = expected.drop(columns=["target"])

    got = pipe.transform(X)
    assert list(got.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False, rtol=1e-9, atol=1e-9,
    )


def test_fit_uses_training_rows_only():
    df = make_frame()
    train, hold = df.iloc[:120].copy(), df.iloc[120:].drop(columns=["target"]).copy()
    hold["x"] = np.nan  # every held-out value is blank
    hold.iloc[0, hold.columns.get_loc("x")] = 10_000.0  # and one extreme value

    cp = plan_clean_full(train, None, "target", task="classification")
    eng = plan_engineer(cp.cleaned, None, "target", "tree", task="classification", signals=cp.signals)
    pipe = build_pipeline_two_phase(
        train.drop(columns=["target"]), train["target"], "target", "tree",
        cp.decisions + eng, task="classification",
    )
    out = pipe.transform(hold)

    train_median = train["x"].median()
    pooled_median = pd.concat([train["x"], df["x"].iloc[120:]]).median()
    filled = out["x"].iloc[1:]
    assert train_median != pooled_median
    assert np.allclose(filled, train_median)
    # the extreme held-out value is pulled back to the limit learned from training rows
    cleaner = pipe.named_steps["cleaner"]
    limit = max(d.evidence["upper_bound"] for d in cleaner.decisions_ if d.kind == "clip_outliers" and d.columns == ["x"])
    assert out["x"].iloc[0] == pytest.approx(limit)


def test_transform_skips_row_removal_and_keeps_every_row():
    df = make_frame()
    dup = pd.concat([df, df.iloc[:30]], ignore_index=True)
    cp = plan_clean_full(dup, None, "target", task="classification")
    assert any(d.kind == "remove_duplicates" for d in cp.decisions)
    X = dup.drop(columns=["target"])
    pipe = build_pipeline_two_phase(X, dup["target"], "target", "tree", cp.decisions, task="classification")
    assert len(pipe.transform(X)) == len(X)


def test_unseen_values_and_missing_columns_in_new_data():
    df = make_frame()
    _, _, pipe, X, _ = fitted(df)
    new = X.iloc[:5].copy()
    new["size"] = ["huge", "small", None, "medium", "gigantic"]
    out = transform_new_data(pipe, new.assign(target=[0, 1, 0, 1, 0]), target="target")
    assert len(out) == 5 and "target" in out.columns
    assert not out.drop(columns=["target"]).isnull().all().any()


@pytest.mark.parametrize("dump,load", [
    (pickle.dumps, pickle.loads),
    (lambda p: _joblib_bytes(p), lambda b: joblib.load(__import__("io").BytesIO(b))),
])
def test_pipeline_serializes_and_reloads(dump, load):
    df = make_frame()
    _, _, pipe, X, _ = fitted(df, "linear")
    reloaded = load(dump(pipe))
    pd.testing.assert_frame_equal(pipe.transform(X), reloaded.transform(X))
    assert list(reloaded.named_steps) == ["cleaner", "engineer"]


def _joblib_bytes(p):
    import io
    buf = io.BytesIO()
    joblib.dump(p, buf)
    return buf.getvalue()


def test_overlay_copies_the_action_in_force_by_kind_and_column():
    df = make_frame()
    cp = plan_clean_full(df, None, "target")
    in_force = [d for d in cp.decisions if d.columns]
    changed = in_force[0]
    forced = type(changed).from_dict(changed.to_dict())
    forced.action = "some_other_action"
    fresh = plan_clean_full(df, None, "target").decisions
    out = overlay_actions(fresh, [forced])
    hit = [d for d in out if (d.kind, tuple(d.columns)) == (forced.kind, tuple(forced.columns))]
    assert hit and all(d.action == "some_other_action" for d in hit)
    assert overlay_actions(fresh, None) is fresh
