"""
Cleaning and engineering steps (Phase 6 / M5): plan purity, apply correctness, threshold
behaviour on destructive Decisions, and hand-built edge cases.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from preflight.clean.cleaner import apply_clean, plan_clean_full
from preflight.clean.config import CleanerConfig
from preflight.engineer.config import FeatureConfig
from preflight.engineer.engineer import apply_engineer, plan_engineer
from preflight.signals import compute_signals


def rng(seed=1):
    return np.random.default_rng(seed)


def kinds(decisions):
    return {(d.kind, tuple(d.columns)): d for d in decisions}


def mixed_frame(n=200):
    r = rng()
    x = r.normal(50, 5, n)
    x[0] = 500.0
    x[r.choice(n, 20, replace=False)[1:]] = np.nan
    return pd.DataFrame({
        "x": x,
        "size": r.choice(["small", "medium", "large", None], n),
        "city": r.choice([f"c{i}" for i in range(30)], n),
        "when": pd.date_range("2022-01-01", periods=n, freq="D"),
        "target": r.integers(0, 2, n),
    })


# --------------------------------------------------------------------------- purity

def test_planning_never_changes_the_table_and_is_repeatable():
    df = mixed_frame()
    before = df.copy(deep=True)
    a = plan_clean_full(df, None, "target", task="classification")
    pd.testing.assert_frame_equal(df, before)
    e1 = plan_engineer(a.cleaned, None, "target", "linear", task="classification", signals=a.signals,
                       feature_config=FeatureConfig(interactions=True, clustering=True))
    pd.testing.assert_frame_equal(df, before)
    b = plan_clean_full(df, None, "target", task="classification")
    e2 = plan_engineer(b.cleaned, None, "target", "linear", task="classification", signals=b.signals,
                       feature_config=FeatureConfig(interactions=True, clustering=True))
    assert [d.fingerprint for d in a.decisions + e1] == [d.fingerprint for d in b.decisions + e2]


def test_applying_never_changes_the_input_or_the_decisions():
    df = mixed_frame()
    cp = plan_clean_full(df, None, "target", task="classification")
    eng = plan_engineer(cp.cleaned, None, "target", "linear", task="classification", signals=cp.signals)
    before, decs = df.copy(deep=True), copy.deepcopy([d.to_dict() for d in cp.decisions + eng])
    out = apply_clean(df, cp.decisions)
    apply_engineer(out, eng, target="target")
    pd.testing.assert_frame_equal(df, before)
    assert decs == [d.to_dict() for d in cp.decisions + eng]


# --------------------------------------------------------------------------- apply == plan

def test_planned_values_are_the_values_apply_uses():
    df = mixed_frame()
    cp = plan_clean_full(df, None, "target", task="classification")
    out = apply_clean(df, cp.decisions)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), cp.cleaned.reset_index(drop=True), check_dtype=False)

    fill = kinds(cp.decisions)[("fill_median", ("x",))]
    n_blank = int(df["x"].isna().sum())
    assert int((out["x"] == fill.evidence["fill_value"]).sum()) >= n_blank
    assert out["x"].isna().sum() == 0
    assert "x_missing" in out.columns and int(out["x_missing"].sum()) == n_blank

    clip = kinds(cp.decisions)[("clip_outliers", ("x",))]
    assert out["x"].max() <= clip.evidence["upper_bound"] + 1e-9
    assert out["x"].min() >= clip.evidence["lower_bound"] - 1e-9


def test_engineering_adds_the_planned_columns():
    df = mixed_frame()
    cp = plan_clean_full(df, None, "target", task="classification")
    eng = plan_engineer(cp.cleaned, None, "target", "linear", task="classification", signals=cp.signals)
    out, specs = apply_engineer(cp.cleaned, eng, target="target")
    k = kinds(eng)
    assert ("expand_datetime", ("when",)) in k and "when" not in out.columns
    assert ("target_encode", ("city",)) in k and pd.api.types.is_numeric_dtype(out["city"])
    assert ("scale_numeric", ("x",)) in k and abs(out["x"].mean()) < 1e-6
    assert out["target"].equals(cp.cleaned["target"])  # the target is never altered
    assert "city" in specs


def test_tree_models_skip_rescaling_and_linear_models_do_not():
    df = mixed_frame()
    cp = plan_clean_full(df, None, "target", task="classification")
    tree = plan_engineer(cp.cleaned, None, "target", "tree", task="classification", signals=cp.signals)
    lin = plan_engineer(cp.cleaned, None, "target", "linear", task="classification", signals=cp.signals)
    assert not any(d.kind == "scale_numeric" for d in tree)
    assert any(d.kind == "scale_numeric" for d in lin)


# --------------------------------------------------------------------------- thresholds

def _dupes(extra):
    base = pd.DataFrame({"a": rng(2).normal(size=200), "target": rng(3).integers(0, 2, 200)})
    d = pd.concat([base, base.iloc[:extra]], ignore_index=True)
    return [x for x in plan_clean_full(d, None, "target").decisions if x.kind == "remove_duplicates"][0]


def test_row_removal_asks_only_above_the_threshold_and_defaults_to_keeping_rows():
    below, above = _dupes(4), _dupes(5)  # 4/204 = 1.96%, 5/205 = 2.44% vs 2%
    assert below.needs_attention is False and below.default == below.action == "remove_duplicates"
    assert above.needs_attention is True and above.default == "keep_rows" and above.action == "remove_duplicates"


def test_the_threshold_is_the_configured_one():
    base = pd.DataFrame({"a": rng(2).normal(size=200), "target": rng(3).integers(0, 2, 200)})
    d = pd.concat([base, base.iloc[:5]], ignore_index=True)
    strict = CleanerConfig(dedupe_impact_threshold=0.5)
    dec = [x for x in plan_clean_full(d, None, "target", strict).decisions if x.kind == "remove_duplicates"][0]
    assert dec.needs_attention is False


def test_column_removal_asks_when_a_quarter_of_the_table_goes():
    r = rng()
    # 1 of 5 columns constant = 20% (no question); 2 of 5 = 40% (question, default keeps them)
    one = pd.DataFrame({"c1": ["k"] * 100, "a": r.normal(size=100), "b": r.normal(size=100),
                        "d": r.normal(size=100), "target": r.integers(0, 2, 100)})
    two = one.assign(c2=["j"] * 100)
    d1 = kinds(plan_clean_full(one, None, "target").decisions)[("drop_constant", ("c1",))]
    d2 = kinds(plan_clean_full(two, None, "target").decisions)[("drop_constant", ("c1",))]
    assert d1.needs_attention is False and d1.default == "drop_column"
    assert d2.needs_attention is True and d2.default == "keep_column" and d2.action == "drop_column"


def test_outlier_pull_back_asks_above_five_percent_and_keeps_values_by_default():
    r = rng()
    base = r.normal(0, 1, 200)
    few, many = base.copy(), base.copy()
    few[:3] = 50.0    # 1.5%
    many[:16] = 50.0  # 8%
    def clip(x):
        df = pd.DataFrame({"x": x, "u": np.arange(200) * 1.5, "target": r.integers(0, 2, 200)})  # "u": no repeated rows
        return kinds(plan_clean_full(df, None, "target").decisions)[("clip_outliers", ("x",))]
    f, m = clip(few), clip(many)
    assert f.needs_attention is False and f.default == "clip_outliers"
    assert m.needs_attention is True and m.default == "keep_outliers" and m.action == "clip_outliers"


# --------------------------------------------------------------------------- edge cases

def run_both(df, target="target", hint="linear"):
    cp = plan_clean_full(df, None, target, task="classification")
    cleaned = apply_clean(df, cp.decisions)
    eng = plan_engineer(cp.cleaned, None, target, hint, task="classification", signals=cp.signals)
    out, _ = apply_engineer(cleaned, eng, target=target)
    return cp, eng, out


def test_all_blank_column_is_proposed_for_removal_not_filled():
    df = pd.DataFrame({"a": [np.nan] * 100, "b": rng().normal(size=100), "target": rng().integers(0, 2, 100)})
    cp, _, out = run_both(df)
    d = kinds(cp.decisions)[("drop_high_missingness", ("a",))]
    assert d.action == "drop_column" and d.evidence["missing_rate"] == 1.0
    assert "a" not in out.columns and not out.isnull().any().any()


def test_constant_column_is_proposed_for_removal():
    df = pd.DataFrame({"a": ["k"] * 100, "b": rng().normal(size=100), "target": rng().integers(0, 2, 100)})
    cp, _, out = run_both(df)
    assert ("drop_constant", ("a",)) in kinds(cp.decisions)
    assert "a" not in out.columns


def test_single_row_table_does_not_crash():
    df = pd.DataFrame({"a": [1.0], "b": ["x"], "target": [1]})
    _, _, out = run_both(df)
    assert len(out) == 1 and "target" in out.columns


def test_high_variety_column_is_encoded_from_the_target_not_expanded():
    r = rng()
    df = pd.DataFrame({"a": [f"v{i}" for i in range(100)], "b": r.normal(size=100), "target": r.integers(0, 2, 100)})
    cp, eng, out = run_both(df, hint="tree")
    assert ("target_encode", ("a",)) in kinds(eng)
    assert out.shape[1] == 3 and pd.api.types.is_numeric_dtype(out["a"])


def test_low_variety_column_is_expanded_for_linear_and_numbered_for_tree():
    r = rng()
    df = pd.DataFrame({"a": r.choice(["p", "q", "r"], 100), "b": r.normal(size=100), "target": r.integers(0, 2, 100)})
    _, eng_lin, out_lin = run_both(df, hint="linear")
    _, eng_tree, out_tree = run_both(df, hint="tree")
    assert ("one_hot_encode", ("a",)) in kinds(eng_lin) and "a" not in out_lin.columns
    assert ("ordinal_encode", ("a",)) in kinds(eng_tree) and "a" in out_tree.columns


def test_perfectly_leaky_feature_asks_and_is_kept_unless_the_person_says_otherwise():
    r = rng()
    y = r.normal(size=100)
    df = pd.DataFrame({"leak": y * 1.0001, "b": r.normal(size=100), "target": y})
    sig = compute_signals(df, None, "target")
    assert sig["leak"].is_leakage_suspect is True
    cp, eng, out = run_both(df, hint="tree")
    d = kinds(eng)[("suspected_leakage", ("leak",))]
    assert d.needs_attention is True and d.action == d.default == "keep_feature"
    assert "leak" in out.columns  # v1 only reported leakage; it never removed the column

    d.action = "drop_leaky_feature"
    dropped, _ = apply_engineer(apply_clean(df, cp.decisions), eng, target="target")
    assert "leak" not in dropped.columns and "b" in dropped.columns


def test_target_is_never_removed_or_filled():
    df = pd.DataFrame({"a": rng().normal(size=100), "target": [np.nan] * 100})
    cp, _, out = run_both(df)
    assert "target" in out.columns
    assert not any(d.columns == ["target"] for d in cp.decisions if d.kind.startswith("drop"))
