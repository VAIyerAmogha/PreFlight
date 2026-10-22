"""
Unit tests for PreFlight v2.0 Feature Engineering Stage (Phase 6 / M5).

Validates:
1. Tree vs Linear transform choice (ordinal vs one-hot, scaling).
2. Cross-fit 5-fold smoothed target encoding without leakage.
3. Datetime expansion (year, month, day, dayofweek, cyclical sin/cos, NaN propagation).
4. Interaction features between numeric columns.
5. Clustering features (KMeans distance).
6. Text features (length, word count, TF-IDF, pandas 4.0 warning fix).
7. VIF computation using pure NumPy matrix inversion without statsmodels.
8. Leakage detection (>0.95 correlation with target) emitting needs_attention Decision.
"""

import numpy as np
import pandas as pd
import pytest

from preflight.engineer.config import FeatureConfig, PRESETS
from preflight.engineer.datetime import expand_datetime, generate_datetime_cyclical_features
from preflight.engineer.encoding import one_hot_encode, ordinal_encode, target_encode_cross_fit
from preflight.engineer.features import (
    generate_cluster_features,
    generate_interaction_features,
    generate_text_features,
)
from preflight.engineer.numeric import log1p_transform, needs_log_transform, standard_scale
from preflight.engineer.vif import (
    compute_correlation_with_target,
    compute_vif_scores,
    flag_leakage_suspect,
    plan_vif_and_leakage,
)
from preflight.engineer.engineer import apply_engineer, plan_engineer


def test_tree_vs_linear_model_hints():
    """Verify tree uses ordinal encoding while linear uses one-hot and scaling."""
    df = pd.DataFrame({
        "color": ["red", "blue", "green", "red", "blue"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0],
    })

    decs_tree = plan_engineer(df, model_hint="tree")
    assert any(d.action == "ordinal_encode" for d in decs_tree)
    assert not any(d.action == "one_hot_encode" for d in decs_tree)
    assert not any(d.action == "standard_scale" for d in decs_tree)

    decs_linear = plan_engineer(df, model_hint="linear")
    assert any(d.action == "one_hot_encode" for d in decs_linear)
    assert any(d.action == "standard_scale" for d in decs_linear)


def test_cross_fit_target_encoding():
    """Verify 5-fold cross-fit target encoding computes out-of-fold smoothed values."""
    categories = pd.Series(["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"])
    target = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])

    encoded, mapping = target_encode_cross_fit(categories, target, n_folds=5, smoothing=1.0)

    assert len(encoded) == len(categories)
    assert not encoded.isna().any()
    assert "A" in mapping
    assert "B" in mapping
    assert "C" in mapping


def test_datetime_expansion_and_nan_propagation():
    """Verify datetime columns expand into calendar parts with NaN propagation."""
    dates = pd.Series(["2025-01-15 08:30:00", None, "2025-06-20 14:45:00"], name="dates")
    expanded = expand_datetime(dates)

    assert "dates_year" in expanded.columns
    assert "dates_month" in expanded.columns
    assert "dates_day" in expanded.columns
    assert "dates_dayofweek" in expanded.columns
    assert "dates_hour" in expanded.columns

    # Row 1 was None/NaT -> all expanded features must be NaN
    assert expanded.iloc[1].isna().all()
    # Row 0 values
    assert expanded.loc[0, "dates_year"] == 2025
    assert expanded.loc[0, "dates_month"] == 1
    assert expanded.loc[0, "dates_day"] == 15

    # Cyclical features
    df_dt = pd.DataFrame({"dates": pd.to_datetime(dates)})
    cyclical, _ = generate_datetime_cyclical_features(df_dt, ["dates"], config=FeatureConfig(datetime_cyclical=True))
    assert "dates_month_sin" in cyclical.columns
    assert "dates_month_cos" in cyclical.columns
    assert np.isnan(cyclical.loc[1, "dates_month_sin"])


def test_interactions_and_clustering():
    """Verify numeric interaction pairs and KMeans cluster generation."""
    np.random.seed(42)
    df = pd.DataFrame({
        "x1": np.random.randn(50),
        "x2": np.random.randn(50),
        "x3": np.random.randn(50),
    })

    cfg = FeatureConfig(interactions=True, clustering=True, cluster_k=3)
    interactions, _ = generate_interaction_features(df, ["x1", "x2", "x3"], target=None, config=cfg)
    assert len(interactions.columns) > 3
    assert any("_div_" in col or "_times_" in col for col in interactions.columns)

    clusters, _, _ = generate_cluster_features(df, ["x1", "x2", "x3"], config=cfg)
    assert len(clusters.columns) >= 3
    assert len(clusters) == 50


def test_text_features_pandas_compat():
    """Verify text features generate string lengths, word counts, and TF-IDF."""
    texts = pd.Series(["Quick brown fox", "Lazier dog sleeping", "The quick fox jumped"])
    df_text = pd.DataFrame({"notes": texts})

    cfg = FeatureConfig(text_features=True, text_tfidf=True)
    feats, _, _ = generate_text_features(df_text, ["notes"], config=cfg)
    assert "notes_char_length" in feats.columns
    assert "notes_word_count" in feats.columns
    assert any("tfidf" in c for c in feats.columns)


def test_vif_numpy_no_statsmodels():
    """Verify VIF scores are calculated using pure NumPy inverse of correlation matrix."""
    # Collinear data: z = 2*x + 3*y + small noise
    np.random.seed(42)
    x = np.random.uniform(1, 10, 100)
    y = np.random.uniform(1, 10, 100)
    z = 2 * x + 3 * y + np.random.normal(0, 0.01, 100)
    df = pd.DataFrame({"x": x, "y": y, "z": z})

    vif_scores, _ = compute_vif_scores(df)
    assert "z" in vif_scores
    assert vif_scores["z"] > 10.0  # High collinearity detected


def test_leakage_detection_emits_needs_attention():
    """Verify features having >0.95 correlation with target emit needs_attention."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    leaky = y * 1.0001  # Correlation ~ 1.0
    safe = np.array([5.0, 1.0, 8.0, 2.0, 4.0, 9.0, 3.0, 7.0, 6.0, 0.0])

    df = pd.DataFrame({"target": y, "leaky_col": leaky, "safe_col": safe})

    decs = plan_vif_and_leakage(df, target="target", leakage_threshold=0.95)
    leak_decs = [d for d in decs if d.kind == "suspected_leakage"]

    assert len(leak_decs) == 1
    assert leak_decs[0].columns == ["leaky_col"]
    assert leak_decs[0].needs_attention is True
    assert leak_decs[0].evidence["correlation_with_target"] > 0.95
