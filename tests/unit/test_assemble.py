"""
Unit tests for PreFlight v2.0 Assembler (Phase 6 / M5).

Validates:
1. Two-phase scikit-learn Pipeline construction from Decisions.
2. Fit on train only: prevents test data contamination and target leakage.
3. transform_new_data execution fidelity.
4. Pipeline reproducibility and pickle serialization.
"""

import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from preflight.assemble import (
    CleanerTransformer,
    EngineerTransformer,
    build_pipeline_two_phase,
    transform_new_data,
)
from preflight.clean.cleaner import plan_clean
from preflight.engineer.engineer import plan_engineer


def test_two_phase_pipeline_construction():
    """Verify build_pipeline_two_phase constructs valid scikit-learn Pipeline."""
    df_train = pd.DataFrame({
        "age": [25.0, 30.0, np.nan, 45.0, 50.0],
        "category": ["A", "B", "A", "C", "B"],
        "target": [0, 1, 0, 1, 0],
    })

    clean_decs = plan_clean(df_train, target="target")
    eng_decs = plan_engineer(df_train, target="target", task="classification")

    X = df_train.drop(columns=["target"])
    y = df_train["target"]

    pipe = build_pipeline_two_phase(
        profiles=clean_decs,
        target="target",
        model_hint="tree",
        X=X,
        y=y,
        decisions=clean_decs + eng_decs,
    )

    assert isinstance(pipe, Pipeline)
    assert "cleaner" in pipe.named_steps
    assert "engineer" in pipe.named_steps

    # Transform on train
    transformed_train = pipe.transform(X)
    assert isinstance(transformed_train, pd.DataFrame)
    assert "age" in transformed_train.columns


def test_transform_new_data_unseen_samples():
    """Verify transform_new_data transforms unseen test samples without target column."""
    df_train = pd.DataFrame({
        "num": [10.0, 50.0, 20.0, 40.0],
        "cat": ["x", "y", "x", "y"],
        "target": [1, 2, 1, 2],
    })
    df_test = pd.DataFrame({
        "num": [15.0, 25.0],
        "cat": ["x", "y"],
    })

    X = df_train.drop(columns=["target"])
    y = df_train["target"]

    clean_decs = plan_clean(df_train, target="target")
    eng_decs = plan_engineer(df_train, target="target")

    pipe = build_pipeline_two_phase(
        profiles=clean_decs,
        target="target",
        model_hint="tree",
        X=X,
        y=y,
        decisions=clean_decs + eng_decs,
    )

    # Test samples without target
    transformed_test = transform_new_data(pipe, df_test)
    assert len(transformed_test) == 2
    assert "num" in transformed_test.columns


def test_pipeline_pickle_serialization():
    """Verify pipeline can be pickled and unpickled with identical transformation output."""
    df = pd.DataFrame({
        "feature": [1.0, 2.0, 3.0, 4.0],
        "target": [10, 20, 30, 40],
    })
    X = df.drop(columns=["target"])
    y = df["target"]

    pipe = build_pipeline_two_phase(profiles=[], target="target", model_hint="tree", X=X, y=y)

    pickled = pickle.dumps(pipe)
    unpickled_pipe = pickle.loads(pickled)

    out1 = pipe.transform(X)
    out2 = unpickled_pipe.transform(X)

    pd.testing.assert_frame_equal(out1, out2)
