"""
Two-phase pipeline assembler for PreFlight v2.0.

Ported from v1 `assembler.py`. The fitted scikit-learn Pipeline is built from
Decisions, so it does what the plan said:

1. The cleaner step is fitted on the training table only (its plan is
   recomputed there, so fill values, limits and rare-value lists never see
   test data) and then applied to new data, minus duplicate-row removal
   (v1: removing rows at inference would silently drop rows the caller
   expects back).
2. The engineer step needs the columns that survived cleaning, so it is
   fitted after the cleaner (the "two phases"), again on training data only.

Decisions handed in (for example the ones a person confirmed or changed) are
matched to the recomputed ones by kind and column, and only their action is
carried over.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from preflight.clean.cleaner import apply_clean, plan_clean_full
from preflight.clean.config import DEFAULT_CLEANER_CONFIG, CleanerConfig
from preflight.decisions import Decision
from preflight.engineer.config import DEFAULT_ENGINEER_CONFIG, EngineerConfig, FeatureConfig
from preflight.engineer.datetime import expand_datetime, generate_datetime_cyclical_features
from preflight.engineer.engineer import apply_engineer, plan_engineer, _target_numeric
from preflight.engineer.features import generate_interaction_features
from preflight.schema import TableSchema
from preflight.signals import ColumnSignals

__all__ = [
    "CleanerTransformer",
    "EngineerTransformer",
    "build_pipeline",
    "build_pipeline_two_phase",
    "transform_new_data",
    "run_assembler",
]


def overlay_actions(fresh: List[Decision], in_force: Optional[Iterable[Decision]]) -> List[Decision]:
    """
    Copy the action in force from `in_force` onto the recomputed `fresh`
    Decisions with the same kind and columns. Recomputed values (evidence)
    are kept; Decisions that exist only on one side keep their own action.
    """
    if not in_force:
        return fresh
    lookup = {(d.kind, tuple(d.columns)): d for d in in_force}
    for nd in fresh:
        old = lookup.get((nd.kind, tuple(nd.columns)))
        if old is not None:
            nd.action = old.action
            nd.status = old.status
    return fresh


class CleanerTransformer(BaseEstimator, TransformerMixin):
    """Cleaning step: fitted on training data, applied to any table."""

    def __init__(
        self,
        decisions: Optional[List[Decision]] = None,
        target: Optional[str] = None,
        schema: Optional[TableSchema] = None,
        config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
        task: Optional[str] = None,
    ) -> None:
        self.decisions = decisions
        self.target = target
        self.schema = schema
        self.config = config
        self.task = task

    def fit(self, X: pd.DataFrame, y=None) -> "CleanerTransformer":
        df = X.copy()
        if y is not None and self.target is not None and self.target not in df.columns:
            df[self.target] = list(y)
        plan = plan_clean_full(df, self.schema, self.target, self.config, task=self.task)
        self.decisions_ = overlay_actions(plan.decisions, self.decisions)
        self.signals_ = plan.signals
        self.columns_to_drop_ = [
            d.columns[0] for d in self.decisions_ if d.action == "drop_column" and d.columns
        ]
        self.output_columns_ = list(self.transform(X).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "decisions_"):
            raise NotFittedError("CleanerTransformer is not fitted yet. Call 'fit' before 'transform'.")
        decs = [d for d in self.decisions_ if d.kind != "remove_duplicates"]
        return apply_clean(X, decs)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "output_columns_"):
            raise NotFittedError("CleanerTransformer is not fitted yet.")
        return np.array(self.output_columns_)


class EngineerTransformer(BaseEstimator, TransformerMixin):
    """Engineering step: fitted on cleaned training data (needs the target for encoding)."""

    def __init__(
        self,
        decisions: Optional[List[Decision]] = None,
        target: Optional[str] = None,
        model_hint: str = "tree",
        schema: Optional[TableSchema] = None,
        signals: Optional[Dict[str, ColumnSignals]] = None,
        cardinality_threshold: int = 20,
        feature_config: Optional[FeatureConfig] = None,
        task: str = "regression",
        config: EngineerConfig = DEFAULT_ENGINEER_CONFIG,
    ) -> None:
        self.decisions = decisions
        self.target = target
        self.model_hint = model_hint
        self.schema = schema
        self.signals = signals
        self.cardinality_threshold = cardinality_threshold
        self.feature_config = feature_config
        self.task = task
        self.config = config

    def fit(self, X: pd.DataFrame, y=None) -> "EngineerTransformer":
        df = X.copy()
        if y is not None and self.target is not None and self.target not in df.columns:
            df[self.target] = list(y)
        fresh = plan_engineer(
            df,
            schema=self.schema,
            target=self.target,
            model_hint=self.model_hint,
            cardinality_threshold=self.cardinality_threshold,
            feature_config=self.feature_config,
            task=self.task,
            signals=self.signals,
            config=self.config,
        )
        self.decisions_ = overlay_actions(fresh, self.decisions)
        needs_y = any(d.action == "target_encode_cross_fit" for d in self.decisions_)
        if needs_y and (y is None and self.target not in df.columns):
            raise ValueError("Target y must be provided for target encoding high-cardinality columns.")

        out, self.specs_ = apply_engineer(df, self.decisions_, target=self.target)
        cols = list(out.columns)
        if self.target in cols and self.target not in X.columns:
            cols.remove(self.target)
        self.output_columns_ = cols

        y_numeric = _target_numeric(df, self.target)
        if y_numeric is not None:
            global_mean = y_numeric.mean()
            for spec in self.specs_.values():
                if isinstance(spec, dict) and spec.get("transform") == "target_encode_cross_fit":
                    spec["global_mean"] = global_mean
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "specs_"):
            raise NotFittedError("EngineerTransformer is not fitted yet.")

        X_in = X.copy()
        X_out = X.copy()

        for col, spec in self.specs_.items():
            if col.startswith("_") or col == self.target or col not in X_out.columns:
                continue
            trans = spec.get("transform")
            if trans == "expand_datetime":
                expanded = expand_datetime(X_out[col])
                X_out = X_out.drop(columns=[col]).join(expanded)
            elif trans == "ordinal_encode":
                X_out[col] = X_out[col].map(spec["mapping"]).fillna(-1).astype(int)
            elif trans == "one_hot_encode":
                dummies = pd.get_dummies(X_out[col], prefix=col, prefix_sep="_")
                X_out = X_out.drop(columns=[col]).join(dummies)
            elif trans == "target_encode_cross_fit":
                X_out[col] = X_out[col].map(spec["mapping"]).fillna(spec.get("global_mean", 0.0))
            elif trans == "linear_numeric":
                if spec.get("log1p"):
                    X_out[col] = np.log1p(X_out[col])
                params = spec["scale_params"]
                X_out[col] = (X_out[col] - params["mean"]) / params["std"]
            elif trans == "drop_leaky":
                X_out = X_out.drop(columns=[col])

        X_out = self._generated(X_in, X_out)

        for expected_col in self.output_columns_:
            if expected_col not in X_out.columns:
                X_out[expected_col] = False
        return X_out.reindex(columns=self.output_columns_)

    def _generated(self, X_in: pd.DataFrame, X_out: pd.DataFrame) -> pd.DataFrame:
        """Generated features (FeatureConfig) for new data, from the fitted state."""
        s = self.specs_

        if "_interactions" in s:
            spec = s["_interactions"]
            cfg = FeatureConfig(interaction_types=spec["interaction_types"])
            gen, _ = generate_interaction_features(X_out, spec["top_columns"], cfg)
            for c in spec["created"]:
                if c in gen.columns:
                    X_out[c] = gen[c]

        if "_datetime" in s:
            spec = s["_datetime"]
            cfg = FeatureConfig(
                datetime_cyclical=spec["cyclical"],
                datetime_deltas=spec["deltas"],
                datetime_reference_col=spec["reference_col"],
            )
            # Generated from the original date columns, i.e. before they were split into parts.
            cols = [c for c in spec["columns"] if c in X_in.columns]
            dt_df, _ = generate_datetime_cyclical_features(X_in, cols, cfg)
            for c in spec["created"]:
                if c in dt_df.columns:
                    X_out[c] = dt_df[c]

        if "_clustering" in s:
            spec = s["_clustering"]
            info = spec["fitted_info"]
            model, feats = info["model"], info["features"]
            if all(f in X_out.columns for f in feats):
                Xc = X_out[feats].fillna(0)
                if "cluster_label" in spec["created"]:
                    X_out["cluster_label"] = model.predict(Xc)
                if "cluster_dist_to_centroid" in spec["created"]:
                    X_out["cluster_dist_to_centroid"] = model.transform(Xc).min(axis=1)

        if "_text_features" in s:
            spec = s["_text_features"]
            created = set(spec["created"])
            vecs = (spec.get("fitted_info") or {}).get("vectorizers", {})
            for col in spec["columns"]:
                if col not in X_out.columns:
                    continue
                series = X_out[col].fillna("").astype(str)
                feats = {
                    f"{col}_char_length": series.str.len(),
                    f"{col}_word_count": series.str.split().str.len(),
                    f"{col}_has_text": (series.str.strip().str.len() > 0).astype(int),
                }
                for name, vals in feats.items():
                    if name in created:
                        X_out[name] = vals
                vec = vecs.get(col)
                if spec.get("tfidf") and vec is not None:
                    mat = vec.transform(series)
                    names = [f"{col}_tfidf_{t}" for t in vec.get_feature_names_out()]
                    tf = pd.DataFrame(mat.toarray(), columns=names, index=series.index)
                    for name in names:
                        if name in created:
                            X_out[name] = tf[name]
        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if not hasattr(self, "output_columns_"):
            raise NotFittedError("EngineerTransformer is not fitted yet.")
        return np.array(self.output_columns_)


def build_pipeline(
    target: Optional[str],
    model_hint: str = "tree",
    decisions: Optional[List[Decision]] = None,
    schema: Optional[TableSchema] = None,
    feature_config: Optional[FeatureConfig] = None,
    cleaner_config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
    cardinality_threshold: int = 20,
    task: str = "regression",
) -> Pipeline:
    """UNFITTED two-step pipeline. Use `build_pipeline_two_phase` for a usable one."""
    clean_in = [d for d in (decisions or []) if d.stage == "clean"]
    eng_in = [d for d in (decisions or []) if d.stage == "engineer"]
    pipe = Pipeline([
        ("cleaner", CleanerTransformer(clean_in, target, schema, cleaner_config)),
        ("engineer", EngineerTransformer(
            eng_in, target, model_hint, schema, None, cardinality_threshold, feature_config, task)),
    ])
    pipe.set_output(transform="pandas")
    return pipe


def build_pipeline_two_phase(
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    model_hint: str = "tree",
    decisions: Optional[List[Decision]] = None,
    schema: Optional[TableSchema] = None,
    feature_config: Optional[FeatureConfig] = None,
    cleaner_config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
    cardinality_threshold: int = 20,
    task: str = "regression",
) -> Pipeline:
    """
    Build and fit the pipeline on training data (X, y): the cleaner first, then
    the engineer on the cleaned columns. Returns a fitted Pipeline.
    """
    clean_in = [d for d in (decisions or []) if d.stage == "clean"]
    eng_in = [d for d in (decisions or []) if d.stage == "engineer"]

    cleaner = CleanerTransformer(clean_in, target, schema, cleaner_config, None)
    cleaner.fit(X, y)
    X_cleaned = cleaner.transform(X)

    engineer = EngineerTransformer(
        eng_in, target, model_hint, schema, cleaner.signals_,
        cardinality_threshold, feature_config, task,
    )
    engineer.fit(X_cleaned, y)

    pipe = Pipeline([("cleaner", cleaner), ("engineer", engineer)])
    pipe.set_output(transform="pandas")
    return pipe


def transform_new_data(
    pipeline: Pipeline,
    new_df: pd.DataFrame,
    target: Optional[str] = None,
) -> pd.DataFrame:
    """Transform unseen data with a fitted pipeline; a target column, if present, is carried through."""
    df_in = new_df.copy()
    target_series = None
    if target is not None and target in df_in.columns:
        target_series = df_in.pop(target)
    df_out = pipeline.transform(df_in)
    if target_series is not None:
        df_out[target] = target_series
    return df_out


def run_assembler(
    df: pd.DataFrame,
    target: str,
    task: str,
    model_hint: str,
    decisions: Optional[List[Decision]] = None,
    schema: Optional[TableSchema] = None,
    feature_config: Optional[FeatureConfig] = None,
    cleaner_config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
    cardinality_threshold: int = 20,
) -> Tuple[pd.DataFrame, Pipeline]:
    """Fit the pipeline on all of `df` and return (transformed table with target, pipeline)."""
    X = df.drop(columns=[target])
    y = df[target]
    pipe = build_pipeline_two_phase(
        X, y, target, model_hint, decisions, schema, feature_config,
        cleaner_config, cardinality_threshold, task,
    )
    out = pipe.transform(X)
    out[target] = y
    return out, pipe
