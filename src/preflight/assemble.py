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

from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from preflight.clean.cleaner import apply_clean, apply_coerce, plan_clean_full
from preflight.clean.config import DEFAULT_CLEANER_CONFIG, CleanerConfig
from preflight.clean.dedupe import apply_dedupe
from preflight.clean.drops import apply_drops
from preflight.clean.impute import apply_impute
from preflight.clean.markers import apply_markers
from preflight.clean.normalize import apply_normalization, identify_rare_categories
from preflight.clean.outliers import apply_outliers
from preflight.decisions import Decision
from preflight.engineer.config import DEFAULT_ENGINEER_CONFIG, EngineerConfig, FeatureConfig
from preflight.engineer.datetime import expand_datetime, generate_datetime_cyclical_features
from preflight.engineer.encoding import target_encode_cross_fit
from preflight.engineer.engineer import _target_numeric, apply_engineer, plan_engineer
from preflight.engineer.features import generate_interaction_features
from preflight.schema import TableSchema
from preflight.signals import ColumnSignals, compute_signals

__all__ = [
    "STATELESS_CLEAN_KINDS",
    "PIPELINE_CLEAN_KINDS",
    "apply_stateless",
    "plan_time_column_kinds",
    "FoldCleaner",
    "build_unfitted_pipeline",
    "CleanerTransformer",
    "EngineerTransformer",
    "build_pipeline",
    "build_pipeline_two_phase",
    "transform_new_data",
    "run_assembler",
]


def overlay_actions(fresh: list[Decision], in_force: Optional[Iterable[Decision]]) -> list[Decision]:
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


# ---------------------------------------------------------------------------
# Fit hygiene (docs/fit_hygiene.md)
#
# Steps before the first one that learns from data run on the full table
# (apply_stateless). Every step from rare-value grouping onwards runs inside the
# UNFITTED pipeline from build_unfitted_pipeline, which callers fit inside their
# CV folds. Which columns get which step is fixed by the plan; only values
# (fill values, limits, rare lists, encodings, scaling, ...) are learned in fit.
# ---------------------------------------------------------------------------

STATELESS_CLEAN_KINDS: tuple[str, ...] = (
    "convert_missing_markers", "coerce_numeric", "remove_duplicates",
    "drop_high_missingness", "drop_id", "drop_constant", "normalize_categories",
)
PIPELINE_CLEAN_KINDS: tuple[str, ...] = (
    "group_rare_categories", "add_missing_indicator", "fill_median", "fill_mode",
    "constant_fill", "clip_outliers",
)


def apply_stateless(df: pd.DataFrame, decisions: list[Decision]) -> pd.DataFrame:
    """The steps that learn nothing from data, in plan order, on the full table."""
    decs = [d for d in decisions if d.stage == "clean" and d.kind in STATELESS_CLEAN_KINDS]
    out = apply_coerce(apply_markers(df.copy(), decs), decs)
    out, _ = apply_dedupe(out, decs)
    out, _ = apply_drops(out, decs)
    return apply_normalization(out, decs)


def plan_time_column_kinds(
    typed: pd.DataFrame,
    decisions: list[Decision],
    schema: Optional[TableSchema],
    target: Optional[str],
    task: Optional[str],
    config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
) -> dict[str, str]:
    """
    Column kinds (numeric, low/high-variety categorical, ...) exactly as the plan
    computed them (after markers and numeric coercion). They choose each column's
    encoder and stay fixed in every fold.
    """
    decs = [d for d in decisions if d.stage == "clean"]
    work = apply_coerce(apply_markers(typed.copy(), decs), decs)
    sigs = compute_signals(
        work, schema=schema, target=target, task=task,
        cardinality_threshold=config.cardinality_threshold,
        rare_threshold=config.rare_category_threshold,
    )
    return {c: s.kind for c, s in sigs.items()}


def _with_evidence(d: Decision, **values: Any) -> Decision:
    nd = Decision.from_dict(d.to_dict())
    nd.evidence = {**nd.evidence, **values}
    return nd


class FoldCleaner(BaseEstimator, TransformerMixin):
    """
    The cleaning steps that learn from data: rare-value grouping, blank flags and
    fills, and clipping. `decisions` are the plan's Decisions (with the actions in
    force); `fit` recomputes their values from the rows it is given only.
    """

    def __init__(
        self,
        decisions: Optional[list[Decision]] = None,
        config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
    ) -> None:
        self.decisions = decisions
        self.config = config

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FoldCleaner":
        decs = [d for d in (self.decisions or []) if d.stage == "clean" and d.kind in PIPELINE_CLEAN_KINDS]
        work = X.copy()
        fitted: list[Decision] = []

        rare = []
        for d in decs:
            if d.kind == "group_rare_categories" and d.columns and d.columns[0] in work.columns:
                found = identify_rare_categories(work[d.columns[0]], threshold=self.config.rare_category_threshold)
                rare.append(_with_evidence(d, rare_categories=found))
        work = apply_normalization(work, rare)
        fitted.extend(rare)

        gaps = []
        for d in decs:
            if d.kind not in ("add_missing_indicator", "fill_median", "fill_mode", "constant_fill"):
                continue
            col = d.columns[0] if d.columns else None
            if d.kind == "fill_median" and col in work.columns:
                s = pd.to_numeric(work[col], errors="coerce")
                gaps.append(_with_evidence(d, fill_value=float(s.median()) if s.notna().any() else 0.0))
            elif d.kind == "fill_mode" and col in work.columns:
                modes = work[col].mode(dropna=True)
                gaps.append(_with_evidence(d, fill_value=np.nan if modes.empty else modes.iloc[0]))
            else:
                gaps.append(d)
        work = apply_impute(work, gaps)
        fitted.extend(gaps)

        for d in decs:
            if d.kind != "clip_outliers" or not d.columns or d.columns[0] not in work.columns:
                continue
            s = pd.to_numeric(work[d.columns[0]], errors="coerce")
            lower, upper = self._limits(s)
            fitted.append(_with_evidence(d, lower_bound=lower, upper_bound=upper, method=self.config.outlier_method))

        self.decisions_ = fitted
        self.output_columns_ = list(self._apply(X).columns)
        return self

    def _limits(self, s: pd.Series) -> tuple[Optional[float], Optional[float]]:
        """Clipping limits from these rows (same rules as clean.outliers.plan_outliers); None when the spread is zero."""
        if self.config.outlier_method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                return None, None
            return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        if self.config.outlier_method == "zscore":
            mean, std = s.mean(), s.std()
            if std == 0 or pd.isna(std):
                return None, None
            return float(mean - 3 * std), float(mean + 3 * std)
        raise ValueError(
            f"Outlier clipping method '{self.config.outlier_method}' is not supported. Use 'iqr' or 'zscore'."
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "decisions_"):
            raise NotFittedError("FoldCleaner is not fitted yet. Call 'fit' before 'transform'.")
        return self._apply(X)

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:
        # Not wrapped by sklearn's set_output, so fit can use it before output_columns_ exists.
        out = apply_normalization(X.copy(), self.decisions_)
        out = apply_impute(out, self.decisions_)
        return apply_outliers(out, self.decisions_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "output_columns_"):
            raise NotFittedError("FoldCleaner is not fitted yet.")
        return np.array(self.output_columns_)


def build_unfitted_pipeline(
    target: Optional[str],
    decisions: list[Decision],
    column_kinds: dict[str, str],
    schema: Optional[TableSchema] = None,
    model_hint: str = "tree",
    feature_config: Optional[FeatureConfig] = None,
    cleaner_config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
    task: str = "regression",
) -> Pipeline:
    """
    UNFITTED pipeline holding every step that learns from data. Its input is the
    table from apply_stateless without the target column. Fit it inside each CV
    fold (it works with sklearn.base.clone); target encoding is cross-fit in fit_transform.
    """
    pipe = Pipeline([
        ("cleaner", FoldCleaner([d for d in decisions if d.stage == "clean"], cleaner_config)),
        ("engineer", EngineerTransformer(
            [d for d in decisions if d.stage == "engineer"], target, model_hint, schema, None,
            cleaner_config.cardinality_threshold, feature_config, task,
            column_kinds=dict(column_kinds),
        )),
    ])
    pipe.set_output(transform="pandas")
    return pipe


class CleanerTransformer(BaseEstimator, TransformerMixin):
    """Cleaning step: fitted on training data, applied to any table."""

    def __init__(
        self,
        decisions: Optional[list[Decision]] = None,
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

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CleanerTransformer":
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

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "output_columns_"):
            raise NotFittedError("CleanerTransformer is not fitted yet.")
        return np.array(self.output_columns_)


class EngineerTransformer(BaseEstimator, TransformerMixin):
    """Engineering step: fitted on cleaned training data (needs the target for encoding)."""

    def __init__(
        self,
        decisions: Optional[list[Decision]] = None,
        target: Optional[str] = None,
        model_hint: str = "tree",
        schema: Optional[TableSchema] = None,
        signals: Optional[dict[str, ColumnSignals]] = None,
        cardinality_threshold: int = 20,
        feature_config: Optional[FeatureConfig] = None,
        task: str = "regression",
        config: EngineerConfig = DEFAULT_ENGINEER_CONFIG,
        column_kinds: Optional[dict[str, str]] = None,
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
        self.column_kinds = column_kinds

    def _fold_signals(self, df: pd.DataFrame) -> Optional[dict[str, ColumnSignals]]:
        """
        With column_kinds (the plan's kinds), signals such as correlation with the
        target are recomputed from these rows only, while each column keeps its plan kind.
        """
        if self.column_kinds is None:
            return self.signals
        sigs = compute_signals(
            df, schema=self.schema, target=self.target, task=self.task,
            cardinality_threshold=self.cardinality_threshold,
        )
        return {c: replace(sg, kind=self.column_kinds.get(c, sg.kind)) for c, sg in sigs.items()}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "EngineerTransformer":
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
            signals=self._fold_signals(df),
            config=self.config,
        )
        self.decisions_ = overlay_actions(fresh, self.decisions)
        # A column the person chose to leave out stays out even when these rows
        # alone would not raise the question.
        have = {(d.kind, tuple(d.columns)) for d in self.decisions_}
        chosen = [
            Decision.from_dict(d.to_dict()) for d in (self.decisions or [])
            if d.action == "drop_leaky_feature" and (d.kind, tuple(d.columns)) not in have
        ]
        self.decisions_ = chosen + self.decisions_
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

    def fit_transform(self, X: pd.DataFrame, y: Any = None, **fit_params: Any) -> pd.DataFrame:
        """
        Fit, then transform the same rows. Target-encoded columns get out-of-fold
        values here (each row encoded without its own target), as in fit-time
        planning; new data passed to transform gets the mapping learned in fit.
        """
        self.fit(X, y)
        out = self.transform(X)
        df = X.copy()
        if y is not None and self.target is not None and self.target not in df.columns:
            df[self.target] = list(y)
        y_numeric = _target_numeric(df, self.target)
        if y_numeric is None:
            return out
        for col, spec in self.specs_.items():
            if isinstance(spec, dict) and spec.get("transform") == "target_encode_cross_fit" and col in out.columns:
                encoded, _ = target_encode_cross_fit(df[col], y_numeric)
                out[col] = encoded.to_numpy()
        return out

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

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "output_columns_"):
            raise NotFittedError("EngineerTransformer is not fitted yet.")
        return np.array(self.output_columns_)


def build_pipeline(
    target: Optional[str],
    model_hint: str = "tree",
    decisions: Optional[list[Decision]] = None,
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
    decisions: Optional[list[Decision]] = None,
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
    decisions: Optional[list[Decision]] = None,
    schema: Optional[TableSchema] = None,
    feature_config: Optional[FeatureConfig] = None,
    cleaner_config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
    cardinality_threshold: int = 20,
) -> tuple[pd.DataFrame, Pipeline]:
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
