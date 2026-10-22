"""
Feature engineering orchestration and backwards-compatible API for PreFlight v2.0.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from preflight.decisions import Decision
from preflight.schema import TableSchema
from preflight.types import ReportEntry
from preflight.engineer.config import FeatureConfig
from preflight.engineer.datetime import expand_datetime, generate_datetime_cyclical_features
from preflight.engineer.encoding import one_hot_encode, ordinal_encode, target_encode_cross_fit
from preflight.engineer.features import (
    generate_cluster_features,
    generate_interaction_features,
    generate_text_features,
)
from preflight.engineer.numeric import log1p_transform, needs_log_transform, standard_scale
from preflight.engineer.vif import plan_vif_and_leakage


class FeatureAugmenterTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for feature config augmentations
    (interactions, cyclical datetimes, clustering, text statistics, TF-IDF).
    """

    def __init__(
        self,
        config: FeatureConfig,
        profiles: Any,
        target: str,
        cluster_info: dict,
        text_info: dict,
        skipped_cols: list[str],
    ):
        self.config = config
        self.profiles = profiles
        self.target = target
        self.cluster_info = cluster_info
        self.text_info = text_info
        self.skipped_cols = skipped_cols

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        numeric_cols = [
            p.name for p in self.profiles
            if getattr(p, "semantic_type", None) == "NUMERIC_FEATURE"
            or getattr(getattr(p, "semantic_type", None), "name", "") == "NUMERIC_FEATURE"
            or (isinstance(p, str) and p in X_out.columns and pd.api.types.is_numeric_dtype(X_out[p]))
        ]

        if self.config.interactions:
            X_int, _ = generate_interaction_features(X_out, numeric_cols, self.target, self.config)
            for col in X_int.columns.difference(X_out.columns):
                if col not in self.skipped_cols:
                    X_out[col] = X_int[col]

        dt_cols = [
            p.name for p in self.profiles
            if getattr(p, "semantic_type", None) in ("DATETIME_NATIVE", "DATETIME_STRING")
            or getattr(getattr(p, "semantic_type", None), "name", "") in ("DATETIME_NATIVE", "DATETIME_STRING")
            or (isinstance(p, str) and p in X_out.columns and pd.api.types.is_datetime64_any_dtype(X_out[p]))
        ]
        if self.config.datetime_cyclical or self.config.datetime_deltas or self.config.datetime_reference_col:
            X_dt, _ = generate_datetime_cyclical_features(X_out, dt_cols, self.config)
            for col in X_dt.columns.difference(X_out.columns):
                if col not in self.skipped_cols:
                    X_out[col] = X_dt[col]

        if self.config.clustering and self.cluster_info and "model" in self.cluster_info:
            model = self.cluster_info["model"]
            features = self.cluster_info["features"]
            X_clust = X_out[features].fillna(0)
            if "cluster_label" not in self.skipped_cols:
                X_out["cluster_label"] = model.predict(X_clust)
            if "cluster_dist_to_centroid" not in self.skipped_cols:
                dists = model.transform(X_clust)
                X_out["cluster_dist_to_centroid"] = dists.min(axis=1)

        if self.config.text_features:
            text_cols = [
                p.name for p in self.profiles
                if getattr(p, "semantic_type", None) == "TEXT"
                or getattr(getattr(p, "semantic_type", None), "name", "") == "TEXT"
                or (isinstance(p, str) and p in X_out.columns and pd.api.types.is_string_dtype(X_out[p]))
            ]
            for col in text_cols:
                if col not in X_out.columns:
                    continue
                series = X_out[col].fillna("").astype(str)
                for feat in ["char_length", "word_count", "has_text"]:
                    new_col = f"{col}_{feat}"
                    if new_col not in self.skipped_cols:
                        if feat == "char_length":
                            X_out[new_col] = series.str.len()
                        elif feat == "word_count":
                            X_out[new_col] = series.str.split().str.len()
                        elif feat == "has_text":
                            X_out[new_col] = (series.str.strip().str.len() > 0).astype(int)

                if self.config.text_tfidf and self.text_info and "vectorizers" in self.text_info:
                    vec = self.text_info["vectorizers"].get(col)
                    if vec:
                        tfidf_matrix = vec.transform(series)
                        feature_names = vec.get_feature_names_out()
                        tfidf_df = pd.DataFrame(
                            tfidf_matrix.toarray(),
                            columns=[f"{col}_tfidf_{term}" for term in feature_names],
                            index=series.index,
                        )
                        for tcol in tfidf_df.columns:
                            if tcol not in self.skipped_cols:
                                X_out[tcol] = tfidf_df[tcol]

        return X_out


def plan_engineer(
    df: pd.DataFrame,
    schema: Optional[TableSchema] = None,
    target: Optional[str] = None,
    model_hint: str = "tree",
    cardinality_threshold: int = 20,
    feature_config: Optional[FeatureConfig] = None,
    task: str = "regression",
) -> List[Decision]:
    """
    Plan feature engineering transformations without mutating data.
    """
    if model_hint not in ("tree", "linear"):
        raise ValueError(f"model_hint must be 'tree' or 'linear', got '{model_hint}'.")

    type_map = {}
    if schema and schema.columns:
        for c in schema.columns:
            eff_type = c.effective_type() or "text"
            type_map[c.name] = eff_type.lower()

    decisions: List[Decision] = []

    # Identify column semantic subsets
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []
    text_cols = []

    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        col_type = type_map.get(col, "")
        if col.endswith("_missing") or col_type == "boolean" or pd.api.types.is_bool_dtype(series):
            # Passthrough boolean
            continue
        if col_type == "constant":
            continue
        if col_type in ("datetime", "date") or pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
        elif col_type in ("int", "float", "numeric") or pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        elif col_type == "text":
            text_cols.append(col)
        else:
            categorical_cols.append(col)

    # 1. Datetime expansion
    for col in datetime_cols:
        dec = Decision(
            id="",
            stage="engineer",
            kind="expand_datetime",
            columns=[col],
            action="expand_datetime",
            evidence={"source_column": col},
            confidence=1.0,
            default="expand_datetime",
            needs_attention=False,
            plain_language=f"Extract calendar parts (year, month, day, day of week, hour, weekend) from '{col}'.",
            status="automatic",
        )
        decisions.append(dec)

    # 2. Categorical encoding
    for col in categorical_cols:
        n_unique = int(df[col].dropna().nunique())
        if n_unique >= cardinality_threshold:
            # Target encode cross-fit
            dec = Decision(
                id="",
                stage="engineer",
                kind="target_encode",
                columns=[col],
                action="target_encode_cross_fit",
                evidence={"unique_count": n_unique, "threshold": cardinality_threshold},
                confidence=0.95,
                default="target_encode_cross_fit",
                needs_attention=False,
                plain_language=f"Replace categories in '{col}' with target group rates using 5-fold cross-fitting.",
                status="automatic",
            )
            decisions.append(dec)
        else:
            if model_hint == "tree":
                dec = Decision(
                    id="",
                    stage="engineer",
                    kind="ordinal_encode",
                    columns=[col],
                    action="ordinal_encode",
                    evidence={"unique_count": n_unique, "model_hint": model_hint},
                    confidence=1.0,
                    default="ordinal_encode",
                    needs_attention=False,
                    plain_language=f"Encode categories in '{col}' as ordered integers for tree models.",
                    status="automatic",
                )
                decisions.append(dec)
            else:
                dec = Decision(
                    id="",
                    stage="engineer",
                    kind="one_hot_encode",
                    columns=[col],
                    action="one_hot_encode",
                    evidence={"unique_count": n_unique, "model_hint": model_hint},
                    confidence=1.0,
                    default="one_hot_encode",
                    needs_attention=False,
                    plain_language=f"Expand categories in '{col}' into separate indicator columns for linear models.",
                    status="automatic",
                )
                decisions.append(dec)

    # 3. Numeric transforms (linear model scaling / log1p)
    if model_hint == "linear":
        for col in numeric_cols:
            series = df[col]
            needs_log = needs_log_transform(series)
            if needs_log:
                dec = Decision(
                    id="",
                    stage="engineer",
                    kind="log_transform",
                    columns=[col],
                    action="log1p_transform",
                    evidence={"skewed": True},
                    confidence=0.95,
                    default="log1p_transform",
                    needs_attention=False,
                    plain_language=f"Apply logarithmic transformation to skewed numerical column '{col}'.",
                    status="automatic",
                )
                decisions.append(dec)

            scale_dec = Decision(
                id="",
                stage="engineer",
                kind="scale_numeric",
                columns=[col],
                action="standard_scale",
                evidence={"model_hint": model_hint},
                confidence=1.0,
                default="standard_scale",
                needs_attention=False,
                plain_language=f"Standardize numerical column '{col}' to zero mean and unit variance.",
                status="automatic",
            )
            decisions.append(scale_dec)

    # 4. Multicollinearity (VIF) & Leakage Detection
    vif_leakage_decs = plan_vif_and_leakage(df, numeric_cols, target=target)
    decisions.extend(vif_leakage_decs)

    return decisions


def apply_engineer(
    df: pd.DataFrame,
    decisions: List[Decision],
    target: Optional[str] = None,
    model_hint: str = "tree",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply planned engineering transformations to DataFrame.
    """
    df_out = df.copy()
    specs: Dict[str, Any] = {}

    target_numeric = None
    if target and target in df_out.columns:
        target_series = df_out[target].copy()
        if not pd.api.types.is_numeric_dtype(target_series):
            unique_vals = sorted(target_series.dropna().unique())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            target_numeric = target_series.map(mapping)
        else:
            target_numeric = target_series

    for d in decisions:
        if not d.columns:
            continue
        col = d.columns[0]
        if col not in df_out.columns or col == target:
            continue

        act = d.action
        if act == "expand_datetime":
            expanded = expand_datetime(df_out[col])
            df_out = df_out.drop(columns=[col]).join(expanded)
            specs[col] = {"transform": "expand_datetime"}

        elif act == "ordinal_encode":
            encoded, mapping = ordinal_encode(df_out[col])
            df_out[col] = encoded
            specs[col] = {"transform": "ordinal_encode", "mapping": mapping}

        elif act == "one_hot_encode":
            dummies = one_hot_encode(df_out[col])
            df_out = df_out.drop(columns=[col]).join(dummies)
            specs[col] = {"transform": "one_hot_encode", "columns": list(dummies.columns)}

        elif act == "target_encode_cross_fit":
            if target_numeric is not None:
                encoded, mapping = target_encode_cross_fit(df_out[col], target_numeric)
                df_out[col] = encoded
                specs[col] = {"transform": "target_encode_cross_fit", "mapping": mapping}

        elif act == "log1p_transform":
            df_out[col] = log1p_transform(df_out[col])
            specs[col] = {"log1p": True}

        elif act == "standard_scale":
            scaled, params = standard_scale(df_out[col])
            df_out[col] = scaled
            if col not in specs:
                specs[col] = {}
            specs[col].update({"transform": "linear_numeric", "scale_params": params})

        elif act == "drop_leaky_feature":
            df_out = df_out.drop(columns=[col])
            specs[col] = {"transform": "drop_leaky"}

    return df_out, specs


def run_engineer(
    df: pd.DataFrame,
    profiles: Any,
    target: str,
    model_hint: str,
    cardinality_threshold: int = 20,
    feature_config: Optional[FeatureConfig] = None,
) -> Tuple[pd.DataFrame, List[Any], Dict[str, Any]]:
    """
    Backwards-compatible run_engineer API supporting v1 tests.
    """
    if model_hint not in ("tree", "linear"):
        raise ValueError(f"The model_hint must be exactly 'tree' or 'linear', but you provided '{model_hint}'.")

    df_out = df.copy()
    original_target_series = None
    if target in df_out.columns:
        original_target_series = df_out[target].copy()

    # Convert profiles to mock TableSchema
    from preflight.schema import ColumnRawMetadata, TableSchema
    cols = []
    if isinstance(profiles, list):
        for idx, p in enumerate(profiles):
            st = getattr(p, "semantic_type", None)
            st_val = getattr(st, "value", str(st)) if st else "text"
            col_name = getattr(p, "name", None) or (p.columns[0] if getattr(p, "columns", None) else f"col_{idx}")
            cols.append(ColumnRawMetadata(name=col_name, position=idx, inferred_type=st_val))
    schema = TableSchema(columns=cols)

    decisions = plan_engineer(
        df_out,
        schema=schema,
        target=target,
        model_hint=model_hint,
        cardinality_threshold=cardinality_threshold,
        feature_config=feature_config,
    )

    df_out, specs = apply_engineer(df_out, decisions, target=target, model_hint=model_hint)

    reports = []
    for d in decisions:
        reports.append({
            "stage": "engineer",
            "column": d.columns[0] if d.columns else "dataset",
            "action": d.action,
            "rationale": d.plain_language,
            "severity": "info",
        })

    # Optional FeatureConfig augmentations
    if feature_config is not None:
        numeric_cols = [c.name for c in schema.columns if c.inferred_type in ("NUMERIC_FEATURE", "int", "float", "numeric")]
        dt_cols = [c.name for c in schema.columns if c.inferred_type in ("DATETIME_NATIVE", "DATETIME_STRING", "datetime")]
        text_cols = [c.name for c in schema.columns if c.inferred_type in ("TEXT", "text")]

        if feature_config.interactions:
            df_out, rep = generate_interaction_features(df_out, numeric_cols, target, feature_config)
            reports.extend(rep)

        if feature_config.datetime_cyclical or feature_config.datetime_deltas or feature_config.datetime_reference_col:
            dt_df, rep = generate_datetime_cyclical_features(df, dt_cols, feature_config)
            new_cols = dt_df.columns.difference(df.columns).difference(df_out.columns)
            df_out = df_out.join(dt_df[new_cols])
            reports.extend(rep)

        if feature_config.clustering:
            df_out, rep, fitted = generate_cluster_features(df_out, numeric_cols, feature_config)
            reports.extend(rep)
            if fitted:
                specs["_clustering"] = {"transform": "cluster_features", "fitted_info": fitted}

        if feature_config.text_features:
            df_out, rep, text_info = generate_text_features(df_out, text_cols, feature_config)
            reports.extend(rep)
            if text_info and text_info.get("vectorizers"):
                specs["_text_features"] = {"transform": "text_features", "fitted_info": text_info}

    if original_target_series is not None:
        df_out[target] = original_target_series

    rep_entries = [
        ReportEntry(
            stage="engineer",
            column=r.get("column", ""),
            action=r.get("action", ""),
            rationale=r.get("summary", ""),
            severity="info",
        )
        for r in reports
    ]
    return df_out, rep_entries, specs


def add_features(
    result: Any,
    feature_config: FeatureConfig,
    profiles: Optional[Any] = None,
    target: Optional[str] = None,
) -> Any:
    """
    Applies FeatureConfig-driven feature engineering to an already prepared PrepResult.
    Fixes Pandas4Warning at v1 engineer.py:698.
    """
    if getattr(result, "pipeline", None) is None:
        raise ValueError("You can only add features if you have already run prepare() to create a pipeline.")

    profiles_to_use = profiles if profiles is not None else getattr(result.report, "profiles", getattr(result.report, "_profiles", None))
    target_to_use = target if target is not None else getattr(result.report, "target", getattr(result.report, "_target", None))

    if profiles_to_use is None or target_to_use is None:
        raise ValueError("We need column profiles and a target name to add features.")

    all_off = not (
        feature_config.interactions
        or feature_config.datetime_cyclical
        or feature_config.datetime_deltas
        or feature_config.datetime_reference_col
        or feature_config.clustering
        or feature_config.text_features
    )

    if all_off:
        return result

    df_out = result.df.copy()
    new_entries = []
    skipped_cols = []
    cluster_info = {}
    text_info = {}

    df_raw = getattr(result.report, "_df", None)
    if df_raw is None:
        df_raw = df_out.copy()
    else:
        df_raw = df_raw.copy()

    numeric_cols = [p.name for p in profiles_to_use if getattr(p, "semantic_type", None) == "NUMERIC_FEATURE" or getattr(getattr(p, "semantic_type", None), "name", "") == "NUMERIC_FEATURE"]
    dt_cols = [p.name for p in profiles_to_use if getattr(p, "semantic_type", None) in ("DATETIME_NATIVE", "DATETIME_STRING") or getattr(getattr(p, "semantic_type", None), "name", "") in ("DATETIME_NATIVE", "DATETIME_STRING")]
    text_cols = [p.name for p in profiles_to_use if getattr(p, "semantic_type", None) == "TEXT" or getattr(getattr(p, "semantic_type", None), "name", "") == "TEXT"]

    # 1. Interactions
    if feature_config.interactions:
        df_int, reps = generate_interaction_features(df_out, numeric_cols, target_to_use, feature_config)
        for col in df_int.columns.difference(df_out.columns):
            if col in result.df.columns:
                skipped_cols.append(col)
                new_entries.append({
                    "stage": "engineer",
                    "column": col,
                    "action": "skipped_duplicate_feature",
                    "rationale": f"Skipped generated interaction feature {col} due to name collision.",
                    "severity": "warning",
                })
            else:
                df_out[col] = df_int[col]
        for rep in reps:
            if rep["column"] not in skipped_cols:
                new_entries.append(rep)

    # 2. Datetimes
    if feature_config.datetime_cyclical or feature_config.datetime_deltas or feature_config.datetime_reference_col:
        df_dt, reps = generate_datetime_cyclical_features(df_raw, dt_cols, feature_config)
        for col in df_dt.columns.difference(df_raw.columns):
            if col in result.df.columns:
                skipped_cols.append(col)
                new_entries.append({
                    "stage": "engineer",
                    "column": col,
                    "action": "skipped_duplicate_feature",
                    "rationale": f"Skipped generated datetime feature {col} due to name collision.",
                    "severity": "warning",
                })
            else:
                df_out[col] = df_dt[col]
        for rep in reps:
            if rep["column"] not in skipped_cols:
                new_entries.append(rep)

    # 3. Clustering
    if feature_config.clustering:
        df_clust, reps, cluster_info = generate_cluster_features(df_out, numeric_cols, feature_config)
        for col in ["cluster_label", "cluster_dist_to_centroid"]:
            if col in df_clust.columns:
                if col in result.df.columns:
                    skipped_cols.append(col)
                    new_entries.append({
                        "stage": "engineer",
                        "column": col,
                        "action": "skipped_duplicate_feature",
                        "rationale": f"Skipped generated cluster feature {col} due to name collision.",
                        "severity": "warning",
                    })
                else:
                    df_out[col] = df_clust[col]
        for rep in reps:
            if rep["column"] not in skipped_cols:
                new_entries.append(rep)

    # 4. Text Features (with Pandas4Warning fix at line 698)
    if feature_config.text_features:
        original_input_cols = set(df_raw.columns)
        df_text, reps, text_info = generate_text_features(
            df_out, text_cols, feature_config, original_cols=original_input_cols
        )
        for col in df_text.columns.difference(df_out.columns):
            df_out[col] = df_text[col]

        _basic_feats = ["char_length", "word_count", "has_text"]
        for orig_col in list(original_input_cols - set(df_out.columns)):
            # Fix Pandas4Warning: use include=['object', 'string']
            for str_col in df_raw.select_dtypes(include=["object", "string"]).columns:
                if orig_col in (f"{str_col}_{f}" for f in _basic_feats):
                    df_out[orig_col] = df_raw[orig_col]
                    already_logged = any(
                        r.get("action") == "skipped_duplicate_feature" and r.get("column") == orig_col
                        for r in reps
                    )
                    if not already_logged:
                        reps.append({
                            "stage": "engineer",
                            "column": orig_col,
                            "action": "skipped_duplicate_feature",
                            "rationale": f"Skipped generated text feature {orig_col} due to name collision with pre-existing column.",
                            "severity": "warning",
                        })
                    break

        for rep in reps:
            new_entries.append(rep)
            if rep.get("action") == "skipped_duplicate_feature":
                skipped_cols.append(rep.get("column"))

    # Construct pipeline with augmenter
    augmenter = FeatureAugmenterTransformer(
        feature_config, profiles_to_use, target_to_use, cluster_info, text_info, skipped_cols
    )
    augmenter.is_fitted_ = True
    new_pipeline = Pipeline(steps=result.pipeline.steps + [("augmenter", augmenter)])

    from preflight.types import PrepResult
    return PrepResult(df=df_out, pipeline=new_pipeline, report=getattr(result, "report", None))
