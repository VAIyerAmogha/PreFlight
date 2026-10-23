"""
Feature engineering for PreFlight v2.0: plan() builds Decisions, apply() runs them.

Ported from v1 `engineer.run_engineer` and the FeatureConfig generators. Column
kinds come from the column signals (schema type plus the numbers v1's Profiler
supplied). `plan_engineer` never changes its input: like the cleaner, it
simulates the steps on a private copy so each step sees what the previous one
left behind (v1 behaviour), and `apply_engineer` replays the same Decisions.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from preflight.decisions import Decision
from preflight.engineer.config import DEFAULT_ENGINEER_CONFIG, EngineerConfig, FeatureConfig
from preflight.engineer.datetime import expand_datetime, generate_datetime_cyclical_features
from preflight.engineer.encoding import one_hot_encode, ordinal_encode, target_encode_cross_fit
from preflight.engineer.features import (
    compute_mutual_info_with_target,
    generate_cluster_features,
    generate_interaction_features,
    generate_text_features,
    rank_interaction_columns,
)
from preflight.engineer.numeric import log1p_transform, needs_log_transform, standard_scale
from preflight.engineer.vif import plan_vif_and_leakage
from preflight.schema import TableSchema
from preflight.signals import (
    CATEGORICAL_HIGH,
    CATEGORICAL_LOW,
    DATETIME_KINDS,
    DATETIME_NATIVE,
    NUMERIC_FEATURE,
    TEXT,
    ColumnSignals,
    compute_signals,
)


def _target_numeric(df: pd.DataFrame, target: Optional[str]) -> Optional[pd.Series]:
    """Target as numbers; text targets get v1's sorted-label codes."""
    if not target or target not in df.columns:
        return None
    s = df[target]
    if pd.api.types.is_numeric_dtype(s):
        return s
    mapping = {val: idx for idx, val in enumerate(sorted(s.dropna().unique()))}
    return s.map(mapping)


def _cfg_from_evidence(ev: Dict[str, Any]) -> FeatureConfig:
    return FeatureConfig(
        interaction_types=list(ev.get("interaction_types", ["ratio", "product"])),
        datetime_cyclical=bool(ev.get("cyclical", False)),
        datetime_deltas=bool(ev.get("deltas", False)),
        datetime_reference_col=ev.get("reference_col"),
        cluster_k=ev.get("cluster_k", "auto"),
        cluster_features=list(ev["features"]) if "features" in ev else "numeric_only",
        text_features=True,
        text_tfidf=bool(ev.get("tfidf", False)),
        text_tfidf_top_k=int(ev.get("tfidf_top_k", 20)),
    )


def _apply_one(
    work: pd.DataFrame,
    d: Decision,
    specs: Dict[str, Any],
    target: Optional[str],
    target_numeric: Optional[pd.Series],
    df_in: pd.DataFrame,
) -> pd.DataFrame:
    """Apply a single Decision to `work` (the shared core of plan simulation and apply)."""
    if not d.columns:
        return work
    col = d.columns[0]
    act = d.action
    if act in ("keep_feature",):
        return work

    # --- column-level transforms -------------------------------------------
    if d.kind in ("expand_datetime", "ordinal_encode", "one_hot_encode", "target_encode",
                  "log_transform", "scale_numeric", "suspected_leakage"):
        if col not in work.columns or col == target:
            return work
        if act == "expand_datetime":
            expanded = expand_datetime(work[col])
            work = work.drop(columns=[col]).join(expanded)
            specs[col] = {"transform": "expand_datetime"}
        elif act == "ordinal_encode":
            encoded, mapping = ordinal_encode(work[col])
            work[col] = encoded
            specs[col] = {"transform": "ordinal_encode", "mapping": mapping}
        elif act == "one_hot_encode":
            dummies = one_hot_encode(work[col])
            work = work.drop(columns=[col]).join(dummies)
            specs[col] = {"transform": "one_hot_encode", "columns": list(dummies.columns)}
        elif act == "target_encode_cross_fit":
            if target_numeric is not None:
                encoded, mapping = target_encode_cross_fit(work[col], target_numeric)
                work[col] = encoded
                specs[col] = {"transform": "target_encode_cross_fit", "mapping": mapping}
        elif act == "log1p_transform":
            work[col] = log1p_transform(work[col])
            specs[col] = {"transform": "linear_numeric", "log1p": True,
                          "scale_params": {"mean": 0.0, "std": 1.0}}
        elif act == "standard_scale":
            scaled, params = standard_scale(work[col])
            work[col] = scaled
            spec = specs.get(col, {"transform": "linear_numeric", "log1p": False})
            spec["scale_params"] = params
            specs[col] = spec
        elif act == "drop_leaky_feature":
            work = work.drop(columns=[col])
            specs[col] = {"transform": "drop_leaky"}
        return work

    # --- generated features (FeatureConfig) --------------------------------
    ev = d.evidence
    if d.kind == "create_interactions" and act == "create_interactions":
        cfg = _cfg_from_evidence(ev)
        out, _ = generate_interaction_features(work, ev.get("top_columns", []), cfg)
        new = [c for c in out.columns if c not in work.columns]
        for c in new:
            work[c] = out[c]
        specs["_interactions"] = {
            "top_columns": list(ev.get("top_columns", [])),
            "interaction_types": list(cfg.interaction_types),
            "created": new,
        }
    elif d.kind == "create_datetime_features" and act == "create_datetime_features":
        cfg = _cfg_from_evidence(ev)
        dt_df, _ = generate_datetime_cyclical_features(df_in, ev.get("columns", []), cfg)
        new = [c for c in dt_df.columns.difference(df_in.columns) if c not in work.columns]
        work = work.join(dt_df[new])
        specs["_datetime"] = {
            "columns": list(ev.get("columns", [])),
            "cyclical": cfg.datetime_cyclical,
            "deltas": cfg.datetime_deltas,
            "reference_col": cfg.datetime_reference_col,
            "created": new,
        }
    elif d.kind == "create_cluster_features" and act == "create_cluster_features":
        cfg = _cfg_from_evidence(ev)
        out, _, fitted = generate_cluster_features(work, ev.get("features", []), cfg)
        created = []
        for c in ("cluster_label", "cluster_dist_to_centroid"):
            if c in out.columns and c not in work.columns:
                work[c] = out[c]
                created.append(c)
        if fitted:
            specs["_clustering"] = {"transform": "cluster_features", "fitted_info": fitted, "created": created}
    elif d.kind == "create_text_features" and act == "create_text_features":
        cfg = _cfg_from_evidence(ev)
        out, reps, info = generate_text_features(
            work, ev.get("columns", []), cfg, original_cols=set(df_in.columns)
        )
        created = [c for c in out.columns if c not in work.columns]
        for c in created:
            work[c] = out[c]
        skipped = [r["column"] for r in reps if r.get("action") == "skipped_duplicate_feature"]
        specs["_text_features"] = {
            "transform": "text_features",
            "columns": list(ev.get("columns", [])),
            "tfidf": cfg.text_tfidf,
            "fitted_info": info,
            "created": created,
            "skipped": skipped,
        }
    return work


def _mk(kind: str, col: List[str], action: str, evidence: Dict[str, Any], plain: str,
        confidence: float = 1.0, default: Optional[str] = None) -> Decision:
    return Decision(
        id="",
        stage="engineer",
        kind=kind,
        columns=col,
        action=action,
        evidence=evidence,
        confidence=confidence,
        default=default or action,
        needs_attention=False,
        plain_language=plain,
        status="automatic",
    )


def plan_engineer(
    df: pd.DataFrame,
    schema: Optional[TableSchema] = None,
    target: Optional[str] = None,
    model_hint: str = "tree",
    cardinality_threshold: int = 20,
    feature_config: Optional[FeatureConfig] = None,
    task: str = "regression",
    signals: Optional[Dict[str, ColumnSignals]] = None,
    config: EngineerConfig = DEFAULT_ENGINEER_CONFIG,
) -> List[Decision]:
    """
    Plan feature engineering without changing `df`.

    `signals` are the column signals computed before cleaning (v1 used its
    Profiler output the same way); when omitted they are computed from `df`.
    """
    if model_hint not in ("tree", "linear"):
        raise ValueError(f"The model_hint must be exactly 'tree' or 'linear', but you provided '{model_hint}'.")

    sigs = signals
    if sigs is None:
        sigs = compute_signals(
            df, schema=schema, target=target, task=None, cardinality_threshold=cardinality_threshold
        )
    sigs = {c: s for c, s in sigs.items() if c != target}

    decisions: List[Decision] = list(plan_vif_and_leakage(df, sigs, target=target, config=config))

    work = df.copy()
    specs: Dict[str, Any] = {}
    target_numeric = _target_numeric(df, target)

    def emit(d: Decision) -> None:
        nonlocal work
        decisions.append(d)
        work = _apply_one(work, d, specs, target, target_numeric, df)

    for col, sig in sigs.items():
        if col not in work.columns:
            continue
        kind = sig.kind

        if kind == DATETIME_NATIVE:
            emit(_mk("expand_datetime", [col], "expand_datetime", {"source_column": col},
                     f"Split date column '{col}' into year, month, day, weekday, hour and weekend parts."))

        elif kind == CATEGORICAL_LOW:
            n_unique = int(df[col].dropna().nunique())
            if model_hint == "tree":
                emit(_mk("ordinal_encode", [col], "ordinal_encode",
                         {"unique_count": n_unique, "model_hint": model_hint},
                         f"Turn the categories in '{col}' into whole numbers (one number per category)."))
            else:
                emit(_mk("one_hot_encode", [col], "one_hot_encode",
                         {"unique_count": n_unique, "model_hint": model_hint},
                         f"Replace '{col}' with one yes/no column per category."))

        elif kind == CATEGORICAL_HIGH:
            if target_numeric is None:
                continue
            n_unique = int(df[col].dropna().nunique())
            emit(_mk("target_encode", [col], "target_encode_cross_fit",
                     {"unique_count": n_unique, "threshold": cardinality_threshold, "folds": 5},
                     f"Replace the many categories in '{col}' with the typical outcome for each category, "
                     f"worked out on separate parts of the data so the answer does not leak in.",
                     confidence=0.95))

        elif kind == NUMERIC_FEATURE and model_hint == "linear":
            if needs_log_transform(work[col], skew_threshold=config.skew_threshold):
                emit(_mk("log_transform", [col], "log1p_transform", {"skewed": True},
                         f"Compress the long tail of '{col}' with a logarithm.", confidence=0.95))
            emit(_mk("scale_numeric", [col], "standard_scale", {"model_hint": model_hint},
                     f"Rescale '{col}' to a common scale (average 0, spread 1)."))

    if feature_config is not None:
        _plan_generated(df, work, sigs, feature_config, target, task, emit)

    return decisions


def _plan_generated(
    df: pd.DataFrame,
    work: pd.DataFrame,
    sigs: Dict[str, ColumnSignals],
    cfg: FeatureConfig,
    target: Optional[str],
    task: str,
    emit,
) -> None:
    """Plan the opt-in generated features, in v1 order: combinations, dates, clusters, text."""
    if cfg.interactions:
        scored = dict(sigs)
        if target and target in df.columns:
            for name, s in scored.items():
                if s.kind == NUMERIC_FEATURE and s.mutual_info_with_target is None and name in df.columns:
                    s.mutual_info_with_target = compute_mutual_info_with_target(df[name], df[target], task=task)
        top = rank_interaction_columns(scored, list(work.columns), cfg)
        if len(top) >= 2:
            before = set(work.columns)
            out, _ = generate_interaction_features(work, top, cfg)
            new = [c for c in out.columns if c not in before]
            emit(_mk("create_interactions", top, "create_interactions",
                     {"top_columns": top, "interaction_types": list(cfg.interaction_types), "new_columns": new},
                     f"Create {len(new)} new columns by combining the {len(top)} most useful number columns "
                     f"({', '.join(cfg.interaction_types)}).", confidence=0.9))

    if cfg.datetime_cyclical or cfg.datetime_deltas or cfg.datetime_reference_col:
        dt_cols = [c for c, s in sigs.items() if s.kind in DATETIME_KINDS and c in df.columns]
        if dt_cols:
            ev = {"columns": dt_cols, "cyclical": cfg.datetime_cyclical, "deltas": cfg.datetime_deltas,
                  "reference_col": cfg.datetime_reference_col}
            dt_df, _ = generate_datetime_cyclical_features(df, dt_cols, cfg)
            new = [c for c in dt_df.columns.difference(df.columns) if c not in work.columns]
            if new:
                ev["new_columns"] = new
                emit(_mk("create_datetime_features", dt_cols, "create_datetime_features", ev,
                         f"Create {len(new)} extra date columns (seasonal position, time between dates).",
                         confidence=0.9))

    if cfg.clustering:
        if cfg.cluster_features == "numeric_only":
            feats = [c for c, s in sigs.items() if s.kind == NUMERIC_FEATURE and c in work.columns]
        else:
            feats = [f for f in cfg.cluster_features if f in work.columns]
        if feats and len(work) >= 2:
            new = [c for c in ("cluster_label", "cluster_dist_to_centroid") if c not in work.columns]
            if new:
                emit(_mk("create_cluster_features", feats, "create_cluster_features",
                         {"features": feats, "cluster_k": cfg.cluster_k, "new_columns": new},
                         f"Group rows that look alike using {len(feats)} number columns, and add each row's "
                         f"group and its distance from the group's centre.", confidence=0.85))

    if cfg.text_features:
        text_cols = [c for c, s in sigs.items() if s.kind == TEXT and c in work.columns]
        if text_cols:
            emit(_mk("create_text_features", text_cols, "create_text_features",
                     {"columns": text_cols, "tfidf": cfg.text_tfidf, "tfidf_top_k": cfg.text_tfidf_top_k},
                     f"Add simple length and word-count columns for the long-text column(s) "
                     f"{', '.join(repr(c) for c in text_cols)}"
                     + (f", plus the {cfg.text_tfidf_top_k} most telling words." if cfg.text_tfidf else "."),
                     confidence=0.9))


def apply_engineer(
    df: pd.DataFrame,
    decisions: List[Decision],
    target: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply engineering Decisions. Returns the new table and the fitted state
    (mappings, scale parameters, cluster model, word lists) keyed like v1's specs.
    The target column is carried through unchanged.
    """
    work = df.copy()
    specs: Dict[str, Any] = {}
    target_numeric = _target_numeric(df, target)
    for d in decisions:
        if d.stage != "engineer":
            continue
        work = _apply_one(work, d, specs, target, target_numeric, df)
    return work, specs
