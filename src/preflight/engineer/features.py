"""
Advanced feature generation for PreFlight v2.0 (interactions, clustering, text features).
"""

import itertools
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OrdinalEncoder

from preflight.engineer.config import FeatureConfig
from preflight.engineer.vif import compute_correlation_with_target


def compute_mutual_info_with_target(
    series: pd.Series,
    target: pd.Series,
    task: str = "regression",
) -> Optional[float]:
    """Compute mutual information with target based on task type."""
    valid_idx = series.notna() & target.notna()
    s = series[valid_idx]
    t = target[valid_idx]

    if len(s) == 0:
        return None

    s_encoded = s
    if not pd.api.types.is_numeric_dtype(s):
        s_encoded = OrdinalEncoder().fit_transform(s.to_frame()).flatten()

    X = np.array(s_encoded).reshape(-1, 1)
    y = np.array(t)

    try:
        if task == "regression":
            if not pd.api.types.is_numeric_dtype(target):
                return None
            mi = mutual_info_regression(X, y)
        elif task == "classification":
            mi = mutual_info_classif(X, y)
        else:
            return None
        return float(mi[0])
    except Exception:
        return None


def generate_interaction_features(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
    target: Optional[str],
    config: FeatureConfig,
    task: str = "regression",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Generate ratio, product, and difference interactions across top-k scored numeric features.
    """
    df_out = df.copy()
    reports = []

    target_series = df_out[target] if target and target in df_out.columns else None

    # Score numeric features by correlation or mutual info with target
    scored = []
    for col in numeric_columns:
        if col == target or col not in df_out.columns:
            continue
        score = 0.0
        if target_series is not None:
            corr = compute_correlation_with_target(df_out[col], target_series)
            if corr is not None:
                score = max(score, abs(corr))
            mi = compute_mutual_info_with_target(df_out[col], target_series, task=task)
            if mi is not None:
                score = max(score, abs(mi))
        scored.append((col, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_cols = [x[0] for x in scored[:config.interaction_top_k] if x[0] in df_out.columns]

    for a, b in itertools.combinations(top_cols, 2):
        if "ratio" in config.interaction_types:
            col_name = f"{a}_div_{b}"
            df_out[col_name] = df_out[a] / df_out[b].replace(0, np.nan)
            reports.append({
                "column": col_name,
                "action": "created_interaction",
                "summary": f"Generated ratio of '{a}' and '{b}'.",
            })
        if "product" in config.interaction_types:
            col_name = f"{a}_times_{b}"
            df_out[col_name] = df_out[a] * df_out[b]
            reports.append({
                "column": col_name,
                "action": "created_interaction",
                "summary": f"Generated product of '{a}' and '{b}'.",
            })
        if "difference" in config.interaction_types:
            col_name = f"{a}_minus_{b}"
            df_out[col_name] = df_out[a] - df_out[b]
            reports.append({
                "column": col_name,
                "action": "created_interaction",
                "summary": f"Generated difference of '{a}' and '{b}'.",
            })

    return df_out, reports


def generate_cluster_features(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
    config: FeatureConfig,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate KMeans cluster assignments and distance-to-centroid features.
    """
    df_out = df.copy()
    reports = []

    if config.cluster_features == "numeric_only":
        features = [col for col in numeric_columns if col in df_out.columns]
    else:
        features = [f for f in config.cluster_features if f in df_out.columns]

    if not features:
        return df_out, reports, {}

    X = df_out[features].fillna(0)
    if len(X) < 2:
        return df_out, reports, {}

    if config.cluster_k == "auto":
        best_k = 2
        best_score = -1.0
        models = {}
        for k in range(2, min(11, len(X))):
            if k >= len(X):
                break
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(X)
            if len(set(labels)) > 1:
                score = float(silhouette_score(X, labels))
                models[k] = (km, score)
                if score > best_score:
                    best_score = score
                    best_k = k
        if best_score == -1.0:
            best_k = 2
            best_model = KMeans(n_clusters=best_k, random_state=42, n_init="auto").fit(X)
        else:
            best_model = models[best_k][0]
    else:
        best_k = int(config.cluster_k)
        best_model = KMeans(n_clusters=best_k, random_state=42, n_init="auto").fit(X)

    df_out["cluster_label"] = best_model.labels_
    dists = best_model.transform(X)
    df_out["cluster_dist_to_centroid"] = dists.min(axis=1)

    reports.append({
        "column": "cluster_label",
        "action": "cluster_features",
        "summary": f"Generated cluster features with k={best_k} across {len(features)} numeric columns.",
    })
    reports.append({
        "column": "cluster_dist_to_centroid",
        "action": "cluster_features",
        "summary": f"Generated distance-to-centroid feature with k={best_k} across {len(features)} numeric columns.",
    })

    return df_out, reports, {"model": best_model, "features": features}


def generate_text_features(
    df: pd.DataFrame,
    text_columns: Sequence[str],
    config: FeatureConfig,
    original_cols: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generate basic text statistics (char_length, word_count, has_text) and optional TF-IDF features.
    Resolves column name collisions using original_cols.
    """
    df_out = df.copy()
    reports = []
    vectorizers = {}

    collision_cols = original_cols if original_cols is not None else set(df.columns)
    valid_text_cols = [c for c in text_columns if c in df_out.columns]

    if not valid_text_cols or not config.text_features:
        return df_out, reports, vectorizers

    for col in valid_text_cols:
        series = df_out[col].fillna("").astype(str)

        added_any = False
        for feat in ["char_length", "word_count", "has_text"]:
            new_col = f"{col}_{feat}"
            if new_col in collision_cols:
                reports.append({
                    "column": new_col,
                    "action": "skipped_duplicate_feature",
                    "summary": f"Skipped generated text feature '{new_col}' due to name collision.",
                })
            else:
                if feat == "char_length":
                    df_out[new_col] = series.str.len()
                elif feat == "word_count":
                    df_out[new_col] = series.str.split().str.len()
                elif feat == "has_text":
                    df_out[new_col] = (series.str.strip().str.len() > 0).astype(int)
                added_any = True

        if added_any:
            reports.append({
                "column": col,
                "action": "generate_text_features",
                "summary": f"Generated character length, word count, and text flag for '{col}'.",
            })

        if config.text_tfidf:
            vec = TfidfVectorizer(max_features=config.text_tfidf_top_k)
            tfidf_matrix = vec.fit_transform(series)
            feature_names = vec.get_feature_names_out()
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f"{col}_tfidf_{term}" for term in feature_names],
                index=series.index,
            )

            cols_to_add = []
            for tcol in tfidf_df.columns:
                if tcol in collision_cols:
                    reports.append({
                        "column": tcol,
                        "action": "skipped_duplicate_feature",
                        "summary": f"Skipped generated TF-IDF feature '{tcol}' due to name collision.",
                    })
                else:
                    cols_to_add.append(tcol)

            if cols_to_add:
                df_out = df_out.join(tfidf_df[cols_to_add])
                reports.append({
                    "column": col,
                    "action": "generate_text_tfidf",
                    "summary": f"Generated top {config.text_tfidf_top_k} TF-IDF features for '{col}'.",
                })
            vectorizers[col] = vec

    return df_out, reports, {"vectorizers": vectorizers}
