"""
Variance Inflation Factor (VIF) and Target Leakage Detection using NumPy.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from preflight.decisions import Decision


def compute_vif_scores(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
) -> Tuple[Dict[str, Optional[float]], bool]:
    """
    Compute Variance Inflation Factor across numeric columns using NumPy.
    Uses inverse of correlation matrix without statsmodels.
    Caps at top 50 numeric features by variance.
    Returns (scores_dict, capped_flag).
    """
    if numeric_columns is None:
        numeric_columns = [c for c in df.select_dtypes(include=[np.number]).columns]
    result = {col: None for col in numeric_columns}
    if not numeric_columns:
        return result, False

    capped = False
    cols_to_compute = list(numeric_columns)

    if len(numeric_columns) > 50:
        capped = True
        variances = df[numeric_columns].var()
        cols_to_compute = variances.nlargest(50).index.tolist()

    df_clean = df[cols_to_compute].dropna()
    if len(df_clean) < 2 or len(cols_to_compute) < 2:
        return result, capped

    corr = df_clean.corr().values
    try:
        inv_corr = np.linalg.inv(corr)
        for i, col in enumerate(cols_to_compute):
            vif = inv_corr[i, i]
            result[col] = float(vif) if vif > 0 else 0.0
    except np.linalg.LinAlgError:
        pass

    return result, capped


def compute_correlation_with_target(
    series: pd.Series,
    target: pd.Series,
) -> Optional[float]:
    """
    Compute Pearson correlation with target for numeric columns.
    """
    if not pd.api.types.is_numeric_dtype(series) or not pd.api.types.is_numeric_dtype(target):
        return None
    val = series.corr(target)
    if pd.isna(val):
        return None
    return float(val)


def flag_leakage_suspect(
    correlation: Optional[float],
    threshold: float = 0.95,
) -> bool:
    """
    Flag whether absolute correlation with target exceeds leakage threshold.
    """
    if correlation is None:
        return False
    return abs(correlation) > threshold


def plan_vif_and_leakage(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    target: Optional[str] = None,
    leakage_threshold: float = 0.95,
) -> List[Decision]:
    """
    Plan multicollinearity diagnostics (VIF) and suspected leakage flags.
    Emits suspected_leakage Decisions when correlation with target exceeds threshold.
    """
    if numeric_columns is None:
        numeric_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    decisions: List[Decision] = []

    target_series = None
    if target and target in df.columns:
        s = df[target]
        if pd.api.types.is_numeric_dtype(s):
            target_series = s
        else:
            # Map categorical target to codes for correlation check
            unique_vals = sorted([x for x in s.dropna().unique()])
            code_map = {v: i for i, v in enumerate(unique_vals)}
            target_series = s.map(code_map)

    # 1. Leakage check on candidate features
    if target_series is not None:
        for col in numeric_columns:
            if col == target or col not in df.columns:
                continue
            corr = compute_correlation_with_target(df[col], target_series)
            if corr is not None and flag_leakage_suspect(corr, threshold=leakage_threshold):
                leak_dec = Decision(
                    id="",
                    stage="engineer",
                    kind="suspected_leakage",
                    columns=[col],
                    action="drop_leaky_feature",
                    evidence={
                        "correlation_with_target": round(corr, 4),
                        "threshold": leakage_threshold,
                        "target": target,
                        "is_leakage_suspect": True,
                    },
                    confidence=0.99,
                    default="drop_leaky_feature",
                    needs_attention=True,
                    plain_language=(
                        f"Exclude feature '{col}' due to very high correlation with target "
                        f"({abs(corr):.2f} > {leakage_threshold:.2f}, potential data leakage)."
                    ),
                    status="automatic",
                )
                decisions.append(leak_dec)

    # 2. VIF check for extreme multicollinearity
    feature_cols = [c for c in numeric_columns if c != target and c in df.columns]
    if len(feature_cols) >= 2:
        vif_scores, capped = compute_vif_scores(df, feature_cols)
        for col, score in vif_scores.items():
            if score is not None and score > 20.0:
                vif_dec = Decision(
                    id="",
                    stage="engineer",
                    kind="feature_overlap",
                    columns=[col],
                    action="keep_collinear_feature",
                    evidence={
                        "redundancy_score": round(score, 2),
                        "capped": capped,
                    },
                    confidence=0.85,
                    default="keep_collinear_feature",
                    needs_attention=False,
                    plain_language=f"Note feature '{col}' has high overlap with other numerical measurements (score {score:.1f}).",
                    status="automatic",
                )
                decisions.append(vif_dec)

    return decisions
