"""
Variance Inflation Factor (VIF) and Target Leakage Detection using NumPy.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from preflight.decisions import Decision
from preflight.engineer.config import DEFAULT_ENGINEER_CONFIG, EngineerConfig

if TYPE_CHECKING:
    from preflight.signals import ColumnSignals


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
    signals: Dict[str, "ColumnSignals"],
    target: Optional[str] = None,
    config: EngineerConfig = DEFAULT_ENGINEER_CONFIG,
) -> List[Decision]:
    """
    Turn the leakage flag and VIF score of each numeric feature into Decisions.

    - Suspected leakage (|correlation with target| above config.leakage_threshold):
      v1 only raised a critical report entry and kept the column. The proposed
      action therefore stays "keep_feature"; the person may choose
      "drop_leaky_feature". Always needs_attention.
    - High overlap (VIF above config.vif_note_threshold): note only, never asks.
    """
    decisions: List[Decision] = []

    for col, sig in signals.items():
        if col == target or col not in df.columns:
            continue
        corr = sig.correlation_with_target
        if corr is not None and abs(corr) > config.leakage_threshold:
            decisions.append(
                Decision(
                    id="",
                    stage="engineer",
                    kind="suspected_leakage",
                    columns=[col],
                    action="keep_feature",
                    evidence={
                        "correlation_with_target": corr,
                        "threshold": config.leakage_threshold,
                        "target": target,
                    },
                    confidence=0.99,
                    default="keep_feature",
                    needs_attention=True,
                    plain_language=(
                        f"Column '{col}' tracks the value to be predicted almost exactly "
                        f"(match {abs(corr):.2f}). It may give away the answer. Keep it, or leave it out?"
                    ),
                    status="automatic",
                )
            )

    for col, sig in signals.items():
        score = sig.vif_score
        if col == target or col not in df.columns or score is None:
            continue
        if score > config.vif_note_threshold:
            decisions.append(
                Decision(
                    id="",
                    stage="engineer",
                    kind="feature_overlap",
                    columns=[col],
                    action="keep_feature",
                    evidence={"redundancy_score": score, "threshold": config.vif_note_threshold},
                    confidence=0.85,
                    default="keep_feature",
                    needs_attention=False,
                    plain_language=(
                        f"Note: column '{col}' largely repeats what other numeric columns already say "
                        f"(overlap score {score:.1f}). It is kept."
                    ),
                    status="automatic",
                )
            )

    return decisions
