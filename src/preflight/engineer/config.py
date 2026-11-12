"""
FeatureConfig dataclass and preset definitions for PreFlight v2.0 Engineering.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class FeatureConfig:
    """
    Configuration options for advanced feature generation.
    """
    interactions: bool = False
    interaction_top_k: int = 5
    interaction_types: list[str] = field(default_factory=lambda: ["ratio", "product"])
    datetime_cyclical: bool = False
    datetime_deltas: bool = False
    datetime_reference_col: Optional[str] = None
    clustering: bool = False
    cluster_k: Union[int, str] = "auto"
    cluster_features: Union[str, list[str]] = "numeric_only"
    text_features: bool = False
    text_tfidf: bool = False
    text_tfidf_top_k: int = 20

    def __post_init__(self) -> None:
        valid_interaction_types = {"ratio", "product", "difference"}
        if not set(self.interaction_types).issubset(valid_interaction_types):
            valid_names = ", ".join(valid_interaction_types)
            raise ValueError(f"The interaction types provided are not recognized. Please use only: {valid_names}.")
        if self.interaction_top_k < 1:
            raise ValueError("The number of top interactions (interaction_top_k) must be at least 1.")
        if self.cluster_k != "auto" and (not isinstance(self.cluster_k, int) or self.cluster_k <= 0):
            raise ValueError("The number of clusters (cluster_k) must either be the word 'auto' or a number greater than 0.")
        if self.cluster_features != "numeric_only" and not isinstance(self.cluster_features, list):
            raise ValueError("The cluster features must either be 'numeric_only' or a list of specific column names.")
        if self.text_tfidf_top_k < 1:
            raise ValueError("The number of text features to generate (text_tfidf_top_k) must be at least 1.")


@dataclass
class EngineerConfig:
    """
    Thresholds for the engineering stage.

    Attributes:
        leakage_threshold: Absolute correlation with the target above which a
            numeric feature is flagged as a suspected leak (v1 constant 0.95).
        vif_note_threshold: Overlap score (VIF) above which a note is added to
            the plan. NEW in v2: v1 only recorded the score. Never tuned.
        skew_threshold: Skewness above which a log transform is used for
            linear models (v1 default 1.0).
    """
    leakage_threshold: float = 0.95
    vif_note_threshold: float = 10.0
    skew_threshold: float = 1.0


DEFAULT_ENGINEER_CONFIG = EngineerConfig()


PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "drop_threshold": 0.8,
        "outlier_method": "zscore",
        "cardinality_threshold": 50,
        "feature_config": None,
    },
    "thorough": {
        "drop_threshold": 0.2,
        "outlier_method": "iqr",
        "cardinality_threshold": 20,
        "feature_config": FeatureConfig(
            interactions=True,
            datetime_cyclical=True,
            datetime_deltas=True,
            clustering=True,
            text_features=True,
            text_tfidf=True,
        ),
    },
}
