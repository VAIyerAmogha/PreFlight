"""
FeatureConfig dataclass and preset definitions for PreFlight v2.0 Engineering.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class FeatureConfig:
    """
    Configuration options for advanced feature generation.
    """
    interactions: bool = False
    interaction_top_k: int = 5
    interaction_types: List[str] = field(default_factory=lambda: ["ratio", "product"])
    datetime_cyclical: bool = False
    datetime_deltas: bool = False
    datetime_reference_col: Optional[str] = None
    clustering: bool = False
    cluster_k: Union[int, str] = "auto"
    cluster_features: Union[str, List[str]] = "numeric_only"
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


PRESETS: Dict[str, Dict[str, Any]] = {
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
