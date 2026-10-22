"""
Type definitions, SemanticType enum, and result containers for PreFlight v2.0.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING, Union
import pandas as pd

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from preflight.report import Report


class SemanticType(Enum):
    """v1-compatible SemanticType enum."""
    NUMERIC_FEATURE = "NUMERIC_FEATURE"
    NUMERIC_ID = "NUMERIC_ID"
    CATEGORICAL_LOW = "CATEGORICAL_LOW"
    CATEGORICAL_HIGH = "CATEGORICAL_HIGH"
    DATETIME_NATIVE = "DATETIME_NATIVE"
    DATETIME_STRING = "DATETIME_STRING"
    BOOLEAN = "BOOLEAN"
    CONSTANT = "CONSTANT"
    TEXT = "TEXT"


@dataclass
class ColumnProfile:
    """v1-compatible ColumnProfile."""
    name: str
    semantic_type: SemanticType
    missing_rate: float = 0.0
    dtype: Any = "object"
    outlier_rate: Optional[float] = None
    cardinality: Optional[int] = None
    rare_categories: Optional[List[Any]] = None
    vif_score: Optional[float] = None
    correlation_with_target: Optional[float] = None
    mutual_info_with_target: Optional[float] = None
    is_leakage_suspect: bool = False
    text_avg_length: Optional[float] = None
    text_avg_word_count: Optional[float] = None
    text_missing_rate: Optional[float] = None


@dataclass
class ReportEntry:
    """v1-compatible ReportEntry."""
    stage: str
    column: str
    action: str
    rationale: str
    severity: str = "info"
    before_stats: Dict[str, Any] = field(default_factory=dict)
    after_stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        valid_stages = {"profiler", "cleaner", "clean", "engineer"}
        valid_severities = {"info", "warning", "critical"}
        if self.stage not in valid_stages:
            raise ValueError(f"The report stage '{self.stage}' is not recognized.")
        if self.severity not in valid_severities:
            raise ValueError(f"The report severity '{self.severity}' is not recognized.")


@dataclass
class PrepResult:
    """v1-compatible PrepResult."""
    df: pd.DataFrame
    pipeline: Optional['Pipeline'] = None
    report: Optional[Any] = None


_UNSET = object()

from preflight.engineer.config import PRESETS, FeatureConfig

__all__ = [
    "ColumnProfile",
    "FeatureConfig",
    "PRESETS",
    "PrepResult",
    "ReportEntry",
    "SemanticType",
    "_UNSET",
]
