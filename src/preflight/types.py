from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from .report import Report

class SemanticType(Enum):
    NUMERIC_FEATURE = "NUMERIC_FEATURE"
    NUMERIC_ID = "NUMERIC_ID"
    CATEGORICAL_LOW = "CATEGORICAL_LOW"
    CATEGORICAL_HIGH = "CATEGORICAL_HIGH"
    DATETIME_NATIVE = "DATETIME_NATIVE"
    DATETIME_STRING = "DATETIME_STRING"
    BOOLEAN = "BOOLEAN"
    CONSTANT = "CONSTANT"

@dataclass
class ColumnProfile:
    name: str
    semantic_type: SemanticType
    missing_rate: float
    dtype: Any
    outlier_rate: Optional[float] = None
    cardinality: Optional[int] = None
    rare_categories: Optional[List[Any]] = None
    vif_score: Optional[float] = None
    correlation_with_target: Optional[float] = None
    mutual_info_with_target: Optional[float] = None
    is_leakage_suspect: bool = False

@dataclass
class ReportEntry:
    stage: Literal["profiler", "cleaner", "engineer"]
    column: str
    action: str
    rationale: str
    severity: Literal["info", "warning", "critical"]
    before_stats: Dict[str, Any] = field(default_factory=dict)
    after_stats: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # NEW: Runtime validation.
    #
    # `Literal` annotations are only used by static type checkers (mypy,
    # pyright, IDEs). Python dataclasses do not automatically enforce them.
    # This method ensures invalid values raise a ValueError at construction,
    # satisfying the pytest tests.
    # ------------------------------------------------------------------
    def __post_init__(self):
        valid_stages = {"profiler", "cleaner", "engineer"}
        valid_severities = {"info", "warning", "critical"}

        if self.stage not in valid_stages:
            raise ValueError(
                f"Invalid stage '{self.stage}'. Expected one of {valid_stages}."
            )

        if self.severity not in valid_severities:
            raise ValueError(
                f"Invalid severity '{self.severity}'. Expected one of {valid_severities}."
            )

@dataclass
class PrepResult:
    df: pd.DataFrame
    pipeline: Optional['Pipeline']
    report: Optional['Report']