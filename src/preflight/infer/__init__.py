"""
PreFlight v2.0 Type Inference Subsystem (Phase 3 / M2).

Exposes:
- detect_missing_markers: Per-column disguised missing value detection.
- MissingMarker: Typed record of detected missing value marker.
- infer_column_type: Parse-success rate type inference with trap handling and abstain logic.
- infer_table_types: Table-scale type inference returning enriched TableSchema.
- TypeInferenceResult: Typed result of column type inference.
- to_canonical_type: Mapping helper from candidate types to 7 ground-truth semantic types.
- Decision: Re-exported Decision record dataclass.
"""

from preflight.decisions import Decision
from .missing import MissingMarker, detect_missing_markers
from .types import (
    TypeInferenceResult,
    infer_column_type,
    infer_table_types,
    to_canonical_type,
)

__all__ = [
    "detect_missing_markers",
    "MissingMarker",
    "infer_column_type",
    "infer_table_types",
    "TypeInferenceResult",
    "to_canonical_type",
    "Decision",
]
