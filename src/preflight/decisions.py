"""
Decision record definition for PreFlight v2.0.

The Decision dataclass is the central record structure used across all pipeline
phases (infer, clean, engineer, assemble). It records proposed transformations,
type inferences, missing marker flags, and abstentions with evidence, confidence,
safe defaults, human-readable explanations, and a stable fingerprint for reuse
across pipeline re-runs.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional


def compute_decision_fingerprint(
    stage: str,
    kind: str,
    columns: List[str],
    action: str,
    default: str,
    evidence: Dict[str, Any],
) -> str:
    """
    Compute a deterministic fingerprint hash across re-runs of the same file.

    Derived exclusively from semantic inputs (stage, kind, column names, action,
    default, and normalized evidence shape/values), NOT from random IDs or
    timestamps.
    """
    # Normalize evidence to ensure deterministic serialization across runs
    norm_ev: Dict[str, Any] = {}
    for k in sorted(evidence.keys()):
        v = evidence[k]
        if isinstance(v, float):
            norm_ev[k] = round(v, 4)
        elif isinstance(v, (int, str, bool)):
            norm_ev[k] = v
        elif isinstance(v, (list, tuple)):
            norm_ev[k] = [
                round(x, 4) if isinstance(x, float) else (str(x) if not isinstance(x, (int, bool)) else x)
                for x in v
            ]
        elif isinstance(v, dict):
            norm_ev[k] = {
                dk: round(dv, 4) if isinstance(dv, float) else (str(dv) if not isinstance(dv, (int, bool)) else dv)
                for dk, dv in sorted(v.items())
            }
        else:
            norm_ev[k] = str(v)

    payload = {
        "stage": stage,
        "kind": kind,
        "columns": sorted(columns),
        "action": action,
        "default": default,
        "evidence": norm_ev,
    }
    raw_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


@dataclass
class Decision:
    """
    Versioned Decision record representing a pipeline inference or transformation.

    Attributes:
        id: Unique identifier for this decision instance.
        stage: Pipeline stage emitting this decision (e.g. "infer", "clean").
        kind: Specific decision category (e.g. "type_inference", "type_abstain").
        columns: Affected column names.
        action: Proposed operation (e.g. "cast_to_int", "keep_as_string").
        evidence: Dictionary of diagnostic scores, stats, and detected anomalies.
        confidence: Numerical confidence score between 0.0 and 1.0.
        default: Safe fallback action if user rejects or pipeline defaults.
        needs_attention: Flag indicating whether human review is recommended.
        plain_language: User-facing textual explanation of the rationale.
        status: Decision lifecycle state ("automatic", "confirmed", "changed").
        timestamp: ISO 8601 UTC creation/modification timestamp.
        fingerprint: Deterministic hash of inputs for cross-run tracking.
    """
    id: str
    stage: str
    kind: str
    columns: List[str]
    action: str
    evidence: Dict[str, Any]
    confidence: float
    default: str
    needs_attention: bool
    plain_language: str
    status: str = "automatic"
    timestamp: Optional[str] = None
    fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.fingerprint is None:
            self.fingerprint = compute_decision_fingerprint(
                stage=self.stage,
                kind=self.kind,
                columns=self.columns,
                action=self.action,
                default=self.default,
                evidence=self.evidence,
            )
        if not self.id:
            col_tag = "_".join(self.columns) if self.columns else "table"
            self.id = f"dec_{self.stage}_{col_tag}_{self.fingerprint[:8]}"

    VALID_STATUSES = {"automatic", "confirmed", "changed"}

    def confirm(self) -> "Decision":
        """
        Transition status to 'confirmed'.

        Updates timestamp to current UTC time. Fingerprint remains invariant.
        """
        self.status = "confirmed"
        self.timestamp = datetime.now(timezone.utc).isoformat()
        return self

    def change(self, new_action: str) -> "Decision":
        """
        Transition status to 'changed' with a modified action.

        Updates action, sets status to 'changed', and updates timestamp.
        Fingerprint remains invariant as it records the original proposed decision identity.
        """
        self.action = new_action
        self.status = "changed"
        self.timestamp = datetime.now(timezone.utc).isoformat()
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert Decision to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        """Reconstruct Decision instance from dictionary."""
        return cls(**data)
