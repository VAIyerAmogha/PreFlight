from typing import Optional
import pandas as pd
import numpy as np
from preflight.types import ReportEntry

class Report:
    """
    Report owns all display and export logic for PreFlight-ML.
    It operates purely on ReportEntry objects and does no data processing.
    """
    def __init__(self, entries: list[ReportEntry]) -> None:
        self._entries = list(entries)

    @property
    def entries(self) -> list[ReportEntry]:
        return list(self._entries)

    def filter_by_severity(self, severity: str) -> list[ReportEntry]:
        valid_severities = {"info", "warning", "critical"}
        if severity not in valid_severities:
            raise ValueError(f"Invalid severity '{severity}'. Expected one of {valid_severities}.")
        return [e for e in self._entries if e.severity == severity]

    def filter_by_stage(self, stage: str) -> list[ReportEntry]:
        valid_stages = {"profiler", "cleaner", "engineer"}
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage '{stage}'. Expected one of {valid_stages}.")
        return [e for e in self._entries if e.stage == stage]

    def filter_by_column(self, column: str) -> list[ReportEntry]:
        return [e for e in self._entries if e.column == column]

    def summary_counts(self) -> dict[str, int]:
        counts = {"info": 0, "warning": 0, "critical": 0}
        for e in self._entries:
            if e.severity in counts:
                counts[e.severity] += 1
            else:
                counts[e.severity] = 1
        return counts

    def show(self, severity_filter: Optional[str] = None) -> None:
        if not self._entries:
            print("No decisions logged in the report.")
            return

        counts = self.summary_counts()
        print(f"{len(self._entries)} decisions logged: {counts.get('info', 0)} info, {counts.get('warning', 0)} warning, {counts.get('critical', 0)} critical")

        if severity_filter is not None:
            entries_to_show = self.filter_by_severity(severity_filter)
        else:
            entries_to_show = self.entries

        if not entries_to_show:
            return

        stages = ["profiler", "cleaner", "engineer"]
        severity_order = {"critical": 0, "warning": 1, "info": 2}

        for stage in stages:
            stage_entries = [e for e in entries_to_show if e.stage == stage]
            if not stage_entries:
                continue

            print(f"\n--- {stage.upper()} ---")
            stage_entries.sort(key=lambda e: severity_order.get(e.severity, 3))

            for e in stage_entries:
                print(f"  [{e.severity.upper()}] Column: {e.column}")
                print(f"    Action: {e.action}")
                print(f"    Rationale: {e.rationale}")

    def to_dict(self) -> dict:
        def _make_serializable(val):
            if isinstance(val, dict):
                return {str(k): _make_serializable(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_make_serializable(v) for v in val]
            if isinstance(val, tuple):
                return tuple(_make_serializable(v) for v in val)
            if isinstance(val, np.integer):
                return int(val)
            if isinstance(val, np.floating):
                return float(val)
            if isinstance(val, np.bool_):
                return bool(val)
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val

        return {
            "summary": self.summary_counts(),
            "total_entries": len(self.entries),
            "entries": [
                {
                    "stage": e.stage,
                    "column": e.column,
                    "action": e.action,
                    "rationale": e.rationale,
                    "severity": e.severity,
                    "before_stats": _make_serializable(e.before_stats),
                    "after_stats": _make_serializable(e.after_stats)
                }
                for e in self.entries
            ]
        }

    def to_dataframe(self) -> pd.DataFrame:
        columns = [
            "stage", "column", "action", "rationale", 
            "severity", "before_stats", "after_stats"
        ]
        
        if not self._entries:
            return pd.DataFrame(columns=columns)
            
        data = [
            {
                "stage": e.stage,
                "column": e.column,
                "action": e.action,
                "rationale": e.rationale,
                "severity": e.severity,
                "before_stats": e.before_stats,
                "after_stats": e.after_stats
            }
            for e in self._entries
        ]
        return pd.DataFrame(data, columns=columns)
