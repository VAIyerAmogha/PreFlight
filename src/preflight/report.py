from typing import Optional
import pandas as pd
import numpy as np
from preflight.types import ReportEntry, ColumnProfile, SemanticType
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import html
import base64
from io import BytesIO

class Report:
    """
    Report owns all display and export logic for PreFlight-ML.
    It operates purely on ReportEntry objects and does no data processing.
    """
    def __init__(
        self,
        entries: list[ReportEntry],
        df: pd.DataFrame | None = None,
        profiles: list[ColumnProfile] | None = None,
        target: str | None = None,
    ) -> None:
        self._entries = list(entries)
        # Store optional context for plotting only, maintaining Phase 5's "reads ReportEntry[] only" 
        # boundary for core report functions (.show(), .to_dict(), .to_dataframe()).
        # These are exclusively used by the visual layer (.plot()).
        self._df = df
        self._profiles = profiles
        self._target = target

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

    def plot(self, kind: str = "all") -> list[Figure]:
        if self._df is None or self._profiles is None or self._target is None:
            raise ValueError("Dataframe, profiles, and target must be provided at construction to generate plots.")
            
        valid_kinds = {"correlation", "missingness", "mutual_info", "class_distribution", "all"}
        if kind not in valid_kinds:
            raise ValueError(f"Invalid kind '{kind}'. Expected one of {valid_kinds}.")
            
        figures = []
        
        if kind in ("missingness", "all"):
            figures.append(plot_missingness_heatmap(self._df))
            
        if kind in ("correlation", "all"):
            numeric_cols = [p.name for p in self._profiles if p.semantic_type in (SemanticType.NUMERIC_FEATURE, SemanticType.NUMERIC_ID)]
            try:
                figures.append(plot_correlation_heatmap(self._df, numeric_cols))
            except ValueError as e:
                if kind == "all":
                    print(f"Skipping correlation plot: {e}")
                else:
                    raise
                    
        if kind in ("mutual_info", "all"):
            try:
                figures.append(plot_mutual_info_bar_chart(self._profiles))
            except ValueError as e:
                if kind == "all":
                    print(f"Skipping mutual info plot: {e}")
                else:
                    raise
                    
        if kind in ("class_distribution", "all"):
            if self._target in self._df.columns:
                target_series = self._df[self._target]
                try:
                    figures.append(plot_class_distribution(target_series))
                except ValueError as e:
                    if kind == "all":
                        print(f"Skipping class distribution plot: {e}")
                    else:
                        raise
            else:
                if kind != "all":
                    raise ValueError(f"Target column '{self._target}' not found in dataframe.")
                else:
                    print(f"Skipping class distribution plot: Target column '{self._target}' not found.")
                    
        return figures

    def to_html(self, include_plots: bool = True) -> str:
        counts = self.summary_counts()
        
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>PreFlight-ML Report</title>",
            "<style>",
            "body { font-family: sans-serif; margin: 20px; line-height: 1.6; color: #333; }",
            "h1, h2 { color: #2c3e50; }",
            "table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            ".severity-critical { background-color: #ffebee; color: #c62828; }",
            ".severity-warning { background-color: #fff3e0; color: #ef6c00; }",
            ".severity-info { background-color: #f5f5f5; color: #424242; }",
            ".summary { margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }",
            ".plot-container { margin-bottom: 30px; text-align: center; }",
            ".plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>PreFlight-ML Preparation Report</h1>",
            "<div class='summary'>",
            "<h2>Summary</h2>",
            f"<p>Total Decisions: {len(self.entries)}</p>",
            "<ul>",
            f"<li>Critical: {counts.get('critical', 0)}</li>",
            f"<li>Warning: {counts.get('warning', 0)}</li>",
            f"<li>Info: {counts.get('info', 0)}</li>",
            "</ul>",
            "</div>"
        ]

        if not self._entries:
            html_parts.append("<p>No decisions logged.</p>")
        else:
            stages = ["profiler", "cleaner", "engineer"]
            severity_order = {"critical": 0, "warning": 1, "info": 2}
            
            for stage in stages:
                stage_entries = [e for e in self.entries if e.stage == stage]
                if not stage_entries:
                    continue
                    
                stage_entries.sort(key=lambda e: severity_order.get(e.severity, 3))
                
                html_parts.append(f"<h2>Stage: {html.escape(stage).capitalize()}</h2>")
                html_parts.append("<table>")
                html_parts.append("<tr><th>Severity</th><th>Column</th><th>Action</th><th>Rationale</th></tr>")
                
                for e in stage_entries:
                    sev_class = f"severity-{e.severity}"
                    html_parts.append(f"<tr class='{sev_class}'>")
                    html_parts.append(f"<td>{html.escape(e.severity).upper()}</td>")
                    html_parts.append(f"<td>{html.escape(e.column)}</td>")
                    html_parts.append(f"<td>{html.escape(e.action)}</td>")
                    html_parts.append(f"<td>{html.escape(e.rationale)}</td>")
                    html_parts.append("</tr>")
                    
                html_parts.append("</table>")
                
        if include_plots:
            html_parts.append("<h2>Visualizations</h2>")
            if self._df is None or self._profiles is None or self._target is None:
                html_parts.append("<p><em>Visualizations unavailable: Plotting context (df, profiles, target) was not provided during Report construction.</em></p>")
            else:
                figures = self.plot(kind="all")
                if not figures:
                    html_parts.append("<p>No visualizations applicable for this dataset.</p>")
                else:
                    for fig in figures:
                        buf = BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        encoded = base64.b64encode(buf.read()).decode("utf-8")
                        html_parts.append("<div class='plot-container'>")
                        html_parts.append(f"<img src='data:image/png;base64,{encoded}' alt='Report Plot'/>")
                        html_parts.append("</div>")
                        plt.close(fig)
        
        html_parts.append("</body>")
        html_parts.append("</html>")
        
        return "\n".join(html_parts)
        
    def save_html(self, path: str, include_plots: bool = True) -> None:
        html_content = self.to_html(include_plots=include_plots)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns: list[str]) -> Figure:
    """
    Computes a Pearson correlation matrix across numeric_columns and renders it as a heatmap.
    Uses its own Figure/Axes to avoid global pyplot state interference.
    """
    if len(numeric_columns) < 2:
        raise ValueError("At least 2 numeric columns are required to plot a correlation heatmap.")
    
    corr = df[numeric_columns].corr()
    
    # Explicitly create figure to avoid global state issues in notebook/script reuse
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    
    ax.set_xticks(np.arange(len(numeric_columns)))
    ax.set_yticks(np.arange(len(numeric_columns)))
    ax.set_xticklabels(numeric_columns, rotation=45, ha="right")
    ax.set_yticklabels(numeric_columns)
    
    ax.set_title("Numeric Features Correlation Heatmap")
    fig.tight_layout()
    return fig

def plot_missingness_heatmap(df: pd.DataFrame) -> Figure:
    """
    Renders a heatmap of null positions across all columns using a binary colormap.
    Uses its own Figure/Axes to avoid global pyplot state interference.
    """
    if len(df) > 1000:
        plot_df = df.sample(1000, random_state=42)
    else:
        plot_df = df
        
    missing_matrix = plot_df.isna().values
    
    # Explicitly create figure to avoid global state issues
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(missing_matrix, cmap="binary", aspect="auto", interpolation="none")
    
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=90)
    ax.set_yticks([])
    ax.set_ylabel(f"Records (n={len(plot_df)})")
    ax.set_title("Missingness Heatmap (Black = Missing)")
    fig.tight_layout()
    return fig

def plot_mutual_info_bar_chart(profiles: list[ColumnProfile]) -> Figure:
    """
    Horizontal bar chart of mutual_info_with_target per column, sorted descending.
    Uses its own Figure/Axes to avoid global pyplot state interference.
    """
    mi_data = []
    for p in profiles:
        if p.mutual_info_with_target is not None:
            mi_data.append((p.name, p.mutual_info_with_target))
            
    if not mi_data:
        raise ValueError("No columns have a computed non-None mutual_info_with_target score.")
        
    # Sort ascending so highest is at top in barh
    mi_data.sort(key=lambda x: x[1], reverse=False)
    
    names = [x[0] for x in mi_data]
    scores = [x[1] for x in mi_data]
    
    # Explicitly create figure to avoid global state issues
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, scores, color="skyblue")
    ax.set_xlabel("Mutual Information")
    ax.set_title("Mutual Information with Target")
    fig.tight_layout()
    return fig

def plot_class_distribution(target: pd.Series) -> Figure:
    """
    Bar chart of value counts for a classification target.
    Uses its own Figure/Axes to avoid global pyplot state interference.
    """
    val_counts = target.value_counts()
    if len(val_counts) > 20:
        raise ValueError("Target has more than 20 unique values, inappropriate for class distribution plot.")
        
    # Explicitly create figure to avoid global state issues
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_labels = [str(x) for x in val_counts.index]
    ax.bar(x_labels, val_counts.values, color="coral")
    ax.set_xlabel("Target Class")
    ax.set_ylabel("Count")
    ax.set_title("Target Class Distribution")
    ax.set_xticks(np.arange(len(val_counts)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    
    fig.tight_layout()
    return fig
