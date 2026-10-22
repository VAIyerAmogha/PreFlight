from typing import Optional
import pandas as pd
import numpy as np
from preflight.types import ReportEntry, ColumnProfile, SemanticType
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import html
import base64
from io import BytesIO

# Presentation Constants
SEVERITY_SYMBOLS = {"info": "·", "warning": "⚠", "critical": "✕"}
SEVERITY_COLORS = {"info": "#f5f5f5", "warning": "#fff3e0", "critical": "#ffebee"}
CHART_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

def _truncate_label(label: str, max_len: int = 20) -> str:
    if not isinstance(label, str):
        label = str(label)
    if len(label) > max_len:
        return label[:max_len-3] + "..."
    return label

def _calc_figsize(n_elements: int) -> tuple[float, float]:
    width = min(20.0, max(8.0, 0.4 * n_elements))
    return (width, 6.0)

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

    @property
    def profiles(self) -> list[ColumnProfile] | None:
        return self._profiles

    @profiles.setter
    def profiles(self, value: list[ColumnProfile] | None) -> None:
        self._profiles = value

    @property
    def target(self) -> str | None:
        return self._target

    @target.setter
    def target(self, value: str | None) -> None:
        self._target = value

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

    def show(self, severity_filter: Optional[str] = None, verbose: bool = False) -> None:
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

            print(f"\n=== {stage.upper()} ===")
            stage_entries.sort(key=lambda e: severity_order.get(e.severity, 3))

            info_count = 0
            for e in stage_entries:
                if not verbose and e.severity == "info":
                    info_count += 1
                    if info_count > 5:
                        continue

                symbol = SEVERITY_SYMBOLS.get(e.severity, "-")
                print(f"  {symbol} [{e.severity.upper()}] Column: {e.column}")
                print(f"    Action: {e.action}")
                print(f"    Rationale: {e.rationale}")
                
            if not verbose and info_count > 5:
                hidden = info_count - 5
                print(f"  ... {hidden} more info-level entries — call .show(verbose=True) to see all")

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
            ".severity-critical { background-color: " + SEVERITY_COLORS["critical"] + "; color: #c62828; }",
            ".severity-warning { background-color: " + SEVERITY_COLORS["warning"] + "; color: #ef6c00; }",
            ".severity-info { background-color: " + SEVERITY_COLORS["info"] + "; color: #424242; }",
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

    def save_pdf(self, path: str, include_appendix: bool = True) -> None:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        counts = self.summary_counts()
        
        with PdfPages(path) as pdf:
            # --- Page 1: Cover Page ---
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            
            ax.text(0.5, 0.9, "PreFlight-ML Preparation Report", ha='center', va='center', fontsize=24, fontweight='bold', color="#2c3e50")
            
            shape_str = f"{self._df.shape[0]} rows × {self._df.shape[1]} columns" if self._df is not None else "Unknown"
            ax.text(0.5, 0.82, f"Dataset Shape: {shape_str}", ha='center', va='center', fontsize=14)
            ax.text(0.5, 0.77, f"Target: {self._target if self._target else 'Unknown'}", ha='center', va='center', fontsize=14)
            
            ax.text(0.5, 0.65, "Severity Counts", ha='center', va='center', fontsize=18, fontweight='bold')
            
            y_start = 0.55
            for sev in ["critical", "warning", "info"]:
                count = counts.get(sev, 0)
                ax.text(0.5, y_start, f"{sev.upper()}: {count}", ha='center', va='center', fontsize=16,
                        bbox=dict(facecolor=SEVERITY_COLORS.get(sev, "#ffffff"), edgecolor='#dddddd', boxstyle='round,pad=1', alpha=1.0))
                y_start -= 0.12
            
            pdf.savefig(fig)
            plt.close(fig)
            
            # --- Following Pages: Pipeline Stages ---
            stages = ["profiler", "cleaner", "engineer"]
            stage_kinds = {
                "profiler": ["missingness", "class_distribution"],
                "cleaner": ["correlation"],
                "engineer": ["mutual_info"]
            }
            
            for stage in stages:
                stage_entries = [e for e in self._entries if e.stage == stage]
                kinds = stage_kinds[stage]
                
                stage_figs = []
                for k in kinds:
                    try:
                        figs = self.plot(kind=k)
                        stage_figs.extend(figs)
                    except Exception:
                        pass
                
                # Stage header
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.axis('off')
                ax.text(0.5, 0.6, f"Stage: {stage.capitalize()}", ha='center', va='center', fontsize=18, fontweight='bold')
                ax.text(0.5, 0.2, f"{len(stage_entries)} decisions logged", ha='center', va='center', fontsize=14)
                pdf.savefig(fig)
                plt.close(fig)
                
                if stage_figs:
                    for s_fig in stage_figs:
                        pdf.savefig(s_fig)
                        plt.close(s_fig)
                else:
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.axis('off')
                    ax.text(0.5, 0.5, "No chartable content", ha='center', va='center', fontsize=12, style='italic', color='gray')
                    pdf.savefig(fig)
                    plt.close(fig)
                    
            # --- Appendix ---
            if include_appendix and self._entries:
                rows_per_page = 35
                for i in range(0, len(self._entries), rows_per_page):
                    chunk = self._entries[i:i+rows_per_page]
                    fig, ax = plt.subplots(figsize=(10, 11))
                    ax.axis('off')
                    
                    if i == 0:
                        ax.text(0.5, 0.95, "Appendix: All Report Entries", ha='center', va='center', fontsize=16, fontweight='bold')
                    
                    y_pos = 0.9 if i == 0 else 0.95
                    
                    ax.text(0.05, y_pos, "SEV", fontsize=9, fontweight='bold')
                    ax.text(0.12, y_pos, "COLUMN", fontsize=9, fontweight='bold')
                    ax.text(0.32, y_pos, "ACTION", fontsize=9, fontweight='bold')
                    ax.text(0.55, y_pos, "RATIONALE", fontsize=9, fontweight='bold')
                    
                    y_pos -= 0.015
                    ax.plot([0.05, 0.95], [y_pos, y_pos], color='black', lw=1)
                    y_pos -= 0.02
                    
                    for e in chunk:
                        sev = e.severity.upper()[:4]
                        col = _truncate_label(e.column, 15) if e.column else ""
                        act = _truncate_label(e.action, 25) if e.action else ""
                        rat = _truncate_label(e.rationale, 60) if e.rationale else ""
                        
                        rect = patches.Rectangle((0.04, y_pos-0.012), 0.92, 0.024, facecolor=SEVERITY_COLORS.get(e.severity, "#ffffff"), edgecolor='none')
                        ax.add_patch(rect)
                        
                        ax.text(0.05, y_pos, sev, fontsize=8, va='center')
                        ax.text(0.12, y_pos, col, fontsize=8, va='center')
                        ax.text(0.32, y_pos, act, fontsize=8, va='center')
                        ax.text(0.55, y_pos, rat, fontsize=8, va='center')
                        
                        y_pos -= 0.024
                        
                    pdf.savefig(fig)
                    plt.close(fig)

def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns: list[str], ax=None) -> Figure:
    """
    Computes a Pearson correlation matrix across numeric_columns and renders it as a heatmap.
    Uses its own Figure/Axes to avoid global pyplot state interference.
    """
    if len(numeric_columns) < 2:
        raise ValueError("At least 2 numeric columns are required to plot a correlation heatmap.")
    
    corr = df[numeric_columns].corr()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=_calc_figsize(len(numeric_columns)))
    else:
        fig = ax.figure

    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    
    ax.set_xticks(np.arange(len(numeric_columns)))
    ax.set_yticks(np.arange(len(numeric_columns)))
    
    labels = [_truncate_label(c) for c in numeric_columns]
    rotation = 45 if len(numeric_columns) > 8 else 0
    ha = "right" if rotation == 45 else "center"
    
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)
    ax.set_yticklabels(labels)
    
    ax.set_title("Numeric Features Correlation Heatmap")
    fig.tight_layout()
    return fig

def plot_missingness_heatmap(df: pd.DataFrame, ax=None) -> Figure:
    """
    Renders a heatmap of null positions across all columns using a binary colormap.
    Uses its own Figure/Axes to avoid global pyplot state interference.
    """
    if len(df) > 1000:
        plot_df = df.sample(1000, random_state=42)
    else:
        plot_df = df
        
    missing_matrix = plot_df.isna().values
    
    if ax is None:
        fig, ax = plt.subplots(figsize=_calc_figsize(len(df.columns)))
    else:
        fig = ax.figure

    im = ax.imshow(missing_matrix, cmap="binary", aspect="auto", interpolation="none")
    
    ax.set_xticks(np.arange(len(df.columns)))
    
    labels = [_truncate_label(c) for c in df.columns]
    rotation = 45 if len(df.columns) > 8 else 0
    ha = "right" if rotation == 45 else "center"
    
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)
    ax.set_yticks([])
    ax.set_ylabel(f"Records (n={len(plot_df)})")
    ax.set_title("Missingness Heatmap (Black = Missing)")
    fig.tight_layout()
    return fig

def plot_mutual_info_bar_chart(profiles: list[ColumnProfile], ax=None) -> Figure:
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
    
    if ax is None:
        height = min(20.0, max(6.0, 0.4 * len(names)))
        fig, ax = plt.subplots(figsize=(8, height))
    else:
        fig = ax.figure

    bars = ax.barh(names, scores, color=CHART_PALETTE[0], label="Mutual Info")
    
    # Add value annotations
    ax.bar_label(bars, fmt='%.3f', padding=3)
    
    trunc_names = [_truncate_label(n) for n in names]
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(trunc_names)
    
    ax.set_xlabel("Mutual Information")
    ax.set_title("Mutual Information with Target")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig

def plot_class_distribution(target: pd.Series, ax=None) -> Figure:
    """
    Bar chart of value counts for a classification target.
    Uses its own Figure/Axes to avoid global pyplot state interference.
    """
    val_counts = target.value_counts()
    if len(val_counts) > 20:
        raise ValueError("Target has more than 20 unique values, inappropriate for class distribution plot.")
        
    if ax is None:
        fig, ax = plt.subplots(figsize=_calc_figsize(len(val_counts)))
    else:
        fig = ax.figure

    
    x_labels = [_truncate_label(str(x)) for x in val_counts.index]
    bars = ax.bar(x_labels, val_counts.values, color=CHART_PALETTE[1], label="Class Count")
    
    # Add value annotations
    ax.bar_label(bars, fmt='%d', padding=3)
    
    ax.set_xlabel("Target Class")
    ax.set_ylabel("Count")
    ax.set_title("Target Class Distribution")
    
    rotation = 45 if len(val_counts) > 8 else 0
    ha = "right" if rotation == 45 else "center"
    
    ax.set_xticks(np.arange(len(val_counts)))
    ax.set_xticklabels(x_labels, rotation=rotation, ha=ha)
    ax.legend()
    
    fig.tight_layout()
    return fig
