import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import numpy as np

from preflight.types import PrepResult, SemanticType
from preflight.report import (
    SEVERITY_COLORS,
    _truncate_label,
    plot_missingness_heatmap,
    plot_correlation_heatmap,
    plot_mutual_info_bar_chart,
    plot_class_distribution
)
# We must import compare from __init__ below or inline, but __init__ imports compare.
# To avoid circular import, we can do inline import or since compare is in __init__, we import from preflight.
import preflight

def save_compare_pdf(result_a: PrepResult, result_b: PrepResult, path: str) -> None:
    if result_a is not None and not isinstance(result_a, PrepResult):
        raise TypeError("The first argument must be a PrepResult object. Please provide a valid PrepResult from preflight.prepare().")
    if result_b is not None and not isinstance(result_b, PrepResult):
        raise TypeError("The second argument must be a PrepResult object. Please provide a valid PrepResult from preflight.prepare().")

    if result_a is None or result_b is None:
        raise ValueError("You must provide two preparation results to compare them. Check that neither result is empty.")

    if result_a.df is None or result_b.df is None:
         raise ValueError("We cannot compare these results because they are missing their datasets.")

    if result_a.report is None or result_b.report is None:
        raise ValueError("We cannot generate a comparison PDF because one or both of the results are missing their report.")

    # 1. Call existing compare() function as source of truth
    diff = preflight.compare(result_a, result_b)

    counts_a = diff.get("report_entry_counts_a", {}) or {}
    counts_b = diff.get("report_entry_counts_b", {}) or {}

    with PdfPages(path) as pdf:
        # --- Page 1: Cover Page ---
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        ax.text(0.5, 0.9, "PreFlight-ML Comparison Report", ha='center', va='center', fontsize=24, fontweight='bold', color="#2c3e50")
        
        # Summary row
        ax.text(0.25, 0.82, "Result A", ha='center', va='center', fontsize=18, fontweight='bold')
        ax.text(0.75, 0.82, "Result B", ha='center', va='center', fontsize=18, fontweight='bold')
        
        shape_a_str = f"{diff['shape_a'][0]} rows × {diff['shape_a'][1]} columns"
        shape_b_str = f"{diff['shape_b'][0]} rows × {diff['shape_b'][1]} columns"
        
        ax.text(0.25, 0.77, f"Shape: {shape_a_str}", ha='center', va='center', fontsize=14)
        ax.text(0.75, 0.77, f"Shape: {shape_b_str}", ha='center', va='center', fontsize=14)
        
        ax.text(0.5, 0.65, "Severity Counts Comparison", ha='center', va='center', fontsize=18, fontweight='bold')
        
        y_start = 0.55
        for sev in ["critical", "warning", "info"]:
            ca = counts_a.get(sev, 0)
            cb = counts_b.get(sev, 0)
            
            ax.text(0.25, y_start, f"{sev.upper()}: {ca}", ha='center', va='center', fontsize=16,
                    bbox=dict(facecolor=SEVERITY_COLORS.get(sev, "#ffffff"), edgecolor='#dddddd', boxstyle='round,pad=1', alpha=1.0))
            
            # Show delta in the middle
            delta = cb - ca
            delta_str = f"{delta:+d}" if delta != 0 else "="
            ax.text(0.5, y_start, delta_str, ha='center', va='center', fontsize=16, fontweight='bold')
            
            ax.text(0.75, y_start, f"{sev.upper()}: {cb}", ha='center', va='center', fontsize=16,
                    bbox=dict(facecolor=SEVERITY_COLORS.get(sev, "#ffffff"), edgecolor='#dddddd', boxstyle='round,pad=1', alpha=1.0))
            
            y_start -= 0.12
            
        # Headline changes
        y_headline = 0.15
        if diff["columns_only_in_a"]:
            ax.text(0.5, y_headline, f"Columns removed in B: {len(diff['columns_only_in_a'])}", ha='center', va='center', fontsize=14, color="red")
            y_headline -= 0.05
        if diff["columns_only_in_b"]:
            ax.text(0.5, y_headline, f"Columns added in B: {len(diff['columns_only_in_b'])}", ha='center', va='center', fontsize=14, color="green")
            y_headline -= 0.05
        if diff["decision_diff"]:
            ax.text(0.5, y_headline, f"Columns with differing decisions: {len(diff['decision_diff'])}", ha='center', va='center', fontsize=14, color="orange")
            y_headline -= 0.05
            
        pdf.savefig(fig)
        plt.close(fig)
        
        # --- Following Pages: Side-by-side charts ---
        # 1. Missingness
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Missingness Heatmap", fontsize=16)
        try:
            plot_missingness_heatmap(result_a.df, ax=ax1)
            ax1.set_title("Result A Missingness")
        except Exception as e:
            ax1.axis('off')
            ax1.text(0.5, 0.5, f"Plot Failed: {e}", ha='center', va='center')
            
        try:
            plot_missingness_heatmap(result_b.df, ax=ax2)
            ax2.set_title("Result B Missingness")
        except Exception as e:
            ax2.axis('off')
            ax2.text(0.5, 0.5, f"Plot Failed: {e}", ha='center', va='center')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # 2. Correlation
        num_a = [p.name for p in (result_a.report.profiles or []) if p.semantic_type in (SemanticType.NUMERIC_FEATURE, SemanticType.NUMERIC_ID)]
        num_b = [p.name for p in (result_b.report.profiles or []) if p.semantic_type in (SemanticType.NUMERIC_FEATURE, SemanticType.NUMERIC_ID)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Numeric Features Correlation", fontsize=16)
        try:
            if len(num_a) >= 2:
                plot_correlation_heatmap(result_a.df, num_a, ax=ax1)
                ax1.set_title("Result A Correlation")
            else:
                raise ValueError("We need at least two numeric columns to generate a correlation heatmap.")
        except Exception as e:
            ax1.axis('off')
            ax1.text(0.5, 0.5, f"Plot Failed/Skipped: {e}", ha='center', va='center')
            
        try:
            if len(num_b) >= 2:
                plot_correlation_heatmap(result_b.df, num_b, ax=ax2)
                ax2.set_title("Result B Correlation")
            else:
                raise ValueError("We need at least two numeric columns to generate a correlation heatmap.")
        except Exception as e:
            ax2.axis('off')
            ax2.text(0.5, 0.5, f"Plot Failed/Skipped: {e}", ha='center', va='center')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # 3. Mutual Info
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("Mutual Information with Target", fontsize=16)
        try:
            if result_a.report.profiles:
                plot_mutual_info_bar_chart(result_a.report.profiles, ax=ax1)
                ax1.set_title("Result A Mutual Info")
            else:
                raise ValueError("We need column profiles to show the mutual information chart.")
        except Exception as e:
            ax1.axis('off')
            ax1.text(0.5, 0.5, f"Plot Failed/Skipped: {e}", ha='center', va='center')
            
        try:
            if result_b.report.profiles:
                plot_mutual_info_bar_chart(result_b.report.profiles, ax=ax2)
                ax2.set_title("Result B Mutual Info")
            else:
                raise ValueError("We need column profiles to show the mutual information chart.")
        except Exception as e:
            ax2.axis('off')
            ax2.text(0.5, 0.5, f"Plot Failed/Skipped: {e}", ha='center', va='center')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # 4. Class Distribution
        if result_a.report.target and result_b.report.target and result_a.report.target == result_b.report.target:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle("Class Distribution", fontsize=16)
            try:
                target_col = result_a.report.target
                if target_col in result_a.df.columns:
                    plot_class_distribution(result_a.df[target_col], ax=ax1)
                    ax1.set_title("Result A Class Dist")
                else:
                    raise ValueError("The target column is missing from the dataset.")
            except Exception as e:
                ax1.axis('off')
                ax1.text(0.5, 0.5, f"Plot Failed/Skipped: {e}", ha='center', va='center')
                
            try:
                target_col = result_b.report.target
                if target_col in result_b.df.columns:
                    plot_class_distribution(result_b.df[target_col], ax=ax2)
                    ax2.set_title("Result B Class Dist")
                else:
                    raise ValueError("The target column is missing from the dataset.")
            except Exception as e:
                ax2.axis('off')
                ax2.text(0.5, 0.5, f"Plot Failed/Skipped: {e}", ha='center', va='center')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # --- Appendix: Differences only ---
        fig, ax = plt.subplots(figsize=(10, 11))
        ax.axis('off')
        ax.text(0.5, 0.95, "Appendix: Column & Decision Differences", ha='center', va='center', fontsize=16, fontweight='bold')
        
        y_pos = 0.90
        
        if diff["columns_only_in_a"]:
            ax.text(0.05, y_pos, "Columns Only in A:", fontsize=12, fontweight='bold')
            y_pos -= 0.03
            ax.text(0.08, y_pos, ", ".join(diff["columns_only_in_a"]), fontsize=10, wrap=True)
            y_pos -= 0.05
            
        if diff["columns_only_in_b"]:
            ax.text(0.05, y_pos, "Columns Only in B:", fontsize=12, fontweight='bold')
            y_pos -= 0.03
            ax.text(0.08, y_pos, ", ".join(diff["columns_only_in_b"]), fontsize=10, wrap=True)
            y_pos -= 0.05
            
        if diff["decision_diff"]:
            ax.text(0.05, y_pos, "Differing Decisions in Shared Columns:", fontsize=12, fontweight='bold')
            y_pos -= 0.03
            for col in diff["decision_diff"][:20]: # Limit to avoid overflow for now
                ax.text(0.08, y_pos, f"{col}", fontsize=10, fontweight='bold')
                y_pos -= 0.02
            if len(diff["decision_diff"]) > 20:
                ax.text(0.08, y_pos, f"... and {len(diff['decision_diff'])-20} more", fontsize=10)
                
        pdf.savefig(fig)
        plt.close(fig)
