"""
Configuration parameters and threshold defaults for PreFlight v2.0 Cleaning.
"""

from dataclasses import dataclass


@dataclass
class CleanerConfig:
    """
    Configuration parameters for cleaning operations.

    Attributes:
        drop_missing_threshold: Missingness proportion above which a column is dropped (v1 default 0.6).
        impute_strategy_threshold: Missingness proportion below which median/mode is used;
            at or above this threshold, constant fill is used (v1 constant 0.3).
        winsorize_missing_limit: Maximum missingness rate that permits winsorization (v1 constant 0.30).
        rare_category_threshold: Minimum category frequency; categories below this are grouped (v1 0.01).
        cardinality_threshold: Distinct values at which a categorical column counts as high-variety (v1 20).
        outlier_method: Outlier clipping method ('iqr' or 'zscore').
        dedupe_impact_threshold: Fraction of total rows removed by deduplication
            that triggers human review (needs_attention=True). Default: 0.02 (2%).
        column_drop_impact_threshold: Fraction of total columns dropped that triggers
            human review (needs_attention=True). Default: 0.25 (a quarter of the table;
            removing a single identifier or constant column from a table of four or more is routine).
        outlier_impact_threshold: Fraction of column rows clipped by winsorization
            that triggers human review (needs_attention=True). Default: 0.05 (5%).
    """
    drop_missing_threshold: float = 0.60
    impute_strategy_threshold: float = 0.30
    winsorize_missing_limit: float = 0.30
    rare_category_threshold: float = 0.01
    outlier_method: str = "iqr"
    cardinality_threshold: int = 20
    dedupe_impact_threshold: float = 0.02
    column_drop_impact_threshold: float = 0.25
    outlier_impact_threshold: float = 0.05


DEFAULT_CLEANER_CONFIG = CleanerConfig()
