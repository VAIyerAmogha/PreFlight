"""
Configuration parameters and threshold defaults for PreFlight v2.0 Cleaning.
"""

from dataclasses import dataclass


@dataclass
class CleanerConfig:
    """
    Configuration parameters for cleaning operations.

    Attributes:
        drop_missing_threshold: Missingness proportion above which a column is dropped.
        impute_strategy_threshold: Missingness proportion below which median/mode is used;
            above this threshold, constant fill is used.
        winsorize_missing_limit: Maximum allowed missingness rate to permit winsorization.
        rare_category_threshold: Minimum category frequency; categories below this are grouped.
        outlier_method: Outlier clipping method ('iqr' or 'zscore').
        dedupe_impact_threshold: Fraction of total rows removed by deduplication
            that triggers human review (needs_attention=True). Default: 0.02 (2%).
        column_drop_impact_threshold: Fraction of total columns dropped that triggers
            human review (needs_attention=True). Default: 0.05 (5%).
        outlier_impact_threshold: Fraction of column rows clipped by winsorization
            that triggers human review (needs_attention=True). Default: 0.05 (5%).
    """
    drop_missing_threshold: float = 0.60
    impute_strategy_threshold: float = 0.30
    winsorize_missing_limit: float = 0.30
    rare_category_threshold: float = 0.01
    outlier_method: str = "iqr"
    dedupe_impact_threshold: float = 0.02
    column_drop_impact_threshold: float = 0.05
    outlier_impact_threshold: float = 0.05


DEFAULT_CLEANER_CONFIG = CleanerConfig()
