"""
Tests for Phase 1 (v1.0.0): TEXT SemanticType detection + basic stats.

Covers:
- TEXT is correctly detected for genuine free-text columns
- TEXT is NOT falsely triggered for CATEGORICAL_HIGH, CATEGORICAL_LOW, or NUMERIC_ID columns
- text_avg_length, text_avg_word_count, text_missing_rate are computed correctly for TEXT columns
- Those same fields remain None for all non-TEXT columns
- A ReportEntry is logged for every TEXT column detected
- Cleaner and Engineer do not crash when a TEXT column is present (pass-through behavior)
"""

import numpy as np
import pandas as pd
import pytest

import preflight as pf
from preflight.types import SemanticType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def text_heavy_df():
    """DataFrame with a genuine free-text column alongside normal columns."""
    reviews = [
        "This product exceeded my expectations in almost every way possible",
        "Terrible experience, would not recommend this to anyone at all",
        "Pretty average, does the job but nothing special about it really",
        "Absolutely loved it, will definitely be buying again very soon",
        "Not worth the price, quality feels cheap and it broke fast",
        "Great value for money, works exactly as described on the box",
        "Customer service was unhelpful and the return process was painful",
        "Solid build quality, arrived on time and packaged very well",
        "Disappointed overall, the item looked different from the pictures",
        "Would buy again, no complaints so far after two months of use",
    ] * 5  # 50 rows, enough for cardinality/length signal

    n = len(reviews)
    return pd.DataFrame({
        "review_text": reviews,
        "category": np.random.choice(["A", "B", "C"], size=n),  # CATEGORICAL_LOW
        "customer_id": [f"CUST-{i:05d}" for i in range(n)],       # NUMERIC_ID-like / high card short strings
        "rating": np.random.randint(1, 6, size=n),
        "price": np.random.uniform(10, 200, size=n),
    })


@pytest.fixture
def text_with_missing_df():
    reviews = [
        "This is a fairly long piece of free text describing an experience",
        None,
        "Another decently long sentence to make sure detection triggers properly",
        "",
        "Yet another example of free text content for this column here",
    ] * 6  # 30 rows
    return pd.DataFrame({
        "comments": reviews,
        "target": np.random.randint(0, 2, size=len(reviews)),
    })


@pytest.fixture
def no_text_df():
    """DataFrame with no genuine text columns — should never trigger TEXT."""
    n = 40
    return pd.DataFrame({
        "id": range(n),
        "status": np.random.choice(["active", "inactive"], size=n),
        "country_code": np.random.choice(["US", "IN", "UK", "DE"], size=n),
        "score": np.random.uniform(0, 1, size=n),
        "target": np.random.randint(0, 2, size=n),
    })


def _profiles_by_name(profiles):
    return {p.name: p for p in profiles}


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestTextDetection:

    def test_free_text_column_detected_as_text(self, text_heavy_df):
        profiles, _entries = pf.run_profiler(text_heavy_df, target="rating", task="regression")
        by_name = _profiles_by_name(profiles)
        assert by_name["review_text"].semantic_type == SemanticType.TEXT

    def test_categorical_low_not_detected_as_text(self, text_heavy_df):
        profiles, _entries = pf.run_profiler(text_heavy_df, target="rating", task="regression")
        by_name = _profiles_by_name(profiles)
        assert by_name["category"].semantic_type != SemanticType.TEXT

    def test_numeric_columns_not_detected_as_text(self, text_heavy_df):
        profiles, _entries = pf.run_profiler(text_heavy_df, target="rating", task="regression")
        by_name = _profiles_by_name(profiles)
        assert by_name["price"].semantic_type != SemanticType.TEXT

    def test_short_high_cardinality_ids_not_detected_as_text(self, text_heavy_df):
        """Short, structured, high-cardinality strings (like IDs) must not be misclassified as TEXT."""
        profiles, _entries = pf.run_profiler(text_heavy_df, target="rating", task="regression")
        by_name = _profiles_by_name(profiles)
        assert by_name["customer_id"].semantic_type != SemanticType.TEXT

    def test_no_text_columns_present(self, no_text_df):
        profiles, _entries = pf.run_profiler(no_text_df, target="target", task="classification")
        assert all(p.semantic_type != SemanticType.TEXT for p in profiles)


# ---------------------------------------------------------------------------
# Stats computation tests
# ---------------------------------------------------------------------------

class TestTextStats:

    def test_text_stats_populated_for_text_column(self, text_heavy_df):
        profiles, _entries = pf.run_profiler(text_heavy_df, target="rating", task="regression")
        by_name = _profiles_by_name(profiles)
        text_profile = by_name["review_text"]

        assert text_profile.text_avg_length is not None
        assert text_profile.text_avg_word_count is not None
        assert text_profile.text_missing_rate is not None
        assert text_profile.text_avg_length > 0
        assert text_profile.text_avg_word_count > 0

    def test_text_missing_rate_correctness(self, text_with_missing_df):
        profiles, _entries = pf.run_profiler(text_with_missing_df, target="target", task="classification")
        by_name = _profiles_by_name(profiles)
        comments_profile = by_name["comments"]

        assert comments_profile.semantic_type == SemanticType.TEXT
        # None and "" should both count toward missing/empty depending on implementation,
        # but at minimum the None values must be reflected.
        assert comments_profile.text_missing_rate > 0
        assert 0.0 <= comments_profile.text_missing_rate <= 1.0

    def test_non_text_columns_have_none_text_stats(self, text_heavy_df):
        profiles, _entries = pf.run_profiler(text_heavy_df, target="rating", task="regression")
        by_name = _profiles_by_name(profiles)

        for col_name in ["category", "customer_id", "price"]:
            profile = by_name[col_name]
            assert profile.text_avg_length is None
            assert profile.text_avg_word_count is None
            assert profile.text_missing_rate is None


# ---------------------------------------------------------------------------
# Report logging tests
# ---------------------------------------------------------------------------

class TestTextReportLogging:

    def test_report_entry_logged_for_text_column(self, text_heavy_df):
        _profiles, entries = pf.run_profiler(text_heavy_df, target="rating", task="regression")
        text_entries = [
            e for e in entries
            if e.column == "review_text" and e.stage == "profiler"
        ]
        assert len(text_entries) >= 1
        assert text_entries[0].severity == "info"
        # Message should be honest about scope: detection only, no transformation yet.
        assert "text" in text_entries[0].rationale.lower()

    def test_no_text_entry_logged_when_no_text_columns(self, no_text_df):
        _profiles, entries = pf.run_profiler(no_text_df, target="target", task="classification")
        text_related = [e for e in entries if "text" in e.rationale.lower() and e.stage == "profiler"]
        assert len(text_related) == 0


# ---------------------------------------------------------------------------
# Pipeline pass-through / no-crash tests
# ---------------------------------------------------------------------------

class TestTextPassThrough:

    def test_prepare_does_not_crash_with_text_column(self, text_heavy_df):
        result = pf.prepare(text_heavy_df, target="rating", task="regression")
        assert result.df is not None
        assert result.pipeline is not None
        assert result.report is not None

    def test_clean_does_not_crash_with_text_column(self, text_heavy_df):
        result = pf.clean(text_heavy_df, target="rating", task="regression")
        assert result.df is not None

    def test_engineer_does_not_crash_with_text_column(self, text_heavy_df):
        result = pf.engineer(text_heavy_df, target="rating", task="regression")
        assert result.df is not None

    def test_prepare_does_not_crash_with_missing_text_values(self, text_with_missing_df):
        result = pf.prepare(text_with_missing_df, target="target", task="classification")
        assert result.df is not None
        assert result.pipeline is not None

    def test_existing_semantic_types_unaffected(self, no_text_df):
        """Sanity check: adding TEXT detection must not change classification of existing types."""
        profiles, _entries = pf.run_profiler(no_text_df, target="target", task="classification")
        by_name = _profiles_by_name(profiles)

        assert by_name["status"].semantic_type in (
            SemanticType.CATEGORICAL_LOW,
            SemanticType.BOOLEAN,
        )
        assert by_name["country_code"].semantic_type == SemanticType.CATEGORICAL_LOW
        assert by_name["score"].semantic_type == SemanticType.NUMERIC_FEATURE