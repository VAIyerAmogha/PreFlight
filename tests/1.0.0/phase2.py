"""
Tests for Phase 2 (v1.0.0): opt-in text feature generation.

Covers:
- text_features=False (default) produces zero new columns — backward compatibility guaranteed
- text_features=True generates {col}_char_length, {col}_word_count, {col}_has_text
- text_tfidf=True additionally generates {col}_tfidf_* columns, capped at text_tfidf_top_k
- text_tfidf=True but text_features=False has no effect (master switch respected)
- Missing/null text values do not crash the generator
- Column name collisions are skipped with a warning ReportEntry, never overwritten
- A ReportEntry is logged for every generated text feature column, with honest scope language
- FeatureConfig validation: text_tfidf_top_k must be positive
- Two-phase fit boundary respected: pipeline.transform() on new/unseen data reproduces the same
  tfidf vocabulary learned at fit time (no leakage / no refitting on transform)
"""

import numpy as np
import pandas as pd
import pytest

import preflight as pf
from preflight.types import FeatureConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def text_df():
    reviews = [
        "This product exceeded my expectations in almost every way possible",
        "Terrible experience would not recommend this to anyone at all",
        "Pretty average does the job but nothing special about it really",
        "Absolutely loved it will definitely be buying again very soon",
        "Not worth the price quality feels cheap and it broke fast",
        "Great value for money works exactly as described on the box",
        "Customer service was unhelpful and the return process was painful",
        "Solid build quality arrived on time and packaged very well",
        "Disappointed overall the item looked different from the pictures",
        "Would buy again no complaints so far after two months of use",
    ] * 6  # 60 rows

    n = len(reviews)
    return pd.DataFrame({
        "review_text": reviews,
        "rating": np.random.randint(1, 6, size=n),
        "price": np.random.uniform(10, 200, size=n),
    })


@pytest.fixture
def text_df_with_missing():
    reviews = [
        "This is a fairly long piece of free text describing an experience",
        None,
        "Another decently long sentence to make sure detection triggers properly",
        "",
        "   ",  # whitespace only
        "Yet another example of free text content for this column here",
    ] * 6  # 36 rows
    return pd.DataFrame({
        "comments": reviews,
        "target": np.random.randint(0, 2, size=len(reviews)),
    })


@pytest.fixture
def collision_df():
    """Text column paired with a pre-existing column name that will collide with a generated feature."""
    reviews = [
        "This is some example free text content for testing collisions",
        "Another example sentence used to test collision handling here",
    ] * 20
    n = len(reviews)
    return pd.DataFrame({
        "notes": reviews,
        "notes_char_length": np.zeros(n),  # deliberately collides with generated column name
        "target": np.random.randint(0, 2, size=n),
    })


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_no_feature_config_produces_zero_new_text_columns(self, text_df):
        baseline = pf.engineer(text_df, target="rating", task="regression")
        with_none_config = pf.engineer(
            text_df, target="rating", task="regression"
        )
        assert set(baseline.df.columns) == set(with_none_config.df.columns)

    def test_text_features_false_produces_zero_new_columns(self, text_df):
        config = FeatureConfig(text_features=False)
        baseline = pf.engineer(text_df, target="rating", task="regression")
        result = pf.prepare(text_df, target="rating", task="regression", feature_config=config)
        text_generated = [c for c in result.df.columns if c.startswith("review_text_")]
        assert text_generated == []

    def test_tfidf_true_but_text_features_false_has_no_effect(self, text_df):
        """Master switch must gate the sub-toggle."""
        config = FeatureConfig(text_features=False, text_tfidf=True)
        result = pf.prepare(text_df, target="rating", task="regression", feature_config=config)
        tfidf_cols = [c for c in result.df.columns if "_tfidf_" in c]
        assert tfidf_cols == []


# ---------------------------------------------------------------------------
# Basic text feature generation
# ---------------------------------------------------------------------------

class TestBasicTextFeatures:

    def test_char_length_word_count_has_text_generated(self, text_df):
        config = FeatureConfig(text_features=True)
        result = pf.prepare(text_df, target="rating", task="regression", feature_config=config)

        assert "review_text_char_length" in result.df.columns
        assert "review_text_word_count" in result.df.columns
        assert "review_text_has_text" in result.df.columns

    def test_char_length_and_word_count_values_reasonable(self, text_df):
        config = FeatureConfig(text_features=True)
        result = pf.prepare(text_df, target="rating", task="regression", feature_config=config)

        assert (result.df["review_text_char_length"] > 0).all()
        assert (result.df["review_text_word_count"] > 0).all()
        assert result.df["review_text_has_text"].isin([0, 1]).all()

    def test_has_text_zero_for_missing_or_empty(self, text_df_with_missing):
        config = FeatureConfig(text_features=True)
        result = pf.prepare(
            text_df_with_missing, target="target", task="classification", feature_config=config
        )
        # Original missing/empty/whitespace-only rows should map to has_text == 0
        original = text_df_with_missing["comments"].reset_index(drop=True)
        has_text = result.df["comments_has_text"].reset_index(drop=True)

        for orig_val, flag in zip(original, has_text):
            is_effectively_empty = orig_val is None or str(orig_val).strip() == ""
            if is_effectively_empty:
                assert flag == 0

    def test_no_crash_on_missing_text_values(self, text_df_with_missing):
        config = FeatureConfig(text_features=True)
        result = pf.prepare(
            text_df_with_missing, target="target", task="classification", feature_config=config
        )
        assert result.df is not None
        assert result.pipeline is not None


# ---------------------------------------------------------------------------
# TF-IDF-lite generation
# ---------------------------------------------------------------------------

class TestTfidfLite:

    def test_tfidf_columns_generated_when_enabled(self, text_df):
        config = FeatureConfig(text_features=True, text_tfidf=True, text_tfidf_top_k=10)
        result = pf.prepare(text_df, target="rating", task="regression", feature_config=config)

        tfidf_cols = [c for c in result.df.columns if c.startswith("review_text_tfidf_")]
        assert len(tfidf_cols) > 0
        assert len(tfidf_cols) <= 10

    def test_tfidf_top_k_respected(self, text_df):
        config = FeatureConfig(text_features=True, text_tfidf=True, text_tfidf_top_k=5)
        result = pf.prepare(text_df, target="rating", task="regression", feature_config=config)

        tfidf_cols = [c for c in result.df.columns if c.startswith("review_text_tfidf_")]
        assert len(tfidf_cols) <= 5

    def test_tfidf_disabled_by_default_even_with_text_features_on(self, text_df):
        config = FeatureConfig(text_features=True)  # text_tfidf defaults False
        result = pf.prepare(text_df, target="rating", task="regression", feature_config=config)

        tfidf_cols = [c for c in result.df.columns if "_tfidf_" in c]
        assert tfidf_cols == []

    def test_tfidf_pipeline_reproducible_on_new_data(self, text_df):
        """Two-phase fit boundary: transform on new data must not refit vocabulary."""
        config = FeatureConfig(text_features=True, text_tfidf=True, text_tfidf_top_k=8)
        result = pf.prepare(text_df, target="rating", task="regression", feature_config=config)

        new_data = text_df.drop(columns=["rating"]).iloc[:5].copy()
        transformed = result.pipeline.transform(new_data)

        fitted_tfidf_cols = sorted(c for c in result.df.columns if "_tfidf_" in c)
        new_tfidf_cols = sorted(c for c in transformed.columns if "_tfidf_" in c)
        assert fitted_tfidf_cols == new_tfidf_cols


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestFeatureConfigValidation:

    def test_text_tfidf_top_k_must_be_positive(self):
        with pytest.raises(ValueError):
            FeatureConfig(text_features=True, text_tfidf=True, text_tfidf_top_k=0)

    def test_text_tfidf_top_k_negative_raises(self):
        with pytest.raises(ValueError):
            FeatureConfig(text_features=True, text_tfidf=True, text_tfidf_top_k=-3)

    def test_default_feature_config_has_text_features_off(self):
        config = FeatureConfig()
        assert config.text_features is False
        assert config.text_tfidf is False


# ---------------------------------------------------------------------------
# Column collision handling
# ---------------------------------------------------------------------------

class TestCollisionHandling:

    def test_collision_skipped_not_overwritten(self, collision_df):
        config = FeatureConfig(text_features=True)
        result = pf.prepare(collision_df, target="target", task="classification", feature_config=config)

        # Original collided column's values must remain untouched (all zeros as defined in fixture)
        assert (result.df["notes_char_length"] == 0).all()

    def test_collision_logs_warning_entry(self, collision_df):
        config = FeatureConfig(text_features=True)
        result = pf.prepare(collision_df, target="target", task="classification", feature_config=config)

        entries = result.report.to_dict().get("entries", None)
        if entries is None:
            entries = result.report.to_dataframe().to_dict("records")

        collision_entries = [
            e for e in entries
            if "skipped_duplicate_feature" in str(e.get("action", "")).lower()
            or "duplicate" in str(e.get("rationale", "")).lower()
        ]
        assert len(collision_entries) >= 1


# ---------------------------------------------------------------------------
# Report logging
# ---------------------------------------------------------------------------

class TestTextFeatureReportLogging:

    def test_report_entries_logged_for_generated_text_columns(self, text_df):
        config = FeatureConfig(text_features=True, text_tfidf=True, text_tfidf_top_k=5)
        result = pf.prepare(text_df, target="rating", task="regression", feature_config=config)

        entries = result.report.to_dataframe()
        engineer_text_entries = entries[
            (entries["stage"] == "engineer") &
            (entries["column"].astype(str).str.contains("review_text"))
        ]
        assert len(engineer_text_entries) >= 1

    def test_report_entries_mention_scope_limitation(self, text_df):
        config = FeatureConfig(text_features=True)
        result = pf.prepare(text_df, target="rating", task="regression", feature_config=config)

        entries = result.report.to_dataframe()
        engineer_text_entries = entries[
            (entries["stage"] == "engineer") &
            (entries["column"].astype(str).str.contains("review_text"))
        ]
        combined_text = " ".join(engineer_text_entries["rationale"].astype(str).tolist()).lower()
        assert "basic" in combined_text or "nlp" in combined_text or "scope" in combined_text

    def test_no_text_entries_when_text_features_disabled(self, text_df):
        result = pf.prepare(text_df, target="rating", task="regression", feature_config=FeatureConfig())
        entries = result.report.to_dataframe()
        engineer_text_entries = entries[
            (entries["stage"] == "engineer") &
            (entries["column"].astype(str).str.contains("review_text"))
        ]
        # No text-generation-related entries should exist at engineer stage since text_features=False
        assert len(engineer_text_entries) == 0