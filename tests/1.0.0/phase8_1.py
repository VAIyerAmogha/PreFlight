"""
Tests for Phase 8.1 (v1.0.0): error message simplification pass.

These tests don't (and can't fully) judge "simplicity" objectively, but they check for
concrete, verifiable proxies of plain-language, user-facing error messages:
- Messages are non-empty and reasonably short (not a wall of text / stack trace dump)
- Messages do not leak raw Python internals (e.g. "NoneType", "traceback", "Object at 0x")
- Messages do not use internal-only jargon terms that a non-expert user wouldn't know
  (e.g. "ColumnProfile", "semantic_type", "two-phase fit", "cross-fit")
- Messages still contain the specific, useful information (column names, preset names, etc.)
- CLI errors never surface a Python traceback to the user
"""

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import preflight as pf
from preflight.types import FeatureConfig, SemanticType
from preflight.cli import app


runner = CliRunner()

# Terms that should never leak into a user-facing error message.
FORBIDDEN_JARGON = [
    "columnprofile", "semantic_type", "two-phase fit", "two phase fit",
    "cross-fit", "crossfit", "nonetype", "traceback", "object at 0x",
    "__init__", "assembler", "self.",
]


def _assert_message_is_clean(message: str):
    assert message, "Error message must not be empty"
    lowered = message.lower()
    for term in FORBIDDEN_JARGON:
        assert term not in lowered, f"Message leaks internal jargon: '{term}' in '{message}'"
    # Reasonable upper bound so we catch accidental raw dumps (e.g. huge exception chains)
    assert len(message) < 500, "Error message is too long — should be simple and short"


@pytest.fixture
def sample_df():
    n = 60
    return pd.DataFrame({
        "num_a": np.random.uniform(0, 100, size=n),
        "cat": np.random.choice(["x", "y", "z"], size=n),
        "target": np.random.randint(0, 2, size=n),
    })


# ---------------------------------------------------------------------------
# Task/target mismatch
# ---------------------------------------------------------------------------

class TestTaskTargetMismatchMessage:

    def test_message_is_clean_and_mentions_column(self):
        n = 300
        df = pd.DataFrame({
            "price": np.random.uniform(10, 500, size=n),
            "num_a": np.random.uniform(0, 1, size=n),
        })
        with pytest.raises(ValueError) as exc_info:
            pf.prepare(df, target="price", task="classification")
        message = str(exc_info.value)
        _assert_message_is_clean(message)
        assert "price" in message
        assert "regression" in message.lower() or "classification" in message.lower()


# ---------------------------------------------------------------------------
# FeatureConfig validation
# ---------------------------------------------------------------------------

class TestFeatureConfigMessages:

    def test_invalid_interaction_types_message_clean(self):
        with pytest.raises(ValueError) as exc_info:
            FeatureConfig(interactions=True, interaction_types=["not_a_real_type"])
        _assert_message_is_clean(str(exc_info.value))

    def test_non_positive_interaction_top_k_message_clean(self):
        with pytest.raises(ValueError) as exc_info:
            FeatureConfig(interactions=True, interaction_top_k=0)
        _assert_message_is_clean(str(exc_info.value))

    def test_invalid_cluster_k_message_clean(self):
        with pytest.raises(ValueError) as exc_info:
            FeatureConfig(clustering=True, cluster_k="not_auto_or_int")
        _assert_message_is_clean(str(exc_info.value))

    def test_non_positive_text_tfidf_top_k_message_clean(self):
        with pytest.raises(ValueError) as exc_info:
            FeatureConfig(text_features=True, text_tfidf=True, text_tfidf_top_k=0)
        _assert_message_is_clean(str(exc_info.value))


# ---------------------------------------------------------------------------
# column_types validation
# ---------------------------------------------------------------------------

class TestColumnTypesMessages:

    def test_nonexistent_column_message_clean_and_specific(self, sample_df):
        with pytest.raises(ValueError) as exc_info:
            pf.prepare(
                sample_df, target="target", task="classification",
                column_types={"does_not_exist": SemanticType.CATEGORICAL_LOW},
            )
        message = str(exc_info.value)
        _assert_message_is_clean(message)
        assert "does_not_exist" in message

    def test_invalid_semantic_type_value_message_clean(self, sample_df):
        with pytest.raises(ValueError) as exc_info:
            pf.prepare(
                sample_df, target="target", task="classification",
                column_types={"cat": "NOT_A_REAL_TYPE"},
            )
        _assert_message_is_clean(str(exc_info.value))

    def test_override_target_column_message_clean(self, sample_df):
        with pytest.raises(ValueError) as exc_info:
            pf.prepare(
                sample_df, target="target", task="classification",
                column_types={"target": SemanticType.CATEGORICAL_LOW},
            )
        _assert_message_is_clean(str(exc_info.value))


# ---------------------------------------------------------------------------
# add_features() guardrails
# ---------------------------------------------------------------------------

class TestAddFeaturesMessages:

    def test_missing_pipeline_message_clean(self, sample_df):
        profile_only_result = pf.profile(sample_df, target="target", task="classification")
        config = FeatureConfig(interactions=True)
        with pytest.raises(ValueError) as exc_info:
            pf.add_features(profile_only_result, config)
        message = str(exc_info.value)
        _assert_message_is_clean(message)
        assert "prepare" in message.lower()  # should guide user toward pf.prepare()


# ---------------------------------------------------------------------------
# preset validation
# ---------------------------------------------------------------------------

class TestPresetMessages:

    def test_invalid_preset_message_lists_options_plainly(self, sample_df):
        with pytest.raises(ValueError) as exc_info:
            pf.prepare(sample_df, target="target", task="classification", preset="bogus")
        message = str(exc_info.value)
        _assert_message_is_clean(message)
        assert "fast" in message.lower()
        assert "thorough" in message.lower()


# ---------------------------------------------------------------------------
# CLI errors never show a traceback
# ---------------------------------------------------------------------------

class TestCliErrorMessages:

    def test_invalid_preset_cli_error_clean(self, sample_df, tmp_path):
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path, index=False)

        result = runner.invoke(app, [
            "prepare", str(csv_path), "--target", "target", "--task", "classification",
            "--preset", "bogus",
        ])
        assert result.exit_code != 0
        _assert_message_is_clean(result.output.strip())

    def test_malformed_column_type_cli_error_clean(self, sample_df, tmp_path):
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path, index=False)

        result = runner.invoke(app, [
            "prepare", str(csv_path), "--target", "target", "--task", "classification",
            "--column-type", "malformed_no_colon",
        ])
        assert result.exit_code != 0
        output = result.output.strip()
        assert "traceback" not in output.lower()
        assert "Traceback (most recent call last)" not in result.output

    def test_invalid_cluster_k_cli_error_clean(self, sample_df, tmp_path):
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path, index=False)

        result = runner.invoke(app, [
            "prepare", str(csv_path), "--target", "target", "--task", "classification",
            "--clustering", "--cluster-k", "not_valid",
        ])
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output


# ---------------------------------------------------------------------------
# Regression: rewritten messages still surface as ValueError (type unchanged)
# ---------------------------------------------------------------------------

class TestExceptionTypesUnchanged:

    def test_task_target_mismatch_still_valueerror(self):
        n = 300
        df = pd.DataFrame({
            "price": np.random.uniform(10, 500, size=n),
            "num_a": np.random.uniform(0, 1, size=n),
        })
        with pytest.raises(ValueError):
            pf.prepare(df, target="price", task="classification")

    def test_feature_config_validation_still_valueerror(self):
        with pytest.raises(ValueError):
            FeatureConfig(interactions=True, interaction_top_k=-1)

    def test_column_types_validation_still_valueerror(self, sample_df):
        with pytest.raises(ValueError):
            pf.prepare(
                sample_df, target="target", task="classification",
                column_types={"nope": SemanticType.TEXT},
            )