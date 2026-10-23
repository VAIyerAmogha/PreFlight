# Phase 6 Port Contract (6.0)

Status: DRAFT, awaiting user decision on section 3. Sources: v1.0.1 `profiler.py`, `types.py`, `cleaner.py`, `engineer.py`, `assembler.py`; v2 `schema.py`, `infer/types.py`, and the existing (pre-audit) `clean/`, `engineer/` code.

## 1. What v1 Cleaner/Engineer/Assembler consume from the Profiler (`ColumnProfile`)

| v1 field | Type | Consumed by |
|---|---|---|
| `semantic_type` | enum of 9: NUMERIC_FEATURE, NUMERIC_ID, CATEGORICAL_LOW, CATEGORICAL_HIGH, DATETIME_NATIVE, DATETIME_STRING, BOOLEAN, CONSTANT, TEXT | Cleaner (every branch), Engineer (every branch), Assembler (both transformers) |
| `missing_rate` | float | Cleaner (drop threshold, impute strategy, indicator, winsorize gate), Assembler fit |
| `rare_categories` | list[str] (<1% of non-null) | Cleaner (CATEGORICAL_HIGH grouping), Assembler transform |
| `correlation_with_target` | float or None (Pearson, numeric only) | Engineer interactions (ranking) |
| `mutual_info_with_target` | float or None | Engineer interactions (ranking) |
| `is_leakage_suspect` | bool (abs corr > 0.95) | Report flag only (critical entry). v1 never dropped leaky features; v2 emits a `suspected_leakage` Decision that keeps the column unless the person chooses to leave it out |
| `vif_score` | float or None (numpy corr-matrix inverse, top 50 by variance) | Report flag only |
| `outlier_rate`, `cardinality` | float / int | Report/profile only (cardinality drives the LOW/HIGH split inside the profiler) |
| `text_avg_length`, `text_avg_word_count`, `text_missing_rate` | float | Report only |
| ReportEntry flags: mixed types, missing > 30%, class imbalance > 10:1 | | Report only |
| `cardinality_threshold` (20) | int | Profiler LOW/HIGH split; Engineer/Assembler `fit` |

## 2. Mapping to v2

| v1 | v2 equivalent | Notes |
|---|---|---|
| NUMERIC_FEATURE | `inferred_type` in {int, float} (canonical "numeric") | v1 also required not high-cardinality ID |
| NUMERIC_ID | `inferred_type == "ID"` (candidate score from infer/) | NOT the same rule: v1 = unique ratio >= 0.95 AND (name matches id/_id/pk/index OR monotonic integer). v2 `ID_NAME_PATTERNS` also includes key, code, fips, zip, postal, rownames and handles leading zeros. Draft `clean/drops.py` uses a third rule (name part in id/pk/index, no cardinality test). |
| CATEGORICAL_LOW / HIGH | `categorical` (no split) | v1 split by `nunique < 20` |
| DATETIME_NATIVE / DATETIME_STRING | `datetime` | v2 typing already converts, so only one native kind exists after Phase 4 apply |
| BOOLEAN | `boolean` | v1 also counted any 2-unique column as BOOLEAN |
| CONSTANT | `constant` | v1: <= 1 unique non-null value or all null |
| TEXT | `text` | v1: avg len > 50, or > 30 with >= 5 words. v2 `text` is also the abstain fallback, so it is broader |
| missing markers | `ColumnRawMetadata.missing_markers` (applied in cleaner) | new in v2, no v1 equivalent |
| column confidence / abstained / anomalies | `confidence`, `abstained`, `anomalies` | no v1 equivalent; use for `needs_attention` and plan evidence |
| target / task | not in schema; passed as `target=`, `task=` | v1 `run_profiler(df, target, task)` |

## 3. v1 fields with NO v2 equivalent (STOP: user decision required)

The v2 schema stores structure and type only. It computes none of these EDA signals:

1. `missing_rate` (per column, after markers)
2. `cardinality` and the CATEGORICAL_LOW/HIGH split (threshold 20)
3. `rare_categories` (<1% frequency list)
4. `outlier_rate`
5. `correlation_with_target`
6. `mutual_info_with_target`
7. `is_leakage_suspect`
8. `vif_score`
9. text stats (avg length, avg word count)
10. class imbalance ratio, mixed-type flag
11. v1 NUMERIC_ID rule (unique ratio + name/monotonic) and BOOLEAN-by-2-unique rule

Current state of the existing draft: it silently recomputes most of these inline from the DataFrame inside each `plan_*` (e.g. `plan_drops` computes missing rate and an ID rule; `plan_engineer` splits categorical by `nunique >= cardinality_threshold`; `plan_vif_and_leakage` computes correlation/VIF). `tests/regression/adapter.py` additionally re-implements the whole v1 profiler inside the tests. None of this is declared in a contract.

### Decision requested
- **Option A (recommended):** compute signals 1-10 in the port in one small, pure `signals.py` (ported verbatim from v1 profiler functions, input = typed DataFrame + schema + target/task). Cleaner/Engineer read `schema` for type and `signals` for numbers. Item 11 uses the v2 `ID`/`boolean` types, not the v1 rules (changes behaviour, logged in port_log.md).
- **Option B:** drop the logic that depends on them (interactions ranking, leakage flag, rare-category grouping, missing-rate-based drop/impute). This removes most of the cleaner.

## 4. Preliminary audit of the existing draft vs v1 (not yet complete)

| File / function | Finding |
|---|---|
| `clean/drops.py:plan_drops` | ID rule differs from v1 (no cardinality test, name-part match only); also drops on `col_type == "id"` |
| `clean/config.py` | Adds 3 impact thresholds (named, documented defaults 2%/5%/5%): consistent with task. `impute_strategy_threshold`, `winsorize_missing_limit` must be checked against v1 constants |
| `clean/cleaner.py:run_cleaner` | Builds a fake `TableSchema` from v1 profiles to feed v2 code: a test shim living in `src/`; should not ship |
| `engineer/vif.py:plan_vif_and_leakage` | VIF emits a `feature_overlap` Decision (v1 only reported it; new behaviour). Leakage emits `suspected_leakage` Decision with `needs_attention=True` (matches task) |
| `engineer/engineer.py:plan_engineer` | categorical split inline (`nunique >= cardinality_threshold`); passthrough rules differ from v1 `run_engineer` order |
| `assemble.py` | stub; `CleanerTransformer`, `EngineerTransformer`, `build_pipeline_two_phase`, `transform_new_data` missing |
| `plan.py` | Fixing/Filling/Removing/Creating stages are stubs (not registered with real planners) |
| Baseline | `tests/unit` (except `test_assemble.py`): 176 passed. `test_assemble.py` and `tests/regression/*` fail at import |
