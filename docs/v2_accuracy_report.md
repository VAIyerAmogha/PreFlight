> **SUPERSEDED (Phase 7b, 2026-10-30):** the numbers here are replaced by docs/v2_hardening_report.md.

> **Held-out set:** every held-out number in this document is from a set that was **seen during review, not independent** (see docs/v2_hardening_report.md).

# PreFlight v2.0 — Corpus Accuracy Report

**Date**: 2026-10-23
**Method**: `scripts/corpus_accuracy.py` ran `preflight.plan.plan()` (full ingest -> infer ->
clean -> engineer plan, per `src/preflight/plan.py`) over every file in `corpus/raw/`,
inside a per-file `try/except` so one bad file can't abort the run. Results were compared
against the per-file ground-truth label JSONs in `corpus/labels/`, split into `tuning` /
`held_out` groups per the frozen `corpus/labels/split.json` (not regenerated, read only).
Raw per-file results: `scripts/_corpus_accuracy_raw.json`. This is a measurement-only
pass — no thresholds, heuristics, or library behavior were tuned to hit a target.

Corpus actually measured: **254 files**, **153 tuning / 101 held_out** (matches
`split.json`'s own `counts`).

## Target vs. actual

| # | Metric | Tuning | Held-out | Combined | Target | PASS/FAIL |
|---|---|---|---|---|---|---|
| 1 | Files loading without unhandled exceptions | 153/153 = 100.0% | 101/101 = 100.0% | 254/254 = 100.0% | 100% | **PASS** (all three) |
| 2 | Dialect accuracy (delimiter+encoding+header_row_index, exact match) | 140/153 = 91.50% | 93/101 = 92.08% | 233/254 = 91.73% | ≥95% | **FAIL** (all three) |
| 3 | Column type accuracy (canonical 7-type match) | 1619/1765 = 91.73% | 978/1085 = 90.14% | 2597/2850 = 91.12% | ≥95% | **FAIL** (all three) |
| 4 | Wrong type with NO flag | 8/1765 = 0.45% | 16/1085 = 1.47% | 24/2850 = 0.84% | ≤1% | tuning **PASS**, held_out **FAIL**, combined **PASS** |
| 5 | Decisions logged with evidence AND confidence | 4737/4737 = 100.0% | 3052/3052 = 100.0% | 7789/7789 = 100.0% | 100% | **PASS** (all three) |
| 6 | Unexplained skipped tests (full `pytest tests/` run) | n/a (suite-wide) | n/a | 2 skips, both explained | 0 unexplained | **PASS** |
| 7 | needs_attention decisions per file: median / max | median 3, max 109 | median 2, max 39 | median 3, max 109 | median ≤10 | **PASS** (median; see outlier list below) |
| 8 | Corpus size vs. handoff target | — | — | 254 files / 2,850 labeled columns | 200 files / 1,000 columns | **PASS** (exceeds on both axes) |

### Notes on metric 6 (test suite skips)

Full run: `python3 -m pytest tests/ -q -rs` → **12 failed, 866 passed, 2 skipped**, in
1590.75s. The 2 skips are both:
```
SKIPPED tests/regression/test_mvp_acceptance.py:503: v2 has no CLI in this phase (no CLI commands are in scope for M5); see docs/port_regressions.md
SKIPPED tests/regression/test_mvp_acceptance.py:526: v2 has no CLI in this phase (no CLI commands are in scope for M5); see docs/port_regressions.md
```
Both carry an explicit reason and point at `docs/port_regressions.md` — **0 unexplained
skips**, target met. (`tests/corpus/test_interact_corpus.py` and
`tests/corpus/test_pipeline_corpus.py` also define `skipif(not RAW.is_dir())` guards, but
`corpus/raw/` exists in this tree, so those guards did not fire — no skip recorded for
them in the run.)

The same run surfaced **12 pre-existing test failures** in
`tests/regression/test_mvp_acceptance.py` (semantic-type mismatches: `total_charges` /
`engine_cc`-style unit-suffixed numeric columns classified `CATEGORICAL_HIGH` instead of
`NUMERIC_FEATURE`, plus related cases). These are outside the 8 requested targets (they
are pipeline-behavior regressions, not corpus-accuracy or skip-count metrics), so they are
reported here for visibility only and are not scored against a PASS/FAIL target in this
report; no fix was attempted per the measurement-only scope of this task.

### Notes on metric 7 (needs_attention load)

26 of 254 files (10.2%) exceed the median-derived "≤10 needs_attention decisions" rule of
thumb for a single file, even though the corpus-wide median (3) comfortably passes the
stated target. Highest-count files (needs_attention decision count, descending):

| file_id | needs_attention count |
|---|---|
| tt_2018_2018_11_20_thanksgiving_meals | 109 |
| fte_bob_ross_elements_by_episode | 67 |
| fte_food_world_cup_food_world_cup_data | 66 |
| fte_bachelorette_bachelorette | 39 |
| fte_masculinity_survey_raw_responses | 39 |
| fte_police_killings_police_killings | 25 |
| tt_2019_2019_10_15_big_epa_cars | 25 |
| fte_avengers_avengers | 19 |
| fte_drug_use_by_age_drug_use_by_age | 19 |
| tt_2019_2019_02_26_full_trains | 19 |
| r_count_affairs | 18 |
| fl_doublequote | 17 |
| fte_star_wars_survey_starwars | 17 |
| tt_2018_2018_05_14_week7_starwars | 17 |
| tt_2018_2018_05_29_week9_comic_characters | 15 |
| tt_2020_2020_05_12_volcano | 15 |
| fte_college_majors_recent_grads | 14 |
| fte_college_majors_grad_students | 13 |
| tt_2020_2020_01_14_passwords | 13 |
| r_cardata_florida | 12 |
| r_count_rwm1984 | 12 |
| tt_2019_2019_05_14_nobel_winners | 12 |
| tt_2019_2019_10_29_nyc_squirrels | 12 |
| r_daag_ant111b | 11 |
| tt_2019_2019_10_08_ipf_lifts | 11 |
| tt_2020_2020_03_03_season_goals | 11 |

These are genuinely wide (many-column) files, so a higher raw count is expected; the
median/max reporting is included as requested rather than treated as a hidden fail.

## Root-cause breakdown for the two FAILed metrics

### Dialect accuracy (21/254 files wrong)

| Mismatch kind | Count | Example file_ids |
|---|---|---|
| encoding (iso-8859-1/utf-8-sig mistaken for utf-8) | 10 | `fl_accents`, `fl_bom`, `fl_latin1`, `fte_avengers_avengers`, `fte_biopics_biopics`, `fte_food_world_cup_food_world_cup_data`, `fte_police_killings_police_killings`, `fte_star_wars_survey_starwars`, `tt_2018_2018_04_23_week4_australian_salary`, `tt_2018_2018_05_14_week7_starwars` |
| delimiter (`;`/`\t` mistaken for `,`) | 7 | `fl_delimiter_custom`, `fl_semicolon`, `fl_tsv`, + 4 more |
| header_row_index (preamble/headerless mistaken for row 0) | 4 | `fl_skip_rows`, `uci_adult`, + 2 more |

The library's own `detect_encoding`/`detect_dialect`/`detect_header` (used here, not v1's
hardcoded `pandas.read_csv()` defaults per the GUIDE's v1.0.1 baseline note) do attempt
real detection, so this is a genuine v2 accuracy gap, not the documented v1.0.1-vs-pandas-
defaults baseline problem. 10 encoding misses dominate: files that are legitimately
iso-8859-1 or utf-8-sig are being detected as `utf-8`.

**Finding written to `docs/hardening_findings.md`**: see below.

### Column type accuracy (253/2850 columns wrong)

The single largest confusion by far is **`ID` mislabeled as `numeric`** (182 of 253
wrong columns, ~72% of all column-type misses). These are columns the ground truth
labels as `ID` (e.g. sequential row indices, numeric foreign keys, codes) that
`infer_column_type()` in `src/preflight/infer/types.py` scores as a clean integer/float
and keeps as numeric because its `ID` heuristic (`is_id_name` name-pattern match, or
`is_sequential_int` + `uniqueness_ratio == 1.0`) doesn't fire — e.g. a numeric ID column
whose name doesn't match any of `ID_NAME_PATTERNS` and whose values, while unique, aren't
perfectly sorted/sequential in the sampled column.

Second-largest confusion: **`categorical` mislabeled as `text`** (45 of 253), concentrated
in columns of short proper-noun-like strings (names, titles) that trip the `avg_len`/
`avg_words` text-vs-categorical heuristic (`is_text_domain` term list includes `"name"`
and `"title"`, which pushes some genuinely-categorical columns like `name`/`meet_name`/
`playlist_name` over to text).

**Finding written to `docs/hardening_findings.md`**: see below.

### Wrong-type-with-no-flag (held_out only fails; 24/2850 columns combined)

Of the 24 wrong-with-no-flag columns, the pattern mirrors the confusion matrix above:
mostly `ID`-as-`numeric` and `categorical`-as-`text` misses where `needs_attention` was
not set on the column's `type_inference` decision (the column scored a *clean, confident*
numeric/text match — by `infer_column_type()`'s own logic there was nothing to flag,
because the type it chose really did parse cleanly; the problem is the chosen type itself
being the wrong semantic category, not an ambiguous parse). This is the same root cause as
the two column-type findings below, not a separate flagging bug — captured as a shared
finding.

## Findings written

Three FAILed numeric targets map to findings appended to `docs/hardening_findings.md`
(append-only; the 2 pre-existing entries — csv field-size-limit, and the
`Categorical`-dtype crash in `normalize.py` — were preserved unchanged):

1. **Dialect — iso-8859-1/utf-8-sig encoding detection misses real-world Latin-1/BOM files.**
2. **Column type — numeric-looking `ID` columns are not recognized as `ID` without a name-pattern hint, and short proper-noun categorical columns tip into `text`.** (This single finding also covers the wrong-type-with-no-flag miss, since it's the same misclassification, not a missing flag.)

No fixes were applied to the library. Both findings are measurement + root-cause only, per
the task's constraint that tuning may only happen later with user sign-off on the tuning
set.
