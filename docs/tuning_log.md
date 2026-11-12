> **Held-out set:** every held-out number in this document is from a set that was **seen during review, not independent** (see docs/v2_hardening_report.md).

# Tuning log (Phase 7c, Stage B)

Scope: tuning set only (153 files, 1,765 labelled columns, `corpus/labels/split.json`).
Held-out per-file results were not used to choose or check any change. Numbers come from
`scripts/tuning_eval.py`, which runs the same `measure_file` as the corpus accuracy suite
but loads only tuning files.

Starting point (matches the Phase 7b tuning numbers): correct 1616 (91.56%), wrong but
flagged 141, wrong with no flag 8 (0.45%), dialect 140/153, errors 0.

## The 8 unflagged wrong-type cases, grouped by cause

| Cause | Cases | Labelled -> detected |
|---|---|---|
| Name-word rule: the column name contains a prose word ("name", "title") and values average 3 or more words, so the column is read as free text | 5 | categorical -> text (person names, place names, event names, film titles) |
| Medium-length phrases (about 36 characters, 4.8 words) read as categorical | 1 | text -> categorical |
| Short repeating code in an ID-named column (2% unique) | 1 | ID -> categorical |
| Near-unique integer measurement labelled ID | 1 | ID -> numeric |

## Changes

| # | Change | Tuning before | Tuning after | Result |
|---|---|---|---|---|
| 1 | `TEXT_NAME_HINT_MIN_WORDS` (new named setting in `infer/config.py`, used in `infer/types.py`): the name-word rule now needs 5 words on average instead of 3 | correct 1616, wrong+flag 141, wrong no flag 8 (0.45%), dialect 140/153 | correct 1624 (92.01%), wrong+flag 138, wrong no flag 3 (0.17%), dialect 140/153 | **Kept.** Fixed 8 columns, broke none, no column changed flag state. On the tuning set, every column the rule decided at 3–4 words was labelled categorical, and no labelled-text column depended on it. |

Considered but not tried, because the tuning data shows they would raise another bucket or
only fit single columns:

- Medium-length phrases: the one text column (36.0 characters, 4.8 words) sits next to
  categorical columns of the same shape (36.1 characters, 4.5 words; 29.6, 4.7). Any length or
  word cut that separates them fits these columns, not a general pattern.
- Short codes in ID-named columns: among tuning columns whose name looks like an ID and
  whose values are under 85% unique, 6 in one file are labelled categorical (`team_id`
  style) and 1 is labelled ID. Making such columns IDs would add more misses than it removes.
- Near-unique integer labelled ID: any rule that reads near-unique integers as IDs would
  also catch real measurements. This looks like a labelling choice.

Fast suite after change 1: 346 passed, 2 skipped, 1 xfailed.

Disclosure: while checking the record format of the previous run, one held-out record
(AER_Affairs.csv, one column: rownames labelled ID, detected numeric, flagged) was printed by
mistake. It was not used to choose or check any change.
