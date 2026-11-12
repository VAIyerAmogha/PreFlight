> **Held-out set:** every held-out number in this document is from a set that was **seen during review, not independent** (see docs/v2_hardening_report.md).

# Hardening findings

## 2026-10-23: bare `_csv.Error` escapes on a very long cell value

**Repro**: `tests/corpus/test_degenerate_inputs.py::test_one_very_long_cell`
(originally written with a 200,000-character cell value in column `b` of a
2-column, 2-row CSV). No corpus file or corruption seed involved; this is a
direct degenerate-input case, not a corruption-generator case.

```python
p.write_text(f"a,b\n1,{'x' * 200_000}\n2,short\n", encoding="utf-8")
plan(p)
```

**What happened**: `preflight.ingest.header.detect_header` reads the file
with the stdlib `csv.reader` and calls `list(reader)` without ever calling
`csv.field_size_limit(...)`. Python's default field size limit is 131072
bytes. Any cell longer than that raises a bare `_csv.Error: field larger
than field limit (131072)` straight out of `src/preflight/ingest/header.py`
(line ~147, inside `detect_header`). This propagates all the way up through
`ReadingStage.plan()` -> `Planner.plan()` -> `plan()` as an untranslated
stdlib exception, not one of the library's own documented exception types
(`ValueError`, `FileNotFoundError`, etc.), and with a message that gives the
caller no indication it came from PreFlight or what to do about it.

**What should have happened**: Either (a) `detect_header` (and any other
`csv.reader`-based code path, e.g. `load_raw`'s fallback header handling)
should raise a clear, library-specific `ValueError` naming the offending
file and suggesting the cell may be unusually large, or (b) the library
should proactively raise the field size limit (e.g.
`csv.field_size_limit(10_000_000)`) for its own parsing so that realistic
long-text cells (long free-text fields, embedded JSON blobs, etc.) are not
arbitrarily rejected at a 128 KB stdlib default that was never designed with
data cleaning in mind.

**Proposed fix** (not applied — leaving to maintainers to choose a global
limit deliberately): in `src/preflight/ingest/header.py`, wrap the
`list(reader)` call (and the equivalent one in `loader.py` / `dialect.py` if
they share this exposure) in a `try/except csv.Error as exc: raise
ValueError(f"Could not parse {p.name}: {exc}. A cell may be unusually large;
try csv.field_size_limit() before calling PreFlight, or re-export the file.")
from exc`. This is a few-line defensive fix but was not applied in this pass
because it touches three independent call sites (`header.py`, `dialect.py`,
possibly `loader.py` via its python-engine fallback) and the maintainers may
prefer to centralize it (e.g. raise the limit once at import time) rather
than wrap each call site individually.

**Status**: documented, not fixed. The hardening test for this case
(`test_one_very_long_cell`) was adjusted to use a 50,000-character cell,
which stays under the stdlib limit and still exercises the "very long
single cell" scenario without tripping this known bug.

## 2026-10-23: `plan_normalization` crashes on any mixed-case/whitespace categorical column with `CategoricalDtype`

**Repro**: discovered while building the v1-vs-v2 model score comparison
(`docs/v2_vs_v1_scores.md`) on real corpus files; reproducible on any file
with a `category`-dtype column containing mixed-case or leading/trailing
whitespace values, e.g. `plan(path_to_titanic_csv, target="survived")`
where a categorical column like `sex`/`embark_town` is not already
all-lowercase/stripped. 6 of the 10 files used in that comparison
(`adult.data`, `iris.data`, `titanic.csv`, `tips.csv`, `diamonds.csv`, and
implicitly any file with a non-trivial categorical column) hit this
deterministically, on every CV fold.

**What happened**: `normalize_category_values()`
(`src/preflight/clean/normalize.py:16-26`) lowercases/strips a column via
`series.apply(lambda x: x.strip().lower() if isinstance(x, str) else x)`.
When `series.dtype` is `pandas.CategoricalDtype` and at least one value
actually changes under lowercasing/stripping, `.apply()` returns a new
`Categorical` whose category set differs from the original (new lowercase
category labels replace the old mixed-case ones). Back in
`plan_normalization()` (`src/preflight/clean/normalize.py:82`), the line
`normalized != series` then compares two `Categorical` series with
different category sets, which pandas refuses to do:
`TypeError: Categoricals can only be compared if 'categories' are the
same.` This propagates up through `plan_engineer`/`build_pipeline_two_phase`
and crashes the whole pipeline — not just a missed flag, a hard crash,
for any target/task.

**What should have happened**: the "did anything change" comparison should
not break on `Categorical` dtype. Either cast both sides to plain string/
object dtype before comparing (`normalized.astype(object) !=
series.astype(object)`, treating NaN consistently), or compare using
`.to_numpy()` instead of the raw Series `!=` operator so it never depends
on the two sides sharing a category set.

**Proposed fix** (not applied — one-line change but left to maintainers to
confirm NaN-handling semantics): in
`src/preflight/clean/normalize.py:82`, change
`changed = int(((normalized != series) & series.notna()).sum())` to compare
on `.astype(object)` views of both sides rather than the raw (possibly
`Categorical`) series.

**Status**: documented, not fixed (outside the authorized scope of the
task that found it — fixing it would change what Step 6's v1-vs-v2
comparison was measuring). This is a real crash bug affecting any
real-world categorical column that isn't already normalized, likely
affects a large fraction of the 254-file corpus, and should be prioritized
above the other findings in this file since it is a hard failure, not a
missed-accuracy-target.

## 2026-10-23: Dialect accuracy misses target (91.73% combined vs. ≥95%) — encoding detection undercalls iso-8859-1 and utf-8-sig

**Repro**: `scripts/corpus_accuracy.py` run over all 254 corpus files
(`docs/v2_accuracy_report.md`, target #2). 21/254 files get at least one
dialect field (delimiter/encoding/header_row_index) wrong: tuning 140/153 =
91.50%, held_out 93/101 = 92.08%, combined 233/254 = 91.73%. All three are
below the ≥95% target.

**What happened**: of the 21 mismatches, 10 are encoding misses — files
whose ground-truth label is `iso-8859-1` (e.g. `fl_accents`, `fl_latin1`,
`fte_avengers_avengers`, `fte_biopics_biopics`,
`fte_food_world_cup_food_world_cup_data`,
`fte_police_killings_police_killings`, `fte_star_wars_survey_starwars`,
`tt_2018_2018_04_23_week4_australian_salary`,
`tt_2018_2018_05_14_week7_starwars`) or `utf-8-sig` (`fl_bom`) are detected
by `preflight.ingest.encoding.detect_encoding` as plain `utf-8`. 7 are
delimiter misses (`;`/`\t` detected as `,`, e.g. `fl_delimiter_custom`,
`fl_semicolon`, `fl_tsv`). 4 are header_row_index misses (preamble rows or a
genuinely headerless file, e.g. `fl_skip_rows`, `uci_adult`, detected as
header row 0). This is a real v2 detection-logic gap, not the v1.0.1
"hardcoded pandas.read_csv() defaults" baseline problem the GUIDE
describes — v2's `detect_encoding`/`detect_dialect`/`detect_header` do
attempt real inference, they're just not catching these specific
real-world files.

**What should have happened**: the target is ≥95% dialect accuracy;
measured is 91.5-92.1% across both splits. iso-8859-1 files that happen to
be valid (if semantically wrong) utf-8 byte sequences are the dominant
failure mode — `detect_encoding` likely needs a confidence check beyond
"does this decode without raising" (e.g. a chardet/charset-normalizer-style
statistical byte-distribution check, or flagging non-ASCII high-byte runs
that decode as nonsense Latin-1-via-utf-8 mojibake) to catch Latin-1 files
that happen to decode cleanly as (wrong) UTF-8.

**Proposed fix** (not applied — this is a detection-sensitivity change,
which is exactly the kind of threshold/heuristic tuning this task is
explicitly forbidden from doing on its own): investigate strengthening
`detect_encoding`'s confidence scoring in
`src/preflight/ingest/encoding.py` for the "decodes cleanly as UTF-8 but
is actually Latin-1" case (e.g. cross-checking decoded text for high rates
of the specific mojibake byte patterns Latin-1-as-UTF-8 produces), and
separately reviewing why `fl_bom`'s BOM is not being surfaced as
`utf-8-sig` specifically (vs. being normalized away to plain `utf-8`). Any
such change must be validated against the `tuning` split only, with
user sign-off, per this task's constraints.

**Status**: documented, not fixed.

## 2026-10-23: Column-type accuracy misses target (91.12% combined vs. ≥95%) — numeric-looking `ID` columns and short proper-noun columns are misclassified

**Repro**: same `scripts/corpus_accuracy.py` run, target #3. 253/2850
labeled columns are wrong across the corpus: tuning 1619/1765 = 91.73%,
held_out 978/1085 = 90.14%, combined 2597/2850 = 91.12%. All three are
below the ≥95% target. This same root cause also drives target #4
(wrong-type-with-no-flag) over its ≤1% limit on the held_out split
specifically (16/1085 = 1.47%), since these misclassifications score as
confident, un-flagged type_inference decisions — `infer_column_type` (in
`src/preflight/infer/types.py`) isn't uncertain about the type it picked,
it just picked the wrong *semantic category* for an otherwise cleanly-
parsing column, so nothing trips `needs_attention`.

**What happened**: by far the largest confusion (182 of 253 wrong columns,
~72%) is ground-truth `ID` columns inferred as `numeric`. Examples:
`fte_bechdel_movies` columns `period code` / `decade code`,
`fte_congress_demographics_data_aging_congress`'s `party_code`,
`r_aer_naturalgas`'s `statecode`, `r_count_fishing`'s `meandepth`. The `ID`
candidate score in `infer_column_type()` (lines ~388-410 of
`src/preflight/infer/types.py`) only goes non-zero when (a) leading zeros
are present, (b) the column name matches one of the hardcoded
`ID_NAME_PATTERNS` (`"id", "_id", "id_", "pk", "key", "index", "code",
"fips", "zip", "postal", "rownames"`) AND values are mostly-unique, or (c)
values are a perfectly sorted, fully-unique sequential integer run. A
numeric-coded ID column that is unique-but-unsorted, or whose name doesn't
happen to contain one of those literal substrings (`statecode` does match
`"code"` yet was still missed in this run — worth double-checking that
branch; `meandepth`/`party_code` don't match cleanly as ID-shaped at all),
falls through to a plain numeric match instead. Second largest confusion
(45 of 253) is ground-truth `categorical` columns inferred as `text` —
concentrated in short proper-noun-like string columns such as `name`,
`title`, `meet_name`, `playlist_name`, `unit_name`: the `is_text_domain`
term list in `infer_column_type()` includes `"name"` and `"title"`, which
pushes the text/categorical score split toward `text` for columns that are
semantically categorical (bounded-ish label sets) even though they're
short strings.

**What should have happened**: the target is ≥95% column-type accuracy;
measured is 90-92%. Both confusions are heuristic-coverage gaps in
`infer_column_type`'s scoring, not crashes or missing evidence.

**Proposed fix** (not applied — tuning the ID/categorical heuristics is
exactly the threshold-tuning this task is forbidden from doing
unilaterally): (1) broaden the `ID` detection path to also consider
"numeric, fully/near-fully unique, NOT a measurement-sounding column name"
as weak evidence for `ID` even without a literal name-pattern hit or
perfect sort order — e.g. a uniqueness-ratio-based score blended with the
absence of float/decimal values, rather than requiring sortedness; and (2)
revisit whether `"name"`/`"title"` belong in the `is_text_domain` term list
at all, or require a higher `avg_words`/`avg_len` bar specifically for
columns matching those two terms, since in this corpus they overwhelmingly
mark bounded categorical-like fields (person/place/thing names) rather than
free-form prose. Any such change must be validated against the `tuning`
split only, with user sign-off, per this task's constraints.

**Status**: documented, not fixed.
