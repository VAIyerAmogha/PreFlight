# PreFlight: Phase 7b report

Date: 30 October 2026. Nothing has been committed; every change is waiting for your review.
This file lives in `docs/` with every other doc and report.

## The short version

PreFlight is faster, safer and better tested than before. It no longer crashes or stalls on any of
the 254 test files. But it still falls short of three of the accuracy goals, so it is **not yet ready
for Phase 8**. 6 of the 10 readiness checks pass.

## How well it does on the 254 test files

The files were split before any work started: 153 "practice" files and 101 "exam" files that were
kept unseen until the very end and then checked once.

| Goal | Target | Practice files | Exam files | All files | Old version (v1) |
|---|---|---|---|---|---|
| Files that run without crashing | all | all 153 | all 101 | all 254 | 244 of 254 |
| Files read with the right settings (separator, text format, header row) | 95% | 91.5% | 92.1% | 91.7% | 91.7% |
| Columns given the right kind (number, date, category, ID, text) | 95% | 91.6% | 89.7% | 90.8% | 80.8% |
| Columns given the wrong kind *without* warning the user | 1% or less | 0.45% | 1.38% | 0.81% | 9.4% |
| Every choice comes with its reasons and a confidence score | all | yes | yes | yes | – |
| Typical number of questions asked per file | 10 or fewer | 3 | 2 | 3 | – |
| Files that stall | none | none | none | none | – |

**What this means:** compared with the old version, PreFlight now labels about 10 more columns in every
100 correctly, and silent mistakes have dropped from about 1 in 11 columns to fewer than 1 in 100.
It is still below the 95% goal for reading files and labelling columns. On the exam files, silent
mistakes are slightly above the 1% limit.

## Where the remaining mistakes come from (practice files only)

- **Row-number columns (112 of 149 column mistakes).** Many files have a first column that just counts
  rows (1, 2, 3 …). The answer key calls these IDs; PreFlight calls them numbers. It already warns the
  user about all but one of them.
- **Short labels read as free text (22 mistakes).** Columns such as answer choices are sometimes treated
  as free text instead of categories.
- **Reading the file (13 files).** The text format is wrong in 7, the separator in 4, and the header row in 2.

No adjustments were made to fix these. You asked to finish within a set budget, so the "tuning" step
was skipped. All the numbers above come from rules that were never adjusted to the test files.

## What was done in this phase

1. **Speed.** Big files are 4 to 6 times faster to check: one went from 73 seconds to 16. PreFlight now
   decides each column's kind from a fixed random sample of up to 10,000 rows, then double-checks the
   whole column.
2. **Fair results for model building.** Any step that learns from the data, like filling gaps with an
   average, now happens inside the model-building step. It no longer happens on the whole table first,
   so test scores are not inflated by peeking at test rows.
3. **Reading fixes.** PreFlight now warns when it cannot tell which text format a file uses. It
   recognises any unit written after a number (not just a fixed list) and treats phone-style codes
   like "+447…" as IDs, not numbers.
4. **Simpler defaults.** PreFlight asks questions when someone is there to answer them and stops with a
   clear message when no one is. Output now goes to a `preflight_out` folder where you run it, never
   next to your data file. The old web-page output was removed, as requested.
5. **Code quality.** All style and type checks now pass with no warnings hidden. Tests cover about 90% of
   the code, and more than 97% of the file-reading and column-labelling parts. Hidden error-swallowing
   spots went from 15 to 10, and each remaining one has a written reason.
6. **Test setup.** The full set of 254-file checks now runs in under 7 minutes, where before it was cut off
   after 50 minutes. Each file is its own test, so a failure names the file. This found one new bug: a
   question about a file's text format could not be shown, and some regional text formats were wrongly
   called "standard". That is now fixed.

## Follow-up cleanup (done after your review)

- **Web-page test fixed.** The old test no longer expects the removed web-page output, as you approved.
- **Leftover folders deleted.** The old stray `corpus/raw/preflight_plan/` folder is gone. So is a
  `preflight_out` folder in the project root, which one test kept creating. That test now writes to a
  temporary folder, and `preflight_out/` is listed in `.gitignore`.
- **Small bug fixed.** When reading a big file in pieces failed partway and PreFlight switched to its
  slower reader, it left the file open. It now closes it first.
- **Docs in one place.** Every doc and report now lives in `docs/`, including this report, the plan,
  the Phase 6 notes, and the review notes from both phases (`docs/review_notes/`). `docs/README.md`
  lists what each file is. Only the files tools expect in the root stay there (README, CHANGELOG,
  LICENSE, CLAUDE.md).

## Still open

1. **The phone-number test still fails.** It feeds 200 random 10-digit numbers. Once saved as a file,
   these look exactly like ordinary large numbers. Making it pass would mean changing how columns are
   labelled after the exam files were scored, which this phase's rules forbid. It is written up in
   `docs/port_regressions.md`.
2. **Oldest supported Python version.** The project says it supports Python 3.9, but only 3.12 is installed
   here, so 3.9 could not be tested. One problem that would have broken 3.9 was found and fixed. I
   suggest raising the minimum to 3.10 or adding a 3.9 test run.
3. **Next step for accuracy.** Improving the three failing scores needs a round of adjustments. The exam
   files have now been looked at once, so a fair re-check would need fresh exam files or your agreement
   to reuse these ones.
4. **Two old tests use an outdated test-setup style** that a future pytest version will drop. They still
   work today.

## Where to find the details

- Full numbers: `docs/v2_hardening_report.md`
- Readiness checklist: `docs/phase8_readiness.md`
- Stage-by-stage notes: `docs/review_phase7b.md` and `docs/review_notes/phase7b/STATE_7b.md`
- Behaviour changes and test changes: `docs/port_regressions.md`
