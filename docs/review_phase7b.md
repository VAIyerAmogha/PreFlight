> **Held-out set:** every held-out number in this document is from a set that was **seen during review, not independent** (see docs/v2_hardening_report.md).

# Phase 7b review

| Stage | Result |
|---|---|
| 1 Speed | plan() on large files 4–6x faster (nbaallelo 72.8 → 16.0 s, ipf_lifts 16.1 → 2.5 s). Types are inferred on a seeded 10,000-row sample, then checked on the full column. 2841/2841 corpus column types unchanged. |
| 2 Fit hygiene | `apply()` returns an unfitted, cloneable Pipeline holding every step that learns from data. `prepare()` also returns a copy fitted on all rows. See docs/fit_hygiene.md. |
| 3 Ingest/infer bugs | Ambiguous single-byte encodings are flagged, with their rivals. Unit detection is generic (no unit names). "+" phone codes are treated as IDs. |
| 4 Defaults/output | Default mode `auto`; output goes to `./preflight_out`; HTML output removed; swallowed exceptions 15 → 10, all justified. |
| 5 Lint/types/coverage | ruff 863 → 0, mypy strict 49 → 0, coverage 90.3% (infer 97.4%, ingest 98.7%). A Python 3.9 import break was fixed. |
| 6 Corpus tests | 2393 per-file tests, `corpus` marker, xdist + timeout, run in 405 s. Fixed an encoding-question bug the run found (avengers.csv). |
| 7 Error analysis | Tuning set only; see docs/v2_hardening_report.md. |
| 8 Tuning | Skipped (owner: finish within token budget). |
| 9 Final measurement | Held-out scored once. Dialect and column-type accuracy miss 95%; held-out wrong-no-flag is 1.38%, above the 1% goal. |

Open items:
- `test_report_has_all_export_methods`: the owner approved the edit, and to_html/save_html were taken out of its list. It passes.
- `test_phone_number_like_column_not_coerced` fails because of its test input; this is documented.
- Cleanup: `corpus/raw/preflight_plan/` and a root `preflight_out/` (leaked by a corpus test, now fixed) were deleted. The loader now closes the file before a fallback re-read. All docs were moved into docs/.
- 3 pre-existing `noqa` remain in src.
- Python 3.9 is untested locally.
