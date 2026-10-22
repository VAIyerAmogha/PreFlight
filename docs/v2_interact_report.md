# PreFlight v2.0 Interaction Layer Report (Phase 5 / M4)

## 1. Executive Summary

This report documents the interaction layer: batched questions, three modes (`ask`, `defaults`, `strict`), four channels (terminal, notebook, review file, callback), and answer persistence keyed by `Decision.fingerprint`.

**Not exercised in a real notebook:** the notebook channel was tested only with mocked widgets and headless construction of real `ipywidgets` objects. The widget UI itself has NOT been run in a real Jupyter notebook.

**Privacy:** `review.json` and the Plan PDF contain sample values from the user's data (question previews). Treat them like the data.

---

## 2. What Was Built

| Path | Purpose |
| :--- | :--- |
| `interact/question.py` | `Question` (2 to 4 options, default always among them, preview for column questions), option builders, batched `ReviewSession`, `validate_answers` |
| `interact/preview.py` | Real before/after sample values per option; conversion rules mirror `TypingStage.apply` |
| `interact/triggers.py` | 8-category trigger registry with plug-in points |
| `interact/modes.py` | `ask` / `defaults` / `strict`, mode auto-selection, `StrictResolutionError` |
| `interact/channels/` | `terminal`, `notebook`, `review_file`, `callback` |
| `interact/store.py` | Versioned `answers.json` in `out_dir` |
| `interact/resolve.py` | `resolve()`, exported from `preflight` |
| `plan.py` | `dry_run()` / `apply()` gain `mode`, `channel`; `write_plan_artifacts`; reading apply now honours the action in force |
| `decisions.py` | `Decision.history` (timestamped status log), `mark_automatic` |
| `report/templates.py`, `plan_pdf.py` | Question phrasing; "Your answers" section |
| `pyproject.toml` | Extras `[interactive]`, `[notebook]` |

---

## 3. Behaviour

**Modes.** An explicit mode always wins. With no mode in a non-interactive context (no TTY, no notebook kernel), `resolve()` selects `strict`. `defaults` sets each open Decision's action to its safe default, keeps status `automatic`, and logs it in `history`. `strict` raises one error listing every unresolved item (id, column, question text).

**API.** `resolve()` writes `plan.json` before rendering the Plan PDF. `dry_run()` and `apply()` with `mode=None` keep Phase 4 behaviour (no resolution). `review_file` and `notebook` never block: they write the file or show the form and raise `ReviewPending`; call `resolve()` again (for the file, `resolve(path_to_review.json)`).

**Answers.** An answer must be exactly one of the listed options. A missing or invalid answer raises an error naming the question, on every channel. An empty Enter at the terminal is not an answer; `default` must be typed. The terminal channel without a TTY raises `NonInteractiveError` instead of waiting.

**Store.** `answers.json` is written only in `out_dir`, never beside the source. A matching fingerprint skips the question and marks the Decision `confirmed`. A changed fingerprint is asked again. A stored answer whose option or action no longer matches is ignored. An unreadable or wrong-version store warns, is backed up to `answers.json.bak`, and questions are asked again.

---

## 4. Triggers

| Category | Status | Input |
| :--- | :--- | :--- |
| low_confidence_structure | Implemented | reading Decisions (text format, separator, header) |
| ambiguous_type | Implemented | type_inference / type_abstain |
| lossy_coercion | Implemented | `unresolved_trap`, leading zeros before numeric cast |
| disguised_missing | Implemented | schema column `missing_markers` (contextual / outlier reasons) |
| high_anomaly_rate | Implemented | raw column values; proposed conversion fails for more than 10% (label only, does not gate asking) |
| ambiguous_structure_or_task | Implemented | ambiguous_task, ragged header |
| **destructive_action** | **Empty plug-in point** | no such Decision exists yet |
| **suspected_leakage** | **Empty plug-in point** | no such Decision exists yet |

---

## 5. Extras and Licences

Verified from the installed `dist-info` LICENSE files:

| Extra | Package | Licence |
| :--- | :--- | :--- |
| `[interactive]` | `rich>=13` | MIT |
| `[notebook]` | `ipywidgets>=8` | BSD-3-Clause |

Core import and the terminal channel work with neither installed (tested with both blocked).

---

## 6. Corpus Results (254 files, per-file subprocess, 180 s timeout)

| Metric | Value |
| :--- | :---: |
| Files completing the worker | 254 / 254 |
| **Files that hang** | **0** |
| Defaults mode completed | 253 / 254 |
| Strict mode raised | 224 / 254 |
| Files with zero questions | 30 |
| Total questions | 839 |
| Questions per file: median / mean / max | 1.0 / 3.30 / 67 |
| Files above 10 questions | 15 |
| Column questions without sample values | 0 / 836 |
| Banned-jargon hits in question text | 0 |

Question categories: ambiguous_type 813, disguised_missing 19, high_anomaly_rate 3, low_confidence_structure 3, lossy_coercion 1.

Files above 10 questions: `elements-by-episode.csv` 67, `thanksgiving_meals.csv` 62, `food-world-cup-data.csv` 41, `raw-responses.csv` 38, `big_epa_cars.csv` 24, `bachelorette.csv` 21, `police_killings.csv` 19, `COUNT_affairs.csv` 18, `full_trains.csv` 14, `week9_comic_characters.csv` 14, `StarWars.csv` 12, `nobel_winners.csv` 12, `nyc_squirrels.csv` 12, `week7_starwars.csv` 12, `ipf_lifts.csv` 11.

---

## 7. Findings (need your attention)

1. **`latin1.csv` fails under defaults mode.** The Phase 4 encoding Decision has `default = use_encoding_utf-8` while the detected format is iso-8859-1. Applying the default makes the file unreadable (clear plain-language `ValueError`). Not patched; needs your decision on the default.
2. **Defaults mode keeps nearly every flagged column as plain text.** 813 of 839 questions are `ambiguous_type`. This comes from upstream inference thresholds, which were not tuned.
3. **Upstream inconsistency:** the encoding Decision for `latin1.csv` is flagged `low_confidence` while its `confidence` is 1.0. Question text avoids quoting the number.
4. `tests/unit/test_assemble.py` and `tests/regression/` import modules that do not exist yet (`CleanerTransformer`, `build_pipeline_two_phase`). They are untracked later-phase files and were excluded from runs.
5. A stray, gitignored `corpus/raw/preflight_plan/` directory exists from an earlier default `out_dir`. Left in place.
6. Header answers other than the detected row use a simple row-index reading; no corpus file raised a header question.
