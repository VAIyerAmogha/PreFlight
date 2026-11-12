> **SUPERSEDED (Phase 7b, 2026-10-30):** the numbers here are replaced by docs/v2_hardening_report.md.

> **Held-out set:** every held-out number in this document is from a set that was **seen during review, not independent** (see docs/v2_hardening_report.md).

# Phase 7 review (review and measurement only; nothing fixed, tuned or edited)

Raw output lives in `.review/` (git-ignored). Review date 2026-10-28, branch v2-dev @ 182903e.
Tier: STANDARD. Areas 1 and 2 reviewed and area 3 only partly (see section 7). Reviewed by the lead directly; no subagents were used.

## 1. Targets (computed from `.review/shard_*/results.json` by `scripts/merge_shards.py`)

Run under load (8 jobs at once). The 7 files that timed out at 120 s all pass solo (section 2), so the column below shows the load-affected run, and the "excluding timeouts" row shows the same numbers with those 7 files removed.

| Target | Goal | Tuning | Held-out | Combined |
|---|---|---|---|---|
| Files without unhandled exceptions | 100% | FAIL 147/153 | FAIL 100/101 | FAIL 247/254 (all 7 misses are timeouts, not exceptions) |
| Dialect accuracy | ≥95% | FAIL 87.58% | FAIL 91.09% | FAIL 88.98% |
| Column type accuracy | ≥95% | FAIL 82.15% | FAIL 88.29% | FAIL 84.49% |
| Wrong type, no flag | ≤1% | PASS 0.40% | FAIL 1.29% | PASS 0.74% |
| Decisions with evidence and confidence | 100% | PASS | PASS | PASS 7329/7329 |
| needs_attention median | ≤10 | PASS 3 (max 109) | PASS 2 (max 39) | PASS 3; 24 files above 10 |
| Timeouts (120 s) | 0 | FAIL 6 | FAIL 1 | FAIL 7 |

Excluding the 7 timed-out files: column type 90.83% (2408/2651) and dialect 226/247 = 91.5% (both still FAIL against 95%).
Solo reruns of the 7 files: `docs/review_notes/phase7/timeouts_solo.md`.

## 2. Run summary

- Fast suite (unit, property, regression; serial; `.review/fast_suite.txt`): 264 tests, 4 failed, 2 skipped, 70.5 s.
- Corpus tests, sharded by test file (`.review/shard_*/pytest.txt`): 611 tests recorded, 1 failed.
  - `shard_0` (`test_pipeline_corpus`): passed, 640 s.
  - `shard_1` (`test_corruption_handling` + `test_label_integrity`): passed, 65 s.
  - `shard_3` (`test_ingest_corpus`, `test_interact_corpus`, `test_degenerate_inputs`): 1 failed, 615 s.
  - `shard_2` (`test_plan_corpus` + `test_infer_corpus`): killed at the 3000 s cap with no result.
  - `test_infer_corpus` alone: killed at 900 s with no result. `test_plan_corpus` was not rerun alone.
- Accuracy harness (`scripts/corpus_accuracy.py` wrapped by `scripts/accuracy_shard.py`, 4 shards): 499 to 578 s each, 7 TIMEOUTs at 120 s.
- Solo reruns of the 7 timed-out files, 300 s cap, nothing else running: all 7 OK, between 15.3 s (`tt_2019_2019_10_08_ipf_lifts`) and 80.4 s (`tt_2020_2020_02_11_hotels`). The `test_interact_corpus` failure ("files hung" at its fixed 180 s limit, for `big_epa_cars.csv`, `hotels.csv`, `nbaallelo.csv`) passes alone: 1 passed in 287 s (`.review/interact_solo.txt`).
- Skipped tests: 2 CLI tests (v2 has no CLI) in the fast suite; 5 conditional skips in the test files (corpus/raw absent, rich or ipywidgets absent).
- Lint: ruff 848 errors (`.review/ruff.txt`); mypy 49 errors in 14 files (`.review/mypy.txt`).
- Coverage: combined 60% (`.review/coverage.txt`), but the killed shard added nothing, so this is understated.
- No speedup is claimed: a serial corpus run was not measured.

## 3. Findings

### HIGH
- **H1. `prepare()` fits cleaner and engineer on the whole table; there is no split.** Evidence: `prepare.py:36-55` passes all rows (`X`, `y`) to `build_pipeline_two_phase`, which calls `cleaner.fit` and `engineer.fit` at `assemble.py:313,320`. The handoff requires no fit on full data before the split. The returned pipeline has seen every row. Fix: take a split or accept pre-split data; at minimum document it. Blocks release: yes if "no fit before split" is a hard requirement.
- **H2. `plan()` is slow on large files and does not sample.** Evidence: cProfile on `tt_2019_2019_10_08_ipf_lifts` (5 MB): 161M function calls; `infer_column_type` (`infer/types.py:231`) was 44.5 of 50.1 s. It runs `_parse_int`, `_parse_float` and `_parse_datetime` on every non-missing value of every column (about 1.23M calls each), with about 23.5M `re.match` calls. The cost is linear in rows × columns, not quadratic, and there is no row cap. `plan()` also calls `_ensure_prep_plan` 7 times (5.5 s). Solo 15 to 80 s; over 120 s under load. Fix: sample for scoring, use precompiled patterns, cache the prep plan. Blocks: no, but it caused the 7 timeouts and the interact test failure under load.
- **H3. Accuracy targets are missed and the older numbers do not reproduce.** See sections 1 and 4. Dialect and column-type accuracy miss ≥95%; held-out wrong-type-no-flag is 1.29% against ≤1%. Blocks: yes against the stated targets.
- **H4. The "held-out" set was never held out from `infer/` and `ingest/`.** Evidence: `git log` shows the last change to `src/preflight/infer` and `src/preflight/ingest` is 9931377 (2026-10-05, the commit that added `docs/v2_infer_report.md`). `corpus/labels/split.json` was first added in 182903e (2026-10-23). `git diff 9931377 HEAD` on those two directories is empty. The split came 18 days after the last threshold change, and the thresholds were tuned against the full corpus or before the split existed. Git cannot show whether specific held-out files drove any threshold. Fix: say in the docs that held-out numbers are not independent for infer and ingest. Blocks: affects whether the held-out claim can be made.

### MEDIUM
- **M1. Encoding fallback check is vacuous.** `ingest/encoding.py:158-166`: for low-confidence cp932, hp-roman8, mac-latin2, cp850 or cp1250, it tests `raw_bytes.decode("iso-8859-1")`. ISO-8859-1 maps every byte, so the test cannot fail and these are always rewritten to iso-8859-1 (information loss). This only covers the lines read; I did not check whether a Decision records the override. Fix: drop the check or require evidence.
- **M2. Default mode differs by entry point.** `interact/modes.py:66-72` picks `ask` if interactive else `strict`. `strict` raises `StrictResolutionError` for any open question (`modes.py:114-117`, `resolve.py:166-167`). It never waits for input, so it does not hang, but any non-interactive `resolve()` on a messy file errors out. `prepare()` defaults to `mode="defaults"` (`prepare.py:64`). Fix: decide one policy.
- **M3. v1 HTML output is still in the tree.** `report/report.py:242` (`to_html`) and `:330` (`save_html`); `types.py:12` imports `Report`; 14 `old_names` hits. Not wired into `plan()` or `prepare()`, but present.
- **M4. Literal thresholds in inference code.** 37 numeric comparisons not drawn from named config, for example `infer/types.py:394,404,405,434,532,549,608` and `infer/missing.py:153,198,202`. They are not tunable or auditable in one place.
- **M5. Lint and type debt.** ruff 848, mypy 49. mypy is configured as 3.10 while the package declares ≥3.9.
- **M6. Regression 4 is a real v2 regression** (see section 4).
- **M7. Phone-like digit runs become numeric with confidence 1.0 and no question.** Tied to `test_phone_number_like_column_not_coerced`; a silent wrong-type risk (`docs/port_regressions.md` item 2 says the same).

### LOW
- `ingest/loader.py:149` swallows any error in the bare-CR probe (falls back to the C engine silently).
- `report/report.py:382` swallows plot errors (`except Exception: pass`).
- `interact/resolve.py:46-48` returns `None` when the raw reload fails (previews then say so).
- `interact/store.py:64` uses `warnings.warn` for an unreadable answer store. Acceptable; the file is kept as `.bak`.
- `engineer/encoding.py:58` uses `random_state=42`. Seeded, not a finding. The one `unseeded_random` hit is a false positive.

### `writes_next_to_source` (19 hits)
- **Real: 3.** Default output directories created beside the source file: `plan.py:1123` (`preflight_plan/`), `prepare.py:99` (`preflight_out/`), `interact/resolve.py:51`. The source file itself is never written. These fail on read-only data directories.
- **Not real: 16.** `plan.py:138` writes to a caller-supplied path. `report/report_pdf.py:549,554,558` write into `out_d`. `interact/channels/review_file.py:61,62` and `interact/store.py:98,105,106` are atomic temp and backup files in the answer-store directory. `report/report.py:317,362,390,395,401,442` are in-memory or PDF-object `savefig`; `report.py:332` writes the user-supplied path for `save_html`.

### Other area-1 and area-2 checks
- `corpus/labels` were not changed after creation. The one flagged file is `corpus/labels/GUIDE.md` (new section on the split in 182903e). No label JSON changed after b659503.
- 27 `src_changed_after_infer_report` hits are only `clean/` and `engineer/` (ports in ad04702 and 0b51162). None are in `infer/` or `ingest/`.
- Evidence and confidence on every Decision: PASS on the corpus (7329/7329).
- Report PDF cannot be disabled: confirmed, `prepare()` has no parameter for it (`prepare.py:58-67`).
- Results saved before the PDF: confirmed, `report/report_pdf.py:543-575` writes cleaned CSV, `schema.json`, `decisions.json` and `pipeline.joblib` first, then renders, and `ReportGenerationError` lists the saved paths.
- Network calls: 0 hits. Dependencies: 9 runtime dependencies with no pins; one non-permissive licence flag; one undeclared import; one unused declared dependency (see `docs/review_notes/phase7/static_findings.md`, "Dependencies").

## 4. Numbers in earlier docs that did not reproduce

`docs/v2_accuracy_report.md` (there is no `docs/v2_hardening_report.md`):

| Metric | Doc (tuning / held-out / combined) | Now |
|---|---|---|
| Files without exceptions | 153/153, 101/101, 254/254 | 147/153, 100/101, 247/254 (timeouts under load; all 7 pass solo) |
| Dialect | 140/153, 93/101, 233/254 | 134/153, 92/101, 226/254 |
| Column type | 1619/1765, 978/1085, 2597/2850 | 1450/1765, 958/1085, 2408/2850 (2408/2651 excluding timeouts) |
| Wrong, no flag | 8/1765, 16/1085, 24/2850 | 7/1765, 14/1085, 21/2850 |
| Evidence and confidence | 4737/4737, 3052/3052, 7789/7789 | 4334/4334, 2995/2995, 7329/7329 |

The "Now" denominators are lower for the last three rows because the timed-out files contribute no decisions or column results. Without a clean run of those 7 files, there is no like-for-like comparison. Dialect is 7 files lower even after excluding timeouts (226 of 247 against 233 of 254); I did not find out why.

Triage of the failures against `docs/port_regressions.md`:

| Failure | Verdict |
|---|---|
| `test_multiclass_string_target_does_not_crash` (149 != 150) | v1 inconsistent; v2 behaviour documented and deliberate (below-threshold dedupe applied silently). The test expects 150 and is stale against that decision, pending your ruling. |
| `test_phone_number_like_column_not_coerced` | Test input (DataFrame of strings) cannot be expressed through CSV; v2 limitation, but see M7. |
| `test_genuinely_messy_numeric_string_still_coerces_correctly` | Deliberate v2 policy (lossy, asks, safe default keeps text); the test expects v1 auto-conversion. The test is wrong against v2 policy, unless you reverse the policy. |
| `test_simple_unit_suffix_column_parses_as_numeric` ("1248 CC") | v2 regressed: `CC` is not in `UNIT_REGEXES`, so the column is categorical with no question. |
| `test_interact_over_corpus` | Not a code failure: passes solo in 287 s; failed under load at its fixed 180 s limit. |

## 5. The six open decisions: what the code does today

1. **Interactive pause behaviour.** `ask` mode is chosen only when stdin and stdout are terminals or in a notebook; otherwise `strict`, which raises on any open question and never waits (`modes.py:55-72`). `prepare()` defaults to `defaults` mode. The review-file channel raises `ReviewPending` until answers exist (`channels/review_file.py:99`).
2. **DataFrame input path.** `plan()` and `prepare()` take a file path (`prepare(path, target, ...)`). DataFrames only appear inside `Plan` and `ApplyResult`. There is no DataFrame entry point.
3. **HTML output.** Not produced by `plan()` or `prepare()`; the v1 `Report.to_html` and `save_html` remain in `report/report.py` (M3).
4. **Python floor.** `requires-python >=3.9`, ruff `py39`, mypy checks as 3.10; the nox matrix is declared but only 3.12.3 was run here.
5. **PDF failure behaviour.** Results are saved first, then the PDF is rendered; on failure `ReportGenerationError` names the saved paths. The PDF cannot be turned off.
6. **LLM profiler deferral.** `grep` for "llm" in `src/`, `README.md`, `PLAN.md` and `docs/GUIDE.md` returns nothing. The code has no LLM profiler.

## 6. Changes made

- Created: `scripts/shard_corpus.py`, `scripts/accuracy_shard.py`, `scripts/merge_shards.py`, `scripts/static_triage.py`, `scripts/timeouts_solo.py`, `docs/review_phase7.md`, and everything under `.review/`.
- Edited: `.gitignore` (added `.review/`).
- Left alone: `src/`, `tests/`, labels, thresholds, `scripts/corpus_accuracy.py` (wrapped, not edited), `scripts/_corpus_accuracy_raw.json`.
- No `git add`, commit or push.

## 7. Not verified, and why

- `test_plan_corpus` and `test_infer_corpus`: no result (killed at the time caps). Their printed metrics are not in this report; the accuracy harness figures stand in for them.
- Serial corpus time and any parallel speedup: not measured.
- A load-free full rerun of the accuracy harness: only the 7 timed-out files were rerun solo, so the targets table is the loaded run.
- Area 3 (`clean/`, `engineer/`, `assemble.py`): only the fit hygiene in `assemble.py` and `prepare.py` was read. Destructive actions and the other leakage paths were not reviewed.
- Areas 4 to 7 (report and failure handling beyond the save order, packaging and CI, test quality, API surface): not reviewed, outside STANDARD.
- Dry run never modifying the source or producing cleaned data: not checked directly.
- Strict mode, answers keyed by fingerprint never misapplied, plan/apply equivalence: only the mode selection was read.
- Python 3.9 compatibility: not run.
- Coverage: understated, see section 2.
