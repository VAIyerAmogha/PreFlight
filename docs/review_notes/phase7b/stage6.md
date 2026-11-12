# Stage 6: corpus test restructure

- Marker `corpus` is registered in pyproject `[tool.pytest.ini_options]`. `addopts = "-m 'not corpus'"`, so plain
  `pytest` runs the fast suite (339 tests). `pytest -m corpus -n 4 --dist loadgroup` runs the corpus suites (2393 tests).
- Every corpus test is now per file (`corpus_file` parametrize; id and xdist group = raw file name):
  ingest (match + schema round-trip), infer, plan (dry_run + PDF), interact (worker subprocess), pipeline
  (worker subprocess), and the new accuracy test (scripts/corpus_accuracy.measure_file, 120 s timeout).
- Cached ingest/infer: tests/corpus/corpus_support.py (`lru_cache` per worker, returning copies).
  `--dist loadgroup` keeps one file's tests on one worker.
- Per-file records go to .pytest_cache/corpus_results/run_<time>/<suite>/<file>.json. A per-file pytest-timeout
  becomes a TIMEOUT record (tests/conftest.py).
- At session end the controller runs scripts/corpus_summary.py. It checks the corpus-wide thresholds the old
  loop tests asserted, unchanged: ingest delimiter >= 99, encoding >= 98, header == 100, dialect >= 98; infer
  2850 columns, accuracy >= 85, wrong-unflagged <= 2; plan 254 completions; no exceptions or hangs. Any FAIL fails the session.
  Partial runs skip the corpus-wide checks and say so. The targets table shows tuning only, and held-out/combined only with --final.
- Dev deps added: pytest-xdist 3.8.0 (MIT), pytest-timeout 2.4.0 (MIT).
- Removed: scripts/shard_corpus.py, scripts/accuracy_shard.py, scripts/merge_shards.py (untracked Phase 7
  helpers). scripts/corpus_accuracy.py now exposes measure_file(); its unused partial summarize() was dropped.
- test_degenerate_inputs: the optional pytest-timeout import (`noqa: F401`) is gone; it now uses the marker directly.
  test_corruption_handling: the sys.path insert (`noqa: E402`) is gone.

## Result
Full corpus run: 2393 passed in 405 s (-n 4, loadgroup). Phase 7 shard 2 was killed at 3000 s.
All 23 corpus-wide checks PASS (docs/review_notes/phase7b/corpus_summary_checkpointA.md).
The first run found 1 failure (avengers.csv interaction: encoding question with a single option). It is fixed;
see docs/port_regressions.md "Encoding question fix found by the per-file corpus run".
