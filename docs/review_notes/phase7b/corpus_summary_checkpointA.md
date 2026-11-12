# Corpus summary for run_20261030_154227

## Checks

- PASS [ingest] timeouts: 0 []
- PASS [ingest] unhandled exceptions: 0 []
- PASS [ingest] delimiter accuracy >= 99.0%: 99.61%
- PASS [ingest] encoding accuracy >= 98.0%: 98.43%
- PASS [ingest] header accuracy == 100.0%: 100.00%
- PASS [ingest] dialect accuracy >= 98.0%: 98.43%
- PASS [infer] timeouts: 0 []
- PASS [infer] unhandled exceptions: 0 []
- PASS [infer] labelled columns == 2850: 2850
- PASS [infer] type accuracy >= 85.0%: 90.84%
- PASS [infer] wrong type, no flag <= 2.0%: 0.74%
- INFO [infer] abstained 1.30%
- PASS [plan] timeouts: 0 []
- PASS [plan] unhandled exceptions: 0 []
- PASS [plan] completions == 254: 254
- INFO [plan] needs_attention median 3.0, max 109, files > 10: 29; PDF pages median 2.0, max 18
- PASS [interact] timeouts: 0 []
- PASS [interact] hung files: 0 []
- PASS [interact] worker failures: 0 []
- INFO [interact] 254/254 files; questions median 3.0, max 109, files > 10: 29, strict raised: 240
- PASS [pipeline] timeouts: 0 []
- PASS [pipeline] some file has a usable target: 142
- PASS [pipeline] unhandled exceptions or hangs: 0 []
- PASS [pipeline] plan and run agree: 0 []
- INFO [pipeline] 142 files with a usable target, 142 completed
- PASS [accuracy] timeouts: 0 []
- PASS [accuracy] unhandled exceptions: 0 []

## Targets (tuning set only (held-out is scored once, with --final))

| Target | Goal | Tuning |
|---|---|---|
| Files without unhandled exceptions | 100% | PASS 153/153 |
| Dialect accuracy | ≥95% | FAIL 140/153 = 91.50% |
| Column type accuracy | ≥95% | FAIL 1616/1765 = 91.56% |
| Wrong type, no flag | ≤1% | PASS 8/1765 = 0.45% |
| Decisions with evidence+confidence | 100% | PASS 4736/4736 = 100.00% |
| needs_attention median | ≤10 | PASS median 3, max 109, files>10: 19 |
| Timeouts | 0 | PASS 0 |
