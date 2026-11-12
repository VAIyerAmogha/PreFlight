# Targets (computed from shard_*/results.json by scripts/merge_shards.py)

| Target | Goal | Tuning | Held-out | Combined |
|---|---|---|---|---|
| Files without unhandled exceptions | 100% | FAIL 147/153 | FAIL 100/101 | FAIL 247/254 |
| Dialect accuracy | ≥95% | FAIL 134/153 = 87.58% | FAIL 92/101 = 91.09% | FAIL 226/254 = 88.98% |
| Column type accuracy | ≥95% | FAIL 1450/1765 = 82.15% | FAIL 958/1085 = 88.29% | FAIL 2408/2850 = 84.49% |
| Wrong type, no flag | ≤1% | PASS 7/1765 = 0.40% | FAIL 14/1085 = 1.29% | PASS 21/2850 = 0.74% |
| Decisions with evidence+confidence | 100% | PASS 4334/4334 = 100.00% | PASS 2995/2995 = 100.00% | PASS 7329/7329 = 100.00% |
| needs_attention median | ≤10 | PASS median 3, max 109, files>10: 16 | PASS median 2.0, max 39, files>10: 8 | PASS median 3, max 109, files>10: 24 |
| Timeouts | 0 | FAIL 6 | FAIL 1 | FAIL 7 |

## Difference vs docs/v2_accuracy_report.md (docs/v2_hardening_report.md does not exist)

| Metric | Set | Doc | Now | Match |
|---|---|---|---|---|
| 1 unhandled | tuning | 153/153 = 100.0% | 147/153 | NO |
| 1 unhandled | held_out | 101/101 = 100.0% | 100/101 | NO |
| 1 unhandled | combined | 254/254 = 100.0% | 247/254 | NO |
| 2 dialect | tuning | 140/153 = 91.50% | 134/153 | NO |
| 2 dialect | held_out | 93/101 = 92.08% | 92/101 | NO |
| 2 dialect | combined | 233/254 = 91.73% | 226/254 | NO |
| 3 coltype | tuning | 1619/1765 = 91.73% | 1450/1765 | NO |
| 3 coltype | held_out | 978/1085 = 90.14% | 958/1085 | NO |
| 3 coltype | combined | 2597/2850 = 91.12% | 2408/2850 | NO |
| 4 wrong_noflag | tuning | 8/1765 = 0.45% | 7/1765 | NO |
| 4 wrong_noflag | held_out | 16/1085 = 1.47% | 14/1085 | NO |
| 4 wrong_noflag | combined | 24/2850 = 0.84% | 21/2850 | NO |
| 5 evidence | tuning | 4737/4737 = 100.0% | 4334/4334 | NO |
| 5 evidence | held_out | 3052/3052 = 100.0% | 2995/2995 | NO |
| 5 evidence | combined | 7789/7789 = 100.0% | 7329/7329 | NO |

## Corpus pytest suites (sharded by test file)
- totals: {'tests': 611, 'failures': 1, 'errors': 0, 'skipped': 0}
- (shard, exit, wall_s): [(0, 0, 640), (1, 0, 65), (2, 124, 3000), (3, 1, 615), ('2b', 124, 900)]
- fast suite (unit+property+regression, serial): tests=264 failures=4 errors=0 skipped=2 time=70.525s
