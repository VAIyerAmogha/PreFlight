# Stage 5: lint, types, coverage
| Check | Before | After |
|---|---|---|
| ruff (project rules E,F,W,I,UP,B) | 863 errors | 0 |
| mypy --strict src/preflight | 49 errors (13 files) | 0 (55 files) |
| coverage overall (fast suite) | 89% | 90.3% (target 85%) |
| coverage infer/ | 97.4% | 97.4% (target 95%) |
| coverage ingest/ | 87.8% | 98.7% (target 95%) |

How: ruff --fix plus a scoped fix for annotation style (List->list, Optional->X|None in files with
`from __future__ import annotations`). The rest were fixed by hand. The dead v1 shim modules
cleaner.py, engineer.py, report.py and assembler.py (`import *`, either shadowed by packages or never imported) were deleted.
mypy: annotations were added. Minimal local stubs in typings/ cover sklearn.base (BaseEstimator,
TransformerMixin, clone) and reportlab.pdfgen.canvas.Canvas (via mypy_path), so subclassing them is
checked. No ignores. The new test file tests/unit/test_ingest_edges.py covers 24 edge-path behaviours.
New noqa / type: ignore / pragma: none. Pre-existing noqa: 3 (clean/impute.py:12,
interact/channels/terminal.py:30 `import rich` availability probe, interact/resolve.py ipywidgets
availability probe). They are left as they were and reported.

Python floor: requires-python >=3.9 (not changed). Only 3.12 is installed, so it cannot be run on 3.9.
Static check (ruff FA102, py39 target): report/report.py used PEP 604 unions at runtime without
`from __future__ import annotations`, so `import preflight` would fail on 3.9 (present in HEAD).
Fixed by adding the future import. FA102 now 0. mypy is pinned to python_version 3.10 (mypy's
minimum), so 3.9-only issues remain unchecked. Recommendation: raise the floor to 3.10 before
Phase 8, or add a 3.9 CI job.
