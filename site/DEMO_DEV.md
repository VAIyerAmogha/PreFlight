# PreFlight live demo: development notes and session log

This file is the single source of truth for the demo website. **Every session
(human or Claude) that works on `site/` reads this file first and appends to the
log at the end before stopping.** It is excluded from the deployed site by
`site/.vercelignore`.

---

## 0. How to use this file (session protocol)

**At the start of a session:**
1. Read §1 (Goal) and §2 (Current status) to see where things stand.
2. Read the **last 2–3 entries** of §9 (Session log). The newest entry's "Next steps" is the plan for this session.
3. Check §8 (Open questions) for anything that blocks you.

**While working:**
- Tick boxes in §7 (Task board) as items get done. Add new tasks there rather than in the log.
- Write down every non-obvious decision in §6 (Decision record) with the reason. Do not delete old ones. If a decision is reversed, mark it `SUPERSEDED by D-xx`.

**Before ending a session (this is required):**
1. Update §2 (Current status): the one-line state and the "Last verified" fields.
2. Append a new entry to the **bottom** of §9 using the template in §9.1. Never edit or reorder old entries, except to fix an obvious typo.
3. If you changed how to run or deploy anything, update §4 or §5 as well.

**For Claude sessions:** start with "read `site/DEMO_DEV.md` and continue from the last log entry". Follow the protocol above without being asked again.

Log entry IDs are `S-NNN`, one higher than the previous entry. Dates are absolute (`YYYY-MM-DD`), never "yesterday".

---

## 1. Goal

A public web page where anyone can upload a CSV and watch `pypreflight` work on it live:
1. Upload a CSV (or pick a bundled sample) and choose the target column.
2. **Plan:** show the detected shape, every `Decision`, and the open questions as a form.
3. **Prepare:** run `prepare()` with the user's answers. Show the raw and cleaned tables side by side and offer downloads (cleaned CSV, Report PDF, Plan PDF, pipeline.joblib, JSON files).

The demo link goes in both the GitHub README and the PyPI README. A `Demo` entry also goes in `[project.urls]` in `pyproject.toml`.

### Constraints
- **No backend.** Python runs in the visitor's browser through [Pyodide](https://pyodide.org). Uploaded files never leave the machine, and the page says so.
- **Hosting:** Vercel, as a static site. Python cannot run server-side on Vercel because pandas, scikit-learn, scipy and matplotlib together exceed the 250 MB function limit (see D-01).
- **The demo does not need a PyPI release to update.** The page installs `pypreflight` from PyPI at load time, so a site-only change is just a git push.

---

## 2. Current status

| Field | Value |
|---|---|
| **State** | Scaffold written. Runner verified under CPython. **Not yet run in a browser / Pyodide.** Not deployed. |
| **Live URL** | _not deployed yet_ (planned: `https://pypreflight.vercel.app`, see Q-01) |
| **Last verified locally (CPython)** | 2026-11-16: `runner.columns/analyze/run` on `samples/autos.csv` → 205 rows, 72 decisions, 11 questions, cleaned 201×26, all 8 artifacts written, 1.7 s |
| **Last verified in browser** | never |
| **pypreflight version the page loads** | latest on PyPI (`PREFLIGHT_SPEC` in `app.js`). **PyPI has 1.0.1 only, so this fails until 2.0.0 ships (Q-03).** |
| **Pyodide version** | 0.27.2 (`PYODIDE_VERSION` in `app.js`) |

---

## 3. Architecture

```
Browser
 ├─ index.html / style.css         static page, 3 steps: Upload → Plan → Result
 ├─ app.js                         boots Pyodide, installs packages, UI wiring
 │    ├─ loadPyodide(CDN jsdelivr)
 │    ├─ loadPackage([numpy, pandas, scipy, scikit-learn, matplotlib, dateutil, joblib, micropip])
 │    ├─ micropip.install("pypreflight")   → pulls typer, charset-normalizer, reportlab (pure Python)
 │    └─ fetch py/runner.py → write into Pyodide FS → import runner
 └─ py/runner.py  (runs inside Pyodide)
      columns(path)               -> JSON list of column names as preflight reads them
      analyze(path, target)       -> JSON: rows, columns, task, decisions[], questions[], raw_preview
      run(path, target, answers)  -> JSON: cleaned_preview, decisions[], files{key: fs path}
```

Data flow:
1. The user picks a file. JS writes it to the Pyodide in-memory FS at `/data/upload.csv`.
2. `columns()` fills the target dropdown. Python supplies the names, not JS, because headerless files are read as `col_0…col_N` (see D-04).
3. `analyze()` calls `preflight.plan()`, then `create_review_session()` to get the same `Question` objects the terminal and notebook channels use.
4. The user answers the questions. `run()` calls `preflight.prepare(mode="ask", channel=<callback>)`. The callback returns the page's answers and falls back to `recommended_default` for any it lacks.
5. Artifacts are written to `/data/preflight_out/` in the virtual FS. JS reads them with `py.FS.readFile` and offers them as Blob downloads.

### Files

| Path | Purpose |
|---|---|
| `site/index.html` | Markup for the three steps |
| `site/style.css` | Minimal functional styles with light/dark tokens. A design pass is still TODO. |
| `site/app.js` | Pyodide boot and all UI logic. The config constants are at the top. |
| `site/py/runner.py` | Python bridge. The only place that touches the preflight API. |
| `site/samples/autos.csv` | UCI Automobile dataset (`imports-85.data`, headerless, `?` = missing). CC BY 4.0, attribution to UCI ML Repository still to add on the page (T-12). |
| `site/vercel.json` | Clean URLs and security headers |
| `site/.vercelignore` | Keeps this file off the live site |
| `site/DEMO_DEV.md` | This file |

### preflight API surface the demo depends on
Check these first if the package changes and the demo breaks:
- `preflight.plan(path, target=None)` → `Plan` with `.decisions`, `.schema.columns[].name`, `.row_count`, `.column_count`, `.task_type`
- `preflight.interact.question.create_review_session(decisions, path, schema=...)` → `.questions: list[Question]`
- `Question` fields: `question_id, column, text, explanation, evidence_summary, options, recommended_default, preview`
- `preflight.prepare(path, target, mode="ask", channel=callable, out_dir=...)` → dict with `apply_result` (`.prepared`, `.data`, `.decisions`) and `paths`
- Callback channel contract: `handler(list[Question]) -> {question_id: option_label}` (`interact/channels/callback.py`)

---

## 4. Running locally

```bash
# From the repo root. Any static server works; Pyodide needs http://, not file://.
python -m http.server 8000 --directory site
# open http://localhost:8000
```

Testing unreleased package code in the browser:
```bash
python -m build --wheel                      # or: hatch build -t wheel
mkdir -p site/wheels && cp dist/pypreflight-*-py3-none-any.whl site/wheels/
# then in app.js set PREFLIGHT_SPEC = "./wheels/<that filename>.whl"
# (do not commit site/wheels/ unless a decision says so; see Q-03)
```

Fast Python-only check (CPython, no browser), the same check S-001 used:
```bash
python - <<'EOF'
import sys, json, pathlib
sys.path.insert(0, "site/py"); import runner
runner.OUT_DIR = pathlib.Path("/tmp/pf_demo_out")
p = "site/samples/autos.csv"
tgt = json.loads(runner.columns(p))[-1]
a = json.loads(runner.analyze(p, tgt)); print(len(a["decisions"]), "decisions,", len(a["questions"]), "questions")
r = json.loads(runner.run(p, tgt, "{}")); print(r["cleaned_preview"]["shape"], list(r["files"]))
EOF
```

---

## 5. Deploying to Vercel

One-time setup:
1. vercel.com → sign in with GitHub → **Add New → Project** → import `VAIyerAmogha/PreFlight`.
2. **Project name:** `pypreflight`. This sets the URL to `pypreflight.vercel.app`. Check that the name is free before you write the URL into the READMEs (Q-01).
3. **Root Directory:** `site`.
4. **Framework Preset:** Other. Leave the build command empty and leave the output directory as default.
5. **Production branch:** `main` (Settings → Git). PRs and `v2-dev` get preview URLs automatically.
6. Optional: Settings → Git → "Ignored Build Step" so that only changes under `site/` trigger a deploy:
   `git diff HEAD^ HEAD --quiet -- .` (run inside the root dir; exit 0 = skip)

After that, every push to `main` that touches `site/` deploys the site.

### README / PyPI link: release order
1. Confirm the Vercel project name and URL.
2. Add the link to `README.md`. Use an absolute URL, because PyPI breaks relative links. Also add it to `pyproject.toml`:
   ```toml
   [project.urls]
   Demo = "https://pypreflight.vercel.app"
   ```
3. Ship it with the next normal release (2.0.0). No extra PyPI release is needed. The link may 404 until the site is live, so deploy before announcing.

---

## 6. Decision record

| ID | Date | Decision | Why |
|---|---|---|---|
| D-01 | 2026-11-16 | Static site plus Pyodide (in-browser Python). No server. | Free hosting, nothing to keep awake, and user CSVs stay on the user's machine. Vercel Python functions cannot hold pandas + sklearn + scipy + matplotlib (250 MB limit) and would time out on larger files. |
| D-02 | 2026-11-16 | Host on Vercel with Root Directory `site/` | Preview URL per PR, a clean `*.vercel.app` URL, and auto-deploy from GitHub. User's choice over GitHub Pages. |
| D-03 | 2026-11-16 | The page installs `pypreflight` from PyPI at runtime and does not bundle it | Site updates never need a PyPI release, and the demo always shows the released behaviour. A local wheel path is the escape hatch for testing unreleased code. |
| D-04 | 2026-11-16 | Python supplies the target-column names (`runner.columns`), not a JS header sniff | Headerless files such as the bundled UCI sample are read as `col_0…col_25`. A JS sniff offered `13495` as a column name and `plan()` raised "Target column not found". |
| D-05 | 2026-11-16 | Questions are answered through the existing callback channel (`mode="ask", channel=callable`) | Uses the package's public interaction path and needs no web-specific code in the library. Unanswered questions fall back to `recommended_default`. |
| D-06 | 2026-11-16 | Runner functions take and return JSON strings | Avoids Pyodide proxy lifetime and conversion pitfalls; the JS only does `JSON.parse`. |

---

## 7. Task board

Ordered roughly by priority. IDs are stable; do not renumber.

**Make it work in the browser**
- [ ] T-01: Run `site/` locally in a browser and confirm Pyodide boots and every package loads.
- [ ] T-02: Confirm `micropip.install("pypreflight")` resolves typer, charset-normalizer and reportlab under Pyodide. If reportlab's optional C accelerator causes trouble, record it.
- [ ] T-03: Confirm matplotlib and the reportlab PDF rendering work in Pyodide, since `prepare()` always renders the Report PDF. If they fail, decide between a fix in the package and a demo workaround, and record a decision.
- [ ] T-04: Move the Python calls off the main thread into a Web Worker. `runner.analyze/run` are synchronous, so the tab freezes during work. This is needed before large files work well.
- [ ] T-05: `plan()` currently runs three times (columns, analyze, run). Cache the Plan in the runner, or pass the plan into prepare, if the API allows.
- [ ] T-06: File size guard. Warn above ~20 MB and refuse above ~100 MB, since browser memory is the limit.

**Make it show the changes clearly ("visible changes")**
- [ ] T-07: Raw vs. cleaned side by side, with changed cells highlighted and dropped rows and columns marked.
- [ ] T-08: Show each question's `preview` (before and after samples per option). The data is already in the JSON.
- [ ] T-09: Per-column summary: type before → after, missing count before → after.
- [ ] T-10: Embed the Report PDF inline (`<iframe>` of the Blob URL) as well as offering the download.

**Polish and ship**
- [ ] T-11: Design pass on `style.css`: layout, loading screen, mobile width.
- [ ] T-12: UCI dataset attribution on the page (CC BY 4.0) and a "your data never leaves your browser" note near the upload.
- [ ] T-13: Add one or two more samples (one with a header row, one classification task).
- [ ] T-14: Create the Vercel project (§5) and confirm the production URL.
- [ ] T-15: Add the demo link to `README.md` and `[project.urls]` before the 2.0.0 release.
- [ ] T-16: Keep `site/` out of the sdist and wheel. Check `[tool.hatch.build]` `include`, which currently lists `/src /tests /docs`, so this is probably already fine; verify.
- [ ] T-17: Decide whether CI should smoke-test `site/py/runner.py` under CPython (the §4 snippet) so that API changes do not silently break the demo.

---

## 8. Open questions

| ID | Question | Status |
|---|---|---|
| Q-01 | Is the Vercel project name `pypreflight` available? If not, pick the URL before editing the READMEs. | open |
| Q-02 | Should the demo pin a specific `pypreflight` version, or always take the latest? Latest is simpler but a bad release breaks the demo. | open (currently latest) |
| Q-03 | **Blocker until 2.0.0 is released.** PyPI has only 1.0.1 (checked 2026-11-16), and `runner.py` uses the v2 API (`plan`, `interact`, `prepare(mode=, channel=)`), so `PREFLIGHT_SPEC = "pypreflight"` will not work today. Choose: (a) commit a dev wheel to `site/wheels/`; (b) have a Vercel build command build the wheel from the repo; (c) do not deploy until 2.0.0 is out. Once 2.0.0 is out, pin `pypreflight>=2`. | open |
| Q-04 | Expose `task` and `model_hint` (tree/linear) as controls on the page? | open |

---

## 9. Session log

Append-only. Newest entry at the bottom.

### 9.1 Entry template

```markdown
### S-NNN · YYYY-MM-DD · <who: name or "Claude (model)">
**Goal:** what this session set out to do (usually the previous entry's Next steps)
**Done:**
- what changed, with file paths
**Verified:** what was actually run and its result. Say plainly if nothing was run.
**Problems / findings:** errors hit, with exact messages, and surprises
**Decisions:** new D-xx IDs added to §6, or "none"
**Tasks:** T-xx ticked / added
**Next steps:**
1. the concrete first thing the next session should do
2. …
```

### S-001 · 2026-11-16 · Claude (Opus 5.5)
**Goal:** Choose a hosting approach and scaffold the demo site. No deploy yet.
**Done:**
- Created `site/` with `index.html`, `style.css`, `app.js`, `py/runner.py`, `vercel.json`, `.vercelignore`, `samples/autos.csv` (copied from `corpus/raw/imports-85.data`), and this file.
- `runner.py` exposes `columns()`, `analyze()` and `run()`, built on `preflight.plan`, `create_review_session` and `prepare(mode="ask", channel=callback)`.
**Verified:** CPython only (repo venv, not Pyodide). The §4 snippet on `samples/autos.csv` gave 26 columns (`col_0…col_25`), 205 rows, 72 decisions, 11 questions, cleaned frame 201×26, and all 8 artifacts written (cleaned_data, schema, decisions, pipeline, pipeline_fitted_on_all_rows, report_pdf, plan_json, plan_pdf) in 1.7 s. **Nothing has been run in a browser yet.**
**Problems / findings:**
- The first version took target names from a JS header sniff. On the headerless sample it offered `13495`, and `plan()` raised `ValueError: Target column '13495' was not found … read as 26 column(s): ['col_0', …]`. Fixed by D-04.
- `plan.task_type` is `None` at plan time (inferred later), so the page shows "inferred later".
**Decisions:** D-01 … D-06
**Tasks:** none ticked. T-01 … T-17 added.
**Next steps:**
1. T-01, T-02, T-03: serve `site/` locally and get one full run working in the browser. The PDF rendering under Pyodide (T-03) is the biggest unknown.
2. PyPI has only 1.0.1, and the runner needs v2. For local browser testing, build a wheel into `site/wheels/` and point `PREFLIGHT_SPEC` at it. Settle Q-03 before deploying.
3. T-04: move the Python calls into a Web Worker once the basic run works.
