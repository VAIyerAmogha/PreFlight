# PreFlight live demo: development notes and session log

This file is the single source of truth for the demo website. **Every session
(human or Claude) that works on `site/` reads this file first and appends to the
log at the end before stopping.** It is excluded from Vercel uploads by
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

One page that is both **the documentation** and **a live playground**. It has an editorial, research-blog feel: a serif reading column, numbered interactive figures and code snippets. Anyone can upload a CSV and watch the real `pypreflight` work on it.

- **Docs:** why, how it works, the five rules, quickstart, API reference, modes and channels, accuracy, limitations. The content comes from the README.
- **Figures:** each is drawn from real package output (fixtures), and each moves or responds to clicks.
- **Playground:** notebook-style cells: read → type → plan → resolve (question cards) → apply (raw/cleaned diff) → outputs (pipeline, downloads, embedded PDFs, the exact Python that ran).

The site link goes in the GitHub README, the PyPI README and `[project.urls]`.

### Constraints
- **No backend.** Python runs in the visitor's browser through Pyodide, in a Web Worker. Files never leave the machine.
- **100% package-driven** (`dev-docs/demo-site-plan.md`): the site only displays what the package returns, with no data logic in TS. `public/py/runner.py` only calls `preflight` and serialises. Its one exception, `_cell_changes`, lines up raw and prepared rows by index for highlighting and decides nothing about the data.
- **Hosting:** Vercel, Next.js static export (`output: "export"`).

---

## 2. Current status

| Field | Value |
|---|---|
| **State** | Two pages: `/` (docs + 7 figures + teaser) and `/try` (7-step interactive studio). Owner palette (shadcn-style, `--radius: 0`, `.dark` class). Verified end to end in headless Chrome. **Not deployed.** Next.js work uncommitted (only the S-001 scaffold is committed, in 71ebd66). |
| **Live URL** | _not deployed yet_ (planned: `https://pypreflight.vercel.app`, see Q-01) |
| **Last verified locally** | 2026-11-16 (S-003): `tsc` + `next build` clean; headless Chrome drove all 7 `/try` steps (quiz, flip cards, decision map, keyboard answers, prepare, results, a second run with a changed answer → 2 rows in Experiments); engine ready 11–12 s with a warm cache; prepare 0.4 s; no console or page errors; 390 px light and dark with no horizontal overflow on `/` and `/try` |
| **Last verified in Pyodide/Node** | 2026-11-16: full `prepare()` incl. both PDFs under Pyodide 0.27.2 (Node), analyze 0.3 s, run 1.8 s |
| **pypreflight the page loads** | bundled wheel `public/wheels/pypreflight-2.0.0.dev0-py3-none-any.whl` (`PREFLIGHT_SPEC` in `lib/config.ts`). PyPI has only 1.0.1 (Q-03). |
| **Stack** | Next.js 16.4 (App Router, Turbopack), React 19, TypeScript, hand-written CSS (`app/globals.css`, owner palette), highlight.js, Pyodide 0.27.2 |

---

## 3. Architecture

```
site/                         Next.js project (Vercel Root Directory)
├── app/
│   ├── layout.tsx            fonts (next/font: Source Serif 4, Inter, JetBrains Mono), theme script (.dark class), metadata
│   ├── page.tsx              `/`: the docs page (server component); imports fixtures at build time
│   ├── try/page.tsx          `/try`: the playground studio
│   └── globals.css           owner palette on :root/.dark → site aliases (--paper, --ink, --brand, …) → components
├── components/
│   ├── ui.tsx                DataTable (diff modes), RawFileView, TypeBadge, Meter, CodeBlock (hljs), Figure, Seg
│   ├── DecisionExplorer.tsx  filterable Decision list + detail (evidence, candidate-score bars, raw JSON)
│   ├── QuestionCard.tsx      a Question with options and the package's before/after preview
│   ├── PageEffects.tsx       top-bar state, reading progress, TOC scroll-spy, reveal-on-scroll, warm-up on /try link hover, ThemeToggle, EngineEyebrow
│   ├── Topbar.tsx            shared header for both pages (Docs · API · Playground · GitHub · theme)
│   ├── CopyButton.tsx
│   ├── figures/              HeroTerminal (Fig 0), ReadTwoWays (1), PipelineStages (2), QuestionsFigure (4),
│   │                         HygieneFigure (5), AccuracyFigure (6), TryTeaser (section 06). Fig 3 = DecisionExplorer.
│   └── try/
│       ├── TryStudio.tsx     all /try state (file, settings, analysis, answers, runs[]), step rail, navigation, toast
│       ├── EngineLoader.tsx  per-package loading grid (Data step) + EngineMini (rail)
│       └── steps/            DataStep, ReadStep (separator/header quiz), TypesStep (guess + flip cards + column detail),
│                             DecisionsStep (lane map + confidence histogram, or list), QuestionsStep (keyboard deck),
│                             ApplyStep (stats, per-column change bars, replayable diff table), ResultsStep
│                             (pipeline flow, downloads, PDFs, Experiments table with deltas, code). common.tsx = header/CountUp/Stat/Verdict.
├── lib/
│   ├── config.ts             PYODIDE_VERSION, PREFLIGHT_SPEC, SAMPLES, MAX_UPLOAD_MB
│   ├── engine.ts             worker client: startEngine, callRunner, writeFile, readFiles, useEngine()
│   ├── hooks.ts              useInView, useVisible, useReducedMotion
│   └── types.ts              TS shapes of runner.py's JSON. Keep in sync with runner.py.
├── public/
│   ├── pyodide-worker.js     classic worker: loads Pyodide from jsDelivr, micropip-installs pypreflight, imports runner
│   ├── py/runner.py          the Python bridge (also run under CPython by the fixture builder)
│   ├── samples/              disguised_missing_units.csv, messy_noheader_semicolon.csv, autos.csv
│   └── wheels/               the v2 wheel (see Q-03)
├── data/fixtures/            real runner output behind the static figures (scripts/build_fixtures.py)
└── scripts/build_fixtures.py
```

**Engine lifecycle:** the worker starts when `/try` mounts, or earlier when someone hovers or focuses any link to `/try` (PageEffects, TryTeaser). The worker lives in module state, so a warm-up on `/` survives the client-side navigation to `/try`. Docs-only visitors never download the ~40 MB. The worker reports each package as it lands (`progress.loaded`) for the loading grid.

**Worker protocol** (`public/pyodide-worker.js` ↔ `lib/engine.ts`): `init`, `call` (runner fn → JSON string), `write` (ArrayBuffer → FS), `read` (paths/dirs → bytes), `rm`; replies are `progress`, `ready`, `fatal`, `result`.

**runner.py entry points** (JSON strings in and out): `version()`, `inspect(path)` → column names as preflight reads them; `analyze(path, target, task, hint)` → reading, raw lines, naive pandas, raw table, schema, decisions, questions; `run(path, target, answers_json, task, hint, out_dir)` → raw/prepared/model-ready frames, changes, pipeline steps, decisions, files, code.

### preflight API surface the demo depends on
Check these first if the package changes and the demo breaks:
- `preflight.plan`, `preflight.prepare(mode="ask", channel=callable, out_dir=...)`, `preflight.__version__`
- `preflight.interact.question.create_review_session`, `Question` fields
- `preflight.interact.resolve._load_raw` (**private**; gives the all-text raw table. See T-21.)
- `Plan.schema.{encoding,dialect,header,columns}`, `ColumnRawMetadata.{effective_name,effective_type,confidence,runner_up,abstained,missing_markers,anomalies,evidence}`
- `prepare()` dict keys: `apply_result` (`.prepared`, `.data`, `.decisions`, `.anomalies`), `pipeline`, `pipeline_fitted_on_all_rows`, `plan`, `paths`

---

## 4. Running locally

```bash
cd site
npm install
npm run dev                  # http://localhost:3000 (hot reload)
# or the production build, exactly what Vercel serves:
npx next build && python -m http.server 8765 --directory out
```

After a package change:
```bash
# from the repo root, with the repo venv active
python -m build --wheel --outdir /tmp/whl && cp /tmp/whl/pypreflight-*.whl site/public/wheels/
#   (update PREFLIGHT_SPEC in site/lib/config.ts if the filename changed; delete the old wheel)
python site/scripts/build_fixtures.py     # regenerate the figure data
```

Headless browser check (what S-002 used): Playwright in any venv (`pip install playwright`), `chromium.launch(channel="chrome")` uses the system Chrome, so no browser download is needed. Load the page, scroll to `#try`, wait for `.engine.ready`, click a sample, click "Run prepare()", wait for `#cell-apply`.

---

## 5. Deploying to Vercel

One-time setup:
1. vercel.com → **Add New → Project** → import `VAIyerAmogha/PreFlight`.
2. **Project name:** `pypreflight` → `pypreflight.vercel.app` (check that it's free first: Q-01).
3. **Root Directory:** `site`. The framework is detected as **Next.js**; leave the build and output settings at their defaults. `vercel.json` sets `"framework": "nextjs"`.
4. **Production branch:** `main`. PRs get preview URLs.
5. Optional: Ignored Build Step `git diff HEAD^ HEAD --quiet -- .` so that only `site/` changes deploy.

**The wheel must be committed.** The root `.gitignore` ignores `wheels/`, and `site/.gitignore` re-includes `public/wheels/`. If the wheel is missing on Vercel, the engine fails at "Installing pypreflight".

### README / PyPI link: release order
1. Confirm the URL.
2. Add the link to `README.md` (absolute URL) and to `pyproject.toml` under `[project.urls]` as `Demo = "https://pypreflight.vercel.app"`.
3. Ship with 2.0.0. No extra PyPI release is needed.

---

## 6. Decision record

| ID | Date | Decision | Why |
|---|---|---|---|
| D-01 | 2026-11-16 | Static site plus Pyodide (in-browser Python). No server. | Free hosting, nothing to keep awake, and user CSVs stay on the user's machine. Vercel Python functions cannot hold pandas + sklearn + scipy + matplotlib (250 MB limit). |
| D-02 | 2026-11-16 | Host on Vercel with Root Directory `site/` | Preview URL per PR, a clean `*.vercel.app` URL, and auto-deploy from GitHub. User's choice. |
| D-03 | 2026-11-16 | The page installs `pypreflight` from PyPI at runtime. ~~bundle~~ **Amended by D-11** while PyPI has only 1.x. | Site updates never need a PyPI release. |
| D-04 | 2026-11-16 | Python supplies the target-column names (`runner.inspect`), not a JS header sniff | Headerless files are read as `col_0…`. A JS sniff offered `13495`, and `plan()` raised "Target column not found". |
| D-05 | 2026-11-16 | Questions are answered through the existing callback channel | Uses the public interaction path; unanswered questions fall back to `recommended_default`. |
| D-06 | 2026-11-16 | Runner functions take and return JSON strings | Avoids Pyodide proxy lifetime and conversion pitfalls. |
| D-07 | 2026-11-16 | **Next.js** (App Router, TS, static export) instead of plain HTML/JS. SUPERSEDES the vanilla scaffold of S-001. | User request. Static export keeps the no-server model. |
| D-08 | 2026-11-16 | Hand-written CSS design system (`app/globals.css`), not Tailwind/shadcn | User's choice. An editorial look (serif prose, numbered figures) with tokens for light and dark. |
| D-09 | 2026-11-16 | Docs figures use **build-time fixtures** generated by running `runner.py` under CPython. The playground runs live. | Figures render instantly without the 40 MB engine and are still real package output. |
| D-10 | 2026-11-16 | Lazy engine start (near the playground, or on a "Try it" click) | Docs readers on mobile shouldn't download 40 MB. |
| D-11 | 2026-11-16 | Bundle the v2 wheel in `public/wheels/` until 2.0.0 is on PyPI | PyPI has only 1.0.1, which lacks the v2 API. Resolves Q-03 for now. |
| D-12 | 2026-11-16 | Next 16 rather than 15 | `npm audit` flagged next 15.x via postcss (high). Next 16.4.6 audits clean. |
| D-14 | 2026-11-16 | The playground moves to its own route `/try`, built as a 7-step guided studio. The home section 06 becomes a teaser. SUPERSEDES the inline notebook playground of S-002. | User request: more interaction, visual activity and involvement. Steps ask the user to predict (separator, header, types) before revealing the package's answer, and runs are kept for comparison. |
| D-15 | 2026-11-16 | Adopt the owner's shadcn-style palette verbatim (`--radius: 0`, dark via `.dark` class). Site tokens are aliases onto it. Colour appears only for data states. | User request. Their `--muted`/`--accent` are background colours, so the site's text-muted and brand tokens were renamed to `--muted-fg` and `--brand` to avoid the clash. ASK = inverted (primary bg). |
| D-16 | 2026-11-16 | The Apply table shows the prepared frame's real dtypes, not the plan's inferred types | Answers can override a type (e.g. "Keep as plain text"); the plan's label would then be wrong. |
| D-13 | 2026-11-16 | Cell-level diff is turned off (with a visible note) when the prepared index was renumbered | The package's missing-target step does `reset_index(drop=True)`, so labels no longer map to raw rows. Reconstructing the mapping in the site would duplicate package logic. The proper fix is in the package (Q-05). |

---

## 7. Task board

**Make it work in the browser**
- [x] T-01: Run in a browser; Pyodide boots and every package loads. (S-002)
- [x] T-02: `micropip` resolves the deps (typer pulls pydantic/pydantic_core from Pyodide's CDN; fine). (S-002)
- [x] T-03: matplotlib and the reportlab PDFs work in Pyodide. (S-002)
- [x] T-04: Python calls run in a Web Worker. (S-002)
- [ ] T-05: `plan()` runs three times per sample (inspect, analyze, prepare). Cache it if the API allows passing a Plan to prepare.
- [x] T-06: File size guard (50 MB refuse). (S-002) A softer warning around 10 MB is still open.

**Show the changes clearly**
- [x] T-07: Raw vs. prepared diff with changed/blanked cells, dropped rows and columns, added columns. (S-002; off for renumbered files, D-13)
- [x] T-08: Question previews (Fig 4 and the playground). (S-002)
- [x] T-09: Per-column type cards with confidence, markers and ASK. (S-002)
- [x] T-10: Report PDF and Plan PDF embedded inline, plus downloads. (S-002)

**Polish and ship**
- [x] T-11: Design pass (editorial layout, dark mode, mobile). (S-002)
- [x] T-12: UCI attribution (footer) and the "never leaves your browser" note. (S-002)
- [ ] T-13: More samples: one with a header row and Latin-1 encoding, and one with dates.
- [ ] T-14: Create the Vercel project (§5) and confirm the URL.
- [ ] T-15: Add the demo link to `README.md` and `[project.urls]` before 2.0.0.
- [ ] T-16: Confirm that `site/` stays out of the sdist and wheel (`[tool.hatch.build] include` lists `/src /tests /docs`, so probably fine).
- [ ] T-17: CI job: `npm ci && npx next build` in `site/`, plus a CPython run of `scripts/build_fixtures.py` so that API changes break CI rather than the demo.
- [ ] T-18: Lighthouse / accessibility pass (keyboard flow through the playground, contrast of `--faint` text).
- [ ] T-19: Open Graph image (a static PNG of Figure 0).
- [ ] T-20: "Run all" autoplay that walks the /try steps for a sample without clicks (from demo-site-plan.md).
- [ ] T-23: Persist the /try session (file name, settings, answers) in the URL or localStorage so a refresh doesn't lose it. Files can't be persisted; samples can.
- [ ] T-24: Real-device check of the PDF iframe (headless Chrome shows blank thumbnails, probably a headless artifact) and of the flip-card 3D on Safari.
- [ ] T-25: Shareable result: a "copy link" that reopens /try on the same sample with the same answers.
- [ ] T-21: Ask for a public way to get the raw all-text table (now `preflight.interact.resolve._load_raw`).
- [ ] T-22: Once 2.0.0 is released, switch `PREFLIGHT_SPEC` to `"pypreflight==2.0.0"`, delete `public/wheels/`, and remove the `!public/wheels/` line in `site/.gitignore`.

---

## 8. Open questions

| ID | Question | Status |
|---|---|---|
| Q-01 | Is the Vercel project name `pypreflight` available? | open |
| Q-02 | After 2.0.0, pin an exact version (safer) or `>=2` (always the latest)? | open (recommend exact pin) |
| Q-03 | How should the demo get v2 before it's on PyPI? | **resolved by D-11** (bundled wheel); revisit at T-22 |
| Q-04 | Expose `task` and `model_hint` on the page? | **resolved:** both are in the playground settings |
| Q-05 | **Package:** `clean/rows.py:apply_missing_target` does `reset_index(drop=True)`, so dropped rows can't be traced back to the source rows. Keep the original row labels instead? It would also re-enable cell diffs for files with a blank target (D-13). Needs a check that pipeline and X/y alignment still hold. | open (owner's call) |
| Q-06 | **Package:** the preview for "weight → Convert to whole numbers" shows `(left blank)` for `69 kg`, but the actual run converts it to `69`. The preview converter doesn't seem to strip units. Bug? | open (owner's call) |

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

### S-002 · 2026-11-16 · Claude (Opus 5.5)
**Goal:** Build the real site: a top-notch editorial UI that is both docs and a live playground with moving figures and code. Mid-session the user asked for **Next.js** and chose to keep the hand-written CSS.
**Done:**
- Proved the package runs in Pyodide (Node, `pyodide@0.27.2`, local v2 wheel): full `prepare()` including both PDFs.
- Rewrote `runner.py` (now `public/py/runner.py`) with `version`, `inspect`, `analyze` and `run`, plus raw/prepared/model-ready frames, cell changes, pipeline steps and the "code that ran".
- Built the Next.js app (see §3): docs page with sections 01–10, Figures 0–6 and the playground. Removed the vanilla `index.html`, `app.js` and `style.css`.
- `scripts/build_fixtures.py` writes `data/fixtures/*.json`; the pages import them at build time.
- `site/.gitignore` re-includes `public/wheels/` (the root `.gitignore` ignores `wheels/`).
- Read `dev-docs/demo-site-plan.md` (written earlier by the owner) and followed its "100% package-driven" rule.
**Verified:**
- `npx tsc --noEmit` clean; `npx next build` → static `/`.
- Headless Chrome (Playwright + system Chrome) against `out/`: engine ready in 16–19 s cold; `disguised_missing_units.csv` → plan, 5 questions, prepare, 137 highlighted cells, 9 downloads (report.pdf, plan.pdf, cleaned_data.csv, decisions.json, schema.json, plan.json, pipeline.joblib, pipeline_fitted_on_all_rows.joblib, answers.json); `autos.csv` → 11 questions, prepare OK, renumbering note shown; no console or page errors; 390 px dark mode with no horizontal overflow. Screenshots were reviewed for every section.
- **Not verified:** Safari/Firefox, real mobile devices, the Vercel deploy, and PDF iframe behaviour on mobile.
**Problems / findings:**
- `npm audit`: next 15.x is flagged via postcss → moved to Next 16 (D-12).
- The diff misaligned on `autos.csv`: the package's missing-target step resets the index. Turned the diff off with a note (D-13) and raised Q-05.
- Package preview mismatch for units (Q-06).
- Layout bug found in screenshots and fixed: `.wide` figures were capped at the prose width because the width rule targeted `section` rather than its children.
**Decisions:** D-07 … D-13 (D-03 amended)
**Tasks:** ticked T-01–T-04, T-06–T-12. Added T-18–T-22.
**Next steps:**
1. Owner: commit `site/`, then do T-14 (create the Vercel project, Root Directory `site`) and check that the deployed engine loads (wheel present).
2. T-15: add the demo URL to the README and `pyproject.toml` before 2.0.0.
3. T-17: add a CI job for the site build and fixtures.
4. Q-05 and Q-06 are package questions for the owner. If Q-05 is fixed, the D-13 fallback switches off by itself (it only triggers on a renumbered index).
5. T-13 and T-20: more samples and a "Run all" autoplay.

### S-003 · 2026-11-16 · Claude (Opus 5.5)
**Goal:** Owner asked for "Try it" to be its own page with much more interaction, visual activity and user involvement, a better UI/UX, and a supplied palette (shadcn-style, radius 0, `.dark`).
**Done:**
- Palette: the owner's tokens go verbatim on `:root` and `.dark`; site aliases are mapped onto them; every radius is `var(--radius)` (0) except circles; the theme script and toggle now use the `.dark` class (OS default, saved choice wins). D-15.
- New route `/try` (`app/try/page.tsx`, `components/try/*`), a 7-step studio with a sticky step rail (file card, step states, live engine status):
  1. Data: sample cards, dropzone, per-package engine loading grid, column chips (click to set target), live `plan()` call preview.
  2. Read: a quiz. Guess the separator and click the header line, then see the reveal (delimiters pop in, verdict ✓/✗ with the package's confidence), with a score chip and a skip option.
  3. Types: guess a type per column from real sample values, then flip the cards (3D) to see the type, confidence, markers, ASK and your result. Click a card for candidate-score bars and values.
  4. Decisions: count-up stats, a lane map (one square per Decision, shade = confidence, inverted = ASK) with hover and pin, a confidence histogram, and a list view toggle.
  5. Questions: a one-at-a-time deck with progress pips, keys 1–4, ←/→ and Enter, an answers sidebar with changed/default badges, and run.
  6. Apply: count-up stats, per-column change bars, a replayable staggered cell highlight, "only rows with changes", and as-read / model-ready views.
  7. Results: an animated pipeline flow, downloads, Report/Plan PDF viewer, an **Experiments** table (every run kept, deltas vs the previous run), "change answers & re-run", and the code that ran.
- Home: shared `Topbar`; hero CTA → `/try`; section 06 is a teaser whose table sweeps raw → prepared using the fixture's change lists; hovering links to `/try` warms up Python.
- Removed `components/playground/`. Exported `DecisionDetail` from DecisionExplorer. The worker reports loaded package names.
**Verified:** `tsc` and `next build` clean (`/`, `/try` static). A Playwright script on system Chrome walked every step on `disguised_missing_units.csv` including a second run (Experiments shows 2 rows), plus mobile/dark on both pages; no page or console errors; no horizontal overflow at 390 px. Screenshots were reviewed for every step.
**Problems / findings:**
- `.reveal` (my quiz class) collided with the page-wide scroll-reveal class (opacity 0), so the verdicts were invisible. Renamed to `.reveal-box`.
- The file size showed "0 B" because the ArrayBuffer is detached when transferred to the worker. The size is now read first.
- The Apply table labelled `weight` as `int` after the user chose "Keep as plain text". It now shows real dtypes (D-16).
- The plan's type evidence for `weight` shows `categorical 0.95` as the top candidate score while the inferred type is `int`. Displayed as given; may relate to Q-06.
**Decisions:** D-14, D-15, D-16
**Tasks:** added T-23–T-25 (T-20 reworded for /try).
**Next steps:**
1. Owner: review `/try` locally (`cd site && npm run dev` → http://localhost:3000/try), then commit `site/`.
2. T-14: create the Vercel project and check the deployed engine (wheel present).
3. T-24: real-device check (Safari flip cards, mobile PDF).
4. T-23/T-25: persist and share a /try session.
5. Q-05 and Q-06 are still open for the owner (package).
