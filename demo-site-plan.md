# Demo Site Plan

An interactive website that walks through PreFlight cleaning a dataset step by step, like a notebook, with animations. **All data actions are done by the published `pypreflight` package. The site only displays results.**

## Architecture
- **Engine:** Pyodide (Python in WebAssembly) running in a Web Worker. It installs the package from PyPI:
  ```python
  import micropip
  await micropip.install("pypreflight==<pinned-version>")
  import preflight
  ```
- **Frontend:** React + Vite, Tailwind + shadcn/ui, Framer Motion (animations), Plotly or Recharts (charts).
- **Hosting:** Cloudflare Pages or Vercel, free tier, static site with no backend.
- **Optional fallback:** FastAPI on a free Hugging Face Space (Docker) for large files or PDF generation, if Pyodide can't handle them.

## Rule: 100% package-driven
- Pin an exact version and show `preflight.__version__` in the UI, with a link to PyPI.
- Call only the public API: `plan`, `dry_run`, `resolve`, `apply`, `prepare`, `Plan`, `Decision`.
- No data logic in JS: no diffs, stats, type guesses or imputation. Every value shown comes from the package's output.
- If the UI needs data the package doesn't expose (per-step diffs, histogram data, etc.), add it to the package and release a new version.
- Interactive questions go through the `callback` channel.
- "Show code" panel on every step with the exact Python that ran.
- Site source is public so the claim can be verified.

## Demo flow (one animated "cell" per stage)
1. **Ingest:** detected encoding, delimiter and header; raw text becomes a table.
2. **Infer:** column type badges; disguised missing values highlighted.
3. **Plan / Dry run:** list of decisions with reasons, and a preview diff.
4. **Resolve:** interactive question cards.
5. **Clean:** cells flash as they change; dropped rows and columns animate out; before/after histograms.
6. **Engineer:** new features slide in; encoding and VIF pruning shown visually.
7. **Assemble / Report:** sklearn Pipeline diagram, report summary, PDF download.

Extras: preset datasets (from `examples/`), CSV upload (processed locally, never uploaded), "Run all" autoplay.

## Pre-release checklist (package)
- [ ] Confirm `pypreflight` name is available on PyPI.
- [ ] Publish a pure-Python wheel (`py3-none-any`), not just an sdist.
- [ ] Test on TestPyPI, then `micropip.install` it in the Pyodide REPL.
- [ ] Verify the PDF report (reportlab + matplotlib) works in Pyodide.
- [ ] Expose any per-step data the UI needs via the public API.

## Milestones
1. Pyodide proof of concept: install from TestPyPI, run `prepare` on `examples/clean.csv`.
2. Stage-by-stage runner in a Web Worker, sending stage results to the UI.
3. Step-by-step UI with animations for each stage.
4. Interactive resolve (question cards via the callback channel).
5. Report and PDF download, preset datasets, upload.
6. Deploy to Cloudflare Pages and add a link from the README.

## Known limits
- First load is about 30–50 MB (cached after that).
- Large CSVs (over about 50 MB) will be slow in the browser; use the HF Space fallback if needed.
