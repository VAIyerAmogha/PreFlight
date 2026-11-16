// Demo page logic. Loads Pyodide, installs pypreflight, and drives site/py/runner.py.
// STATUS: scaffold. Not yet run end to end; see DEMO_DEV.md (TODO list + log).

const PYODIDE_VERSION = "0.27.2";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
// Packages shipped with Pyodide (loaded from its CDN, fast).
const PYODIDE_PACKAGES = ["micropip", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "python-dateutil", "joblib"];
// Where pypreflight comes from. "pypreflight" = latest release on PyPI.
// For unreleased code, build a wheel into site/wheels/ and point this at it, e.g.
// "./wheels/pypreflight-2.0.0.dev0-py3-none-any.whl".
const PREFLIGHT_SPEC = "pypreflight";
const SAMPLE_URL = "./samples/autos.csv";
const UPLOAD_PATH = "/data/upload.csv";

const $ = (id) => document.getElementById(id);
let py = null;
let runner = null;

function setStatus(text, step) {
  $("load-status").textContent = text;
  if (step !== undefined) $("load-progress").value = step;
}

function showError(err) {
  const box = $("error");
  box.hidden = false;
  box.textContent = String(err && err.message ? err.message : err);
  console.error(err);
}

async function boot() {
  setStatus("Loading Python runtime…", 0);
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  py = await loadPyodide({ indexURL: PYODIDE_URL });

  setStatus("Loading pandas, scikit-learn, scipy…", 1);
  await py.loadPackage(PYODIDE_PACKAGES);

  setStatus("Installing pypreflight…", 2);
  const micropip = py.pyimport("micropip");
  await micropip.install(PREFLIGHT_SPEC);

  setStatus("Loading demo runner…", 3);
  const src = await (await fetch("./py/runner.py")).text();
  py.FS.mkdirTree("/data");
  py.FS.writeFile("/home/pyodide/runner.py", src);
  runner = py.pyimport("runner");

  setStatus("Ready.", 4);
  $("loader").hidden = true;
  $("step-upload").hidden = false;
}

async function loadCsvText(text) {
  $("error").hidden = true;
  py.FS.writeFile(UPLOAD_PATH, text);
  const select = $("target");
  select.innerHTML = "";
  // Ask preflight for the names, not a JS header sniff: headerless files get col_0, col_1, ...
  const names = JSON.parse(runner.columns(UPLOAD_PATH));
  for (const col of names) {
    const opt = document.createElement("option");
    opt.value = opt.textContent = col;
    select.appendChild(opt);
  }
  select.disabled = false;
  select.selectedIndex = select.options.length - 1;
  $("btn-analyze").disabled = false;
}

function renderTable(container, preview) {
  if (!preview) { container.textContent = ""; return; }
  const rows = preview.rows.map((r) => `<tr>${r.map((v) => `<td>${escapeHtml(v)}</td>`).join("")}</tr>`).join("");
  container.innerHTML = `<table><thead><tr>${preview.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderDecisions(container, decisions) {
  const rows = decisions.map((d) =>
    `<tr class="${d.needs_attention ? "attention" : ""}"><td>${escapeHtml(d.stage)}</td><td>${escapeHtml(d.columns.join(", "))}</td><td>${escapeHtml(d.action)}</td><td>${d.confidence}</td><td>${escapeHtml(d.plain_language)}</td></tr>`
  ).join("");
  container.innerHTML = `<table><thead><tr><th>Stage</th><th>Columns</th><th>Action</th><th>Confidence</th><th>Why</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderQuestions(form, questions) {
  if (!questions.length) { form.innerHTML = "<p>None. Every decision was confident.</p>"; return; }
  form.innerHTML = questions.map((q) => `
    <fieldset>
      <legend>${escapeHtml(q.text)}</legend>
      <p class="hint">${escapeHtml(q.explanation)}</p>
      ${q.options.map((o) => `
        <label><input type="radio" name="${escapeHtml(q.question_id)}" value="${escapeHtml(o)}" ${o === q.recommended_default ? "checked" : ""}>
        ${escapeHtml(o)}${o === q.recommended_default ? " (recommended)" : ""}</label><br>`).join("")}
    </fieldset>`).join("");
}

function collectAnswers(form) {
  const answers = {};
  for (const input of form.querySelectorAll("input[type=radio]:checked")) answers[input.name] = input.value;
  return answers;
}

function renderDownloads(container, files) {
  container.innerHTML = "<h3>Downloads</h3>";
  for (const [key, path] of Object.entries(files)) {
    let bytes;
    try { bytes = py.FS.readFile(path); } catch { continue; }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([bytes]));
    a.download = path.split("/").pop();
    a.textContent = a.download;
    a.title = key;
    container.append(a, " ");
  }
}

function escapeHtml(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

$("csv-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (file) await loadCsvText(await file.text());
});

$("btn-sample").addEventListener("click", async () => {
  try { await loadCsvText(await (await fetch(SAMPLE_URL)).text()); } catch (err) { showError(err); }
});

$("btn-analyze").addEventListener("click", async () => {
  $("error").hidden = true;
  try {
    const res = JSON.parse(runner.analyze(UPLOAD_PATH, $("target").value));
    $("plan-summary").textContent = `${res.rows} rows × ${res.columns} columns · task: ${res.task ?? "inferred later"}`;
    renderTable($("raw-preview"), res.raw_preview);
    renderDecisions($("decisions"), res.decisions);
    renderQuestions($("questions"), res.questions);
    $("step-plan").hidden = false;
  } catch (err) { showError(err); }
});

$("btn-run").addEventListener("click", async () => {
  $("error").hidden = true;
  try {
    const answers = JSON.stringify(collectAnswers($("questions")));
    const res = JSON.parse(runner.run(UPLOAD_PATH, $("target").value, answers));
    renderTable($("cleaned-preview"), res.cleaned_preview);
    renderDownloads($("downloads"), res.files);
    $("step-result").hidden = false;
  } catch (err) { showError(err); }
});

boot().catch(showError);
