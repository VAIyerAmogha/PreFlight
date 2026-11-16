/* Pyodide worker: runs the real pypreflight package off the main thread.
 *
 * Plain (classic) worker kept in public/ on purpose: Pyodide's loader is fetched
 * from its CDN with importScripts and is not run through the Next.js bundler.
 *
 * Protocol (lib/engine.ts is the other side):
 *   main -> worker  {type:"init", pyodideVersion, preflightSpec, runnerUrl}
 *   main -> worker  {type:"call", id, fn, args}          runner.<fn>(*args) -> JSON string
 *   main -> worker  {type:"write", id, path, data}       data: ArrayBuffer
 *   main -> worker  {type:"read", id, paths}             -> {path: Uint8Array}
 *   worker -> main  {type:"progress", step, total, text}
 *   worker -> main  {type:"ready", versions}
 *   worker -> main  {type:"result", id, ok, value | error}
 */

let py = null;
let runner = null;

const PACKAGES = ["micropip", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "python-dateutil", "joblib"];

function progress(step, total, text, loaded) {
  self.postMessage({ type: "progress", step, total, text, loaded });
}

async function init({ pyodideVersion, preflightSpec, runnerUrl }) {
  const TOTAL = PACKAGES.length + 4;
  const indexURL = `https://cdn.jsdelivr.net/pyodide/v${pyodideVersion}/full/`;
  progress(0, TOTAL, "Downloading the Python runtime");
  importScripts(`${indexURL}pyodide.js`);
  py = await self.loadPyodide({ indexURL });

  // Pyodide reports "Loaded a, b, c" as packages land; pass the names on so the
  // page can tick them off one by one.
  const loaded = new Set();
  progress(1, TOTAL, "Loading pandas, NumPy, SciPy, scikit-learn", []);
  await py.loadPackage(PACKAGES, {
    messageCallback: (msg) => {
      const m = /^Loaded (.+)$/.exec(msg);
      if (!m) return;
      m[1].split(",").map((x) => x.trim()).filter(Boolean).forEach((n) => loaded.add(n));
      const done = PACKAGES.filter((p) => loaded.has(p)).length;
      progress(1 + done, TOTAL, `Loaded ${m[1]}`, [...loaded]);
    },
    errorCallback: () => {},
  });
  progress(PACKAGES.length + 1, TOTAL, "Scientific stack ready", PACKAGES);

  progress(PACKAGES.length + 1, TOTAL, "Installing pypreflight");
  const micropip = py.pyimport("micropip");
  await micropip.install(preflightSpec);

  progress(PACKAGES.length + 2, TOTAL, "Loading the demo bridge");
  const src = await (await fetch(runnerUrl)).text();
  py.FS.mkdirTree("/data");
  py.FS.writeFile("/home/pyodide/runner.py", src);
  runner = py.pyimport("runner");

  progress(TOTAL, TOTAL, "Ready");
  self.postMessage({ type: "ready", versions: JSON.parse(runner.version()) });
}

function readTree(path, out) {
  const st = py.FS.stat(path);
  if (py.FS.isDir(st.mode)) {
    for (const name of py.FS.readdir(path)) if (name !== "." && name !== "..") readTree(`${path}/${name}`, out);
  } else {
    out[path] = py.FS.readFile(path);
  }
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    try { await init(msg); } catch (err) { self.postMessage({ type: "fatal", error: String(err && err.message || err) }); }
    return;
  }
  const reply = (ok, payload, transfer) => self.postMessage({ type: "result", id: msg.id, ok, ...payload }, transfer || []);
  try {
    if (msg.type === "call") {
      const fn = runner[msg.fn];
      const value = fn(...msg.args);
      reply(true, { value });
    } else if (msg.type === "write") {
      const dir = msg.path.slice(0, msg.path.lastIndexOf("/"));
      if (dir) py.FS.mkdirTree(dir);
      py.FS.writeFile(msg.path, new Uint8Array(msg.data));
      reply(true, { value: null });
    } else if (msg.type === "read") {
      const out = {};
      for (const p of msg.paths) { try { readTree(p, out); } catch { /* missing file: skipped */ } }
      reply(true, { value: out }, Object.values(out).map((u) => u.buffer));
    } else if (msg.type === "rm") {
      try { readTree(msg.path, {}); } catch { reply(true, { value: null }); return; }
      const rm = (p) => { const st = py.FS.stat(p); if (py.FS.isDir(st.mode)) { for (const n of py.FS.readdir(p)) if (n !== "." && n !== "..") rm(`${p}/${n}`); py.FS.rmdir(p); } else py.FS.unlink(p); };
      rm(msg.path);
      reply(true, { value: null });
    }
  } catch (err) {
    // Python exceptions arrive as PythonError with the traceback in .message; keep the last line readable.
    const text = String(err && err.message || err);
    const lines = text.trim().split("\n");
    reply(false, { error: lines[lines.length - 1], traceback: text });
  }
};
