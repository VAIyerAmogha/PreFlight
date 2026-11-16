"use client";

import { useState } from "react";
import { MAX_UPLOAD_MB, SAMPLES } from "@/lib/config";
import { useEngine } from "@/lib/engine";
import { fmtBytes, Seg } from "../../ui";
import EngineLoader from "../EngineLoader";
import type { Hint, Source, Task } from "../TryStudio";
import { StepHeader } from "./common";

interface Props {
  source: Source | null; columns: string[]; target: string; task: Task; hint: Hint; busy: null | "file" | "plan" | "run";
  onSample: (file: string, target: string) => void; onFile: (f: File | undefined) => void;
  setTarget: (v: string) => void; setTask: (v: Task) => void; setHint: (v: Hint) => void;
  onAnalyze: () => void; analyzed: boolean;
}

export default function DataStep(p: Props) {
  const engine = useEngine();
  const [over, setOver] = useState(false);
  const ready = engine.phase === "ready";

  return (
    <>
      <StepHeader n={1} title="Bring a messy file">
        Pick one of the bundled traps or drop in your own CSV. It is read by the real <code>pypreflight</code> running inside this tab, and never uploaded.
      </StepHeader>

      <div className="data-grid">
        <div>
          <h3 className="sec">Samples</h3>
          <div className="sample-cards">
            {SAMPLES.map((s) => {
              const on = p.source?.sample && p.source.name === s.file;
              return (
                <button key={s.file} className={`sample-card${on ? " on" : ""}`} onClick={() => p.onSample(s.file, s.target)} disabled={p.busy === "file"}>
                  <span className="tag">{s.tag}</span>
                  <span className="nm">{s.file}</span>
                  <span className="ds">{s.desc}</span>
                  <span className="go">{on ? "Loaded" : "Load →"}</span>
                </button>
              );
            })}
          </div>
          <h3 className="sec">Or your own</h3>
          <label
            className={`dropzone${over ? " over" : ""}${p.source && !p.source.sample ? " has" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); p.onFile(e.dataTransfer.files[0]); }}
          >
            <input type="file" accept=".csv,.tsv,.txt,.data,text/csv" hidden onChange={(e) => { p.onFile(e.target.files?.[0]); e.target.value = ""; }} />
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></svg>
            {p.source && !p.source.sample ? (
              <span><b>{p.source.name}</b> · {fmtBytes(p.source.size)} — drop another to replace</span>
            ) : (
              <span><b>Drop a CSV</b> or click to browse</span>
            )}
            <small>CSV, TSV or any delimited text · up to {MAX_UPLOAD_MB} MB · stays on your machine</small>
          </label>
          {p.busy === "file" && <div className="spin-line"><span className="spinner" /> {ready ? "Reading the file…" : "Waiting for Python, then reading the file…"}</div>}
        </div>

        <div className="data-side">
          <EngineLoader />
          <div className={`settings${p.columns.length ? "" : " dim"}`}>
            <h3 className="sec">Settings</h3>
            {p.columns.length > 0 && (
              <div className="col-chips" aria-label="Columns as PreFlight reads them">
                {p.columns.slice(0, 40).map((c, i) => (
                  <button key={c} className={`col-chip${c === p.target ? " on" : ""}`} style={{ animationDelay: `${i * 18}ms` }} onClick={() => p.setTarget(c)} title="Use as target">
                    {c}
                  </button>
                ))}
                {p.columns.length > 40 && <span className="col-chip more">+{p.columns.length - 40}</span>}
              </div>
            )}
            <div className="field">
              <label htmlFor="try-target">Target column <span className="hint">(what a model would predict)</span></label>
              <select id="try-target" value={p.target} disabled={!p.columns.length} onChange={(e) => p.setTarget(e.target.value)}>
                {!p.columns.length && <option>Load a file first</option>}
                {p.columns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div className="field-row">
              <div className="field">
                <label>Task</label>
                <Seg<Task> label="Task" value={p.task} onChange={p.setTask} options={[{ v: "", label: "Infer" }, { v: "classification", label: "Classify" }, { v: "regression", label: "Regress" }]} />
              </div>
              <div className="field">
                <label>Model hint</label>
                <Seg<Hint> label="Model hint" value={p.hint} onChange={p.setHint} options={[{ v: "tree", label: "tree" }, { v: "linear", label: "linear" }]} />
              </div>
            </div>
            <button className="btn primary big" disabled={!p.source || !p.target || p.busy !== null} onClick={p.onAnalyze}>
              {p.busy === "plan" ? <><span className="spinner inv" /> Running plan()…</> : p.analyzed ? "Re-run plan() →" : "Run plan() →"}
            </button>
            <div className="code-inline">
              preflight.plan(&quot;{p.source?.name ?? "your.csv"}&quot;, target=&quot;{p.target || "…"}&quot;{p.task ? `, task="${p.task}"` : ""}{p.hint !== "tree" ? `, model_hint="${p.hint}"` : ""})
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
