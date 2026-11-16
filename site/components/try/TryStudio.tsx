"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { MAX_UPLOAD_MB } from "@/lib/config";
import { callRunner, EngineError, readFiles, startEngine, useEngine, writeFile } from "@/lib/engine";
import type { Analysis, RunResult } from "@/lib/types";
import { fmtBytes } from "../ui";
import { EngineMini } from "./EngineLoader";
import ApplyStep from "./steps/ApplyStep";
import DataStep from "./steps/DataStep";
import DecisionsStep from "./steps/DecisionsStep";
import QuestionsStep from "./steps/QuestionsStep";
import ReadStep from "./steps/ReadStep";
import ResultsStep from "./steps/ResultsStep";
import TypesStep from "./steps/TypesStep";

export type StepId = "data" | "read" | "types" | "decisions" | "questions" | "apply" | "results";
export type Task = "" | "classification" | "regression";
export type Hint = "tree" | "linear";
export interface Source { name: string; path: string; size: number; sample: boolean; rows: number }
export interface OutFile { key: string; name: string; url: string; size: number; ext: string }
export interface Run {
  id: number;
  result: RunResult;
  files: OutFile[];
  changedAnswers: Record<string, string>;
  settings: { target: string; task: Task; hint: Hint };
  seconds: number;
}
export interface Failure { message: string; traceback?: string }

export const STEPS: { id: StepId; label: string; sub: string }[] = [
  { id: "data", label: "Data", sub: "Pick a file and a target" },
  { id: "read", label: "Read", sub: "Separator & header" },
  { id: "types", label: "Types", sub: "One per column" },
  { id: "decisions", label: "Decisions", sub: "The whole plan" },
  { id: "questions", label: "Questions", sub: "Your call" },
  { id: "apply", label: "Apply", sub: "What changed" },
  { id: "results", label: "Results", sub: "Pipeline, files, runs" },
];

const ORDER = ["report.pdf", "plan.pdf", "cleaned_data.csv", "decisions.json", "schema.json", "plan.json", "pipeline.joblib", "pipeline_fitted_on_all_rows.joblib", "answers.json"];
const safeName = (n: string) => n.replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 80) || "upload.csv";

export default function TryStudio() {
  const engine = useEngine();
  const [step, setStep] = useState<StepId>("data");
  const [source, setSource] = useState<Source | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [target, setTarget] = useState("");
  const [task, setTask] = useState<Task>("");
  const [hint, setHint] = useState<Hint>("tree");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [planSeconds, setPlanSeconds] = useState<number>(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [runs, setRuns] = useState<Run[]>([]);
  const [busy, setBusy] = useState<null | "file" | "plan" | "run">(null);
  const [error, setError] = useState<Failure | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [visited, setVisited] = useState<Set<StepId>>(() => new Set(["data"]));
  const outSeq = useRef(0);
  const stageRef = useRef<HTMLDivElement>(null);
  const runsRef = useRef<Run[]>([]);
  runsRef.current = runs;

  useEffect(() => { startEngine().catch(() => {}); }, []);
  useEffect(() => () => runsRef.current.forEach((r) => r.files.forEach((f) => URL.revokeObjectURL(f.url))), []);
  useEffect(() => { if (!toast) return; const t = setTimeout(() => setToast(null), 4200); return () => clearTimeout(t); }, [toast]);

  const fail = (e: unknown) => {
    const err = e as EngineError;
    setError({ message: err?.message ?? String(e), traceback: err?.traceback });
  };

  const go = useCallback((s: StepId) => {
    setStep(s);
    setVisited((v) => (v.has(s) ? v : new Set(v).add(s)));
    stageRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  // ------------------------------------------------------------- data
  const clearDownstream = () => { setAnalysis(null); setAnswers({}); setVisited(new Set(["data"])); };

  const loadBytes = useCallback(async (name: string, data: ArrayBuffer, sample: boolean, defaultTarget?: string) => {
    setError(null); setBusy("file"); clearDownstream();
    const path = `/data/in/${safeName(name)}`;
    const size = data.byteLength; // read before the buffer is transferred to the worker
    try {
      await writeFile(path, data);
      const info = await callRunner<{ columns: string[]; rows: number }>("inspect", path);
      runsRef.current.forEach((r) => r.files.forEach((f) => URL.revokeObjectURL(f.url)));
      setRuns([]);
      setColumns(info.columns);
      setSource({ name: safeName(name), path, size, sample, rows: info.rows });
      setTarget(defaultTarget && info.columns.includes(defaultTarget) ? defaultTarget : info.columns[info.columns.length - 1] ?? "");
    } catch (e) { fail(e); } finally { setBusy(null); }
  }, []);

  const pickSample = async (file: string, defaultTarget: string) => {
    try { await loadBytes(file, await (await fetch(`/samples/${file}`)).arrayBuffer(), true, defaultTarget); } catch (e) { fail(e); }
  };
  const pickFile = async (f: File | undefined) => {
    if (!f) return;
    if (f.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError({ message: `${f.name} is ${fmtBytes(f.size)}. The browser demo is limited to ${MAX_UPLOAD_MB} MB; use pip install --pre pypreflight for larger files.` });
      return;
    }
    await loadBytes(f.name, await f.arrayBuffer(), false);
  };

  const changeSetting = <T,>(set: (v: T) => void) => (v: T) => { set(v); clearDownstream(); };

  // ------------------------------------------------------------- plan
  const analyze = async () => {
    if (!source || !target) return;
    setBusy("plan"); setError(null);
    const t0 = performance.now();
    try {
      const a = await callRunner<Analysis>("analyze", source.path, target, task, hint);
      setAnalysis(a);
      setPlanSeconds((performance.now() - t0) / 1000);
      setAnswers(Object.fromEntries(a.questions.map((q) => [q.question_id, q.recommended_default])));
      go("read");
    } catch (e) { fail(e); } finally { setBusy(null); }
  };

  // ------------------------------------------------------------- prepare
  const runPrepare = async () => {
    if (!source || !analysis) return;
    setBusy("run"); setError(null);
    const id = ++outSeq.current;
    const outDir = `/data/out/${id}`;
    const t0 = performance.now();
    try {
      const changed = Object.fromEntries(
        analysis.questions.filter((q) => answers[q.question_id] && answers[q.question_id] !== q.recommended_default).map((q) => [q.question_id, answers[q.question_id]])
      );
      const r = await callRunner<RunResult>("run", source.path, target, JSON.stringify(changed), task, hint, outDir);
      const blobs = await readFiles([outDir]);
      const files: OutFile[] = Object.entries(blobs).map(([p, bytes]) => {
        const name = p.split("/").pop()!;
        const ext = name.includes(".") ? name.split(".").pop()! : "";
        const type = ext === "pdf" ? "application/pdf" : ext === "csv" ? "text/csv" : ext === "json" ? "application/json" : "application/octet-stream";
        return { key: p, name, ext, size: bytes.byteLength, url: URL.createObjectURL(new Blob([bytes as BlobPart], { type })) };
      }).sort((a, b) => ORDER.indexOf(a.name) - ORDER.indexOf(b.name));
      const seconds = (performance.now() - t0) / 1000;
      setRuns((rs) => [...rs, { id, result: r, files, changedAnswers: changed, settings: { target, task, hint }, seconds }]);
      setToast(`prepare() finished in ${seconds.toFixed(1)} s · ${files.length} files written`);
      go("apply");
    } catch (e) { fail(e); } finally { setBusy(null); }
  };

  // ------------------------------------------------------------- layout
  const hasPlan = !!analysis;
  const hasRun = runs.length > 0;
  const unlocked = (s: StepId) => s === "data" || (["read", "types", "decisions", "questions"].includes(s) ? hasPlan : hasPlan && hasRun);
  const idx = STEPS.findIndex((s) => s.id === step);
  const next = STEPS[idx + 1];
  const prev = STEPS[idx - 1];
  const run = runs[runs.length - 1];

  return (
    <div className="studio">
      <aside className="rail" aria-label="Steps">
        {source ? (
          <div className="rail-file">
            <div className="k">File</div>
            <div className="v">{source.name}</div>
            <div className="sub">{fmtBytes(source.size)} · {source.rows} rows · {columns.length} cols</div>
            {target && <div className="sub">target <b>{target}</b>{task ? ` · ${task}` : ""} · {hint}</div>}
          </div>
        ) : (
          <div className="rail-file empty"><div className="k">File</div><div className="sub">None yet: start with step 1</div></div>
        )}
        <ol className="rail-steps">
          {STEPS.map((s, i) => {
            const on = unlocked(s.id);
            const done = on && visited.has(s.id) && (s.id !== "data" || hasPlan) && (s.id !== "questions" || hasRun);
            const state = s.id === step ? "current" : done ? "done" : on ? "open" : "locked";
            return (
              <li key={s.id}>
                <button className={`rail-step ${state}`} disabled={!on} onClick={() => go(s.id)} aria-current={s.id === step ? "step" : undefined}>
                  <span className="n">{state === "done" ? "✓" : i + 1}</span>
                  <span className="t"><b>{s.label}</b><small>{s.sub}</small></span>
                </button>
              </li>
            );
          })}
        </ol>
        <EngineMini />
      </aside>

      <section className="stage" ref={stageRef} aria-live="polite">
        <div className="mobile-steps" role="tablist" aria-label="Steps">
          {STEPS.map((s, i) => (
            <button key={s.id} role="tab" aria-selected={s.id === step} disabled={!unlocked(s.id)} onClick={() => go(s.id)}>
              {i + 1}. {s.label}
            </button>
          ))}
        </div>

        {error && (
          <div className="pg-error" role="alert">
            {error.message}
            {error.traceback && (
              <details style={{ marginTop: 8 }}>
                <summary style={{ cursor: "pointer" }}>Python traceback</summary>
                <pre style={{ whiteSpace: "pre-wrap", margin: "8px 0 0", fontSize: 11 }}>{error.traceback}</pre>
              </details>
            )}
          </div>
        )}
        {engine.phase === "error" && <div className="pg-error" role="alert">The Python engine could not start: {engine.error}</div>}

        <div className="stage-body" key={step}>
          {step === "data" && (
            <DataStep
              source={source} columns={columns} target={target} task={task} hint={hint} busy={busy}
              onSample={pickSample} onFile={pickFile}
              setTarget={changeSetting(setTarget)} setTask={changeSetting(setTask)} setHint={changeSetting(setHint)}
              onAnalyze={analyze} analyzed={hasPlan}
            />
          )}
          {step === "read" && analysis && <ReadStep a={analysis} seconds={planSeconds} />}
          {step === "types" && analysis && <TypesStep a={analysis} />}
          {step === "decisions" && analysis && <DecisionsStep a={analysis} />}
          {step === "questions" && analysis && (
            <QuestionsStep a={analysis} answers={answers} setAnswers={setAnswers} onRun={runPrepare} running={busy === "run"} runs={runs.length} />
          )}
          {step === "apply" && analysis && run && <ApplyStep a={analysis} run={run} />}
          {step === "results" && analysis && run && (
            <ResultsStep a={analysis} runs={runs} onChangeAnswers={() => go("questions")} onChangeSettings={() => go("data")} />
          )}
        </div>

        <nav className="stage-nav" aria-label="Step navigation">
          {prev ? <button className="btn" onClick={() => go(prev.id)}>← {prev.label}</button> : <span />}
          {next && unlocked(next.id) && (
            <button className="btn primary" onClick={() => go(next.id)}>{next.label} →</button>
          )}
          {next && !unlocked(next.id) && step === "questions" && (
            <button className="btn primary" onClick={runPrepare} disabled={busy === "run"}>{busy === "run" ? "Running prepare()…" : "Run prepare() →"}</button>
          )}
        </nav>
      </section>

      {toast && <div className="toast" role="status">✓ {toast}</div>}
    </div>
  );
}
