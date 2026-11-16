"use client";

import { useState } from "react";
import type { Analysis } from "@/lib/types";
import { CodeBlock, fmtBytes, Seg } from "../../ui";
import type { Run } from "../TryStudio";
import { StepHeader } from "./common";

const count = (m: Record<string, unknown[]> | undefined) => Object.values(m ?? {}).reduce((s, x) => s + x.length, 0);

export default function ResultsStep({ a, runs, onChangeAnswers, onChangeSettings }: { a: Analysis; runs: Run[]; onChangeAnswers: () => void; onChangeSettings: () => void }) {
  const [sel, setSel] = useState(runs[runs.length - 1].id);
  const run = runs.find((x) => x.id === sel) ?? runs[runs.length - 1];
  const r = run.result;
  const [pdf, setPdf] = useState<"report.pdf" | "plan.pdf">("report.pdf");
  const pdfFile = run.files.find((f) => f.name === pdf);
  const mr = r.model_ready && !("error" in r.model_ready) ? r.model_ready : null;
  const qText = Object.fromEntries(a.questions.map((q) => [q.question_id, q.column ?? q.category]));

  return (
    <>
      <StepHeader n={7} title="Pipeline, files and experiments">
        <code>prepare()</code> returns an <b>unfitted</b> scikit-learn Pipeline holding every step that learns from data, and writes everything below. Change an answer or a
        setting and run again: each run is kept so you can compare them.
      </StepHeader>

      <div className="flow" aria-label="Pipeline">
        <div className="fnode data"><small>input</small><b>prepared</b><span>{r.prepared.shape[0]} × {r.prepared.shape[1] - 1} + target</span></div>
        {r.pipeline.map((p) => (
          <div key={p.name} className="fseg">
            <span className="wire"><i /><i /><i /></span>
            <div className="fnode"><small>{p.name}</small><b>{p.class}</b><span>{p.module}</span></div>
          </div>
        ))}
        <div className="fseg">
          <span className="wire"><i /><i /><i /></span>
          <div className="fnode data"><small>output</small><b>model-ready</b><span>{mr ? `${mr.shape[0]} × ${mr.shape[1]}` : "—"}</span></div>
        </div>
      </div>

      <div className="results-grid">
        <div>
          <h3 className="sec">Files written by this run</h3>
          <div className="downloads">
            {run.files.map((f, i) => (
              <a key={f.key} className="dl" href={f.url} download={f.name} style={{ animationDelay: `${i * 40}ms` }}>
                <span className="ext">{f.ext}</span>
                <span className="nm">{f.name}</span>
                <span className="sz">{fmtBytes(f.size)}</span>
              </a>
            ))}
          </div>
          {pdfFile && (
            <>
              <div className="table-tools"><Seg<"report.pdf" | "plan.pdf"> label="PDF" value={pdf} onChange={setPdf} options={[{ v: "report.pdf", label: "Report PDF" }, { v: "plan.pdf", label: "Plan PDF" }]} /></div>
              <iframe className="pdf-frame" src={pdfFile.url} title={pdf} />
            </>
          )}
        </div>
        <div>
          <h3 className="sec">Experiments</h3>
          <div className="runs">
            <table className="runs-tbl">
              <thead><tr><th>Run</th><th>Answers</th><th>Rows</th><th>Changed</th><th>Model cols</th><th>Time</th></tr></thead>
              <tbody>
                {runs.map((x, k) => {
                  const prev = runs[k - 1];
                  const rows = x.result.prepared.shape[0];
                  const chg = count(x.result.changes.changed);
                  const cols = x.result.model_ready && !("error" in x.result.model_ready) ? x.result.model_ready.shape[1] : null;
                  const pcols = prev?.result.model_ready && !("error" in prev.result.model_ready) ? prev.result.model_ready.shape[1] : null;
                  const delta = (v: number | null, p: number | null | undefined) => (prev && v !== null && p !== null && p !== undefined && v !== p ? <span className={`delta ${v > p ? "up" : "down"}`}>{v > p ? "+" : ""}{v - p}</span> : null);
                  const nAns = Object.keys(x.changedAnswers).length;
                  return (
                    <tr key={x.id} className={x.id === run.id ? "on" : ""} onClick={() => setSel(x.id)}>
                      <td>#{k + 1}</td>
                      <td title={Object.entries(x.changedAnswers).map(([q, v]) => `${qText[q] ?? q}: ${v}`).join("\n")}>
                        {nAns ? `${nAns} changed` : "defaults"}
                        <div className="source">{x.settings.target} · {x.settings.task || "infer"} · {x.settings.hint}</div>
                      </td>
                      <td>{rows} {delta(rows, prev?.result.prepared.shape[0])}</td>
                      <td>{x.result.changes.available ? <>{chg} {delta(chg, prev ? count(prev.result.changes.changed) : null)}</> : "—"}</td>
                      <td>{cols ?? "—"} {delta(cols, pcols)}</td>
                      <td>{x.seconds.toFixed(1)}s</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="exp-actions">
            <button className="btn primary" onClick={onChangeAnswers}>Change answers & re-run</button>
            <button className="btn" onClick={onChangeSettings}>Try other settings</button>
          </div>
          <p className="hint-line">Tip: switch the model hint to <code>linear</code> in step 1 and compare the model-ready column count. One-hot encoding adds columns.</p>
          <h3 className="sec">The Python that ran</h3>
          <CodeBlock code={r.code} />
        </div>
      </div>
    </>
  );
}
