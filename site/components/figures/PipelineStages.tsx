"use client";

import { useEffect, useState, type ReactNode } from "react";
import type { Fixture } from "@/lib/types";
import { useReducedMotion, useVisible } from "@/lib/hooks";
import { conf, delimiterLabel, Figure, TypeBadge } from "../ui";

interface Stage {
  key: string;
  name: string;
  sub: string;
  icon: ReactNode;
  title: string;
  body: string;
  out: string;
}

const I = (d: string) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
);

const STAGES: Stage[] = [
  { key: "read", name: "READ", sub: "encoding, separator, header", icon: I("M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9zM14 3v6h6M8 13h8M8 17h5"),
    title: "Read the file as it really is", body: "Detects the text encoding, the separator and which line (if any) holds column names. Title rows, blank rows and footers are found and skipped.", out: "→ raw table, every cell still text" },
  { key: "type", name: "TYPE", sub: "per-column inference", icon: I("M4 7V5h16v2M9 19h6M12 5v14"),
    title: "Score every candidate type", body: "For each column: find disguised missing values (-999, ?, N/A), score each candidate type, and pick a clear winner. If the margin is too thin, abstain and keep the text.", out: "→ TableSchema" },
  { key: "plan", name: "PLAN", sub: "simulated on a copy", icon: I("M9 11l3 3L22 4M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"),
    title: "Propose the cleaning and features", body: "Markers → units → duplicates → column drops → text tidying → fills → clipping → leakage → encoding, simulated on a private copy. Nothing is changed yet.", out: "→ Plan of ordered Decisions · plan.json · plan.pdf" },
  { key: "resolve", name: "RESOLVE", sub: "one batched review", icon: I("M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"),
    title: "Ask once", body: "Every Decision that needs a person goes into one review. Stored answers are reused by fingerprint; with nobody to ask, the run stops with a list instead of hanging.", out: "→ answers.json" },
  { key: "apply", name: "APPLY", sub: "table + unfitted pipeline", icon: I("M13 2 3 14h9l-1 8 10-12h-9z"),
    title: "Split by what learns from data", body: "Steps that learn nothing run on the table. Steps that learn a statistic go into an unfitted scikit-learn Pipeline for you to fit per fold.", out: "→ prepared DataFrame · Pipeline · report.pdf" },
];

export default function PipelineStages({ fx }: { fx: Fixture }) {
  const [active, setActive] = useState(0);
  const [auto, setAuto] = useState(true);
  const [ref, visible] = useVisible<HTMLDivElement>();
  const reduced = useReducedMotion();

  useEffect(() => {
    if (!auto || !visible || reduced) return;
    const id = setInterval(() => setActive((a) => (a + 1) % STAGES.length), 3600);
    return () => clearInterval(id);
  }, [auto, visible, reduced]);

  const s = STAGES[active];
  return (
    <Figure
      n={2}
      title="From file to pipeline"
      tools={<button className="chip" aria-pressed={auto} onClick={() => setAuto((v) => !v)}>{auto ? "Auto-play on" : "Auto-play off"}</button>}
      caption={<>The five stages, with what each one produced for <code>{fx.sample}</code>. Click a stage to pin it.</>}
    >
      <div ref={ref}>
        <div className="pipeline">
          <div className="track"><i style={{ width: `${(active / (STAGES.length - 1)) * 100}%` }} /></div>
          {STAGES.map((st, i) => (
            <button
              key={st.key}
              className={`stage-btn${i === active ? " active" : i < active ? " done" : ""}`}
              onClick={() => { setActive(i); setAuto(false); }}
              aria-pressed={i === active}
            >
              <span className="node">{st.icon}</span>
              <span className="nm">{st.name}</span>
              <span className="sub">{st.sub}</span>
            </button>
          ))}
        </div>
        <div className="stage-detail fade-swap" key={s.key}>
          <div>
            <h5>{s.title}</h5>
            <p>{s.body}</p>
            <div className="out">{s.out}</div>
          </div>
          <div className="ex">
            <div className="k">In this file</div>
            <StageExample k={s.key} fx={fx} />
          </div>
        </div>
      </div>
    </Figure>
  );
}

function StageExample({ k, fx }: { k: string; fx: Fixture }) {
  const a = fx.analysis;
  const r = fx.result;
  if (k === "read") {
    const rd = a.reading;
    return (
      <div style={{ display: "grid", gap: 6, fontFamily: "var(--mono)", fontSize: 12.5 }}>
        <div>encoding <b>{rd.encoding.name}</b> <span className="source">{conf(rd.encoding.confidence)}</span></div>
        <div>separator <b>{delimiterLabel(rd.delimiter.char)}</b> <span className="source">{conf(rd.delimiter.confidence)}</span></div>
        <div>header <b>{rd.header.row === null ? "none" : `row ${rd.header.row}`}</b> <span className="source">{conf(rd.header.confidence)}</span></div>
        <div className="source">{a.rows} rows × {a.columns} columns</div>
      </div>
    );
  }
  if (k === "type") {
    return (
      <div style={{ display: "grid", gap: 7 }}>
        {a.schema.map((c) => (
          <div key={c.name} style={{ display: "flex", gap: 8, alignItems: "center", fontFamily: "var(--mono)", fontSize: 12.5 }}>
            <span style={{ minWidth: 86 }}>{c.name}</span>
            <TypeBadge type={c.type} />
            <span className="source">{conf(c.confidence)}</span>
            {c.missing_markers.length > 0 && <span className="source" style={{ color: "var(--amber)" }}>missing: {c.missing_markers.map((m) => m.marker).join(", ")}</span>}
          </div>
        ))}
      </div>
    );
  }
  if (k === "plan") {
    const by: Record<string, number> = {};
    a.decisions.forEach((d) => (by[d.stage] = (by[d.stage] ?? 0) + 1));
    const ex = a.decisions.find((d) => d.stage === "clean");
    return (
      <>
        <div className="stage-sum">
          {Object.entries(by).map(([st, n]) => <span key={st} className="chip" style={{ cursor: "default" }}>{st} <span className="count">{n}</span></span>)}
        </div>
        {ex && <p style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontSize: 15, margin: 0 }}>“{ex.plain_language}”</p>}
      </>
    );
  }
  if (k === "resolve") {
    return (
      <>
        <div style={{ fontWeight: 600, marginBottom: 8 }}>{a.questions.length} questions, asked together</div>
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13 }}>
          {a.questions.slice(0, 5).map((q) => <li key={q.question_id}>{q.text}</li>)}
        </ul>
      </>
    );
  }
  const ch = r.changes;
  return (
    <div style={{ display: "grid", gap: 6, fontSize: 13 }}>
      <div><b>{r.prepared.shape[0]} × {r.prepared.shape[1]}</b> prepared table (from {ch.raw_rows} × {a.columns})</div>
      {ch.dropped_columns.length > 0 && <div>dropped: <code>{ch.dropped_columns.join(", ")}</code></div>}
      <div style={{ fontFamily: "var(--mono)", fontSize: 12 }}>
        Pipeline({r.pipeline.map((p) => p.class).join(" → ")})
      </div>
      {r.model_ready && !("error" in r.model_ready) && (
        <div className="source">model-ready columns: {r.model_ready.columns.join(", ")}</div>
      )}
    </div>
  );
}
