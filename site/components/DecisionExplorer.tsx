"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Decision } from "@/lib/types";
import { conf, Meter } from "./ui";

export default function DecisionExplorer({ decisions }: { decisions: Decision[] }) {
  const stages = useMemo(() => [...new Set(decisions.map((d) => d.stage))], [decisions]);
  const nAsk = decisions.filter((d) => d.needs_attention).length;
  const [stage, setStage] = useState<string>("all");
  const [askOnly, setAskOnly] = useState(false);
  const [selected, setSelected] = useState<string | undefined>(() => decisions.find((d) => d.needs_attention)?.id ?? decisions[0]?.id);
  const listRef = useRef<HTMLDivElement>(null);

  const visible = decisions.filter((d) => (stage === "all" || d.stage === stage) && (!askOnly || d.needs_attention));
  const current = visible.find((d) => d.id === selected) ?? visible[0];

  useEffect(() => {
    if (!decisions.some((d) => d.id === selected)) setSelected(decisions.find((d) => d.needs_attention)?.id ?? decisions[0]?.id);
  }, [decisions, selected]);

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const i = visible.findIndex((d) => d.id === current?.id);
    const next = visible[Math.max(0, Math.min(visible.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)))];
    if (next) {
      setSelected(next.id);
      listRef.current?.querySelector<HTMLButtonElement>(`[data-id="${CSS.escape(next.id)}"]`)?.focus();
    }
  };

  return (
    <div className="explorer">
      <div className="list">
        <div className="filters">
          <button className="chip" aria-pressed={stage === "all" && !askOnly} onClick={() => { setStage("all"); setAskOnly(false); }}>
            All <span className="count">{decisions.length}</span>
          </button>
          <button className="chip" aria-pressed={askOnly} onClick={() => setAskOnly((v) => !v)}>
            Needs a person <span className="count">{nAsk}</span>
          </button>
          {stages.map((s) => (
            <button key={s} className="chip" aria-pressed={stage === s} onClick={() => setStage(stage === s ? "all" : s)}>
              {s} <span className="count">{decisions.filter((d) => d.stage === s).length}</span>
            </button>
          ))}
        </div>
        <div role="listbox" aria-label="Decisions" ref={listRef} onKeyDown={onKey}>
          {visible.map((d) => (
            <button key={d.id} data-id={d.id} className="drow" role="option" aria-selected={d.id === current?.id} onClick={() => setSelected(d.id)}>
              <span className={`stage-tag ${d.stage}`}>{d.stage}</span>
              <span className="what">{d.action}{d.needs_attention && <span className="badge ask">ASK</span>}</span>
              <span className="cf">
                {conf(d.confidence)}
                <Meter value={d.confidence} />
              </span>
              <span className="cols">
                {d.columns.length ? d.columns.join(", ") : "whole file"} · {d.kind}
              </span>
            </button>
          ))}
          {visible.length === 0 && <div className="pg-empty">No decisions match.</div>}
        </div>
      </div>
      <div className="detail" aria-live="polite">{current && <DecisionDetail key={current.id} d={current} />}</div>
    </div>
  );
}

export function DecisionDetail({ d }: { d: Decision }) {
  const [grown, setGrown] = useState(false);
  useEffect(() => {
    const t = requestAnimationFrame(() => setGrown(true));
    return () => cancelAnimationFrame(t);
  }, []);
  const ev = d.evidence ?? {};
  const scores = ev.candidate_scores as Record<string, number> | undefined;
  const entries = scores ? Object.entries(scores).sort((a, b) => b[1] - a[1]) : [];
  const top = entries[0]?.[1];
  const simple = Object.entries(ev).filter(([k, v]) => k !== "candidate_scores" && (v === null || typeof v !== "object"));

  return (
    <div className="fade-swap">
      <p className="quote">{d.plain_language}</p>
      <dl className="kv">
        <dt>action</dt><dd>{d.action}</dd>
        <dt>default</dt><dd>{d.default}</dd>
        <dt>confidence</dt><dd>{conf(d.confidence)}</dd>
        <dt>needs_attention</dt><dd>{d.needs_attention ? <span className="badge ask">True</span> : "False"}</dd>
        <dt>status</dt><dd>{d.status}</dd>
        <dt>columns</dt><dd>{JSON.stringify(d.columns)}</dd>
      </dl>
      {entries.length > 0 && (
        <>
          <h4>Candidate type scores</h4>
          <div className="bars">
            {entries.map(([k, v]) => (
              <div key={k} className={`bar${v === top && v > 0 ? " top" : ""}`}>
                <span>{k}</span>
                <span className="track"><i style={{ width: grown ? `${Math.round(v * 100)}%` : 0 }} /></span>
                <span className="v">{conf(v)}</span>
              </div>
            ))}
          </div>
        </>
      )}
      {simple.length > 0 && (
        <>
          <h4>Evidence</h4>
          <dl className="kv">
            {simple.map(([k, v]) => (
              <Frag key={k} k={k} v={typeof v === "number" ? String(+v.toFixed(4)) : String(v)} />
            ))}
          </dl>
        </>
      )}
      <details className="json">
        <summary>Raw Decision JSON</summary>
        <pre>{JSON.stringify(d, null, 2)}</pre>
      </details>
    </div>
  );
}

function Frag({ k, v }: { k: string; v: string }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </>
  );
}
