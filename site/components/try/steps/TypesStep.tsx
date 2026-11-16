"use client";

import { useMemo, useState } from "react";
import type { Analysis, ColumnInfo } from "@/lib/types";
import { conf, Meter, TypeBadge } from "../../ui";
import { StepHeader } from "./common";

const GUESSES = ["int", "float", "categorical", "boolean", "datetime", "ID", "text"];

/** Up to n distinct non-empty raw values per column, from the raw table preflight read. */
function samplesFor(a: Analysis, name: string, n = 4): string[] {
  const t = a.raw_table;
  if (!t) return [];
  const ci = t.columns.indexOf(name);
  if (ci < 0) return [];
  const seen: string[] = [];
  for (const row of t.rows) {
    const v = row[ci];
    if (v === null || v === "") continue;
    const s = String(v);
    if (!seen.includes(s)) seen.push(s);
    if (seen.length >= n) break;
  }
  return seen;
}

export default function TypesStep({ a }: { a: Analysis }) {
  const [guess, setGuess] = useState<Record<string, string>>({});
  const [flipped, setFlipped] = useState<Set<string>>(new Set());
  const [focus, setFocus] = useState<string | null>(null);
  const askCols = useMemo(() => new Set(a.questions.map((q) => q.column).filter(Boolean) as string[]), [a]);
  const samples = useMemo(() => Object.fromEntries(a.schema.map((c) => [c.name, samplesFor(a, c.name)])), [a]);

  const all = flipped.size === a.schema.length;
  const guessed = a.schema.filter((c) => guess[c.name] && flipped.has(c.name));
  const right = guessed.filter((c) => guess[c.name] === c.type).length;
  const flip = (n: string) => setFlipped((f) => new Set(f).add(n));
  const focusCol = a.schema.find((c) => c.name === focus) ?? null;

  return (
    <>
      <StepHeader
        n={3}
        title="A type for every column"
        aside={guessed.length > 0 ? <div className="score-chip">You matched <b>{right}</b>/{guessed.length}</div> : undefined}
      >
        Each card shows real values from the file. Guess a type, then flip it to see what PreFlight inferred and how sure it was. An inverted <span className="badge ask">ASK</span>{" "}
        means it wants a person to confirm.
      </StepHeader>

      <div className="types-bar">
        <span className="source">{flipped.size}/{a.schema.length} revealed · {Object.keys(guess).length} guessed</span>
        <button className="btn" onClick={() => setFlipped(new Set(a.schema.map((c) => c.name)))} disabled={all}>Reveal all</button>
      </div>

      <div className="flip-grid">
        {a.schema.map((c, i) => {
          const isFlipped = flipped.has(c.name);
          const g = guess[c.name];
          const ask = askCols.has(c.name) || c.abstained;
          return (
            <div key={c.name} className={`flip${isFlipped ? " flipped" : ""}${focus === c.name ? " focus" : ""}`} style={{ animationDelay: `${i * 35}ms` }}>
              <div className="flip-inner">
                <div className="face front">
                  <div className="fc-name" title={c.name}>{c.name}</div>
                  <ul className="fc-samples">{samples[c.name].map((s) => <li key={s}>{s}</li>)}{samples[c.name].length === 0 && <li className="na">(all blank)</li>}</ul>
                  <select aria-label={`Your guess for ${c.name}`} value={g ?? ""} onChange={(e) => setGuess((x) => ({ ...x, [c.name]: e.target.value }))}>
                    <option value="">Your guess…</option>
                    {GUESSES.map((t) => <option key={t} value={t}>{t}</option>)}
                  </select>
                  <button className="flip-btn" onClick={() => flip(c.name)}>Flip ↻</button>
                </div>
                <div className="face back" onClick={() => setFocus(focus === c.name ? null : c.name)} role="button" tabIndex={isFlipped ? 0 : -1}
                     onKeyDown={(e) => { if (e.key === "Enter") setFocus(focus === c.name ? null : c.name); }}>
                  <div className="fc-top"><span className="fc-name" title={c.name}>{c.name}</span>{ask && <span className="badge ask">ASK</span>}</div>
                  <div className="fc-type"><TypeBadge type={c.type} /> <span className="source">{conf(c.confidence)}</span></div>
                  <Meter value={c.confidence} />
                  <div className="fc-meta">
                    {c.runner_up && <span>runner-up {c.runner_up}</span>}
                    {c.missing_markers.map((m) => <span key={m.marker} className="mk">“{m.marker}” = missing{m.count ? ` ×${m.count}` : ""}</span>)}
                  </div>
                  {g && <div className={`fc-you ${g === c.type ? "ok" : "miss"}`}>{g === c.type ? "✓ you got it" : `✗ you said ${g}`}</div>}
                  <span className="fc-more">details</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {focusCol && <ColumnDetail c={focusCol} samples={samples[focusCol.name]} onClose={() => setFocus(null)} />}
    </>
  );
}

function ColumnDetail({ c, samples, onClose }: { c: ColumnInfo; samples: string[]; onClose: () => void }) {
  const scores = c.candidate_scores ? Object.entries(c.candidate_scores).sort((x, y) => y[1] - x[1]) : [];
  const top = scores[0]?.[1];
  return (
    <div className="col-detail fade-swap">
      <div className="cd-head">
        <div><span className="step-kicker">Column</span><h3>{c.name}</h3></div>
        <button className="icon-btn" onClick={onClose} aria-label="Close">✕</button>
      </div>
      <div className="cd-grid">
        <div>
          <h4>Candidate type scores</h4>
          <div className="bars">
            {scores.map(([k, v], i) => (
              <div key={k} className={`bar${v === top && v > 0 ? " top" : ""}`}>
                <span>{k}</span>
                <span className="track"><i className="grow" style={{ width: `${Math.round(v * 100)}%`, animationDelay: `${i * 50}ms` }} /></span>
                <span className="v">{conf(v)}</span>
              </div>
            ))}
          </div>
          <p className="hint-line">Margin over the runner-up: <b>{conf(c.margin)}</b>. A thin margin is what makes PreFlight abstain and ask.</p>
        </div>
        <div>
          <h4>Values in the file</h4>
          <ul className="fc-samples big">{samples.map((s) => <li key={s}>{s}</li>)}</ul>
          {c.missing_markers.length > 0 && (
            <>
              <h4>Disguised missing values</h4>
              <ul className="plain">{c.missing_markers.map((m) => <li key={m.marker}><code>{m.marker}</code> {m.count ? `×${m.count}` : ""} <span className="source">{m.reason}</span></li>)}</ul>
            </>
          )}
          {c.anomalies.length > 0 && (
            <>
              <h4>Anomalies</h4>
              <ul className="plain">{c.anomalies.map((x) => <li key={x}>{x}</li>)}</ul>
            </>
          )}
          <p className="hint-line">{c.non_missing ?? "—"} of {c.total_rows ?? "—"} values present.</p>
        </div>
      </div>
    </div>
  );
}
