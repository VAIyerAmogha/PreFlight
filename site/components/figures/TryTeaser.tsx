"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { startEngine, useEngine } from "@/lib/engine";
import { useReducedMotion, useVisible } from "@/lib/hooks";
import type { Fixture } from "@/lib/types";

const STEPS = ["Data", "Read", "Types", "Decisions", "Questions", "Apply", "Results"];

/** Home-page teaser for /try: a real before→after slice that sweeps column by column. */
export default function TryTeaser({ fx }: { fx: Fixture }) {
  const e = useEngine();
  const [ref, visible] = useVisible<HTMLDivElement>();
  const reduced = useReducedMotion();
  const [col, setCol] = useState(-1);
  const p = fx.result.prepared;
  const raw = fx.result.raw_table!;
  const cols = p.columns.filter((c) => raw.columns.includes(c)).slice(0, 4);
  const rows = p.index.slice(0, 7);
  const rawRow = new Map(raw.index.map((ix, i) => [String(ix), raw.rows[i]]));
  const ch = fx.result.changes;
  const has = (m: Record<string, (string | number)[]>, c: string, ix: string | number) => (m[c] ?? []).some((x) => String(x) === String(ix));

  useEffect(() => {
    if (!visible || reduced) { if (reduced) setCol(cols.length); return; }
    const id = setInterval(() => setCol((c) => (c >= cols.length + 1 ? -1 : c + 1)), 900);
    return () => clearInterval(id);
  }, [visible, reduced, cols.length]);

  const cell = (ix: string | number, c: string) => {
    const after = p.rows[p.index.indexOf(ix)][p.columns.indexOf(c)];
    const before = rawRow.get(String(ix))?.[raw.columns.indexOf(c)];
    return { before: before ?? "", after };
  };
  const shownAfter = (k: number) => k < col;
  const fmt = (v: unknown) => (v === null ? "NA" : typeof v === "boolean" ? (v ? "True" : "False") : String(v));

  return (
    <div className="teaser" ref={ref}>
      <div className="teaser-copy">
        <div className="step-kicker">Playground</div>
        <h3>Run it on your own file, step by step.</h3>
        <p>
          A guided workspace with the real package inside the page. Guess the separator and the types before PreFlight does, answer its questions, watch every changed cell
          light up, then change an answer and compare runs.
        </p>
        <ol className="teaser-steps">{STEPS.map((s, i) => <li key={s}><span>{i + 1}</span>{s}</li>)}</ol>
        <div className="cta-row">
          <Link className="btn primary big" href="/try" onPointerEnter={() => startEngine().catch(() => {})}>Open the playground →</Link>
          <span className="source">{e.phase === "ready" ? "Python is already warm" : "Your file never leaves the browser"}</span>
        </div>
      </div>
      <div className="teaser-demo" aria-hidden="true">
        <div className="panel-title"><span>{fx.sample}</span><span className="source">as read → prepared</span></div>
        <table className="data mini">
          <thead><tr>{cols.map((c, k) => <th key={c} className={k === col ? "sweep" : ""}>{c}</th>)}</tr></thead>
          <tbody>
            {rows.map((ix) => (
              <tr key={String(ix)}>
                {cols.map((c, k) => {
                  const { before, after } = cell(ix, c);
                  const show = shownAfter(k);
                  const v = show ? fmt(after) : String(before);
                  const mark = !show ? "" : has(ch.blanked, c, ix) ? " blank" : has(ch.changed, c, ix) ? " chg" : "";
                  return <td key={c} className={`${k === col ? "sweep" : ""}${mark}`}>{v}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
