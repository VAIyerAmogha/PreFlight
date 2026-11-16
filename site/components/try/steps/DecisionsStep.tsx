"use client";

import { useMemo, useState } from "react";
import type { Analysis, Decision } from "@/lib/types";
import DecisionExplorer, { DecisionDetail } from "../../DecisionExplorer";
import { Seg } from "../../ui";
import { CountUp, Stat, StepHeader } from "./common";

const LANES = [
  { id: "reading", label: "Reading", sub: "file layout" },
  { id: "infer", label: "Typing", sub: "per column" },
  { id: "clean", label: "Cleaning", sub: "markers, fills, drops" },
  { id: "engineer", label: "Engineering", sub: "encoders, features" },
];
const BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001];

export default function DecisionsStep({ a }: { a: Analysis }) {
  const [view, setView] = useState<"map" | "list">("map");
  const [sel, setSel] = useState<string>(() => a.decisions.find((d) => d.needs_attention)?.id ?? a.decisions[0]?.id);
  const [hover, setHover] = useState<Decision | null>(null);
  const ds = a.decisions;
  const nAsk = ds.filter((d) => d.needs_attention).length;
  const mean = ds.reduce((s, d) => s + d.confidence, 0) / Math.max(1, ds.length);
  const lanes = useMemo(() => {
    const known = new Set(LANES.map((l) => l.id));
    const extra = [...new Set(ds.map((d) => d.stage))].filter((s) => !known.has(s)).map((s) => ({ id: s, label: s, sub: "" }));
    return [...LANES, ...extra].filter((l) => ds.some((d) => d.stage === l.id));
  }, [ds]);
  const hist = BINS.map((hi, i) => {
    const lo = i === 0 ? 0 : BINS[i - 1];
    const inBin = ds.filter((d) => d.confidence >= lo && d.confidence < hi);
    return { label: i === 0 ? "< .5" : `${lo.toFixed(1)}–${Math.min(hi, 1).toFixed(1)}`, n: inBin.length, ask: inBin.filter((d) => d.needs_attention).length };
  });
  const maxBin = Math.max(1, ...hist.map((h) => h.n));
  const current = ds.find((d) => d.id === sel);

  return (
    <>
      <StepHeader n={4} title="The whole plan, before anything changes" aside={<Seg<"map" | "list"> label="View" value={view} onChange={setView} options={[{ v: "map", label: "Map" }, { v: "list", label: "List" }]} />}>
        Every proposal <code>plan()</code> made, from file layout to encoders. Each square is a Decision; darker means more confident, and inverted ones need a person.
        Nothing has been applied yet.
      </StepHeader>

      <div className="stats-row">
        <Stat k="Decisions"><CountUp value={ds.length} /></Stat>
        <Stat k="Need a person"><CountUp value={nAsk} /></Stat>
        <Stat k="Mean confidence"><CountUp value={mean} decimals={2} /></Stat>
        <Stat k="Stages">{lanes.length}</Stat>
      </div>

      {view === "list" ? (
        <div className="fig" style={{ boxShadow: "none" }}><DecisionExplorer decisions={ds} /></div>
      ) : (
        <div className="dmap-wrap">
          <div className="dmap">
            {lanes.map((l, li) => (
              <div key={l.id} className="lane-row">
                <div className="lane-label"><b>{l.label}</b><small>{l.sub}</small></div>
                <div className="lane-cells">
                  {ds.filter((d) => d.stage === l.id).map((d, i) => (
                    <button
                      key={d.id}
                      className={`dcell${d.needs_attention ? " ask" : ""}${d.id === sel ? " sel" : ""}`}
                      style={{ ["--c" as string]: d.confidence, animationDelay: `${li * 120 + i * 22}ms` }}
                      onClick={() => setSel(d.id)}
                      onMouseEnter={() => setHover(d)}
                      onMouseLeave={() => setHover(null)}
                      onFocus={() => setHover(d)}
                      onBlur={() => setHover(null)}
                      aria-label={`${d.action} on ${d.columns.join(", ") || "file"}, confidence ${d.confidence}`}
                    />
                  ))}
                </div>
              </div>
            ))}
            <div className="dmap-hover">
              {hover ? <><code>{hover.action}</code> · {hover.columns.join(", ") || "whole file"} · {hover.confidence.toFixed(2)}</> : <span className="source">Hover a square · click to pin</span>}
            </div>
            <div className="hist">
              <div className="hist-title">Confidence distribution</div>
              <div className="hist-bars">
                {hist.map((h, i) => (
                  <div key={h.label} className="hb">
                    <div className="hb-col">
                      <i className="all grow-y" style={{ height: `${(h.n / maxBin) * 100}%`, animationDelay: `${i * 70}ms` }} />
                      <i className="ask grow-y" style={{ height: `${(h.ask / maxBin) * 100}%`, animationDelay: `${i * 70}ms` }} />
                    </div>
                    <span className="hb-n">{h.n}</span>
                    <span className="hb-l">{h.label}</span>
                  </div>
                ))}
              </div>
              <div className="hist-legend"><span><i className="all" />all</span><span><i className="ask" />needs a person</span></div>
            </div>
          </div>
          <div className="dmap-detail">{current && <DecisionDetail key={current.id} d={current} />}</div>
        </div>
      )}
    </>
  );
}
