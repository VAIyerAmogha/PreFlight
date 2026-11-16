"use client";

import { useMemo, useState } from "react";
import type { Analysis, Frame } from "@/lib/types";
import { DataTable, DiffLegend, Seg } from "../../ui";
import type { Run } from "../TryStudio";
import { CountUp, Stat, StepHeader } from "./common";

type View = "changes" | "raw" | "model";

export default function ApplyStep({ a, run }: { a: Analysis; run: Run }) {
  const r = run.result;
  const ch = r.changes;
  const [view, setView] = useState<View>("changes");
  const [onlyChanged, setOnlyChanged] = useState(false);
  const [replay, setReplay] = useState(0);
  const nChanged = Object.values(ch.changed ?? {}).reduce((s, x) => s + x.length, 0);
  const nBlank = Object.values(ch.blanked ?? {}).reduce((s, x) => s + x.length, 0);

  // Per-column counts for the bar chart (display only: counts of the runner's change lists).
  const perCol = useMemo(() => {
    const cols = new Set([...Object.keys(ch.changed ?? {}), ...Object.keys(ch.blanked ?? {})]);
    return [...cols]
      .map((c) => ({ c, changed: ch.changed?.[c]?.length ?? 0, blanked: ch.blanked?.[c]?.length ?? 0 }))
      .sort((x, y) => y.changed + y.blanked - (x.changed + x.blanked));
  }, [ch]);
  const maxCol = Math.max(1, ...perCol.map((p) => p.changed + p.blanked));

  // "Only rows with changes": filter the preview rows (display only).
  const prepared: Frame = useMemo(() => {
    if (!onlyChanged || !ch.available) return r.prepared;
    const hit = new Set<string>();
    for (const ids of [...Object.values(ch.changed), ...Object.values(ch.blanked)]) ids.forEach((x) => hit.add(String(x)));
    const keep = r.prepared.index.map((ix, k) => (hit.has(String(ix)) ? k : -1)).filter((k) => k >= 0);
    return { ...r.prepared, index: keep.map((k) => r.prepared.index[k]), rows: keep.map((k) => r.prepared.rows[k]) };
  }, [onlyChanged, ch, r.prepared]);

  return (
    <>
      <StepHeader n={6} title="What prepare() changed" aside={<button className="btn" onClick={() => setReplay((x) => x + 1)}>↻ Replay</button>}>
        The table below is <code>out[&quot;apply_result&quot;].prepared</code>, with every changed cell marked against the file as it was read. Hover a marked cell to see the
        original. This run took {run.seconds.toFixed(2)} s{Object.keys(run.changedAnswers).length ? ` with ${Object.keys(run.changedAnswers).length} of your answers` : " with the default answers"}.
      </StepHeader>

      <div className="stats-row" key={`s${replay}`}>
        <Stat k="Rows" sub={<>from {ch.raw_rows}</>}><CountUp value={r.prepared.shape[0]} /></Stat>
        <Stat k="Columns" sub={<>from {a.columns}</>}><CountUp value={r.prepared.shape[1]} /></Stat>
        <Stat k="Cells changed" sub={ch.available ? "in the preview" : "not traceable"}>{ch.available ? <CountUp value={nChanged} /> : "—"}</Stat>
        <Stat k="Became missing" sub={ch.available ? "disguised blanks" : "not traceable"}>{ch.available ? <CountUp value={nBlank} /> : "—"}</Stat>
        <Stat k="Dropped" sub={`${ch.dropped_columns.length} column(s)`}><CountUp value={ch.dropped_rows_total} /> <small>rows</small></Stat>
      </div>

      {perCol.length > 0 && (
        <div className="colbars" key={`b${replay}`}>
          <div className="panel-title"><span>Changes by column</span><span className="data-legend inline"><span><i className="l-chg" />changed</span><span><i className="l-blank" />became missing</span></span></div>
          {perCol.map((p, i) => (
            <div key={p.c} className="colbar">
              <span className="nm">{p.c}</span>
              <span className="tr">
                <i className="chg grow" style={{ width: `${(p.changed / maxCol) * 100}%`, animationDelay: `${i * 60}ms` }} />
                <i className="blank grow" style={{ width: `${(p.blanked / maxCol) * 100}%`, animationDelay: `${i * 60 + 120}ms` }} />
              </span>
              <span className="n">{p.changed + p.blanked}</span>
            </div>
          ))}
        </div>
      )}

      <div className="table-tools">
        <Seg<View> label="Table" value={view} onChange={setView} options={[{ v: "changes", label: "Prepared · changes" }, { v: "raw", label: "As read" }, { v: "model", label: "Model-ready" }]} />
        {view === "changes" && ch.available && (
          <label className="toggle"><input type="checkbox" checked={onlyChanged} onChange={(e) => setOnlyChanged(e.target.checked)} /> Only rows with changes</label>
        )}
      </div>
      <div className={`replay-wrap v-${view}`} key={`${view}${replay}${onlyChanged}`}>
        {view === "changes" && <DataTable frame={prepared} mode="prepared" changes={ch} raw={r.raw_table} maxHeight={520} />}
        {view === "raw" && <DataTable frame={r.raw_table} mode="raw" changes={ch} maxHeight={520} />}
        {view === "model" && (
          <>
            <DataTable frame={r.model_ready} mode="model" baseColumns={r.prepared.columns} maxHeight={520} />
            <p className="hint-line">This is <code>pipeline_fitted_on_all_rows.transform(prepared)</code>, shown for illustration. For evaluation, fit the unfitted pipeline inside each CV fold.</p>
          </>
        )}
      </div>
      <DiffLegend changes={ch} />
    </>
  );
}
