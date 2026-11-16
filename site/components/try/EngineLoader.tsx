"use client";

import { useEffect, useState } from "react";
import { PYODIDE_VERSION } from "@/lib/config";
import { ENGINE_PACKAGES, useEngine } from "@/lib/engine";

function useElapsed(running: boolean, startedAt: number | null) {
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (!running || startedAt === null) return;
    const id = setInterval(() => setNow((performance.now() - startedAt) / 1000), 100);
    return () => clearInterval(id);
  }, [running, startedAt]);
  return now;
}

/** Big loading visual for the Data step: every package ticks off as it lands. */
export default function EngineLoader() {
  const e = useEngine();
  const loading = e.phase === "loading" || e.phase === "idle";
  const elapsed = useElapsed(loading, e.startedAt);
  const pct = e.phase === "ready" ? 100 : Math.round((e.step / Math.max(1, e.total)) * 100);
  const loaded = new Set(e.loaded);
  const installing = e.step >= ENGINE_PACKAGES.length + 1;

  return (
    <div className={`engine-loader${e.phase === "ready" ? " ready" : ""}`}>
      <div className="el-head">
        <div>
          <div className="el-k">{e.phase === "ready" ? "Python engine ready" : e.phase === "error" ? "Engine failed" : "Starting Python in your browser"}</div>
          <div className="el-sub">
            Pyodide {PYODIDE_VERSION} · CPython 3.12 on WebAssembly · {e.phase === "ready" ? `ready in ${e.readyIn?.toFixed(1)} s` : `${elapsed.toFixed(1)} s`}
          </div>
        </div>
        <div className="el-pct">{pct}<small>%</small></div>
      </div>
      <div className="el-bar"><i style={{ width: `${pct}%` }} /></div>
      <div className="el-grid">
        {ENGINE_PACKAGES.map((p) => {
          const ok = e.phase === "ready" || loaded.has(p);
          return (
            <div key={p} className={`el-pkg${ok ? " ok" : ""}`}>
              <span className="tick">{ok ? "✓" : ""}</span>
              {p}
            </div>
          );
        })}
        <div className={`el-pkg wide${e.phase === "ready" ? " ok" : installing ? " busy" : ""}`}>
          <span className="tick">{e.phase === "ready" ? "✓" : ""}</span>
          pypreflight {e.versions?.preflight ?? ""}
        </div>
      </div>
      <p className="el-note">
        {e.phase === "ready"
          ? "Everything below runs on your machine. Nothing you load here is uploaded."
          : "About 40 MB on the first visit, then cached by your browser. You can pick a file while this loads."}
      </p>
    </div>
  );
}

/** Compact engine status for the step rail. */
export function EngineMini() {
  const e = useEngine();
  const pct = e.phase === "ready" ? 100 : Math.round((e.step / Math.max(1, e.total)) * 100);
  return (
    <div className={`rail-engine ${e.phase}`}>
      <div className="row">
        <span className="dot" />
        <span>{e.phase === "ready" ? `pypreflight ${e.versions?.preflight}` : e.phase === "error" ? "Engine failed" : `Python ${pct}%`}</span>
      </div>
      <div className="mini-bar"><i style={{ width: `${pct}%` }} /></div>
      <div className="sub">{e.phase === "ready" ? `pandas ${e.versions?.pandas} · in-browser` : e.text}</div>
    </div>
  );
}
