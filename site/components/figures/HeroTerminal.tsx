"use client";

import { useEffect, useState } from "react";
import type { Decision } from "@/lib/types";
import { useInView, useReducedMotion } from "@/lib/hooks";

// Figure 0: the exact loop from the README's "Look before you leap", with its
// real output (from the fixture) revealed line by line.
const pad = (s: string, n: number) => (s.length >= n ? s + " " : s + " ".repeat(n - s.length));
const pyList = (cols: string[]) => `[${cols.map((c) => `'${c}'`).join(", ")}]`;

export default function HeroTerminal({ decisions, file, version }: { decisions: Decision[]; file: string; version: string }) {
  const [ref, seen] = useInView<HTMLDivElement>("0px");
  const reduced = useReducedMotion();
  const [shown, setShown] = useState(0);
  const total = decisions.length + 4;

  useEffect(() => {
    if (!seen) return;
    if (reduced) { setShown(total); return; }
    // Commands appear slowly, then the output streams quickly.
    let i = 0;
    let id: ReturnType<typeof setTimeout>;
    const tick = () => {
      i += 1;
      setShown(i);
      if (i < total) id = setTimeout(tick, i < 2 ? 520 : i === 2 ? 380 : 85);
    };
    id = setTimeout(tick, 300);
    return () => clearTimeout(id);
  }, [seen, reduced, total]);

  const on = (k: number) => `ln${shown > k ? " on" : ""}`;
  return (
    <div className="term" aria-label={`Output of preflight.plan on ${file}`}>
      <div className="term-head">
        <i /><i /><i />
        <span className="title">python · {file}</span>
        <span className="right">pypreflight {version}</span>
      </div>
      <div className="term-body" ref={ref}>
        <div className={`${on(0)} cmd`}>
          <span className="c">&gt;&gt;&gt; </span>p = preflight.<span className="k">plan</span>(<span className="s">&quot;{file}&quot;</span>, target=<span className="s">&quot;outcome&quot;</span>)
        </div>
        <div className={`${on(1)} cmd`}>
          <span className="c">&gt;&gt;&gt; </span><span className="k">for</span> d <span className="k">in</span> p.decisions: print(d.stage, d.kind, d.columns, d.action, d.confidence, <span className="s">&quot;ASK&quot;</span> <span className="k">if</span> d.needs_attention <span className="k">else</span> <span className="s">&quot;&quot;</span>)
        </div>
        <div className={on(2)}> </div>
        {decisions.map((d, i) => (
          <div key={d.id} className={on(i + 3)}>
            <span className="stage">{pad(d.stage, 9)}</span>
            <span className="act">{pad(d.kind, 22)}{pad(pyList(d.columns), 15)}{pad(d.action, 26)}</span>
            <span className="conf">{d.confidence.toFixed(2)} </span>
            {d.needs_attention && <span className="ask">ASK</span>}
          </div>
        ))}
        <div className={on(decisions.length + 3)}>
          <span className="c">&gt;&gt;&gt; </span><span className="cursor" />
        </div>
      </div>
    </div>
  );
}
