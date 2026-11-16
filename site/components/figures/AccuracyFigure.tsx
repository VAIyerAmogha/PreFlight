"use client";

import { useInView } from "@/lib/hooks";
import { Figure } from "../ui";

// Numbers from the README's Accuracy table (254 files, 2,850 labelled columns).
const METRICS: { label: string; v1: number | null; v2: number; max: number; unit: string; lower?: boolean; fmt?: (x: number) => string }[] = [
  { label: "Files that crash", v1: 10, v2: 0, max: 254, unit: "/ 254", lower: true, fmt: (x) => `${x}` },
  { label: "Column type exactly right", v1: 80.77, v2: 91.23, max: 100, unit: "%" },
  { label: "Column type right or flagged for review", v1: null, v2: 99.54, max: 100, unit: "%" },
  { label: "Wrong type with no flag (silent failures)", v1: 9.4, v2: 0.46, max: 12, unit: "%", lower: true },
  { label: "File format right (encoding + separator + header)", v1: 91.73, v2: 91.73, max: 100, unit: "%" },
];

export default function AccuracyFigure() {
  const [ref, seen] = useInView<HTMLDivElement>();
  return (
    <Figure
      n={6}
      title="v1.0.1 vs v2 on the 254-file corpus"
      caption={
        <>
          In plain terms, fewer than 1 column in 100 gets a wrong type without a warning. Caveat: the held-out part of the corpus was examined during
          development, so it is not a fully independent test. Labels are in{" "}
          <a href="https://github.com/VAIyerAmogha/PreFlight/tree/main/corpus/labels"><code>corpus/labels/</code></a>.
        </>
      }
    >
      <div className="acc" ref={ref}>
        {METRICS.map((m) => {
          const f = m.fmt ?? ((x: number) => `${x.toFixed(2)}`);
          return (
            <div className="acc-row" key={m.label}>
              <div className="lbl"><span>{m.label}</span><span className="better">{m.lower ? "lower is better" : "higher is better"}</span></div>
              {(["v1", "v2"] as const).map((k) => {
                const v = m[k];
                return (
                  <div key={k} className={`acc-bar ${k}`}>
                    <span>{k === "v1" ? "v1.0.1" : "v2"}</span>
                    <span className="t"><i style={{ width: seen && v !== null ? `${Math.max(0.6, (v / m.max) * 100)}%` : 0 }} /></span>
                    <span className="n">{v === null ? "n/a" : `${f(v)}${m.unit === "%" ? "%" : ` ${m.unit}`}`}</span>
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </Figure>
  );
}
