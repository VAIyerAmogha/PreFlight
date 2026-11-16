"use client";

import { useEffect, useState } from "react";
import { useReducedMotion, useVisible } from "@/lib/hooks";
import { Figure } from "../ui";

// Step lists from docs/fit_hygiene.md (steps 1-5 run on the table, 6-22 in the pipeline).
const TABLE_STEPS = ["convert_missing_markers", "coerce_numeric", "remove_duplicates", "drop_column", "normalize_categories"];
const PIPE_STEPS = [
  "group_rare_categories", "add_missing_indicator", "fill_median / fill_mode", "constant_fill", "clip_outliers",
  "drop_leaky_feature", "feature_overlap (VIF)", "expand_datetime", "ordinal_encode", "one_hot_encode",
  "target_encode_cross_fit", "log1p_transform", "standard_scale", "create_interactions", "create_datetime_features",
  "create_cluster_features", "create_text_features",
];

export default function HygieneFigure({ rows }: { rows: number }) {
  const [fold, setFold] = useState(0);
  const [ref, visible] = useVisible<HTMLDivElement>();
  const reduced = useReducedMotion();
  useEffect(() => {
    if (!visible || reduced) return;
    const id = setInterval(() => setFold((f) => (f + 1) % 5), 1500);
    return () => clearInterval(id);
  }, [visible, reduced]);
  const test = Math.round(rows / 5);

  return (
    <Figure
      n={5}
      title="What runs where"
      caption={
        <>
          Left: steps applied to the table, which produce <code>apply_result.prepared</code>. Right: steps held in the returned pipeline. Below: in
          each fold a fresh clone is fitted on the blue rows only and scored on the orange ones. The full table with reasons is in{" "}
          <a href="https://github.com/VAIyerAmogha/PreFlight/blob/main/docs/fit_hygiene.md"><code>docs/fit_hygiene.md</code></a>.
        </>
      }
    >
      <div ref={ref}>
        <div className="hygiene">
          <div className="lane table">
            <h5>Runs on the table <span className="badge">learns nothing</span></h5>
            <p className="why">Each value is handled on its own, so nothing leaks between rows.</p>
            <ol>{TABLE_STEPS.map((s, i) => <li key={s}><b>{i + 1}</b>{s}</li>)}</ol>
          </div>
          <div className="lane pipe">
            <h5>Goes in the pipeline <span className="badge ask">fit per fold</span></h5>
            <p className="why">Medians, limits, encodings and vocabularies come from training rows only.</p>
            <ol style={{ columnGap: 12 }}>{PIPE_STEPS.map((s, i) => <li key={s}><b>{i + 6}</b>{s}</li>)}</ol>
          </div>
        </div>
        <div className="folds" aria-label="Five-fold cross-validation">
          {[0, 1, 2, 3, 4].map((f) => (
            <div key={f} className={`fold${f === fold ? "" : " dim"}`}>
              <span>fold {f + 1}</span>
              {[0, 1, 2, 3, 4].map((s) => <span key={s} className={`seg5${s === f ? " test" : ""}`} />)}
              <span className="res">{f === fold ? `clone().fit(${rows - test} rows) → score ${test}` : ""}</span>
            </div>
          ))}
        </div>
      </div>
    </Figure>
  );
}
