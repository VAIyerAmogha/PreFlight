"use client";

import type { Question } from "@/lib/types";

export default function QuestionCard({
  q, selected, onSelect, name,
}: { q: Question; selected: string; onSelect: (option: string) => void; name?: string }) {
  return (
    <div className="qcard">
      <div>
        <span className="stage-tag infer">
          {q.category}
          {q.column ? ` · ${q.column}` : ""}
        </span>
        <div className="q">{q.text}</div>
        <p className="expl">{q.explanation}</p>
        <p className="ev">{q.evidence_summary}</p>
        <div className="opts" role="radiogroup" aria-label={q.text}>
          {q.options.map((o) => (
            <label key={o} className="opt">
              <input type="radio" name={name ?? q.question_id} value={o} checked={o === selected} onChange={() => onSelect(o)} />
              <span>{o}</span>
              {o === q.recommended_default && <span className="badge ok rec">recommended</span>}
            </label>
          ))}
        </div>
      </div>
      <div className="pv">
        <PreviewTable key={selected} q={q} option={selected} />
      </div>
    </div>
  );
}

function PreviewTable({ q, option }: { q: Question; option: string }) {
  const p = q.preview;
  if (!p || !p.before || !p.after) {
    const act = q.option_actions?.[option];
    return (
      <>
        <h4>What this option does</h4>
        <p className="expl">
          Applies the action <code>{act ?? option}</code>
          {q.column && (
            <>
              {" "}to <code>{q.column}</code>
            </>
          )}
          .
        </p>
      </>
    );
  }
  const after = p.after[option] ?? [];
  return (
    <>
      <table className="preview-tbl">
        <thead>
          <tr>
            <th>In your file</th>
            <th />
            <th>After “{option}”</th>
          </tr>
        </thead>
        <tbody>
          {p.before.map((b, i) => {
            const a = after[i] ?? "";
            const lost = /left blank/i.test(String(a));
            return (
              <tr key={i}>
                <td>{b}</td>
                <td className="arrow">→</td>
                <td className={`after flash${lost ? " lost" : ""}`} style={{ animationDelay: `${i * 60}ms` }}>
                  {a}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {p.note && <p className="preview-note">{p.note}</p>}
    </>
  );
}
