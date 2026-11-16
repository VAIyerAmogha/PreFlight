"use client";

import { useEffect, useState } from "react";
import type { Analysis } from "@/lib/types";
import QuestionCard from "../../QuestionCard";
import { StepHeader } from "./common";

interface Props {
  a: Analysis;
  answers: Record<string, string>;
  setAnswers: (f: (p: Record<string, string>) => Record<string, string>) => void;
  onRun: () => void;
  running: boolean;
  runs: number;
}

export default function QuestionsStep({ a, answers, setAnswers, onRun, running, runs }: Props) {
  const qs = a.questions;
  const [i, setI] = useState(0);
  const [seen, setSeen] = useState<Set<number>>(() => new Set([0]));
  const q = qs[i];
  const pick = (opt: string) => q && setAnswers((p) => ({ ...p, [q.question_id]: opt }));
  const goto = (n: number) => { const k = Math.max(0, Math.min(qs.length - 1, n)); setI(k); setSeen((s) => new Set(s).add(k)); };
  const changed = qs.filter((x) => answers[x.question_id] && answers[x.question_id] !== x.recommended_default);

  // Keyboard: 1-4 choose, arrows move, Enter = next (or run on the last card).
  useEffect(() => {
    if (!qs.length) return;
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el && (el.tagName === "SELECT" || el.tagName === "TEXTAREA" || (el.tagName === "INPUT" && (el as HTMLInputElement).type !== "radio"))) return;
      const n = Number(e.key);
      if (n >= 1 && n <= (q?.options.length ?? 0)) { pick(q!.options[n - 1]); e.preventDefault(); }
      else if (e.key === "ArrowRight") { goto(i + 1); e.preventDefault(); }
      else if (e.key === "ArrowLeft") { goto(i - 1); e.preventDefault(); }
      else if (e.key === "Enter" && el?.tagName !== "BUTTON") { if (i < qs.length - 1) goto(i + 1); else if (!running) onRun(); e.preventDefault(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!qs.length) {
    return (
      <>
        <StepHeader n={5} title="Nothing to ask">Every Decision was confident enough to make on its own. That is the common case for clean files.</StepHeader>
        <div className="empty-card">
          <p>Run <code>prepare()</code> to apply the plan and get the cleaned table, the pipeline and both PDFs.</p>
          <button className="btn primary big" onClick={onRun} disabled={running}>{running ? "Running prepare()…" : "Run prepare() →"}</button>
        </div>
      </>
    );
  }

  return (
    <>
      <StepHeader n={5} title="Your call" aside={<div className="score-chip">{changed.length} changed from default</div>}>
        PreFlight abstained on these instead of guessing. Each starts on the safe default; change any you disagree with. The preview shows real values under your choice.
        Your answers reach <code>prepare()</code> through the callback channel.
      </StepHeader>

      <div className="deck">
        <div className="deck-main">
          <div className="deck-progress" role="tablist" aria-label="Questions">
            {qs.map((x, k) => {
              const ch = answers[x.question_id] !== x.recommended_default;
              return (
                <button key={x.question_id} role="tab" aria-selected={k === i} className={`pip${k === i ? " on" : ""}${seen.has(k) ? " seen" : ""}${ch ? " ch" : ""}`} onClick={() => goto(k)} title={x.column ?? x.category}>
                  <span>{k + 1}</span>
                </button>
              );
            })}
            <span className="deck-count">{i + 1} / {qs.length}</span>
          </div>

          {q && (
            <div className="deck-card fade-swap" key={q.question_id}>
              <QuestionCard q={q} selected={answers[q.question_id] ?? q.recommended_default} onSelect={pick} name={`deck-${q.question_id}`} />
              <div className="kbd-row">
                {q.options.map((o, k) => <span key={o}><kbd>{k + 1}</kbd> {o}</span>)}
              </div>
            </div>
          )}

          <div className="deck-nav">
            <button className="btn" onClick={() => goto(i - 1)} disabled={i === 0}>← Previous <kbd>←</kbd></button>
            {i < qs.length - 1 ? (
              <button className="btn primary" onClick={() => goto(i + 1)}>Next question <kbd>Enter</kbd></button>
            ) : (
              <button className="btn primary" onClick={onRun} disabled={running}>{running ? "Running prepare()…" : runs ? "Run prepare() again" : "Run prepare()"} <kbd>Enter</kbd></button>
            )}
          </div>
        </div>

        <aside className="deck-side">
          <h4>All answers</h4>
          <ol className="ans-list">
            {qs.map((x, k) => {
              const v = answers[x.question_id] ?? x.recommended_default;
              const ch = v !== x.recommended_default;
              return (
                <li key={x.question_id}>
                  <button className={k === i ? "on" : ""} onClick={() => goto(k)}>
                    <span className="col">{x.column ?? x.category}</span>
                    <span className="val">{v}</span>
                    {ch ? <span className="badge ask">changed</span> : <span className="badge">default</span>}
                  </button>
                </li>
              );
            })}
          </ol>
          <button className="btn primary big" onClick={onRun} disabled={running} style={{ width: "100%" }}>
            {running ? <><span className="spinner inv" /> Running prepare()…</> : "Run prepare() with these →"}
          </button>
          <p className="hint-line">Same as <code>mode=&quot;defaults&quot;</code> if you change nothing.</p>
        </aside>
      </div>
    </>
  );
}
