"use client";

import { useState } from "react";
import type { Question } from "@/lib/types";
import QuestionCard from "../QuestionCard";
import { Figure } from "../ui";

export default function QuestionsFigure({ questions, file }: { questions: Question[]; file: string }) {
  // Start on a column question with a preview (the most illustrative one).
  const first = Math.max(0, questions.findIndex((q) => q.preview && q.preview.before?.length));
  const [idx, setIdx] = useState(first);
  const [picked, setPicked] = useState<Record<string, string>>({});
  const q = questions[idx];
  if (!q) return null;

  return (
    <Figure
      n={4}
      title="A question, with its preview"
      tools={
        <div className="qtabs">
          {questions.map((x, i) => (
            <button key={x.question_id} className="chip" aria-pressed={i === idx} onClick={() => setIdx(i)}>
              {x.column ?? x.category}
            </button>
          ))}
        </div>
      }
      caption={
        <>
          The questions PreFlight raised for <code>{file}</code>. Pick an option to see what it does to real values from the file. The preview
          text is the package’s own: “(left blank)” means a value could not be converted and would be recorded as a problem value.
        </>
      }
    >
      <div key={q.question_id} className="fade-swap">
        <QuestionCard
          q={q}
          name={`fig4-${q.question_id}`}
          selected={picked[q.question_id] ?? q.recommended_default}
          onSelect={(o) => setPicked((p) => ({ ...p, [q.question_id]: o }))}
        />
      </div>
    </Figure>
  );
}
