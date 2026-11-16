"use client";

import { useState } from "react";
import type { Analysis } from "@/lib/types";
import { conf, delimiterLabel, Meter } from "../../ui";
import { StepHeader, Verdict } from "./common";

const SEPS = [",", ";", "\t", "|"];

export default function ReadStep({ a, seconds }: { a: Analysis; seconds: number }) {
  const rd = a.reading;
  const [sepGuess, setSepGuess] = useState<string | null>(null);
  const [hdrGuess, setHdrGuess] = useState<number | null | undefined>(undefined); // undefined = not guessed, null = "no header"
  const sepShown = sepGuess !== null;
  const hdrShown = hdrGuess !== undefined;
  const truthHdr = rd.header.row;
  const SKIP = -1;
  const sepSkipped = sepGuess === "";
  const hdrSkipped = hdrGuess === SKIP;
  const asked = (sepShown && !sepSkipped ? 1 : 0) + (hdrShown && !hdrSkipped ? 1 : 0);
  const score = (sepShown && sepGuess === rd.delimiter.char ? 1 : 0) + (hdrShown && hdrGuess === truthHdr ? 1 : 0);

  return (
    <>
      <StepHeader
        n={2}
        title="How is this file laid out?"
        aside={asked > 0 ? <div className="score-chip">Your score <b>{score}</b>/{asked}</div> : undefined}
      >
        Before PreFlight tells you, look at the raw bytes and make two calls. Then see what <code>plan()</code> decided, and how sure it was. It took{" "}
        {seconds.toFixed(2)} s.
      </StepHeader>

      <div className="read-grid">
        <div>
          <div className="panel-title">
            <span>{a.file}</span>
            <span className="source">first {a.raw_lines.length} lines as stored on disk</span>
          </div>
          <div className={`rawfile big${sepShown ? " sep-on" : ""}${hdrShown ? "" : " pickable"}`}>
            {a.raw_lines.map((line, i) => {
              const parts = sepShown ? line.split(rd.delimiter.char) : [line];
              const isHdr = hdrShown && truthHdr === i;
              const picked = hdrShown && hdrGuess === i && truthHdr !== i;
              return (
                <button
                  key={i}
                  className={`row${isHdr ? " hdr" : ""}${picked ? " wrong" : ""}`}
                  style={{ animationDelay: `${i * 30}ms` }}
                  disabled={hdrShown}
                  onClick={() => setHdrGuess(i)}
                  title={hdrShown ? undefined : "This line holds the column names"}
                >
                  <span className="no">{i + 1}</span>
                  <span className="txt">
                    {parts.map((p, j) => (
                      <span key={j}>
                        {p}
                        {j < parts.length - 1 && <span className="d" style={{ animationDelay: `${(i * 4 + j) * 12}ms` }}>{rd.delimiter.char === "\t" ? "→" : rd.delimiter.char}</span>}
                      </span>
                    ))}
                  </span>
                </button>
              );
            })}
          </div>
          {!hdrShown && <p className="hint-line">Tip: click the line that holds the column names, or say there isn’t one →</p>}
        </div>

        <div className="quiz">
          <div className={`quiz-card${sepShown ? " done" : ""}`}>
            <div className="q-n">Q1</div>
            <h3>What separates the fields?</h3>
            <div className="choice-row">
              {SEPS.map((s) => (
                <button
                  key={s}
                  className={`choice${sepGuess === s ? " picked" : ""}${sepShown && s === rd.delimiter.char ? " truth" : ""}`}
                  disabled={sepShown}
                  onClick={() => setSepGuess(s)}
                >
                  <code>{s === "\t" ? "\\t" : s}</code> {delimiterLabel(s)}
                </button>
              ))}
            </div>
            {sepShown && (
              <div className="reveal-box">
                <Verdict ok={sepSkipped ? null : sepGuess === rd.delimiter.char}>
                  PreFlight: <b>{delimiterLabel(rd.delimiter.char)}</b> with confidence {conf(rd.delimiter.confidence)} (margin {conf(rd.delimiter.margin)} over the next separator).
                </Verdict>
                <Meter value={rd.delimiter.confidence} />
              </div>
            )}
          </div>

          <div className={`quiz-card${hdrShown ? " done" : ""}`}>
            <div className="q-n">Q2</div>
            <h3>Which line holds the column names?</h3>
            {!hdrShown ? (
              <div className="choice-row">
                <button className="choice" onClick={() => setHdrGuess(0)}>Line 1</button>
                <button className="choice" onClick={() => setHdrGuess(null)}>There isn’t one</button>
                <span className="hint">…or click a line in the file</span>
              </div>
            ) : (
              <div className="reveal-box">
                <Verdict ok={hdrSkipped ? null : hdrGuess === truthHdr}>
                  {!hdrSkipped && <>You said {hdrGuess === null ? "no header" : `line ${(hdrGuess ?? 0) + 1}`}. </>}PreFlight:{" "}
                  <b>{truthHdr === null ? "no header, so columns are named col_0, col_1, …" : `line ${truthHdr + 1}`}</b> with confidence {conf(rd.header.confidence)}.
                  {rd.header.title_rows.length > 0 && ` It also skipped ${rd.header.title_rows.length} title row(s).`}
                </Verdict>
                <Meter value={rd.header.confidence} />
              </div>
            )}
          </div>

          {(!sepShown || !hdrShown) && (
            <button className="linkish" onClick={() => { if (!sepShown) setSepGuess(""); if (!hdrShown) setHdrGuess(SKIP); }}>Skip the quiz, just show me</button>
          )}

          <div className="mini-tiles">
            <div><span className="k">Encoding</span><b>{rd.encoding.name}</b><span className="source">{conf(rd.encoding.confidence)}{rd.encoding.has_bom ? " · BOM" : ""}</span></div>
            <div><span className="k">Shape</span><b>{a.rows} × {a.columns}</b><span className="source">rows × columns</span></div>
          </div>
        </div>
      </div>
    </>
  );
}
