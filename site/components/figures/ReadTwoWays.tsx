"use client";

import { useState } from "react";
import type { Fixture } from "@/lib/types";
import { conf, DataTable, delimiterLabel, Figure, RawFileView, Seg } from "../ui";

type View = "file" | "pandas" | "preflight";

export default function ReadTwoWays({ fx }: { fx: Fixture }) {
  const [view, setView] = useState<View>("file");
  const a = fx.analysis;
  const types = Object.fromEntries(a.schema.map((c) => [c.name, c.type]));
  const naive = a.naive_pandas;

  return (
    <Figure
      n={1}
      title="The same file, read two ways"
      tools={
        <Seg<View>
          label="View"
          value={view}
          onChange={setView}
          options={[
            { v: "file", label: "The file" },
            { v: "pandas", label: <code>pd.read_csv</code> },
            { v: "preflight", label: <code>preflight.plan</code> },
          ]}
        />
      }
      caption={
        <>
          <code>{fx.sample}</code>: semicolon-separated, no header row. Plain pandas returns one column whose name is the first data row.
          PreFlight detects the separator and the missing header, names the columns and types them. Switch views to compare.
        </>
      }
    >
      <div key={view} className="fade-swap">
        {view === "file" && (
          <>
            <RawFileView lines={a.raw_lines.slice(0, 8)} delimiter={a.reading.delimiter.char} headerRow={null} />
            <div className="note">The first lines of the file as bytes on disk. Nothing marks a header, and the separator is not a comma.</div>
          </>
        )}
        {view === "pandas" && (
          <>
            {naive.ok ? <DataTable frame={naive.frame} maxHeight={300} /> : <div className="note bad">{naive.error}</div>}
            {naive.ok && (
              <div className="note bad">
                <Warn />
                <span>
                  <b>{naive.frame.shape[1]} column</b>
                  {naive.frame.shape[1] === 1 ? "" : "s"}, named <code>{naive.frame.columns[0]}</code>. The first data row became the header, and every value is
                  still a string of joined fields. No error is raised.
                </span>
              </div>
            )}
          </>
        )}
        {view === "preflight" && (
          <>
            <div className="tiles" style={{ marginBottom: 14 }}>
              <div className="tile"><div className="k">Encoding</div><div className="v">{a.reading.encoding.name}</div><div className="sub">confidence {conf(a.reading.encoding.confidence)}</div></div>
              <div className="tile"><div className="k">Separator</div><div className="v"><code>{a.reading.delimiter.char}</code> {delimiterLabel(a.reading.delimiter.char)}</div><div className="sub">confidence {conf(a.reading.delimiter.confidence)}</div></div>
              <div className="tile"><div className="k">Header row</div><div className="v">{a.reading.header.row === null ? "none" : `line ${a.reading.header.row + 1}`}</div><div className="sub">confidence {conf(a.reading.header.confidence)}</div></div>
            </div>
            <DataTable frame={fx.result.prepared} types={types} maxHeight={300} />
            <div className="note good">
              <Check />
              <span>
                {a.columns} columns, {a.rows} rows. Names <code>col_0</code>…<code>col_{a.columns - 1}</code> are assigned because the file has no header.
                Each column carries the type PreFlight inferred.
              </span>
            </div>
          </>
        )}
      </div>
    </Figure>
  );
}

export const Warn = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: "var(--red)" }}>
    <path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
  </svg>
);
export const Check = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" style={{ color: "var(--green)" }}>
    <path d="M20 6 9 17l-5-5" />
  </svg>
);
