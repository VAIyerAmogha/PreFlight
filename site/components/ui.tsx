// Shared display primitives. They only *show* values produced by the package
// (dev-docs/demo-site-plan.md: "100% package-driven").

import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import python from "highlight.js/lib/languages/python";
import type { CSSProperties, ReactNode } from "react";
import CopyButton from "./CopyButton";
import type { Cell, Changes, MaybeFrame } from "@/lib/types";

hljs.registerLanguage("python", python);
hljs.registerLanguage("bash", bash);

export const conf = (x: unknown) => (typeof x === "number" ? x.toFixed(2) : "—");

export function fmtBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function delimiterLabel(ch: string) {
  return ({ ",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe" } as Record<string, string>)[ch] ?? JSON.stringify(ch);
}

const TYPE_CLASSES = new Set(["int", "float", "categorical", "boolean", "ID", "datetime", "text", "constant"]);
export function TypeBadge({ type, className = "" }: { type: string | null | undefined; className?: string }) {
  const t = type ?? "text";
  return <span className={`badge ${TYPE_CLASSES.has(t) ? `t-${t}` : "t-text"} ${className}`}>{t}</span>;
}

export function Meter({ value, low = 0.8 }: { value: number; low?: number }) {
  return (
    <div className={`meter${value < low ? " low" : ""}`}>
      <i style={{ width: `${Math.round((value ?? 0) * 100)}%` }} />
    </div>
  );
}

// ------------------------------------------------------------------ code
export function highlightCode(code: string, lang: "python" | "bash") {
  return hljs.highlight(code, { language: lang }).value;
}

export function CodeBlock({ code, lang = "python", label }: { code: string; lang?: "python" | "bash"; label?: string }) {
  return (
    <div className="code">
      <div className="code-head">
        <span>{label ?? lang}</span>
        <CopyButton text={code} />
      </div>
      <pre>
        <code className={`hljs language-${lang}`} dangerouslySetInnerHTML={{ __html: highlightCode(code, lang) }} />
      </pre>
    </div>
  );
}

// ------------------------------------------------------------------ raw file
export function RawFileView({ lines, delimiter, headerRow }: { lines: string[]; delimiter: string; headerRow: number | null }) {
  return (
    <div className="rawfile">
      {lines.map((line, i) => {
        const parts = line.split(delimiter);
        return (
          <div key={i} className={`row${headerRow === i ? " hdr" : ""}`} style={{ animationDelay: `${i * 35}ms` }}>
            <span className="no">{i + 1}</span>
            {parts.map((p, j) => (
              <span key={j}>
                {p}
                {j < parts.length - 1 && <span className="d">{delimiter === "\t" ? "→" : delimiter}</span>}
              </span>
            ))}
            {line === "" && " "}
          </div>
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------------ data table
type TableMode = "plain" | "raw" | "prepared" | "model";
interface TableProps {
  frame: MaybeFrame;
  mode?: TableMode;
  changes?: Changes | null;
  types?: Record<string, string | null>;
  raw?: MaybeFrame;
  baseColumns?: string[];
  maxHeight?: number;
}

const show = (v: Cell, mode: TableMode) =>
  v === null ? (mode === "raw" ? "" : "NA") : typeof v === "boolean" ? (v ? "True" : "False") : String(v);

export function DataTable({ frame, mode = "plain", changes = null, types = {}, raw = null, baseColumns, maxHeight }: TableProps) {
  if (!frame) return <div className="note">Not available for this file.</div>;
  if ("error" in frame) return <div className="note bad">{frame.error}</div>;
  const ch = changes?.available ? changes : null;
  const droppedCols = new Set(ch?.dropped_columns ?? []);
  const droppedRows = new Set((ch?.dropped_rows ?? []).map(String));
  const changed = new Map<string, Set<string>>();
  const blanked = new Map<string, Set<string>>();
  if (ch) {
    for (const [c, ids] of Object.entries(ch.changed)) changed.set(c, new Set(ids.map(String)));
    for (const [c, ids] of Object.entries(ch.blanked)) blanked.set(c, new Set(ids.map(String)));
  }
  const added = new Set(
    mode === "model" && baseColumns ? frame.columns.filter((c) => !baseColumns.includes(c)) : mode === "prepared" && ch ? ch.added_columns : []
  );
  const rawRows = new Map<string, Cell[]>();
  const rawCols = new Map<string, number>();
  if (raw && !("error" in raw) && mode === "prepared") {
    raw.index.forEach((ix, i) => rawRows.set(String(ix), raw.rows[i]));
    raw.columns.forEach((c, i) => rawCols.set(c, i));
  }
  const colCls = (c: string) => (mode === "raw" && droppedCols.has(c) ? "col-dropped" : added.has(c) ? "col-added" : "");
  const style: CSSProperties | undefined = maxHeight ? { maxHeight } : undefined;

  return (
    <div className="table-wrap" style={style}>
      <table className="data">
        <thead>
          <tr>
            <th className="idx">#</th>
            {frame.columns.map((c, i) => (
              <th key={c + i} className={colCls(c)}>
                {c}
                {types[c] ? <TypeBadge type={types[c]} /> : mode !== "raw" && frame.dtypes ? <span className="badge">{frame.dtypes[i]}</span> : null}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {frame.rows.map((row, r) => {
            const ix = String(frame.index[r]);
            return (
              <tr key={ix + ":" + r} className={mode === "raw" && droppedRows.has(ix) ? "dropped" : undefined}>
                <td className="idx">{ix}</td>
                {row.map((v, ci) => {
                  const col = frame.columns[ci];
                  const cls = [colCls(col)];
                  let title: string | undefined;
                  if (v === null) cls.push("na");
                  if (mode === "prepared") {
                    const was = rawRows.get(ix)?.[rawCols.get(col) ?? -1];
                    if (blanked.get(col)?.has(ix)) { cls.push("blank"); title = `was: ${was ?? ""} → missing`; }
                    else if (changed.get(col)?.has(ix)) { cls.push("chg"); title = `was: ${was ?? ""}`; }
                  }
                  // --ci/--ri drive the staggered "replay" animation on highlighted cells.
                  const st = title ? ({ "--ci": ci, "--ri": Math.min(r, 40) } as CSSProperties) : undefined;
                  return (
                    <td key={ci} className={cls.join(" ") || undefined} title={title} style={st}>
                      {show(v, mode)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      {frame.shape[0] > frame.rows.length && (
        <div className="source" style={{ padding: "8px 12px" }}>
          showing {frame.rows.length} of {frame.shape[0]} rows
        </div>
      )}
    </div>
  );
}

export function DiffLegend({ changes }: { changes: Changes }) {
  if (!changes) return null;
  if (!changes.available) return changes.reason ? <div className="note">{changes.reason}</div> : null;
  const nChanged = Object.values(changes.changed).reduce((a, b) => a + b.length, 0);
  const nBlank = Object.values(changes.blanked).reduce((a, b) => a + b.length, 0);
  return (
    <div className="data-legend">
      <span><i className="l-chg" />value changed · {nChanged}</span>
      <span><i className="l-blank" />became missing · {nBlank}</span>
      <span><i className="l-drop" />dropped · {changes.dropped_rows_total} row(s), {changes.dropped_columns.length} column(s)</span>
      <span><i className="l-add" />added column</span>
      <span className="source">hover a highlighted cell to see the original value</span>
    </div>
  );
}

// ------------------------------------------------------------------ figure frame
export function Figure({
  n, title, tools, caption, children, bodyClass = "fig-body", id,
}: { n: number | string; title: string; tools?: ReactNode; caption: ReactNode; children: ReactNode; bodyClass?: string; id?: string }) {
  return (
    <figure className="wide reveal" id={id}>
      <div className="fig">
        <div className="fig-head">
          <span className="label">Figure {n}</span>
          <span className="title">{title}</span>
          {tools && <div className="tools">{tools}</div>}
        </div>
        <div className={bodyClass}>{children}</div>
      </div>
      <figcaption>
        <b>Figure {n}.</b> {caption}
      </figcaption>
    </figure>
  );
}

export function Seg<T extends string>({ value, options, onChange, label }: { value: T; options: { v: T; label: ReactNode }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.v} aria-pressed={value === o.v} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
