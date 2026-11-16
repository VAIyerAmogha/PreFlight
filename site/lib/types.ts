// Shapes of the JSON that public/py/runner.py returns. Keep in sync with it.

export type Cell = string | number | boolean | null;

export interface Frame {
  columns: string[];
  dtypes: string[];
  index: (string | number)[];
  rows: Cell[][];
  shape: [number, number];
}
export type MaybeFrame = Frame | { error: string } | null;

export interface Decision {
  id: string;
  stage: string;
  kind: string;
  columns: string[];
  action: string;
  default: string;
  confidence: number;
  needs_attention: boolean;
  status: string;
  plain_language: string;
  evidence: Record<string, unknown>;
  history: unknown[];
}

export interface Question {
  question_id: string;
  decision_id: string;
  category: string;
  column: string | null;
  text: string;
  explanation: string;
  evidence_summary: string;
  options: string[];
  option_actions: Record<string, string>;
  recommended_default: string;
  preview: { before: string[]; after: Record<string, string[]>; note?: string } | null;
}

export interface ColumnInfo {
  name: string;
  position: number;
  type: string | null;
  confidence: number;
  runner_up: string | null;
  abstained: boolean;
  missing_markers: { marker: string; count?: number; reason?: string }[];
  anomalies: string[];
  candidate_scores: Record<string, number> | null;
  margin: number | null;
  non_missing: number | null;
  total_rows: number | null;
}

export interface Reading {
  encoding: { name: string; confidence: number; has_bom: boolean };
  delimiter: { char: string; confidence: number; margin: number };
  header: { row: number | null; confidence: number; title_rows: number[]; blank_rows: number[] };
}

export interface Analysis {
  file: string;
  rows: number | null;
  columns: number;
  task: string | null;
  reading: Reading;
  raw_lines: string[];
  naive_pandas: { ok: true; frame: Frame } | { ok: false; error: string };
  raw_table: Frame | null;
  schema: ColumnInfo[];
  decisions: Decision[];
  questions: Question[];
}

export interface Changes {
  available: boolean;
  reason?: string;
  dropped_columns: string[];
  added_columns: string[];
  dropped_rows: (string | number)[];
  dropped_rows_total: number;
  changed: Record<string, (string | number)[]>;
  blanked: Record<string, (string | number)[]>;
  raw_rows: number;
}

export interface RunResult {
  task: string | null;
  raw_table: Frame | null;
  prepared: Frame;
  model_ready: MaybeFrame;
  changes: Changes;
  pipeline: { name: string; class: string; module: string }[];
  decisions: Decision[];
  anomalies: unknown[];
  files: Record<string, string>;
  code: string;
}

export interface Fixture {
  meta: { preflight: string; pandas: string };
  sample: string;
  target: string;
  analysis: Analysis;
  result: RunResult;
}
