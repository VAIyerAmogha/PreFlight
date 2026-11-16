"use client";

// Main-thread client for public/pyodide-worker.js. One worker per page, shared
// by every component through useEngine().

import { useSyncExternalStore } from "react";
import { PREFLIGHT_SPEC, PYODIDE_VERSION } from "./config";

export type EnginePhase = "idle" | "loading" | "ready" | "error";
export interface EngineState {
  phase: EnginePhase;
  step: number;
  total: number;
  text: string;
  versions: { preflight: string; pandas: string } | null;
  error: string | null;
  startedAt: number | null;
  readyIn: number | null;
  loaded: string[];
}

export class EngineError extends Error {
  traceback?: string;
  constructor(message: string, traceback?: string) {
    super(message);
    this.traceback = traceback;
  }
}

type Pending = { resolve: (v: unknown) => void; reject: (e: unknown) => void };

let state: EngineState = { phase: "idle", step: 0, total: 1, text: "Not started", versions: null, error: null, startedAt: null, readyIn: null, loaded: [] };
const listeners = new Set<() => void>();
let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, Pending>();
let readyPromise: Promise<void> | null = null;
let resolveReady: (() => void) | null = null;
let rejectReady: ((e: unknown) => void) | null = null;

function set(patch: Partial<EngineState>) {
  state = { ...state, ...patch };
  listeners.forEach((l) => l());
}

export function startEngine(): Promise<void> {
  if (readyPromise) return readyPromise;
  readyPromise = new Promise<void>((res, rej) => { resolveReady = res; rejectReady = rej; });
  set({ phase: "loading", text: "Starting Python", startedAt: performance.now() });
  worker = new Worker("/pyodide-worker.js");
  worker.onmessage = (e: MessageEvent) => {
    const m = e.data;
    if (m.type === "progress") set({ step: m.step, total: m.total, text: m.text, ...(m.loaded ? { loaded: m.loaded } : {}) });
    else if (m.type === "ready") {
      set({ phase: "ready", versions: m.versions, text: "Ready", readyIn: state.startedAt ? (performance.now() - state.startedAt) / 1000 : null });
      resolveReady?.();
    } else if (m.type === "fatal") {
      set({ phase: "error", error: m.error, text: "Failed to start" });
      rejectReady?.(new EngineError(m.error));
    } else if (m.type === "result") {
      const p = pending.get(m.id);
      if (!p) return;
      pending.delete(m.id);
      if (m.ok) p.resolve(m.value);
      else p.reject(new EngineError(m.error, m.traceback));
    }
  };
  worker.onerror = (e) => {
    set({ phase: "error", error: e.message || "Worker failed", text: "Failed to start" });
    rejectReady?.(new EngineError(e.message));
  };
  const spec = PREFLIGHT_SPEC.endsWith(".whl") ? new URL(PREFLIGHT_SPEC, location.href).href : PREFLIGHT_SPEC;
  worker.postMessage({ type: "init", pyodideVersion: PYODIDE_VERSION, preflightSpec: spec, runnerUrl: new URL("/py/runner.py", location.href).href });
  return readyPromise;
}

function send<T>(msg: Record<string, unknown>, transfer: Transferable[] = []): Promise<T> {
  return startEngine().then(
    () =>
      new Promise<T>((resolve, reject) => {
        const id = nextId++;
        pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
        worker!.postMessage({ ...msg, id }, transfer);
      })
  );
}

/** Call a runner.py entry point. It returns a JSON string, parsed here. */
export async function callRunner<T>(fn: string, ...args: (string | number)[]): Promise<T> {
  const raw = await send<string>({ type: "call", fn, args });
  return JSON.parse(raw) as T;
}
export const writeFile = (path: string, data: ArrayBuffer) => send<null>({ type: "write", path, data }, [data]);
export const readFiles = (paths: string[]) => send<Record<string, Uint8Array>>({ type: "read", paths });
export const removePath = (path: string) => send<null>({ type: "rm", path });

export function useEngine(): EngineState {
  return useSyncExternalStore(
    (l) => { listeners.add(l); return () => listeners.delete(l); },
    () => state,
    () => state
  );
}

/** Packages the worker loads, in order (for the loading visual). */
export const ENGINE_PACKAGES = ["micropip", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "python-dateutil", "joblib"];
