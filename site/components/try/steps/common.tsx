"use client";

import { useEffect, useState, type ReactNode } from "react";
import { useReducedMotion } from "@/lib/hooks";

export function StepHeader({ n, title, children, aside }: { n: number; title: string; children?: ReactNode; aside?: ReactNode }) {
  return (
    <header className="step-head">
      <div>
        <div className="step-kicker">Step {n} of 7</div>
        <h1>{title}</h1>
        {children && <p className="step-lede">{children}</p>}
      </div>
      {aside && <div className="step-aside">{aside}</div>}
    </header>
  );
}

/** Animated number that counts up from 0 when it mounts or the value changes. */
export function CountUp({ value, decimals = 0, ms = 900 }: { value: number; decimals?: number; ms?: number }) {
  const reduced = useReducedMotion();
  const [v, setV] = useState(0);
  useEffect(() => {
    if (reduced) { setV(value); return; }
    let raf = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const k = Math.min(1, (t - t0) / ms);
      setV(value * (1 - Math.pow(1 - k, 3)));
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, ms, reduced]);
  return <>{v.toFixed(decimals)}</>;
}

export function Stat({ k, children, sub }: { k: string; children: ReactNode; sub?: ReactNode }) {
  return (
    <div className="stat">
      <div className="k">{k}</div>
      <div className="v">{children}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

/** ok=null means "no guess was made" (the person skipped). */
export function Verdict({ ok, children }: { ok: boolean | null; children: ReactNode }) {
  const cls = ok === null ? "neutral" : ok ? "ok" : "miss";
  return <div className={`verdict ${cls}`}><span className="ic">{ok === null ? "i" : ok ? "✓" : "✗"}</span><span>{children}</span></div>;
}
