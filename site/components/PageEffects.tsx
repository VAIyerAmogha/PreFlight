"use client";

import { useEffect } from "react";
import { startEngine, useEngine } from "@/lib/engine";

/** Page-wide behaviour: top bar state, reading progress, TOC scroll-spy, reveal-on-scroll. */
export default function PageEffects() {
  useEffect(() => {
    const bar = document.getElementById("topbar");
    const prog = document.getElementById("read-progress");
    const onScroll = () => {
      bar?.classList.toggle("scrolled", window.scrollY > 8);
      const max = document.documentElement.scrollHeight - window.innerHeight;
      if (prog) prog.style.width = `${max > 0 ? (window.scrollY / max) * 100 : 0}%`;
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });

    const links = [...document.querySelectorAll<HTMLAnchorElement>(".toc a[href^='#']")];
    const sections = links.map((a) => document.querySelector<HTMLElement>(a.hash)).filter(Boolean) as HTMLElement[];
    const spy = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          links.forEach((a) => a.classList.toggle("active", a.hash === `#${e.target.id}`));
        }
      },
      { rootMargin: "-20% 0px -70% 0px" }
    );
    sections.forEach((s) => spy.observe(s));

    const reveal = new IntersectionObserver(
      (entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); reveal.unobserve(e.target); } }),
      { rootMargin: "0px 0px -8% 0px" }
    );
    document.querySelectorAll(".reveal").forEach((el) => reveal.observe(el));

    // Hovering or focusing any link to the playground starts Python early. The worker
    // lives in module state, so it survives the client-side navigation to /try.
    const warm = (e: Event) => {
      const a = (e.target as HTMLElement | null)?.closest?.("a[href='/try']");
      if (a) startEngine().catch(() => {});
    };
    document.addEventListener("pointerover", warm, { passive: true });
    document.addEventListener("focusin", warm);

    return () => {
      window.removeEventListener("scroll", onScroll);
      spy.disconnect();
      reveal.disconnect();
      document.removeEventListener("pointerover", warm);
      document.removeEventListener("focusin", warm);
    };
  }, []);
  return null;
}

export function ThemeToggle() {
  return (
    <button
      className="icon-btn"
      aria-label="Toggle dark mode"
      title="Toggle theme"
      onClick={() => {
        const dark = document.documentElement.classList.toggle("dark");
        try { localStorage.setItem("pf-theme", dark ? "dark" : "light"); } catch { /* storage blocked */ }
      }}
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
        <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />
      </svg>
    </button>
  );
}

/** Hero pill and TOC footer: live state of the in-browser Python engine. */
export function EngineEyebrow() {
  const e = useEngine();
  const text =
    e.phase === "ready" ? `pypreflight ${e.versions?.preflight} is running in your browser`
    : e.phase === "loading" ? `Python warming up in your browser · ${Math.round((e.step / Math.max(1, e.total)) * 100)}%`
    : e.phase === "error" ? "Python engine unavailable in this browser"
    : "Runs the real package in your browser · no server";
  return (
    <span className="eyebrow">
      <span className={`dot${e.phase === "loading" ? " loading" : ""}`} style={e.phase === "error" ? { background: "var(--red)" } : undefined} />
      <span>{text}</span>
    </span>
  );
}
