import Link from "next/link";
import { REPO } from "@/lib/config";
import { ThemeToggle } from "./PageEffects";

export function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 24 24" aria-hidden="true">
      <rect width="24" height="24" fill="currentColor" />
      <path d="M6 15.5 18 8l-4.5 10-2-4-5.5 1.5Z" fill="var(--brand-fg)" />
    </svg>
  );
}

export default function Topbar({ page }: { page: "docs" | "try" }) {
  return (
    <header className="topbar" id="topbar">
      <Link className="brand" href="/" aria-label="PreFlight home">
        <BrandMark />
        PreFlight
      </Link>
      <nav aria-label="Primary">
        <Link href="/#how" aria-current={page === "docs" ? "page" : undefined}>Docs</Link>
        <Link href="/#api">API</Link>
        <Link href="/try" className="nav-try" aria-current={page === "try" ? "page" : undefined}>
          Playground
        </Link>
        <a href={REPO}>GitHub</a>
      </nav>
      <span className="spacer" />
      <ThemeToggle />
      {page === "docs" && <div className="progress-bar" id="read-progress" />}
    </header>
  );
}
