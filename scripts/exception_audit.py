"""List exception handlers in src/ that swallow errors (same rules as scripts/static_triage.py).

Categories: except_Exception_pass (broad and empty body), swallowed_exception (narrow and
empty body), broad_except_no_reraise (broad, has a body, never re-raises). Each hit is
JUSTIFIED when the handler carries a comment starting with "# Swallowed:" on the except line
or inside the handler, explaining why ignoring the error is safe.

Usage: python scripts/exception_audit.py [OUT.md]
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARK = "# Swallowed:"


def audit():
    hits = []
    for f in sorted((ROOT / "src").rglob("*.py")):
        text = f.read_text(encoding="utf-8")
        lines = text.splitlines()
        for n in ast.walk(ast.parse(text)):
            if not isinstance(n, ast.ExceptHandler):
                continue
            t = ast.unparse(n.type) if n.type else None
            trivial = all(
                isinstance(b, (ast.Pass, ast.Continue)) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))
                for b in n.body
            )
            if t is None:
                cat = "bare_except"
            elif t in ("Exception", "BaseException") and trivial:
                cat = "except_Exception_pass"
            elif trivial:
                cat = "swallowed_exception"
            elif t in ("Exception", "BaseException") and not any(isinstance(x, ast.Raise) for x in ast.walk(n)):
                cat = "broad_except_no_reraise"
            else:
                continue
            span = lines[n.lineno - 1:(n.end_lineno or n.lineno)]
            why = next((ln.split(MARK, 1)[1].strip() for ln in span if MARK in ln), None)
            hits.append((cat, f"{f.relative_to(ROOT)}:{n.lineno}", t, why))
    return hits


def main():
    hits = audit()
    out = ["| category | location | except | justification |", "|---|---|---|---|"]
    out += [f"| {c} | {loc} | {t} | {w or 'UNJUSTIFIED'} |" for c, loc, t, w in hits]
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"hits {len(hits)} justified {sum(1 for h in hits if h[3])} unjustified {sum(1 for h in hits if not h[3])}")


if __name__ == "__main__":
    main()
