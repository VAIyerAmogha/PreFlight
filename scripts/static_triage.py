"""Static triage (grep/AST, no LLM). Writes docs/review_notes/phase7/static_findings.md; prints counts per category."""
import ast
import re
import subprocess
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "preflight"
OUT = ROOT / "docs" / "review_notes" / "phase7" / "static_findings.md"
hits: dict = defaultdict(list)
rel = lambda p: str(p.relative_to(ROOT))  # noqa: E731
src_files = sorted(SRC.rglob("*.py"))
test_files = sorted((ROOT / "tests").rglob("*.py"))
all_py = src_files + test_files + sorted((ROOT / "scripts").glob("*.py"))


def line_grep(cat, pat, files, flags=0):
    rx = re.compile(pat, flags)
    for f in files:
        for i, ln in enumerate(f.read_text(errors="replace").splitlines(), 1):
            if rx.search(ln):
                hits[cat].append(f"{rel(f)}:{i}: {ln.strip()[:110]}")


# 1 exceptions
for f in src_files:
    tree = ast.parse(f.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.ExceptHandler):
            t = ast.unparse(n.type) if n.type else None
            trivial = all(isinstance(b, (ast.Pass, ast.Continue)) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)) for b in n.body)
            if t is None:
                hits["bare_except"].append(f"{rel(f)}:{n.lineno}")
            elif t in ("Exception", "BaseException") and trivial:
                hits["except_Exception_pass"].append(f"{rel(f)}:{n.lineno}")
            elif trivial:
                hits["swallowed_exception"].append(f"{rel(f)}:{n.lineno}: except {t}")
            elif t in ("Exception", "BaseException") and not any(isinstance(x, ast.Raise) for x in ast.walk(n)):
                hits["broad_except_no_reraise"].append(f"{rel(f)}:{n.lineno}")
line_grep("warnings_warn", r"warnings\.warn\(|\bwarn\(", src_files)

# 2 markers
line_grep("type_ignore", r"#\s*type:\s*ignore", src_files + test_files)
line_grep("noqa", r"#\s*noqa", src_files + test_files)
line_grep("pragma_no_cover", r"pragma:\s*no cover", src_files)
line_grep("nosec", r"nosec", all_py)
line_grep("todo_fixme_xxx", r"\b(TODO|FIXME|XXX)\b", all_py)
for f in src_files:  # commented-out code: 3+ consecutive comment lines that parse as code
    lines = f.read_text().splitlines()
    i = 0
    while i < len(lines):
        j = i
        while j < len(lines) and lines[j].strip().startswith("#"):
            j += 1
        if j - i >= 3:
            body = "\n".join(ln.strip()[1:].strip() for ln in lines[i:j])
            try:
                t = ast.parse(body)
                if t.body and not all(isinstance(b, ast.Expr) for b in t.body):
                    hits["commented_out_code"].append(f"{rel(f)}:{i+1}-{j}")
            except SyntaxError:
                pass
        i = max(j, i + 1)

# 3 skips
line_grep("skip_xfail", r"pytest\.(mark\.)?(skip|skipif|xfail)|pytest\.importorskip|@unittest\.skip", test_files)

# 4 numeric literals in comparisons (infer/clean/engineer)
for sub in ("infer", "clean", "engineer"):
    for f in sorted((SRC / sub).rglob("*.py")):
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.Compare):
                for c in [n.left, *n.comparators]:
                    if isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) and not isinstance(c.value, bool) and c.value not in (0, 1, -1):
                        hits["magic_number_compare"].append(f"{rel(f)}:{n.lineno}: {ast.unparse(n)[:90]}")

# 5 old names
line_grep("old_names", r"\bProfiler\b|\bprofiler\b|HTML report|\.to_html\(|html_report|\bgenerate_report\b", src_files)
line_grep("old_names_docs", r"\bProfiler\b|HTML report", [ROOT / "README.md", *sorted((ROOT / "docs").glob("GUIDE.md"))])

# 6 network, randomness, writes, temp
line_grep("network", r"^\s*(import|from)\s+(requests|urllib|socket|http|ftplib|smtplib|httpx|aiohttp)\b|urlopen|urlretrieve", src_files)
line_grep("unseeded_random", r"\brandom\.(random|randint|choice|shuffle|sample)\(|np\.random\.(?!seed|RandomState|default_rng\(\s*\w)|default_rng\(\s*\)|RandomState\(\s*\)|\bshuffle=True", src_files)
line_grep("writes_next_to_source", r"with_suffix|\.parent\s*/|with_name|write_text|write_bytes|\.to_csv\(|\.savefig\(|open\([^)]*['\"][wax]b?['\"]", src_files)
line_grep("fixed_temp_paths", r"['\"]/tmp|['\"]/var/tmp|C:\\\\Temp|gettempdir", src_files + test_files)

# 7 exports vs docs
try:
    sys.path.insert(0, str(ROOT / "src"))
    import preflight
    docs = "\n".join((ROOT / p).read_text() for p in ("README.md", "docs/GUIDE.md") if (ROOT / p).exists())
    exp = list(getattr(preflight, "__all__", []))
    for name in exp:
        if not re.search(rf"\b{re.escape(name)}\b", docs):
            hits["export_undocumented"].append(f"preflight.{name} not in README/GUIDE")
    public = [n for n in dir(preflight) if not n.startswith("_") and n not in exp and not isinstance(getattr(preflight, n), type(sys))]
    for n in public:
        hits["export_not_in___all__"].append(f"preflight.{n} importable but not in __all__")
    for m in re.findall(r"preflight\.(\w+)\(", docs):
        if m not in dir(preflight):
            hits["docs_name_not_exported"].append(f"docs mention preflight.{m}()")
except Exception as e:  # noqa: BLE001
    hits["export_check_error"].append(repr(e))

# 8 dependencies
import tomllib  # noqa: E402

pp = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
declared = {}
for spec in pp["dependencies"]:
    declared[re.split(r"[<>=!~\[ ]", spec)[0].lower()] = ("runtime", spec)
for grp, specs in pp.get("optional-dependencies", {}).items():
    for spec in specs:
        declared.setdefault(re.split(r"[<>=!~\[ ]", spec)[0].lower(), (grp, spec))
PERMISSIVE = re.compile(r"MIT|BSD|Apache|ISC|PSF|Python Software|MPL|Zlib|HPND|Unlicense", re.I)
dep_rows = []
for name, (grp, spec) in sorted(declared.items()):
    try:
        md = metadata.metadata(name)
        lic = md.get("License-Expression") or md.get("License") or ""
        if not lic or len(lic) > 80:
            lic = "; ".join(c.split("::")[-1].strip() for c in md.get_all("Classifier", []) if c.startswith("License")) or (lic[:60] + "…" if lic else "UNKNOWN")
        ver = md["Version"]
    except metadata.PackageNotFoundError:
        lic, ver = "NOT INSTALLED", "?"
    flags = []
    if not PERMISSIVE.search(lic):
        flags.append("NON-PERMISSIVE/UNKNOWN")
    if not re.search(r"==|~=", spec):
        flags.append("unpinned" if not re.search(r"[<>]=?", spec) else "lower-bound-only")
    dep_rows.append(f"{name} [{grp}] {ver} | {lic} | {spec} | {','.join(flags)}")
    if "NON-PERMISSIVE/UNKNOWN" in flags:
        hits["dep_nonpermissive"].append(f"{name}: {lic}")
    if "unpinned" in flags and grp == "runtime":
        hits["dep_unpinned_runtime"].append(name)
imports = defaultdict(set)
for f in src_files:
    for n in ast.walk(ast.parse(f.read_text())):
        mods = [n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else [a.name for a in n.names] if isinstance(n, ast.Import) else []
        for m in mods:
            imports[m.split(".")[0]].add(rel(f))
dist_map = metadata.packages_distributions()
used_dists = set()
for m, fs in imports.items():
    if m in sys.stdlib_module_names or m == "preflight":
        continue
    ds = {d.lower().replace("_", "-") for d in dist_map.get(m, [m])}
    used_dists |= ds
    if not ds & {k.replace("_", "-") for k in declared}:
        hits["dep_undeclared_import"].append(f"{m} used in {sorted(fs)[0]}")
for name, (grp, _) in declared.items():
    if grp == "runtime" and name.replace("_", "-") not in used_dists:
        hits["dep_declared_unused_in_src"].append(name)

# 9 git history
def git(*a):
    r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None

if git("rev-parse", "--is-inside-work-tree") == "true":
    ch = git("log", "--diff-filter=M", "--name-only", "--pretty=format:@%h %ad %s", "--date=short", "--", "corpus/labels")
    cur = None
    for ln in (ch or "").splitlines():
        if ln.startswith("@"):
            cur = ln
        elif ln.strip():
            hits["label_modified_after_creation"].append(f"{ln} <- {cur}")
    ts = git("log", "-1", "--format=%H %ad", "--date=short", "--diff-filter=A", "--", "docs/v2_infer_report.md")
    if ts:
        h = ts.split()[0]
        ch = git("log", f"{h}..HEAD", "--pretty=format:@%h %ad %s", "--date=short", "--name-only", "--", "src/preflight/infer", "src/preflight/clean", "src/preflight/engineer", "src/preflight/ingest")
        cur = None
        for ln in (ch or "").splitlines():
            if ln.startswith("@"):
                cur = ln
            elif ln.strip():
                hits["src_changed_after_infer_report"].append(f"{ln} <- {cur}")
    else:
        hits["git_note"].append("docs/v2_infer_report.md add-commit not found")
else:
    hits["git_note"].append("git history unavailable")

with OUT.open("w") as fh:
    fh.write("# Static findings\n\n## Dependencies (name [group] ver | licence | spec | flags)\n")
    fh.writelines(f"- {r}\n" for r in dep_rows)
    for cat in sorted(hits):
        fh.write(f"\n## {cat} ({len(hits[cat])})\n")
        fh.writelines(f"- {h}\n" for h in hits[cat][:200])
for cat in sorted(hits):
    print(f"{cat}: {len(hits[cat])}")
