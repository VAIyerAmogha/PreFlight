"""
Deterministic corruption generator for hardening tests.

Takes real corpus files from `corpus/raw/` and produces byte-level corrupted
variants covering a fixed catalogue of real-world damage patterns. Every
generator is seeded with `random.Random(seed)` — no wall-clock or OS entropy —
so the same (file, kind, seed) always yields byte-identical output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CorruptionVariant:
    """One corrupted byte-string variant plus a label describing its kind."""
    kind: str
    data: bytes
    description: str


def _read_lines(raw_text: str) -> list[str]:
    return raw_text.splitlines()


# ---------------------------------------------------------------------------
# Individual corruption kinds
# ---------------------------------------------------------------------------

def corrupt_wrong_encoding(raw_text: str, seed: int) -> bytes:
    """Re-encode as latin-1/cp1252, injecting non-ASCII bytes so a utf-8
    decode attempt will choke or mangle characters."""
    rng = random.Random(seed)
    # Sprinkle a few high-byte latin-1 characters (e.g. accented letters) into
    # the text so the file is not accidentally pure-ASCII (which is valid
    # under both encodings and wouldn't exercise the detector).
    chars = list(raw_text)
    extras = ["é", "ñ", "ü", "ö", "ç", "ß"]
    for _ in range(min(5, len(chars))):
        if not chars:
            break
        idx = rng.randrange(len(chars))
        chars.insert(idx, rng.choice(extras))
    mangled = "".join(chars)
    encoding = rng.choice(["latin-1", "cp1252"])
    return mangled.encode(encoding, errors="replace")


def corrupt_mixed_delimiters(raw_text: str, seed: int) -> bytes:
    """Swap the delimiter on a random subset of lines to a different one."""
    rng = random.Random(seed)
    lines = _read_lines(raw_text)
    if not lines:
        return raw_text.encode("utf-8")
    delim_candidates = [",", ";", "\t", "|"]
    primary = max(delim_candidates, key=lambda d: lines[0].count(d))
    alt = rng.choice([d for d in delim_candidates if d != primary])
    out_lines = []
    for i, line in enumerate(lines):
        if i > 0 and rng.random() < 0.3:
            out_lines.append(line.replace(primary, alt, rng.randint(1, 3)))
        else:
            out_lines.append(line)
    return "\n".join(out_lines).encode("utf-8")


def corrupt_extra_title_rows(raw_text: str, seed: int) -> bytes:
    """Prepend shifted title/preamble rows above the real header."""
    rng = random.Random(seed)
    lines = _read_lines(raw_text)
    titles = [
        "Company Confidential Report",
        f"Generated on 2024-0{rng.randint(1, 9)}-01",
        "",
        "Source: internal export",
    ]
    n = rng.randint(1, len(titles))
    prefix = titles[:n]
    return "\n".join(prefix + lines).encode("utf-8")


SENTINELS = ["N/A", "-999", "NULL", "n/a", "NaN", "#N/A", "null"]


def corrupt_sentinel_values(raw_text: str, seed: int) -> bytes:
    """Replace a deterministic subset of clean cell values with sentinel markers."""
    rng = random.Random(seed)
    lines = _read_lines(raw_text)
    if len(lines) < 2:
        return raw_text.encode("utf-8")
    delim = max([",", ";", "\t", "|"], key=lambda d: lines[0].count(d))
    out = [lines[0]]
    for line in lines[1:]:
        cells = line.split(delim)
        for _ in range(min(2, len(cells))):
            if not cells:
                break
            idx = rng.randrange(len(cells))
            if cells[idx].strip():
                cells[idx] = rng.choice(SENTINELS)
        out.append(delim.join(cells))
    return "\n".join(out).encode("utf-8")


def corrupt_ragged_rows(raw_text: str, seed: int) -> bytes:
    """Drop or add fields on a subset of rows so column counts disagree."""
    rng = random.Random(seed)
    lines = _read_lines(raw_text)
    if len(lines) < 2:
        return raw_text.encode("utf-8")
    delim = max([",", ";", "\t", "|"], key=lambda d: lines[0].count(d))
    out = [lines[0]]
    for line in lines[1:]:
        cells = line.split(delim)
        action = rng.choice(["drop", "add", "none"])
        if action == "drop" and len(cells) > 1:
            cells = cells[:-1]
        elif action == "add":
            cells = cells + ["extra_field"]
        out.append(delim.join(cells))
    return "\n".join(out).encode("utf-8")


def corrupt_duplicate_columns(raw_text: str, seed: int) -> bytes:
    """Duplicate one header column name onto another header position."""
    rng = random.Random(seed)
    lines = _read_lines(raw_text)
    if not lines:
        return raw_text.encode("utf-8")
    delim = max([",", ";", "\t", "|"], key=lambda d: lines[0].count(d))
    header = lines[0].split(delim)
    if len(header) >= 2:
        src = rng.randrange(len(header))
        dst = rng.randrange(len(header))
        header[dst] = header[src]
    lines[0] = delim.join(header)
    return "\n".join(lines).encode("utf-8")


def corrupt_embedded_nul(raw_text: str, seed: int) -> bytes:
    """Insert embedded NUL bytes at deterministic positions."""
    rng = random.Random(seed)
    data = bytearray(raw_text.encode("utf-8"))
    if not data:
        return bytes(data)
    n_nuls = min(5, len(data))
    for _ in range(n_nuls):
        idx = rng.randrange(len(data))
        data.insert(idx, 0)
    return bytes(data)


def corrupt_line_endings_crlf(raw_text: str, seed: int) -> bytes:
    lines = _read_lines(raw_text)
    return "\r\n".join(lines).encode("utf-8")


def corrupt_line_endings_lf(raw_text: str, seed: int) -> bytes:
    lines = _read_lines(raw_text)
    return "\n".join(lines).encode("utf-8")


def corrupt_line_endings_cr(raw_text: str, seed: int) -> bytes:
    lines = _read_lines(raw_text)
    return "\r".join(lines).encode("utf-8")


def corrupt_quoted_delims_and_newlines(raw_text: str, seed: int) -> bytes:
    """Wrap a cell value in quotes containing the delimiter and an embedded newline."""
    rng = random.Random(seed)
    lines = _read_lines(raw_text)
    if len(lines) < 2:
        return raw_text.encode("utf-8")
    delim = max([",", ";", "\t", "|"], key=lambda d: lines[0].count(d))
    data_idx = rng.randrange(1, len(lines))
    cells = lines[data_idx].split(delim)
    if cells:
        col = rng.randrange(len(cells))
        quoted = f'"value with {delim} and\nembedded newline"'
        cells[col] = quoted
    lines[data_idx] = delim.join(cells)
    return "\n".join(lines).encode("utf-8")


CORRUPTIONS: dict[str, Callable[[str, int], bytes]] = {
    "wrong_encoding": corrupt_wrong_encoding,
    "mixed_delimiters": corrupt_mixed_delimiters,
    "extra_title_rows": corrupt_extra_title_rows,
    "sentinel_values": corrupt_sentinel_values,
    "ragged_rows": corrupt_ragged_rows,
    "duplicate_columns": corrupt_duplicate_columns,
    "embedded_nul": corrupt_embedded_nul,
    "crlf_endings": corrupt_line_endings_crlf,
    "lf_endings": corrupt_line_endings_lf,
    "cr_endings": corrupt_line_endings_cr,
    "quoted_delims_newlines": corrupt_quoted_delims_and_newlines,
}


def generate_corruption(raw_text: str, kind: str, seed: int = 1234) -> CorruptionVariant:
    """Produce one deterministic corrupted variant of `raw_text`."""
    fn = CORRUPTIONS[kind]
    data = fn(raw_text, seed)
    return CorruptionVariant(kind=kind, data=data, description=fn.__doc__ or kind)


def all_kinds() -> list[str]:
    return list(CORRUPTIONS.keys())
