"""
Encoding detection module for PreFlight v2.0 ingest pipeline.

Checks for Byte Order Marks (BOM) first, then applies charset-normalizer
for non-BOM byte streams. If confidence is below threshold, flags as
low_confidence rather than guessing. Returns a typed EncodingResult object.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Union

import charset_normalizer

# How many other plausible encodings a low-confidence result lists for the person
MAX_ENCODING_ALTERNATIVES = 4

# Another candidate encoding is a rival when its charset-normalizer "chaos" (mess)
# score is within this margin of the best one AND it decodes the sample to different
# text. Single-byte code pages often tie this way (for example cp1252 vs cp775 for
# Western text), so a rival makes the result low confidence instead of a silent guess.
ENCODING_RIVAL_CHAOS_MARGIN = 0.10

# Recognized BOM byte signatures
BOM_SIGNATURES = [
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
]


@dataclass(frozen=True)
class EncodingResult:
    """Typed result object for encoding detection."""
    encoding: str
    confidence: float
    has_bom: bool
    low_confidence: bool
    raw_detected: Optional[str] = None
    # Other encodings the detector also found plausible (best first), for the person to choose from
    alternatives: tuple[str, ...] = ()


def normalize_encoding_name(name: str) -> str:
    """Normalize encoding string to standard python naming conventions."""
    if not name:
        return "utf-8"
    cleaned = name.strip().lower().replace("_", "-")
    alias_map = {
        "utf-8-sig": "utf-8-sig",
        "utf8": "utf-8",
        "utf-8": "utf-8",
        "ascii": "utf-8",
        "us-ascii": "utf-8",
        "latin-1": "iso-8859-1",
        "latin1": "iso-8859-1",
        "iso-8859-1": "iso-8859-1",
        "iso8859-1": "iso-8859-1",
        "windows-1252": "cp1252",
        "cp-1252": "cp1252",
        "windows-1250": "cp1250",
        "cp-1250": "cp1250",
    }
    return alias_map.get(cleaned, cleaned)


def detect_encoding(
    source: Union[str, Path, bytes, BinaryIO],
    sample_size: int = 131072,
    confidence_threshold: float = 0.70,
) -> EncodingResult:
    """
    Detect character encoding of tabular source bytes.

    Priority:
    1. Check for standard Byte Order Marks (BOM).
    2. Check if bytes decode strictly as valid UTF-8 without error.
    3. Apply charset-normalizer for multi-byte or single-byte legacy encodings.
    4. Flag low_confidence when confidence is below confidence_threshold.

    Args:
        source: File path, raw bytes, or binary stream.
        sample_size: Number of bytes to sample from file (default: 128 KB).
        confidence_threshold: Threshold below which detection is flagged.

    Returns:
        EncodingResult containing detected encoding, confidence, and flags.
    """
    # 1. Read byte sample
    raw_bytes: bytes
    if isinstance(source, bytes):
        raw_bytes = source
    elif isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        with open(p, "rb") as f:
            raw_bytes = f.read(sample_size)
    elif hasattr(source, "read"):
        raw_bytes = source.read(sample_size)
    else:
        raise TypeError(f"Unsupported source type for encoding detection: {type(source)}")

    if not raw_bytes:
        # Empty input defaults to utf-8 with low confidence
        return EncodingResult(
            encoding="utf-8",
            confidence=0.5,
            has_bom=False,
            low_confidence=True,
            raw_detected=None,
        )

    # 2. Check for BOM signatures
    for bom_bytes, bom_enc in BOM_SIGNATURES:
        if raw_bytes.startswith(bom_bytes):
            return EncodingResult(
                encoding=bom_enc,
                confidence=1.0,
                has_bom=True,
                low_confidence=False,
                raw_detected=bom_enc,
            )

    # 3. Check for pure ASCII or strict UTF-8 validity
    is_pure_ascii = all(b < 128 for b in raw_bytes)
    if is_pure_ascii:
        return EncodingResult(
            encoding="utf-8",
            confidence=1.0,
            has_bom=False,
            low_confidence=False,
            raw_detected="ascii",
        )

    # Try strict UTF-8 decoding
    try:
        raw_bytes.decode("utf-8")
        # Strict UTF-8 multi-byte sequence valid across entire sample
        return EncodingResult(
            encoding="utf-8",
            confidence=1.0,
            has_bom=False,
            low_confidence=False,
            raw_detected="utf-8",
        )
    except UnicodeDecodeError:  # Swallowed: not valid UTF-8, so the next detector below decides
        pass

    # 4. Use charset-normalizer for non-UTF8 input
    matches = charset_normalizer.from_bytes(raw_bytes)
    best_match = matches.best()

    if best_match is not None:
        raw_enc = best_match.encoding
        norm_enc = normalize_encoding_name(raw_enc)

        # Calculate confidence metric
        chaos = getattr(best_match, "chaos", 0.0) or 0.0
        coherence = getattr(best_match, "coherence", 0.0) or 0.0
        calc_conf = max(0.0, min(1.0, 1.0 - (chaos * 0.5)))

        # If linguistic coherence is zero on very short or ambiguous sample, flag low confidence.
        # A low-confidence result is kept as detected (never swapped for another encoding
        # here); the reading Decision flags it and lists the alternatives to choose from.
        best_text = str(best_match)
        rivals: list[tuple[float, str]] = []
        others: list[str] = []
        for m in matches:
            alt = normalize_encoding_name(m.encoding)
            if alt == norm_enc or alt in others or any(alt == r for _, r in rivals):
                continue
            close = abs((getattr(m, "chaos", 0.0) or 0.0) - chaos) <= ENCODING_RIVAL_CHAOS_MARGIN
            if close and str(m) != best_text:
                rivals.append((-(getattr(m, "coherence", 0.0) or 0.0), alt))
            else:
                others.append(alt)
        is_low_conf = (calc_conf < confidence_threshold) or (coherence == 0.0) or bool(rivals)
        alternatives = [r for _, r in sorted(rivals)] + others

        return EncodingResult(
            encoding=norm_enc,
            confidence=round(calc_conf, 4),
            has_bom=False,
            low_confidence=is_low_conf,
            raw_detected=raw_enc,
            alternatives=tuple(alternatives[:MAX_ENCODING_ALTERNATIVES]),
        )

    # 5. Fallback for unresolvable streams
    return EncodingResult(
        encoding="iso-8859-1",
        confidence=0.3,
        has_bom=False,
        low_confidence=True,
        raw_detected=None,
    )
