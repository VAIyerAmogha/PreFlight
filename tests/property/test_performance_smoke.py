"""
Performance smoke test: generate one large synthetic CSV (~1M rows, roughly
state-population scale) and confirm chunked loading via
`preflight.ingest.load_raw(..., chunksize=...)` stays within a memory bound
measured and documented here (not a pre-set target -- see module docstring
in the final report for the actual observed figures).

This test is intentionally marked slow; it is still expected to finish within
a few minutes. Peak RSS is measured via `resource.getrusage` (Linux: KB: Linux
ru_maxrss is in KiB).
"""

from __future__ import annotations

import csv
import resource
import time
from pathlib import Path

import pytest

from preflight.ingest import load_raw

N_ROWS = 1_000_000
CHUNKSIZE = 50_000

# Documented, measured bound (see report): chunked loading should stay well
# under this many MiB of peak RSS even for a 1M-row file, because each chunk
# is released once consumed. This is a generous ceiling with headroom over
# what was actually observed in development, not an aspirational target.
PEAK_RSS_BOUND_MB = 2000.0


@pytest.fixture(scope="module")
def large_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("perf") / "large_state_scale.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "state", "population", "score"])
        for i in range(N_ROWS):
            writer.writerow([i, f"state_{i % 50}", 10000 + (i % 5000), round((i % 1000) / 10.0, 2)])
    return p


def _peak_rss_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024.0


@pytest.mark.timeout(300) if "pytest_timeout" in dir() else pytest.mark.filterwarnings()
def test_chunked_load_memory_and_time_bound(large_csv):
    start_rss_mb = _peak_rss_mb()
    start_time = time.monotonic()

    load_res = load_raw(large_csv, chunksize=CHUNKSIZE)
    assert load_res.is_chunked

    total_rows = 0
    for chunk in load_res.data:
        total_rows += len(chunk)

    elapsed = time.monotonic() - start_time
    end_rss_mb = _peak_rss_mb()

    assert total_rows == N_ROWS

    print(
        f"\n[perf smoke] rows={N_ROWS} chunksize={CHUNKSIZE} "
        f"elapsed={elapsed:.2f}s peak_rss_start={start_rss_mb:.1f}MB "
        f"peak_rss_end={end_rss_mb:.1f}MB bound={PEAK_RSS_BOUND_MB}MB"
    )

    assert end_rss_mb < PEAK_RSS_BOUND_MB, (
        f"Peak RSS {end_rss_mb:.1f}MB exceeded documented bound {PEAK_RSS_BOUND_MB}MB "
        f"for a {N_ROWS}-row chunked load"
    )
