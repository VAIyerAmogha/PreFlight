"""
Nothing may ever block waiting for input in a non-interactive context (Phase 5 / M4).

Each case runs the pipeline in a subprocess with stdin closed (or an open pipe that
never delivers data) under a hard timeout. A hang fails the test.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from conftest import make_messy_csv

HARD_TIMEOUT = 120  # seconds; planning a tiny file takes a few seconds

SCRIPT = textwrap.dedent(
    """
    import os, sys
    if {close_fd0}:
        os.close(0)
    from preflight import resolve
    mode = {mode!r}
    kwargs = dict(out_dir={out!r}, target="target")
    if mode is not None:
        kwargs["mode"] = mode
    if {channel!r} is not None:
        kwargs["channel"] = {channel!r}
    try:
        resolve({src!r}, **kwargs)
    except Exception as exc:
        print("RAISED", type(exc).__name__, file=sys.stderr)
        sys.exit(3)
    print("OK")
    """
)


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return make_messy_csv(d / "messy.csv")


def run(src, tmp_path, mode, channel=None, stdin="devnull", close_fd0=False):
    code = SCRIPT.format(close_fd0=close_fd0, mode=mode, channel=channel, out=str(tmp_path / "out"), src=str(src))
    stdin_arg = {"devnull": subprocess.DEVNULL, "pipe": subprocess.PIPE, "inherit-none": None}[stdin]
    proc = subprocess.Popen(
        [sys.executable, "-c", code], stdin=stdin_arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        out, err = proc.communicate(timeout=HARD_TIMEOUT)  # for "pipe": stdin is open and silent
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        pytest.fail(f"HANG: subprocess still running after {HARD_TIMEOUT}s (mode={mode}, channel={channel})")
    return proc.returncode, out, err


@pytest.mark.parametrize("stdin", ["devnull", "pipe"])
def test_default_context_selects_strict_and_exits(src, tmp_path, stdin):
    code, out, err = run(src, tmp_path, mode=None, stdin=stdin)
    assert code == 3
    assert "StrictResolutionError" in err


def test_default_context_with_closed_stdin_fd(src, tmp_path):
    code, out, err = run(src, tmp_path, mode=None, close_fd0=True)
    assert code == 3 and "StrictResolutionError" in err


@pytest.mark.parametrize("stdin", ["devnull", "pipe"])
def test_explicit_strict_exits(src, tmp_path, stdin):
    code, out, err = run(src, tmp_path, mode="strict", stdin=stdin)
    assert code == 3 and "StrictResolutionError" in err


def test_defaults_mode_completes(src, tmp_path):
    code, out, err = run(src, tmp_path, mode="defaults")
    assert code == 0 and "OK" in out
    assert (tmp_path / "out" / "plan.pdf").exists()


@pytest.mark.parametrize("stdin", ["devnull", "pipe"])
def test_ask_terminal_without_a_tty_errors_instead_of_waiting(src, tmp_path, stdin):
    code, out, err = run(src, tmp_path, mode="ask", channel="terminal", stdin=stdin)
    assert code == 3 and "NonInteractiveError" in err


def test_ask_review_file_writes_and_exits(src, tmp_path):
    code, out, err = run(src, tmp_path, mode="ask", channel="review_file")
    assert code == 3 and "ReviewPending" in err
    assert (tmp_path / "out" / "review.json").exists()
