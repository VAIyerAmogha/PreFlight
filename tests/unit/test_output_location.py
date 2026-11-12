"""
Output location and the 'auto' default mode (Phase 7b): outputs go to preflight_out/ in
the current working directory, nothing is ever created next to the source file, and a
non-interactive run with open questions stops with an error that says how to proceed.
"""
from pathlib import Path

import pytest
from conftest import make_messy_csv

from preflight.interact import modes
from preflight.interact.modes import StrictResolutionError, resolve_mode
from preflight.interact.resolve import resolve
from preflight.plan import dry_run
from preflight.prepare import prepare


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = tmp_path / "data"
    data.mkdir()
    src = make_messy_csv(data / "messy.csv")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return src, data, work


def _listing(d: Path):
    return sorted(p.name for p in d.iterdir())


@pytest.mark.parametrize("entry", ["prepare", "dry_run", "resolve"])
def test_nothing_is_created_next_to_the_source_file(layout, entry, capsys):
    src, data, work = layout
    before = _listing(data)
    if entry == "prepare":
        prepare(src, target="target", mode="defaults")
    elif entry == "dry_run":
        dry_run(src, target="target", mode="defaults")
    else:
        resolve(src, mode="defaults", target="target")
    assert _listing(data) == before == ["messy.csv"]
    out = work / "preflight_out"
    assert (out / "plan.json").exists() and (out / "plan.pdf").exists()
    if entry == "prepare":
        assert (out / "report.pdf").exists() and (out / "pipeline.joblib").exists()
    if entry != "resolve":
        assert str(out) in capsys.readouterr().out  # the output folder is printed


def test_auto_is_ask_when_interactive_and_strict_otherwise(monkeypatch):
    monkeypatch.setattr(modes, "is_interactive_context", lambda: False)
    assert resolve_mode("auto") == resolve_mode(None) == "strict"
    monkeypatch.setattr(modes, "is_interactive_context", lambda: True)
    assert resolve_mode("auto") == resolve_mode(None) == "ask"
    assert resolve_mode("defaults") == "defaults"


@pytest.mark.parametrize("entry", ["prepare", "dry_run"])
def test_default_mode_is_auto_and_strict_errors_say_how_to_proceed(layout, entry, monkeypatch):
    src, data, _ = layout
    monkeypatch.setattr(modes, "is_interactive_context", lambda: False)
    with pytest.raises(StrictResolutionError) as err:
        if entry == "prepare":
            prepare(src, target="target")
        else:
            dry_run(src, target="target")
    msg = str(err.value)
    assert "mode='defaults'" in msg and "review file" in msg
    assert _listing(data) == ["messy.csv"]
