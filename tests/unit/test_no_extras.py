"""
Core import and the terminal channel work with neither `rich` nor `ipywidgets` installed
(Phase 5 / M4). Simulated by making both un-importable in a fresh interpreter.
"""

import subprocess
import sys
import textwrap

from conftest import make_messy_csv

BLOCK = textwrap.dedent(
    """
    import sys
    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("rich", "ipywidgets", "IPython"):
                raise ImportError(f"{name} blocked for this test")
    sys.meta_path.insert(0, Block())
    for m in [m for m in sys.modules if m.split(".")[0] in ("rich", "ipywidgets", "IPython")]:
        del sys.modules[m]
    """
)


def run(code, timeout=120):
    return subprocess.run(
        [sys.executable, "-c", BLOCK + textwrap.dedent(code)],
        capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
    )


def test_core_import_and_exports_without_extras():
    r = run(
        """
        import preflight
        assert callable(preflight.resolve)
        import preflight.interact.channels as ch
        assert not any(m.split(".")[0] in ("rich", "ipywidgets") for m in sys.modules)
        print("OK")
        """
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr


def test_terminal_channel_works_without_extras(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    src = make_messy_csv(d / "messy.csv")
    r = run(
        f"""
        import io
        from preflight import resolve
        from preflight.interact.channels import TerminalChannel
        from preflight.interact.channels.terminal import rich_available
        assert rich_available() is False
        out = io.StringIO()
        ch = TerminalChannel(stdin=io.StringIO("default\\n" * 50), stdout=out)
        plan, pdf = resolve({str(src)!r}, mode="ask", channel=ch, out_dir={str(tmp_path / "out")!r}, target="target")
        assert pdf.exists()
        assert "recommended" in out.getvalue()
        assert not any(m.split(".")[0] in ("rich", "ipywidgets") for m in sys.modules)
        print("OK")
        """
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr


def test_notebook_channel_gives_install_hint_without_ipywidgets(tmp_path):
    r = run(
        """
        from preflight.interact.channels import NotebookChannel
        from preflight.interact.question import ReviewSession, Question
        q = Question("q1", "d", "f", "ambiguous_type", None, "t", "e", "s", ["a", "b"],
                     {"a": "x", "b": "y"}, "a")
        try:
            NotebookChannel().review(ReviewSession("/x", [q]))
        except ImportError as exc:
            assert "pypreflight[notebook]" in str(exc)
            print("OK")
        """
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr
