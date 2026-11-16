"""Session wiring for the corpus suites (see tests/corpus/corpus_support.py).

It lives at tests/ level so that the xdist controller loads it at startup. The controller
does not collect tests/corpus itself.

- Each session gets a fresh results directory under .pytest_cache/corpus_results/.
  xdist workers receive it through `workerinput`.
- A per-file test that hits its pytest-timeout limit is saved as a TIMEOUT record.
- When the session ends, the controller runs scripts/corpus_summary.py on the records. Any
  corpus-wide threshold that fails also fails the session.
"""

import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

# Same values as tests/corpus/corpus_support.py (that module imports preflight and is only
# importable once tests/corpus is collected, so it is not imported here).
ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / ".pytest_cache" / "corpus_results"
RUN_DIR_ENV = "PREFLIGHT_CORPUS_RUN_DIR"
_SUMMARY_SCRIPT = ROOT / "scripts" / "corpus_summary.py"


def _is_worker(config: pytest.Config) -> bool:
    return hasattr(config, "workerinput")


def pytest_configure(config: pytest.Config) -> None:
    if _is_worker(config):
        os.environ[RUN_DIR_ENV] = config.workerinput[RUN_DIR_ENV]
    elif RUN_DIR_ENV not in os.environ:
        run = RESULTS_ROOT / time.strftime("run_%Y%m%d_%H%M%S")
        os.environ[RUN_DIR_ENV] = str(run)


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    """xdist hook: pass the run directory to each worker."""
    node.workerinput[RUN_DIR_ENV] = os.environ[RUN_DIR_ENV]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> Any:
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    suite = getattr(item.module, "SUITE", None)
    callspec = getattr(item, "callspec", None)
    if suite is None or callspec is None or "corpus_file" not in callspec.params:
        return
    if "Timeout" in report.longreprtext:
        name = callspec.id
        out = Path(os.environ[RUN_DIR_ENV]) / suite
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps({"file": name, "status": "TIMEOUT"}), encoding="utf-8")


def _load_summary_module() -> Any:
    spec = importlib.util.spec_from_file_location("corpus_summary", _SUMMARY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _is_worker(session.config):
        return
    run = Path(os.environ[RUN_DIR_ENV])
    if not run.is_dir():
        return
    summary = _load_summary_module().summarize(run)
    (run / "summary.md").write_text("\n".join(summary["lines"]) + "\n", encoding="utf-8")
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep("=", "corpus summary")
        for line in summary["check_lines"]:
            reporter.write_line(line)
        reporter.write_line(f"full summary: {run / 'summary.md'}")
    if summary["failed"] and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
