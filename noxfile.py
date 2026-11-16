"""Nox sessions for tests, lint, and type checking.

Python matrix: 3.10 is the project's declared floor (requires-python = ">=3.10"
in pyproject.toml); CI runs 3.10, 3.11 and 3.12. Only 3.12 is installed on the
development machine, so the 3.10 and 3.11 sessions are verified in CI only.
"""

import nox

PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]

nox.options.sessions = ["tests", "lint", "type_check"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(".[dev]")
    session.run("pytest", "-q", *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def lint(session: nox.Session) -> None:
    """Run ruff lint checks."""
    session.install("ruff")
    session.run("ruff", "check", "src/", "tests/")


@nox.session(python=PYTHON_VERSIONS)
def type_check(session: nox.Session) -> None:
    """Run mypy strict type checking on src/preflight."""
    session.install(".[dev]")
    session.run("mypy", "--strict", "src/preflight")
