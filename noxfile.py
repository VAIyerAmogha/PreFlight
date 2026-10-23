"""Nox sessions for tests, lint, and type checking.

Python matrix: 3.9 is the project's declared floor (requires-python = ">=3.9"
in pyproject.toml). 3.12 is the newest interpreter actually installed and
verified on this machine at the time this file was written (checked via
`which -a python3.9 python3.10 python3.11 python3.12 python3.13`: only
python3.12 was found). Sessions for 3.9/3.10/3.11/3.13 are declared for CI
(which can provision them) but are UNVERIFIED locally - nox will fail to
find/create those interpreters on this machine unless they are installed.
"""

import nox

PYTHON_VERSIONS = ["3.9", "3.12"]

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
