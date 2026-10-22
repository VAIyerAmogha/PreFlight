"""Preflight package."""

from .decisions import Decision
from .interact.resolve import resolve
from .plan import Plan, apply, dry_run, plan
from .schema import TableSchema

__version__ = "2.0.0.dev0"

__all__ = [
    "Decision",
    "Plan",
    "TableSchema",
    "apply",
    "dry_run",
    "plan",
    "resolve",
    "__version__",
]
