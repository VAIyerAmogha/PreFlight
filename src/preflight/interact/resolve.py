"""
resolve(): turn a Plan's open questions into final answers.

Steps:
  1. Plan (from a Plan object, a data file, or a review.json written earlier).
  2. Collect every needs_attention Decision into ONE batched session.
  3. Apply stored answers whose fingerprint matches exactly (marked 'confirmed').
  4. Resolve what is left according to the mode (ask / defaults / strict).
  5. Persist new answers, then write plan.json and only then render the Plan PDF.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd

from preflight.interact.channels.base import BaseChannel, ReviewPending
from preflight.interact.channels.callback import CallbackChannel
from preflight.interact.channels.notebook import NotebookChannel
from preflight.interact.channels.review_file import (
    ReviewFileChannel,
    load_review_file,
)
from preflight.interact.channels.terminal import TerminalChannel
from preflight.interact.modes import (
    apply_answer,
    apply_ask_mode,
    apply_defaults_mode,
    apply_strict_mode,
    in_notebook_kernel,
    resolve_mode,
)
from preflight.interact.question import ReviewSession, create_review_session
from preflight.interact.store import AnswerStore
from preflight.plan import ApplyContext, Plan, ReadingStage, plan as make_plan, write_plan_artifacts

ChannelArg = Union[str, BaseChannel, Callable[..., Any], None]


def _load_raw(plan_obj: Plan) -> Optional[pd.DataFrame]:
    """Raw values for previews. If the file cannot be re-read, previews say so."""
    try:
        return ReadingStage().apply(
            ApplyContext(source_path=Path(plan_obj.source_path)), plan_obj.decisions
        )
    except Exception:
        return None


def _default_out_dir(source: Path) -> Path:
    return source.parent / "preflight_plan"


def _pick_channel(channel: ChannelArg, out_dir: Path, meta: Dict[str, Any]) -> BaseChannel:
    if isinstance(channel, BaseChannel):
        return channel
    if channel is None:
        if in_notebook_kernel():
            try:
                import ipywidgets  # noqa: F401

                return NotebookChannel()
            except ImportError:
                pass  # no widgets installed: fall back to the terminal prompt
        return TerminalChannel()
    if callable(channel):
        return CallbackChannel(channel)
    name = str(channel).lower()
    if name == "terminal":
        return TerminalChannel()
    if name == "notebook":
        return NotebookChannel()
    if name == "review_file":
        return ReviewFileChannel(out_dir / "review.json", meta=meta)
    raise ValueError(
        f"Unknown channel {channel!r}; use 'terminal', 'notebook', 'review_file', or a callable."
    )


def resolve(
    source_or_plan: Union[Plan, str, Path],
    mode: Optional[str] = None,
    channel: ChannelArg = None,
    out_dir: Optional[Union[str, Path]] = None,
    target: Optional[str] = None,
    task: Optional[str] = None,
) -> Tuple[Plan, Path]:
    """
    Resolve every question in a Plan and write the final plan.json and Plan PDF.

    source_or_plan: a Plan (resolved in place), a data file, or the path of a review.json
        whose answers are filled in (mode is then 'ask' through the review file).
    mode: 'ask', 'defaults' or 'strict'. None selects strict in a non-interactive
        context (no TTY, no notebook kernel) and ask otherwise.
    channel: 'terminal', 'notebook', 'review_file', a Channel object, or a callable
        taking the list of Questions and returning {question_id: option}.
    out_dir: where plan.json, plan.pdf, answers.json and review.json go. Never the
        source file's own folder.

    Returns (Plan, path of the Plan PDF).
    Raises StrictResolutionError (strict), ReviewPending (answers not given yet; nothing
    is waited for), or ValueError subclasses naming the question for bad answers.
    """
    meta: Dict[str, Any] = {}
    review_path: Optional[Path] = None

    if isinstance(source_or_plan, Plan):
        plan_obj = source_or_plan
        source = Path(plan_obj.source_path).resolve()
    else:
        p = Path(source_or_plan).resolve()
        if p.suffix.lower() == ".json" and p.exists():
            data = load_review_file(p)  # raises ReviewFileError if it is not a review file
            review_path = p
            source = Path(data["source_path"])
            target = target if target is not None else data.get("target")
            task = task if task is not None else data.get("task")
            if out_dir is None and data.get("out_dir"):
                out_dir = data["out_dir"]
            mode = mode or "ask"
            channel = channel if channel is not None else "review_file"
        else:
            source = p
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        plan_obj = make_plan(source, target=target, task=task)

    out_d = Path(out_dir).resolve() if out_dir is not None else _default_out_dir(source)
    if out_d == source.parent:
        raise ValueError(
            f"out_dir must not be the source file's own folder ({out_d}); "
            "PreFlight never writes next to the source file."
        )
    out_d.mkdir(parents=True, exist_ok=True)

    ids = [d.id for d in plan_obj.decisions]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Plan has duplicate decision ids, cannot resolve unambiguously: {dupes}")
    index = {d.id: d for d in plan_obj.decisions}

    active_mode = resolve_mode(mode)
    meta = {"out_dir": str(out_d), "target": plan_obj.target_column, "task": plan_obj.task_type}

    session = create_review_session(
        plan_obj.decisions,
        plan_obj.source_path,
        raw_df=_load_raw(plan_obj) if any(d.needs_attention for d in plan_obj.decisions) else None,
        schema=plan_obj.schema,
    )

    # Stored answers first, in every mode.
    store = AnswerStore(out_d)
    remaining = []
    for q in session.questions:
        rec = store.lookup(q)
        if rec is None:
            remaining.append(q)
        else:
            apply_answer(index[q.decision_id], q, rec.answer, source="stored answer (same fingerprint)", reuse=True)
    session = ReviewSession(source_path=session.source_path, questions=remaining)

    if active_mode == "strict":
        apply_strict_mode(session)
    elif active_mode == "defaults":
        apply_defaults_mode(session, plan_obj.decisions)
    else:  # ask
        if session.questions:
            chan = _pick_channel(channel, out_d, meta)
            if review_path is not None and isinstance(chan, ReviewFileChannel):
                chan.path = review_path
            answers = chan.review(session)  # validated: every question has one of its options
            statuses = apply_ask_mode(session, plan_obj.decisions, answers, source=f"ask: {chan.name} channel")
            for q in session.questions:
                store.record(q, answers[q.question_id], statuses[q.question_id])
            store.save()

    plan_json, plan_pdf = write_plan_artifacts(plan_obj, out_d)
    return plan_obj, plan_pdf


__all__ = ["resolve", "ReviewPending"]
