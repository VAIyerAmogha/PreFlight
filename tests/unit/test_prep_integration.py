"""
Plan integration of the cleaning and engineering steps (Phase 6 / M5, step 6.4).

Covers: the four plan sections are populated, dry_run == actual run across ALL stages,
dry_run purity, trigger wiring for destructive and leakage Decisions, and plain-language
text for every new Decision kind.
"""

import hashlib
import json
import re
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pypdf
import pytest

from preflight.engineer.config import FeatureConfig
from preflight.interact.triggers import TriggerCategory, TriggerRegistry
from preflight.plan import apply, dry_run, plan
from preflight.report.templates import render_decision_text, render_question_text, section_for_decision

from test_jargon_lint import STRICT_BANNED_PATTERNS


def _messy_csv(tmp_path: Path, rows: int = 120) -> Path:
    rng = np.random.default_rng(0)
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    lines = ["row_id,city,size,price,when,notes,constant,leak,target"]
    cities = ["Paris", " paris", "Rome", "Oslo", "Lima", "Kyiv"] + [f"Town{i}" for i in range(30)]
    for i in range(rows):
        price = "" if i % 9 == 0 else f"{rng.normal(100, 15):.2f}"
        size = f"{rng.normal(50, 10):.1f}" if i != 5 else "9999.0"
        y = i % 2
        notes = "short" if i % 3 else "a considerably longer free text note that runs on and on for many words here"
        lines.append(f"{i},{cities[i % len(cities)]},{size},{price},2020-0{(i % 9) + 1}-1{i % 9},{notes},same,{y * 1.0001 + i * 1e-6:.6f},{y}")
    lines.append(lines[3])  # one repeated row
    f = d / "messy.csv"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


def _comparable(d):
    skip = {"status", "timestamp", "history"}
    return {f.name: getattr(d, f.name) for f in fields(d) if f.name not in skip}


def _pdf_text(path: Path) -> str:
    return "\n".join((p.extract_text() or "") for p in pypdf.PdfReader(str(path)).pages)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


FULL = FeatureConfig(interactions=True, datetime_cyclical=True, datetime_deltas=True, clustering=True,
                     text_features=True, text_tfidf=True)


def test_all_four_sections_are_populated(tmp_path):
    src = _messy_csv(tmp_path)
    p = plan(src, target="target", model_hint="linear", feature_config=FULL)
    sections = {section_for_decision(d) for d in p.decisions}
    assert {"fixing", "filling gaps", "removing", "creating"} <= sections


@pytest.mark.parametrize("hint,cfg", [("tree", None), ("linear", FULL)])
def test_dry_run_equals_actual_run_across_all_stages(tmp_path, hint, cfg):
    src = _messy_csv(tmp_path)
    dry, _ = dry_run(src, target="target", out_dir=tmp_path / "dry", mode="defaults", model_hint=hint, feature_config=cfg)
    run = apply(plan(src, target="target", model_hint=hint, feature_config=cfg), mode="defaults", out_dir=tmp_path / "run")
    assert len(dry.decisions) == len(run.decisions)
    assert any(d.stage in ("clean", "engineer") for d in dry.decisions)
    for a, b in zip(dry.decisions, run.decisions):
        assert _comparable(a) == _comparable(b)
    assert run.prepared is not None and len(run.prepared) > 0


def test_dry_run_is_pure_for_cleaning_and_engineering(tmp_path):
    src = _messy_csv(tmp_path)
    before_src = _sha(src)
    out = tmp_path / "out"
    dry_run(src, target="target", out_dir=out, model_hint="linear", feature_config=FULL)
    assert _sha(src) == before_src
    assert sorted(p.name for p in out.iterdir()) == ["plan.json", "plan.pdf"]
    def _plan_without_timestamps():
        def strip(x):
            if isinstance(x, dict):
                return {k: strip(v) for k, v in x.items() if k != "timestamp"}
            if isinstance(x, list):
                return [strip(v) for v in x]
            return x
        return strip(json.loads((out / "plan.json").read_text()))

    first = _plan_without_timestamps()
    dry_run(src, target="target", out_dir=out, model_hint="linear", feature_config=FULL)
    assert _sha(src) == before_src
    assert _plan_without_timestamps() == first  # deterministic


def test_plan_json_round_trips_with_prep_settings(tmp_path):
    from preflight.plan import Plan
    src = _messy_csv(tmp_path)
    p = plan(src, target="target", model_hint="linear", feature_config=FULL)
    q = Plan.from_json(p.to_json())
    assert q.model_hint == "linear" and q.feature_config == FULL
    assert [d.id for d in q.decisions] == [d.id for d in p.decisions]


def test_plan_pdf_lists_new_sections_in_plain_language(tmp_path):
    src = _messy_csv(tmp_path)
    _, pdf = dry_run(src, target="target", out_dir=tmp_path / "o", mode="defaults", model_hint="linear", feature_config=FULL)
    text = _pdf_text(pdf).lower()
    for pat in STRICT_BANNED_PATTERNS:
        assert re.search(pat, text) is None, f"banned term {pat!r} in Plan PDF"
    after = text.split("what we will do", 1)[1].split("what we won't touch", 1)[0]
    after = re.sub(r"\s+", " ", after)
    for head, nxt in (("fixing", "filling gaps"), ("filling gaps", "removing"), ("removing", "creating")):
        seg = after.split(head, 1)[1].split(nxt, 1)[0]
        assert "nothing to do at this stage" not in seg, f"section {head!r} is empty"
    assert "nothing to do at this stage" not in after.split("creating", 1)[1][:60]


def test_every_new_decision_kind_has_jargon_free_text(tmp_path):
    src = _messy_csv(tmp_path)
    p = plan(src, target="target", model_hint="linear", feature_config=FULL)
    seen = set()
    for d in p.decisions:
        if d.stage not in ("clean", "engineer"):
            continue
        seen.add(d.kind)
        texts = [render_decision_text(d).summary, d.plain_language]
        if d.needs_attention:
            q = render_question_text(d)
            texts += [q.question or "", q.explanation or "", *q.options]
        for t in texts:
            for pat in STRICT_BANNED_PATTERNS:
                assert re.search(pat, t.lower()) is None, f"{d.kind}: banned {pat!r} in {t!r}"
    assert len(seen) >= 8


def test_destructive_decision_asks_and_default_keeps_the_data(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    rows = ["a,b,target"] + [f"{i % 4},{i % 5},{i % 2}" for i in range(10)] * 3  # heavy repetition
    f = d / "dupes.csv"
    f.write_text("\n".join(rows) + "\n")
    p = plan(f, target="target")
    dup = [x for x in p.decisions if x.kind == "remove_duplicates"]
    assert len(dup) == 1 and dup[0].needs_attention
    assert dup[0].default == "keep_rows" and dup[0].action == "remove_duplicates"
    reg = TriggerRegistry()
    assert reg.classify(dup[0]) == TriggerCategory.DESTRUCTIVE_ACTION

    kept = apply(plan(f, target="target"), mode="defaults", out_dir=tmp_path / "o1")
    assert len(kept.prepared) == 30  # safe default keeps every row
    chosen = apply(plan(f, target="target"), mode="ask",
                   channel=lambda qs: {q.question_id: next(o for o in q.options if "Remove the repeated" in o)
                                       if "repeated" in " ".join(q.options) else q.recommended_default for q in qs},
                   out_dir=tmp_path / "o2")
    assert len(chosen.prepared) == 10


def test_leakage_decision_asks_and_answer_is_applied(tmp_path):
    src = _messy_csv(tmp_path)
    p = plan(src, target="target")
    leak = [x for x in p.decisions if x.kind == "suspected_leakage"]
    assert len(leak) == 1 and leak[0].columns == ["leak"] and leak[0].needs_attention
    assert TriggerRegistry().classify(leak[0]) == TriggerCategory.SUSPECTED_LEAKAGE

    kept = apply(plan(src, target="target"), mode="defaults", out_dir=tmp_path / "o1")
    assert "leak" in kept.prepared.columns

    proposed = {d.id: d.action for d in p.decisions}

    def drop_it(qs):
        out = {}
        for q in qs:
            keep_proposal = next((o for o, a in q.option_actions.items() if a == proposed.get(q.decision_id)), q.recommended_default)
            out[q.question_id] = next((o for o in q.options if "Leave this column out" in o), keep_proposal)
        return out

    dropped = apply(plan(src, target="target"), mode="ask", channel=drop_it, out_dir=tmp_path / "o2")
    assert "leak" not in dropped.prepared.columns
