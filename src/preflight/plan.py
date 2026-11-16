"""
PreFlight v2.0 Execution Plan and Dry-Run Orchestration Engine.

Exposes the stage-based planner and apply lifecycle:
- Stage interface: plan(context) -> List[Decision], apply(context, decisions) -> Any.
- Pure planning: plan() never mutates source files or produces cleaned data.
- typing apply: converts raw strings to inferred semantic dtypes without silent data loss.
  Any value that fails conversion is recorded in an anomalies record with its row and column.
- Task inference: infers classification vs. regression when a target column is specified,
  emitting a needs_attention Decision when inference is ambiguous.
- Registry architecture: stages are registered in execution order; cleaning and engineering steps (PrepStage) plan from one shared in-memory simulation.
- dry_run: writes plan.json and Plan PDF, with strict failure isolation.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from dateutil import parser as date_parser

from preflight.assemble import build_unfitted_pipeline, plan_time_column_kinds
from preflight.clean.cleaner import apply_coerce, plan_clean_full
from preflight.clean.config import DEFAULT_CLEANER_CONFIG, CleanerConfig
from preflight.clean.dedupe import apply_dedupe
from preflight.clean.drops import apply_drops
from preflight.clean.impute import apply_impute
from preflight.clean.markers import apply_markers
from preflight.clean.normalize import apply_normalization
from preflight.clean.outliers import apply_outliers
from preflight.clean.rows import apply_missing_target, plan_class_imbalance, plan_grouped_rows
from preflight.decisions import Decision
from preflight.engineer.config import FeatureConfig
from preflight.engineer.engineer import apply_engineer, plan_engineer
from preflight.infer import infer_table_types
from preflight.infer.types import strip_unit
from preflight.ingest import HeaderResult, detect_dialect, detect_encoding, detect_header, load_raw
from preflight.ingest.dialect import WHITESPACE_DELIMITER, split_rows
from preflight.report.plan_pdf import render_plan_pdf
from preflight.report.templates import (
    template_ambiguous_task,
    template_dialect,
    template_encoding,
    template_header,
)
from preflight.schema import (
    ColumnRawMetadata,
    DialectSchema,
    EncodingSchema,
    HeaderSchema,
    TableSchema,
)

# ===========================================================================
# Context & Result Objects
# ===========================================================================

@dataclass
class PlanContext:
    """Execution context passed through stage planning methods."""
    source_path: Path
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    raw_data: Optional[pd.DataFrame] = None
    schema: Optional[TableSchema] = None
    decisions: list[Decision] = field(default_factory=list)
    model_hint: str = "tree"
    feature_config: Optional[FeatureConfig] = None
    cleaner_config: CleanerConfig = field(default_factory=lambda: DEFAULT_CLEANER_CONFIG)
    # Cleaning and engineering Decisions, computed once on first use (see _ensure_prep_plan)
    prep_decisions: Optional[list[Decision]] = None


@dataclass
class ApplyContext:
    """Execution context passed through stage apply methods."""
    source_path: Path
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    raw_data: Optional[pd.DataFrame] = None
    typed_data: Optional[pd.DataFrame] = None
    schema: Optional[TableSchema] = None
    decisions: list[Decision] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    prepared_data: Optional[pd.DataFrame] = None
    engineer_specs: dict[str, Any] = field(default_factory=dict)
    model_hint: str = "tree"
    feature_config: Optional[FeatureConfig] = None


@dataclass
class ApplyResult:
    """
    Result returned by full pipeline apply execution.

    data: the typed table (Phase 4 contract). prepared: the full table after the steps
    that learn nothing from data (markers, numeric coercion, repeated rows, column
    removal, text tidying; see docs/fit_hygiene.md). pipeline: an UNFITTED sklearn
    Pipeline with every step that learns from data (rare values, blank fills,
    clipping, encodings, rescaling, generated columns); fit it on training rows only,
    for example inside each CV fold. Its input is `prepared` without the target column.
    specs: kept for compatibility; always empty now that nothing is fitted in apply.
    """
    data: pd.DataFrame
    decisions: list[Decision]
    anomalies: list[dict[str, Any]]
    prepared: Optional[pd.DataFrame] = None
    specs: dict[str, Any] = field(default_factory=dict)
    pipeline: Optional[Any] = None


@dataclass
class Plan:
    """
    Serializable execution plan representing structural and semantic pipeline proposals.
    """
    source_path: str
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    row_count: Optional[int] = None
    column_count: int = 0
    schema: Optional[TableSchema] = None
    decisions: list[Decision] = field(default_factory=list)
    model_hint: str = "tree"
    feature_config: Optional[FeatureConfig] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert Plan to dictionary."""
        return {
            "source_path": self.source_path,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "schema": self.schema.to_dict() if self.schema is not None else None,
            "decisions": [d.to_dict() for d in self.decisions],
            "model_hint": self.model_hint,
            "feature_config": asdict(self.feature_config) if self.feature_config is not None else None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize Plan to JSON formatted string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: Union[str, Path]) -> Path:
        """Save Plan to plan.json file."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        """Reconstruct Plan from dictionary."""
        schema_data = data.get("schema")
        schema = TableSchema.from_dict(schema_data) if schema_data else None
        decs_data = data.get("decisions", [])
        decisions = [
            Decision.from_dict(d) if isinstance(d, dict) else d for d in decs_data
        ]
        return cls(
            source_path=data["source_path"],
            target_column=data.get("target_column"),
            task_type=data.get("task_type"),
            row_count=data.get("row_count"),
            column_count=data.get("column_count", len(decisions)),
            schema=schema,
            decisions=decisions,
            model_hint=data.get("model_hint", "tree"),
            feature_config=FeatureConfig(**data["feature_config"]) if data.get("feature_config") else None,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Plan":
        """Deserialize Plan from JSON string."""
        return cls.from_dict(json.loads(json_str))


# ===========================================================================
# Stage Base Class & Implementations
# ===========================================================================

class Stage:
    """Base interface for all pipeline stages."""
    name: str = "base"
    section: str = "reading"

    def plan(self, context: PlanContext) -> list[Decision]:
        """Generate pipeline Decision proposals without mutating source or state."""
        raise NotImplementedError

    def apply(self, context: ApplyContext, decisions: list[Decision]) -> Any:
        """Execute stage transformations given active decisions."""
        raise NotImplementedError


class ReadingStage(Stage):
    """
    Stage 1: Reading.
    Inspects text format, column separator, and table header rows.
    """
    name: str = "reading"
    section: str = "reading"

    def plan(self, context: PlanContext) -> list[Decision]:
        p = context.source_path
        enc_res = detect_encoding(p)
        dialect_res = detect_dialect(p, encoding=enc_res.encoding)
        header_res = detect_header(
            p,
            delimiter=dialect_res.delimiter,
            quotechar=dialect_res.quotechar,
            escapechar=dialect_res.escapechar,
            doublequote=dialect_res.doublequote,
            encoding=enc_res.encoding,
        )

        # 1. Encoding Decision
        enc_needs_att = enc_res.low_confidence or (enc_res.confidence < 0.80)
        enc_dec = Decision(
            id="",
            stage="reading",
            kind="encoding",
            columns=[],
            action=f"use_encoding_{enc_res.encoding}",
            evidence={
                "encoding": enc_res.encoding,
                "confidence": enc_res.confidence,
                "has_bom": enc_res.has_bom,
                "low_confidence": enc_res.low_confidence,
                "detected_as": enc_res.raw_detected,
                "alternatives": list(enc_res.alternatives),
            },
            confidence=enc_res.confidence,
            # The safe default is the detected encoding: it is the one that decoded the
            # sample (UTF-8 is only chosen when the bytes are valid UTF-8).
            default=f"use_encoding_{enc_res.encoding}",
            needs_attention=enc_needs_att,
            plain_language=template_encoding(
                enc_res.encoding,
                enc_res.confidence,
                has_bom=enc_res.has_bom,
                needs_attention=enc_needs_att,
            ).summary,
            status="automatic",
        )

        # 2. Dialect Decision
        dial_needs_att = dialect_res.flagged or (dialect_res.confidence < 0.80)
        dial_dec = Decision(
            id="",
            stage="reading",
            kind="dialect",
            columns=[],
            action=f"separate_by_{dialect_res.delimiter}",
            evidence={
                "delimiter": dialect_res.delimiter,
                "quotechar": dialect_res.quotechar,
                "confidence": dialect_res.confidence,
                "margin": dialect_res.margin,
                "flagged": dialect_res.flagged,
            },
            confidence=dialect_res.confidence,
            # Comma is the usual safe default; but whitespace is only ever detected when
            # comma (and every other explicit separator) failed to split the rows, so
            # falling back to comma there would read the whole file as one column.
            default=(
                f"separate_by_{WHITESPACE_DELIMITER}"
                if dialect_res.delimiter == WHITESPACE_DELIMITER
                else "separate_by_,"
            ),
            needs_attention=dial_needs_att,
            plain_language=template_dialect(
                dialect_res.delimiter,
                dialect_res.confidence,
                needs_attention=dial_needs_att,
            ).summary,
            status="automatic",
        )

        # 3. Header Decision
        hdr_action = (
            f"use_header_row_{header_res.header_row_index}"
            if header_res.header_row_index is not None
            else "no_header"
        )
        hdr_needs_att = (
            header_res.flagged
            or header_res.is_ragged
            or (header_res.confidence < 0.80)
        )
        hdr_dec = Decision(
            id="",
            stage="reading",
            kind="header",
            columns=[],
            action=hdr_action,
            evidence={
                "header_row_index": header_res.header_row_index,
                "confidence": header_res.confidence,
                "title_rows": header_res.title_rows,
                "ragged_rows": header_res.ragged_rows,
                "is_ragged": header_res.is_ragged,
                "flagged": header_res.flagged,
            },
            confidence=header_res.confidence,
            default="use_header_row_0",
            needs_attention=hdr_needs_att,
            plain_language=template_header(
                header_res.header_row_index,
                title_rows_count=len(header_res.title_rows),
                is_ragged=header_res.is_ragged,
                needs_attention=hdr_needs_att,
            ).summary,
            status="automatic",
        )

        # Load raw data in-memory without saving or modifying files
        load_res = load_raw(
            p,
            encoding=enc_res.encoding,
            dialect=dialect_res,
            header_result=header_res,
        )
        context.raw_data = load_res.data
        context.schema = TableSchema(
            source_path=str(p.resolve()),
            row_count=load_res.row_count,
            column_count=load_res.column_count,
            # Record what was actually detected: left to the dataclass defaults, the
            # schema would claim confidence 1.0 and a header in row 0 for every file.
            encoding=EncodingSchema(
                detected=enc_res.encoding,
                confidence=enc_res.confidence,
                has_bom=enc_res.has_bom,
                low_confidence=enc_res.low_confidence,
                raw_detected=enc_res.raw_detected,
            ),
            dialect=DialectSchema(
                delimiter=dialect_res.delimiter,
                quotechar=dialect_res.quotechar,
                escapechar=dialect_res.escapechar,
                doublequote=dialect_res.doublequote,
                confidence=dialect_res.confidence,
                flagged=dialect_res.flagged,
                margin=dialect_res.margin,
            ),
            header=HeaderSchema(
                header_row_index=header_res.header_row_index,
                confidence=header_res.confidence,
                title_rows=list(header_res.title_rows),
                footer_rows=list(header_res.footer_rows),
                blank_rows=list(header_res.blank_rows),
                total_rows=list(header_res.total_rows),
                ragged_rows=list(header_res.ragged_rows),
                is_ragged=header_res.is_ragged,
                flagged=header_res.flagged,
            ),
            columns=[
                ColumnRawMetadata(name=name, position=idx)
                for idx, name in enumerate(load_res.column_names)
            ],
        )

        return [enc_dec, dial_dec, hdr_dec]

    def apply(self, context: ApplyContext, decisions: list[Decision]) -> pd.DataFrame:
        """
        Read the file using the actions in force on the reading Decisions (text format,
        separator, column-names row), so a confirmed or changed answer is what is used.
        """
        p = context.source_path
        enc: Optional[str] = None
        delim: Optional[str] = None
        hdr_action: Optional[str] = None
        detected_hdr_action: Optional[str] = None

        for d in decisions:
            if d.kind == "encoding":
                enc = _action_value(d.action, "use_encoding_") or d.evidence.get("encoding")
            elif d.kind == "dialect":
                delim = _action_value(d.action, "separate_by_") or d.evidence.get("delimiter")
            elif d.kind == "header":
                hdr_action = d.action
                idx = d.evidence.get("header_row_index")
                detected_hdr_action = "no_header" if idx is None else f"use_header_row_{idx}"

        dialect_param = {"delimiter": delim} if delim else None
        header_result = None
        if hdr_action is not None and hdr_action != detected_hdr_action:
            header_result = _header_for_action(p, enc, delim, hdr_action)
        try:
            load_res = load_raw(p, encoding=enc, dialect=dialect_param, header_result=header_result)
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Could not read {p.name} with the chosen text format. Choose a different "
                f"text format for this file (mode='ask'), or review the plan. ({exc.reason})"
            ) from exc
        context.raw_data = load_res.data
        return load_res.data


def _action_value(action: str, prefix: str) -> Optional[str]:
    """'separate_by_;' with prefix 'separate_by_' -> ';' (None if the prefix is absent)."""
    return action[len(prefix):] if action.startswith(prefix) else None


def _header_for_action(
    path: Path, encoding: Optional[str], delimiter: Optional[str], action: str
) -> HeaderResult:
    """HeaderResult for a column-names row chosen by the user, not the one detected."""
    detected = detect_header(path, delimiter=delimiter or ",", encoding=encoding or "utf-8")
    if action == "no_header":
        n = len(detected.column_names)
        return replace(
            detected,
            header_row_index=None,
            title_rows=[],
            column_names=[f"column_{i + 1}" for i in range(n)],
        )
    row = _action_value(action, "use_header_row_")
    if row is None or not row.isdigit():
        raise ValueError(f"Unknown column-names action: {action!r}")
    n = int(row)
    with open(path, newline="", encoding=encoding or "utf-8") as fh:
        rows = split_rows([ln.rstrip("\r\n") for ln in fh], delimiter or ",")
    if n >= len(rows):
        raise ValueError(f"Row {n + 1} does not exist in {path.name} ({len(rows)} rows)")
    return replace(
        detected,
        header_row_index=n,
        title_rows=list(range(n)),
        column_names=[c.strip() for c in rows[n]],
    )


class TypingStage(Stage):
    """
    Stage 2: Typing.
    Evaluates column types, missing markers, anomalies, and task inference.
    Applies typed parsing with lossless anomaly logging.
    """
    name: str = "typing"
    section: str = "reading"

    def plan(self, context: PlanContext) -> list[Decision]:
        df = context.raw_data
        if df is None:
            return []

        # Run type inference on raw data
        inferred_schema = infer_table_types(df, context.schema)
        context.schema = inferred_schema

        typing_decisions: list[Decision] = []
        for dec in inferred_schema.decisions:
            if dec.stage == "infer":
                typing_decisions.append(dec)

        # Target Column & Task Type Inference
        if context.target_column:
            target = context.target_column
            if target not in df.columns:
                # Usually a misread file (wrong separator or header row), not a typo:
                # without this, every later step would run and then train on nothing.
                cols = [str(c) for c in df.columns]
                shown = cols[:10] + (["..."] if len(cols) > 10 else [])
                raise ValueError(
                    f"Target column '{target}' was not found in {context.source_path.name}. "
                    f"The file was read as {len(cols)} column(s): {shown}. "
                    "If that looks wrong, the separator or column-names row was misread: "
                    "re-run with mode='ask' to choose them. Otherwise check the target's name."
                )
            if target in df.columns:
                target_series = df[target]
                unique_vals = [
                    str(x) for x in target_series.dropna().unique() if str(x).strip()
                ]
                num_unique = len(unique_vals)

                # Find inferred semantic type of target column
                col_meta = next(
                    (c for c in inferred_schema.columns if c.name == target), None
                )
                t_type = col_meta.inferred_type if col_meta else "text"

                # The target must keep at least two distinct values once its missing
                # markers are blanked. Otherwise every later step would train on a
                # constant (a "perfect" but meaningless score), so stop here, loudly.
                markers = {
                    str(m.get("marker", "")) for m in (col_meta.missing_markers if col_meta else [])
                }
                kept_vals = [v for v in unique_vals if v not in markers]
                if len(kept_vals) < 2:
                    blanked = sorted(v for v in unique_vals if v in markers)
                    raise ValueError(
                        f"Target column '{target}' has {len(kept_vals)} distinct value(s) "
                        f"left after reading ({kept_vals[:5]}); a model cannot be trained on it. "
                        + (
                            f"Values read as missing: {blanked}. If these are real labels, "
                            "review the target's missing markers. "
                            if blanked else ""
                        )
                        + "Check that the right column was chosen as the target."
                    )

                if context.task_type and context.task_type.lower() != "none":
                    specified = context.task_type.lower()
                    task_dec = Decision(
                        id="",
                        stage="infer",
                        kind="task_inference",
                        columns=[target],
                        action=specified,
                        evidence={
                            "target": target,
                            "num_unique": num_unique,
                            "unique_samples": unique_vals[:5],
                            "specified_by_user": True,
                        },
                        confidence=1.0,
                        default=specified,
                        needs_attention=False,
                        plain_language=f"Target column '{target}' configured for {specified}.",
                        status="automatic",
                    )
                    typing_decisions.append(task_dec)
                else:
                    # Infer task from target distribution
                    # Ambiguity rule: integer or numeric with <= 10 unique values, or binary non-bool
                    is_ambiguous = (
                        t_type in ("int", "numeric") and num_unique <= 10
                    ) or (num_unique == 2 and t_type != "boolean")

                    if is_ambiguous:
                        action = "classification" if num_unique <= 10 else "regression"
                        task_dec = Decision(
                            id="",
                            stage="infer",
                            kind="ambiguous_task",
                            columns=[target],
                            action=action,
                            evidence={
                                "target": target,
                                "num_unique": num_unique,
                                "unique_samples": unique_vals[:5],
                                "inferred_type": t_type,
                                "ambiguous": True,
                            },
                            confidence=0.50,
                            default="classification",
                            needs_attention=True,
                            plain_language=template_ambiguous_task(
                                column_name=target,
                                unique_count=num_unique,
                                example_values=unique_vals[:5],
                            ).summary,
                            status="automatic",
                        )
                        typing_decisions.append(task_dec)
                    else:
                        inferred_task = (
                            "regression"
                            if t_type in ("int", "float", "numeric")
                            else "classification"
                        )
                        task_dec = Decision(
                            id="",
                            stage="infer",
                            kind="task_inference",
                            columns=[target],
                            action=inferred_task,
                            evidence={
                                "target": target,
                                "num_unique": num_unique,
                                "unique_samples": unique_vals[:5],
                                "inferred_type": t_type,
                                "ambiguous": False,
                            },
                            confidence=0.95,
                            default=inferred_task,
                            needs_attention=False,
                            plain_language=f"Target column '{target}' designated for {inferred_task}.",
                            status="automatic",
                        )
                        typing_decisions.append(task_dec)

        return typing_decisions

    def apply(self, context: ApplyContext, decisions: list[Decision]) -> pd.DataFrame:
        """
        Apply inferred or confirmed types to raw string DataFrame.

        Never discards information silently: any value failing conversion is
        recorded in context.anomalies with its row, column, and raw value.
        """
        raw_df = context.raw_data
        if raw_df is None:
            raise ValueError("Cannot apply typing without raw DataFrame in context.")

        typed_df = pd.DataFrame(index=raw_df.index)

        # Index type decisions by column
        col_decisions: dict[str, Decision] = {}
        for d in decisions:
            if d.stage == "infer" and d.kind in ("type_inference", "type_abstain"):
                if d.columns:
                    col_decisions[d.columns[0]] = d

        for col_name in raw_df.columns:
            series = raw_df[col_name]
            dec = col_decisions.get(col_name)

            if dec is None:
                typed_df[col_name] = series
                continue

            action = dec.action
            evidence = dec.evidence or {}
            dec_comma = bool(evidence.get("decimal_comma_detected", False))
            # A unit column is read as numbers; the unit stays in the Decision's evidence
            unit = evidence.get("embedded_unit_detected") if evidence.get("unit_column") else None

            # Missing value sentinels from column metadata
            missing_markers: list[str] = []
            if context.schema:
                col_meta = next(
                    (c for c in context.schema.columns if c.name == col_name), None
                )
                if col_meta:
                    missing_markers = [
                        m.get("marker", "") for m in col_meta.missing_markers
                    ]
            missing_set = set(missing_markers) | {""}

            converted_vals: list[Any] = []

            for r_idx, val in enumerate(series):
                s_val = str(val).strip()

                # Missing value check
                if s_val in missing_set:
                    converted_vals.append(pd.NA)
                    continue

                if unit and action in ("cast_to_int", "cast_to_float"):
                    s_val = strip_unit(s_val, unit) or s_val

                # Type conversions
                if action == "cast_to_int":
                    try:
                        clean_int = s_val.replace(",", "")
                        c_val = int(clean_int)
                        converted_vals.append(c_val)
                    except (ValueError, TypeError) as err:
                        context.anomalies.append(
                            {
                                "row": r_idx,
                                "column": col_name,
                                "value": s_val,
                                "target_type": "int",
                                "message": f"Could not convert '{s_val}' to integer: {err}",
                            }
                        )
                        converted_vals.append(pd.NA)

                elif action == "cast_to_float":
                    try:
                        clean_flt = s_val
                        if dec_comma:
                            # European decimal comma format: 1.234,56 or 12,5
                            if "." in clean_flt and "," in clean_flt:
                                clean_flt = clean_flt.replace(".", "").replace(",", ".")
                            else:
                                clean_flt = clean_flt.replace(",", ".")
                        else:
                            clean_flt = clean_flt.replace(",", "")
                        f_val = float(clean_flt)
                        converted_vals.append(f_val)
                    except (ValueError, TypeError) as err:
                        context.anomalies.append(
                            {
                                "row": r_idx,
                                "column": col_name,
                                "value": s_val,
                                "target_type": "float",
                                "message": f"Could not convert '{s_val}' to float: {err}",
                            }
                        )
                        converted_vals.append(pd.NA)

                elif action == "cast_to_boolean":
                    lower_s = s_val.lower()
                    if lower_s in ("true", "yes", "y", "t", "1"):
                        converted_vals.append(True)
                    elif lower_s in ("false", "no", "n", "f", "0"):
                        converted_vals.append(False)
                    else:
                        context.anomalies.append(
                            {
                                "row": r_idx,
                                "column": col_name,
                                "value": s_val,
                                "target_type": "boolean",
                                "message": f"Could not convert '{s_val}' to boolean",
                            }
                        )
                        converted_vals.append(pd.NA)

                elif action == "cast_to_datetime":
                    try:
                        dt_val = date_parser.parse(s_val)
                        converted_vals.append(dt_val)
                    except (ValueError, OverflowError) as err:
                        context.anomalies.append(
                            {
                                "row": r_idx,
                                "column": col_name,
                                "value": s_val,
                                "target_type": "datetime",
                                "message": f"Could not convert '{s_val}' to datetime: {err}",
                            }
                        )
                        converted_vals.append(pd.NaT)

                elif action in ("keep_as_categorical", "cast_to_categorical"):
                    converted_vals.append(s_val)

                else:
                    # String, ID, Constant, or Abstained
                    converted_vals.append(s_val)

            # Assign Series with appropriate dtype
            if action == "cast_to_int":
                typed_df[col_name] = pd.Series(converted_vals, dtype="Int64", index=raw_df.index)
            elif action == "cast_to_float":
                typed_df[col_name] = pd.Series(converted_vals, dtype="Float64", index=raw_df.index)
            elif action == "cast_to_boolean":
                typed_df[col_name] = pd.Series(converted_vals, dtype="boolean", index=raw_df.index)
            elif action == "cast_to_datetime":
                typed_df[col_name] = pd.to_datetime(pd.Series(converted_vals, index=raw_df.index))
            elif action in ("keep_as_categorical", "cast_to_categorical"):
                typed_df[col_name] = pd.Series(converted_vals, dtype="category", index=raw_df.index)
            else:
                typed_df[col_name] = pd.Series(converted_vals, dtype="object", index=raw_df.index)

        context.typed_data = typed_df
        return typed_df


# ===========================================================================
# Plug-in Registry & Stubs for Future Stages
# ===========================================================================

def _working_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Typed table -> the plain numpy-backed dtypes the cleaning and engineering code
    works on (as in v1): nullable integers/floats become int/float (float when blanks
    exist), nullable booleans become bool (object when blanks exist), categories
    become plain text.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            out[col] = s.astype(object)
        elif pd.api.types.is_bool_dtype(s) and pd.api.types.is_extension_array_dtype(s.dtype):
            out[col] = s.astype(object) if s.isna().any() else s.astype(bool)
        elif pd.api.types.is_integer_dtype(s) and pd.api.types.is_extension_array_dtype(s.dtype):
            out[col] = s.astype("float64") if s.isna().any() else s.astype("int64")
        elif pd.api.types.is_float_dtype(s) and pd.api.types.is_extension_array_dtype(s.dtype):
            out[col] = s.astype("float64")
    return out


def _resolved_task(context: Union[PlanContext, ApplyContext]) -> str:
    """classification | regression for the target, from the user or the task Decision."""
    if context.task_type and context.task_type.lower() in ("classification", "regression"):
        return context.task_type.lower()
    for d in context.decisions:
        if d.kind in ("task_inference", "ambiguous_task") and d.action in ("classification", "regression"):
            return d.action
    return "classification"


_ACTION_TYPE = {
    "keep_as_constant": "constant",
    "cast_to_boolean": "boolean",
    "keep_as_id": "ID",
    "cast_to_int": "int",
    "cast_to_float": "float",
    "cast_to_datetime": "datetime",
    "keep_as_categorical": "categorical",
    "cast_to_categorical": "categorical",
    "keep_as_string": "text",
    "keep_as_text": "text",
}


def _schema_with_actions(schema: TableSchema, decisions: Sequence[Decision]) -> TableSchema:
    """Copy of the schema whose column types follow the typing actions currently in force."""
    out = TableSchema.from_dict(schema.to_dict())
    by_col = {c.name: c for c in out.columns}
    for d in decisions:
        if d.stage == "infer" and d.kind in ("type_inference", "type_abstain") and d.columns:
            col = by_col.get(d.columns[0])
            if col is not None and d.action in _ACTION_TYPE:
                col.type_override = _ACTION_TYPE[d.action]
    return out


def _unfitted_pipeline(context: ApplyContext, decisions: list[Decision]) -> Optional[Any]:
    """The unfitted pipeline of data-learning steps for this run (docs/fit_hygiene.md)."""
    if context.typed_data is None:
        return None
    typed = _working_frame(context.typed_data)
    target = context.target_column if context.target_column in typed.columns else None
    schema = _schema_with_actions(context.schema, decisions) if context.schema is not None else None
    task = _resolved_task(context)
    kinds = plan_time_column_kinds(typed, decisions, schema, target, task)
    return build_unfitted_pipeline(
        target, decisions, kinds, schema=schema, model_hint=context.model_hint,
        feature_config=context.feature_config, task=task,
    )


def _ensure_prep_plan(context: PlanContext) -> list[Decision]:
    """
    Plan cleaning and engineering once per planning run. The typed table is built in
    memory from the typing Decisions (nothing is written), then the steps are
    simulated in v1 order by the cleaner and engineer planners.
    """
    if context.prep_decisions is not None:
        return context.prep_decisions
    if context.raw_data is None or context.schema is None:
        context.prep_decisions = []
        return context.prep_decisions

    actx = ApplyContext(
        source_path=context.source_path,
        target_column=context.target_column,
        task_type=context.task_type,
        raw_data=context.raw_data,
        schema=context.schema,
        decisions=context.decisions,
    )
    typed = TypingStage().apply(actx, context.decisions)
    work = _working_frame(typed)
    schema = _schema_with_actions(context.schema, context.decisions)

    target = context.target_column if context.target_column in work.columns else None
    cp = plan_clean_full(work, schema, target, context.cleaner_config)
    # Notes on the table as read (before fills, which would create false repeats).
    notes = plan_class_imbalance(work, target, _resolved_task(context)) + plan_grouped_rows(
        work, target, cp.signals
    )
    eng = plan_engineer(
        cp.cleaned,
        schema=schema,
        target=target,
        model_hint=context.model_hint,
        cardinality_threshold=context.cleaner_config.cardinality_threshold,
        feature_config=context.feature_config,
        task=_resolved_task(context),
        signals=cp.signals,
    )
    context.prep_decisions = list(cp.decisions) + notes + list(eng)
    return context.prep_decisions


class PrepStage(Stage):
    """
    One step of cleaning/engineering. Steps are registered in execution order (the
    order v1 ran them in); `section` is the heading they appear under in the Plan PDF,
    so one section can be served by several steps.
    """
    kinds: tuple[str, ...] = ()
    # True for steps that learn from data: apply() leaves them to the returned
    # unfitted pipeline (docs/fit_hygiene.md) and does not run them on the full table.
    in_pipeline: bool = False

    def runner(
        self, df: pd.DataFrame, decisions: list[Decision], context: ApplyContext
    ) -> pd.DataFrame:
        raise NotImplementedError

    def plan(self, context: PlanContext) -> list[Decision]:
        return [d for d in _ensure_prep_plan(context) if d.kind in self.kinds]

    def apply(self, context: ApplyContext, decisions: list[Decision]) -> Any:
        if context.typed_data is None:
            return None
        if context.prepared_data is None:
            context.prepared_data = _working_frame(context.typed_data)
        if self.in_pipeline:
            return context.prepared_data
        mine = [d for d in decisions if d.stage in ("clean", "engineer") and d.kind in self.kinds]
        context.prepared_data = self.runner(context.prepared_data, mine, context)
        return context.prepared_data


class FixingStage(PrepStage):
    """Fixing: blank-marker conversion and number reading (first cleaning steps)."""
    name: str = "fixing"
    section: str = "fixing"
    kinds = ("convert_missing_markers", "coerce_numeric")

    def runner(self, df: pd.DataFrame, decisions: list[Decision], context: ApplyContext) -> pd.DataFrame:
        return apply_coerce(apply_markers(df, decisions), decisions)


class RemovingStage(PrepStage):
    """Removing: rows without a target value, repeated rows and unusable columns (plus notes)."""
    name: str = "removing"
    section: str = "removing"
    kinds = (
        "missing_target_rows", "remove_duplicates", "drop_high_missingness", "drop_id", "drop_constant",
        # Notes only: listed so they stay in the plan; the runner does nothing with them.
        "class_imbalance", "grouped_rows",
    )

    def runner(self, df: pd.DataFrame, decisions: list[Decision], context: ApplyContext) -> pd.DataFrame:
        df, _ = apply_missing_target(df, decisions)
        df, _ = apply_dedupe(df, decisions)
        df, _ = apply_drops(df, decisions)
        return df


class TidyingStage(PrepStage):
    """Fixing: tidy category text and combine very uncommon values."""
    name: str = "tidying"
    section: str = "fixing"
    kinds = ("normalize_categories", "group_rare_categories")

    def runner(self, df: pd.DataFrame, decisions: list[Decision], context: ApplyContext) -> pd.DataFrame:
        # Rare-value grouping learns from data, so it runs in the pipeline instead.
        return apply_normalization(df, [d for d in decisions if d.kind == "normalize_categories"])


class FillingGapsStage(PrepStage):
    """Filling gaps: blank-value flags and fills."""
    name: str = "filling_gaps"
    section: str = "filling gaps"
    kinds = ("add_missing_indicator", "fill_median", "fill_mode", "constant_fill")

    in_pipeline = True

    def runner(self, df: pd.DataFrame, decisions: list[Decision], context: ApplyContext) -> pd.DataFrame:
        return apply_impute(df, decisions)


class OutlierStage(PrepStage):
    """Fixing: pull extreme numbers back into range (runs after gaps are filled, as in v1)."""
    name: str = "outliers"
    section: str = "fixing"
    kinds = ("clip_outliers",)

    in_pipeline = True

    def runner(self, df: pd.DataFrame, decisions: list[Decision], context: ApplyContext) -> pd.DataFrame:
        return apply_outliers(df, decisions)


class LeakageStage(PrepStage):
    """Removing: columns that may give away the target (kept unless the person chooses otherwise)."""
    name: str = "leakage"
    section: str = "removing"
    kinds = ("suspected_leakage", "feature_overlap")

    in_pipeline = True

    def runner(self, df: pd.DataFrame, decisions: list[Decision], context: ApplyContext) -> pd.DataFrame:
        out, _ = apply_engineer(df, decisions, target=context.target_column)
        return out


class CreatingStage(PrepStage):
    """Creating: encodings, rescaling and generated columns."""
    name: str = "creating"
    section: str = "creating"
    kinds = (
        "expand_datetime", "ordinal_encode", "one_hot_encode", "target_encode",
        "log_transform", "scale_numeric", "create_interactions",
        "create_datetime_features", "create_cluster_features", "create_text_features",
    )

    in_pipeline = True

    def runner(self, df: pd.DataFrame, decisions: list[Decision], context: ApplyContext) -> pd.DataFrame:
        out, specs = apply_engineer(df, decisions, target=context.target_column)
        context.engineer_specs = specs
        return out


class StageRegistry:
    """Registry managing execution order and extensibility of pipeline stages."""

    def __init__(self) -> None:
        self._stages: list[Stage] = []

    def register(self, stage: Stage) -> None:
        """Register a pipeline stage."""
        self._stages.append(stage)

    def get_stages(self) -> list[Stage]:
        """Return registered stages in pipeline execution order."""
        return list(self._stages)


def build_default_registry() -> StageRegistry:
    """Construct the standard registry; stages are listed in execution order."""
    reg = StageRegistry()
    reg.register(ReadingStage())
    reg.register(TypingStage())
    reg.register(FixingStage())
    reg.register(RemovingStage())
    reg.register(TidyingStage())
    reg.register(FillingGapsStage())
    reg.register(OutlierStage())
    reg.register(LeakageStage())
    reg.register(CreatingStage())
    return reg


# ===========================================================================
# Planner Orchestration Class
# ===========================================================================

class Planner:
    """High-level orchestrator coordinating plan generation and apply execution."""

    def __init__(self, registry: Optional[StageRegistry] = None) -> None:
        self.registry = registry if registry is not None else build_default_registry()

    def plan(
        self,
        path: Union[str, Path],
        target: Optional[str] = None,
        task: Optional[str] = None,
        model_hint: str = "tree",
        feature_config: Optional[FeatureConfig] = None,
        cleaner_config: CleanerConfig = DEFAULT_CLEANER_CONFIG,
    ) -> Plan:
        """
        Generate a complete Plan for a tabular dataset without mutating anything.
        """
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

        context = PlanContext(
            source_path=p,
            target_column=target,
            task_type=task,
            model_hint=model_hint,
            feature_config=feature_config,
            cleaner_config=cleaner_config,
        )
        all_decisions: list[Decision] = []

        for stage in self.registry.get_stages():
            stage_decisions = stage.plan(context)
            all_decisions.extend(stage_decisions)
            context.decisions = list(all_decisions)

        row_count = (
            context.schema.row_count
            if context.schema and context.schema.row_count is not None
            else (len(context.raw_data) if context.raw_data is not None else None)
        )
        col_count = (
            context.schema.column_count
            if context.schema and context.schema.column_count
            else (len(context.raw_data.columns) if context.raw_data is not None else 0)
        )

        if context.schema is not None:
            context.schema.decisions = list(all_decisions)

        return Plan(
            source_path=str(p),
            target_column=target,
            task_type=task,
            row_count=row_count,
            column_count=col_count,
            schema=context.schema,
            decisions=all_decisions,
            model_hint=model_hint,
            feature_config=feature_config,
        )

    def apply(
        self,
        plan_or_context: Union[Plan, ApplyContext],
        decisions: Optional[list[Decision]] = None,
    ) -> ApplyResult:
        """
        Execute pipeline stages using decisions from plan or context.
        """
        if isinstance(plan_or_context, Plan):
            plan_obj = plan_or_context
            p = Path(plan_obj.source_path).resolve()
            decs = decisions if decisions is not None else plan_obj.decisions
            context = ApplyContext(
                source_path=p,
                target_column=plan_obj.target_column,
                task_type=plan_obj.task_type,
                schema=plan_obj.schema,
                decisions=decs,
                model_hint=plan_obj.model_hint,
                feature_config=plan_obj.feature_config,
            )
        else:
            context = plan_or_context
            decs = decisions if decisions is not None else context.decisions

        for stage in self.registry.get_stages():
            stage.apply(context, decs)

        res_df = (
            context.typed_data
            if context.typed_data is not None
            else (context.raw_data if context.raw_data is not None else pd.DataFrame())
        )
        return ApplyResult(
            data=res_df,
            decisions=decs,
            anomalies=context.anomalies,
            prepared=context.prepared_data,
            specs=context.engineer_specs,
            pipeline=_unfitted_pipeline(context, decs),
        )


def replan_prep(plan_obj: Plan, cleaner_config: CleanerConfig = DEFAULT_CLEANER_CONFIG) -> None:
    """
    Recompute the cleaning and engineering Decisions of `plan_obj` in place, from the
    reading and typing Decisions as they now stand (answers may have changed them).
    """
    base = [d for d in plan_obj.decisions if d.stage not in ("clean", "engineer")]
    ctx = PlanContext(
        source_path=Path(plan_obj.source_path),
        target_column=plan_obj.target_column,
        task_type=plan_obj.task_type,
        schema=plan_obj.schema,
        decisions=base,
        model_hint=plan_obj.model_hint,
        feature_config=plan_obj.feature_config,
        cleaner_config=cleaner_config,
    )
    ctx.raw_data = ReadingStage().apply(ApplyContext(source_path=ctx.source_path), base)
    plan_obj.decisions = base + list(_ensure_prep_plan(ctx))


# ===========================================================================
# Top-Level Functional API
# ===========================================================================

def plan(
    path: Union[str, Path],
    target: Optional[str] = None,
    task: Optional[str] = None,
    model_hint: str = "tree",
    feature_config: Optional[FeatureConfig] = None,
) -> Plan:
    """Generate pure pipeline plan. model_hint ('tree' or 'linear') and feature_config shape the creating step."""
    planner = Planner()
    return planner.plan(path, target=target, task=task, model_hint=model_hint, feature_config=feature_config)


DEFAULT_OUT_DIR_NAME = "preflight_out"


def default_out_dir() -> Path:
    """Where outputs go when no out_dir is given: preflight_out/ in the current working
    directory. Never the source file's folder (it may be read-only or shared)."""
    return Path.cwd().resolve() / DEFAULT_OUT_DIR_NAME


def write_plan_artifacts(plan_obj: Plan, out_dir: Union[str, Path]) -> tuple[Path, Path]:
    """
    Write plan.json, THEN render the Plan PDF, into out_dir.

    If the PDF render fails, plan.json is already on disk and the error names it.
    Returns (plan.json path, plan.pdf path).
    """
    out_d = Path(out_dir).resolve()
    out_d.mkdir(parents=True, exist_ok=True)
    plan_json_path = out_d / "plan.json"
    plan_pdf_path = out_d / "plan.pdf"

    plan_obj.save_json(plan_json_path)

    try:
        renderer = globals().get("render_plan_pdf", render_plan_pdf)
        renderer(plan_obj, plan_pdf_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to render Plan PDF to {plan_pdf_path}, but plan was saved to {plan_json_path}: {exc}"
        ) from exc
    return plan_json_path, plan_pdf_path


def apply(
    plan_or_context: Union[Plan, ApplyContext],
    decisions: Optional[list[Decision]] = None,
    mode: Optional[str] = None,
    channel: Any = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> ApplyResult:
    """
    Execute pipeline transformations.

    mode=None applies the Decisions as they stand (unchanged behaviour). With
    mode='ask' | 'defaults' | 'strict', every open question is resolved first
    (see preflight.resolve), the resolved plan.json and Plan PDF are written to out_dir,
    and the file is then processed with the final answers.
    """
    if mode is not None:
        if not isinstance(plan_or_context, Plan):
            raise ValueError("mode/channel need a Plan; pass the result of plan() or dry_run()")
        from preflight.interact.resolve import resolve

        resolve(plan_or_context, mode=mode, channel=channel, out_dir=out_dir)
    planner = Planner()
    return planner.apply(plan_or_context, decisions=decisions)


def dry_run(
    path: Union[str, Path],
    target: Optional[str] = None,
    task: Optional[str] = None,
    out_dir: Optional[Union[str, Path]] = None,
    mode: Optional[str] = "auto",
    channel: Any = None,
    model_hint: str = "tree",
    feature_config: Optional[FeatureConfig] = None,
) -> tuple[Plan, Path]:
    """
    Run pipeline ingest, infer, cleaning and engineering in plan-only form, write plan.json and Plan PDF,
    and return the Plan object plus the PDF path.

    Purity:
    - Never modifies the source file.
    - Never produces cleaned data.

    mode: 'auto' (default) asks when the session is interactive (terminal or notebook)
    and is strict otherwise; 'ask' | 'defaults' | 'strict' choose directly. The
    questions are resolved first (see preflight.resolve) and plan.json and the Plan PDF
    show the final answers. mode=None writes the plan with its questions still open
    (Phase 4 behaviour).

    out_dir defaults to preflight_out/ in the current working directory; nothing is
    ever written next to the source file. The output folder is printed at the end.

    Failure handling:
    - Writes plan.json BEFORE rendering the PDF.
    - If PDF rendering fails, plan.json still exists on disk, and a clear error
      naming the saved plan.json file is raised.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {p}")

    out_d = Path(out_dir).resolve() if out_dir is not None else default_out_dir()
    out_d.mkdir(parents=True, exist_ok=True)

    plan_obj = plan(p, target=target, task=task, model_hint=model_hint, feature_config=feature_config)

    if mode is not None:
        from preflight.interact.resolve import resolve

        result = resolve(plan_obj, mode=mode, channel=channel, out_dir=out_d)
        print(f"PreFlight dry run: output written to {out_d}")
        return result

    _, plan_pdf_path = write_plan_artifacts(plan_obj, out_d)
    print(f"PreFlight dry run: output written to {out_d}")
    return plan_obj, plan_pdf_path
