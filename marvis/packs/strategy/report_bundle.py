"""Canonical StrategyReportBundle V2 contract.

The report layer is deliberately evidence-only.  It accepts typed report
fields and references to already-authenticated artifacts; it never reads a
dataset, evaluates a rule, scores a model, or calculates a business metric.
Renderers may rearrange these values for people, but they may not turn missing
information into zero or upgrade a development observation to OOT evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import hmac
import json
import math
import re
from typing import Any

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.project_context import (
    REPORT_FIELD_AVAILABILITIES,
    validate_missing_information_record,
    validate_red_flag,
    validate_report_field,
    validate_source_ref,
)


STRATEGY_REPORT_BUNDLE_SCHEMA_VERSION = "strategy.report-bundle.v2"
STRATEGY_REPORT_SECTION_SCHEMA_VERSION = "strategy.report-section.v2"
STRATEGY_REPORT_TABLE_SCHEMA_VERSION = "strategy.report-table.v2"
STRATEGY_REPORT_PRODUCER_VERSION = "marvis.strategy.report-bundle/2"
STRATEGY_REPORT_DATA_CLASSIFICATION = (
    "deidentified_aggregate_business_evidence"
)

REPORT_STATUSES = frozenset({"draft", "partial", "final"})
EFFECT_STAGES = frozenset(
    {"estimated", "backtested", "oot_validated", "post_launch_observed"}
)
REPORT_SECTION_KEYS = (
    "current_project",
    "historical_versions",
    "sample_design",
    "univariate_and_models",
    "candidate_combinations",
    "impact_assessment",
    "final_document",
)
REPORT_CORE_SHEET_KEYS = (
    "00_summary",
    "01_current_state",
    "02_history",
    "03_sample",
    "04_univariate_model",
    "05_candidates",
    "06_strategy",
    "07_waterfall_swap",
    "08_impact",
    "09_economics",
    "10_validation",
    "11_evidence",
)
REPORT_TABLE_CONTENT_CLASSES = frozenset(
    {
        "metric_summary",
        "rule_summary",
        "bin_summary",
        "monthly_summary",
        "segment_summary",
        "lineage",
        "assumption_summary",
    }
)
REPORT_TABLE_GRANULARITIES = frozenset({"aggregate"})
STRATEGY_TYPES = frozenset(
    {"approval", "reject", "limit", "pricing", "segmentation"}
)
SAMPLE_POPULATIONS = frozenset({"approval", "risk"})
SAMPLE_PARTITIONS = frozenset({"development", "validation", "oot"})

MAX_REPORT_BUNDLE_JSON_BYTES = 16 * 1024 * 1024
MAX_REPORT_BUNDLE_JSON_DEPTH = 40
MAX_REPORT_BUNDLE_JSON_NODES = 500_000
MAX_REPORT_FIELDS = 100_000
MAX_REPORT_TABLES = 2_000
MAX_REPORT_ROWS = 200_000
MAX_REPORT_COLUMNS = 500
MAX_REPORT_REFS = 20_000
MAX_REPORT_RED_FLAGS = 10_000
MAX_REPORT_SHEETS = 100

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_REPORT_ID_RE = re.compile(r"^strategy-report-[0-9a-f]{24}$")
_APPENDIX_SHEET_KEY_RE = re.compile(r"^appendix_[a-z0-9_]{1,20}$")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "report_id",
        "report_revision",
        "task_id",
        "strategy_id",
        "strategy_version",
        "strategy_type",
        "title",
        "status",
        "effect_stages",
        "sections",
        "dataset_refs",
        "strategy_artifact_refs",
        "tool_run_refs",
        "missing_information",
        "completeness_summary",
        "generated_at",
        "producer_version",
        "data_classification",
        "previous_report_id",
        "content_sha256",
    }
)
_SECTION_FIELDS = frozenset(
    {
        "schema_version",
        "key",
        "title",
        "availability",
        "summary_fields",
        "tables",
        "stage_evidence",
        "red_flags",
        "source_refs",
    }
)
_NAMED_FIELD_FIELDS = frozenset({"field_id", "label", "field"})
_TABLE_FIELDS = frozenset(
    {
        "schema_version",
        "table_id",
        "title",
        "sheet_key",
        "granularity",
        "content_class",
        "effect_stage",
        "columns",
        "rows",
        "source_refs",
    }
)
_COLUMN_FIELDS = frozenset({"key", "label", "unit", "precision"})
_ROW_FIELDS = frozenset({"row_id", "cells"})
_STAGE_EVIDENCE_FIELDS = frozenset(
    {"effect_stage", "population", "partition", "binding"}
)
_ESTIMATED_STAGE_BINDING_FIELDS = frozenset(
    {"kind", "scenario_ref", "result_ref"}
)
_SAMPLE_STAGE_BINDING_FIELDS = frozenset(
    {"kind", "dataset_ref", "frozen_artifact_ref", "result_ref"}
)
_POST_LAUNCH_STAGE_BINDING_FIELDS = frozenset(
    {
        "kind",
        "deployment_ref",
        "environment_ref",
        "effective_period",
        "monitoring_ref",
    }
)
_STAGE_REF_KINDS = {
    "scenario_ref": frozenset({"strategy_scenario"}),
    "estimated_result_ref": frozenset({"pool_impact", "strategy_impact"}),
    "dataset_ref": frozenset({"dataset"}),
    "frozen_artifact_ref": frozenset(
        {
            "strategy",
            "strategy_candidate_asset",
            "strategy_candidate_pool",
        }
    ),
    "backtest_result_ref": frozenset(
        {
            "backtest",
            "pool_impact",
            "strategy_impact",
            "voting_candidate_search",
        }
    ),
    "validation_result_ref": frozenset(
        {"model_score_evidence", "strategy_impact", "strategy_validation"}
    ),
    "deployment_ref": frozenset({"deployment", "strategy_deployment"}),
    "environment_ref": frozenset({"deployment_environment", "environment"}),
    "monitoring_ref": frozenset({"monitoring_run", "strategy_monitoring"}),
}
_EFFECTIVE_PERIOD_FIELDS = frozenset({"start", "end"})
_COMPLETENESS_FIELDS = frozenset(
    {
        "field_counts",
        "section_counts",
        "missing_information_counts",
        "blocking_counts",
        "has_strategy_blocker",
        "has_impact_blocker",
        "has_validation_blocker",
    }
)
_AVAILABILITY_ORDER = (
    "present",
    "unavailable",
    "not_applicable",
    "not_matured",
)
_MISSING_STATUSES = ("pending", "provided", "unavailable")
_BLOCKING_LEVELS = ("strategy", "impact", "validation", "report_optional")
_REQUIRED_FINAL_SECTION_KEYS = frozenset(
    {
        "current_project",
        "sample_design",
        "candidate_combinations",
        "impact_assessment",
        "final_document",
    }
)
_SECTION_TABLE_SHEETS = {
    "current_project": frozenset({"01_current_state"}),
    "historical_versions": frozenset({"02_history"}),
    "sample_design": frozenset({"03_sample"}),
    "univariate_and_models": frozenset({"04_univariate_model"}),
    "candidate_combinations": frozenset({"05_candidates", "06_strategy"}),
    "impact_assessment": frozenset(
        {
            "07_waterfall_swap",
            "08_impact",
            "09_economics",
            "10_validation",
        }
    ),
    "final_document": frozenset(
        {"00_summary", "06_strategy", "10_validation"}
    ),
}


class StrategyReportBundleError(StrategyError):
    """A report bundle is structurally invalid or contradicts its evidence."""


def build_named_report_field(
    *,
    field_id: str,
    label: str,
    field: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one labelled field without altering its availability semantics."""

    return _named_field(
        {"field_id": field_id, "label": label, "field": field},
        "named report field",
    )


def build_strategy_report_table(
    *,
    table_id: str,
    title: str,
    sheet_key: str,
    granularity: str,
    content_class: str,
    columns: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    source_refs: Sequence[Mapping[str, Any]],
    effect_stage: str | None = None,
) -> dict[str, Any]:
    """Build a strict evidence table whose cells are all typed ReportFields."""

    return _report_table(
        {
            "schema_version": STRATEGY_REPORT_TABLE_SCHEMA_VERSION,
            "table_id": table_id,
            "title": title,
            "sheet_key": sheet_key,
            "granularity": granularity,
            "content_class": content_class,
            "effect_stage": effect_stage,
            "columns": list(columns),
            "rows": list(rows),
            "source_refs": list(source_refs),
        },
        "report table",
    )


def build_strategy_report_section(
    *,
    key: str,
    title: str,
    availability: str,
    summary_fields: Sequence[Mapping[str, Any]] = (),
    tables: Sequence[Mapping[str, Any]] = (),
    stage_evidence: Sequence[Mapping[str, Any]] = (),
    red_flags: Sequence[Mapping[str, Any]] = (),
    source_refs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build one of the seven fixed report sections."""

    return _report_section(
        {
            "schema_version": STRATEGY_REPORT_SECTION_SCHEMA_VERSION,
            "key": key,
            "title": title,
            "availability": availability,
            "summary_fields": list(summary_fields),
            "tables": list(tables),
            "stage_evidence": list(stage_evidence),
            "red_flags": list(red_flags),
            "source_refs": list(source_refs),
        },
        "report section",
    )


def build_strategy_report_bundle(
    *,
    task_id: str,
    report_revision: int,
    strategy_id: str | None,
    strategy_version: str | None,
    strategy_type: str | None,
    title: Mapping[str, Any],
    status: str,
    sections: Sequence[Mapping[str, Any]],
    dataset_refs: Sequence[Mapping[str, Any]] = (),
    strategy_artifact_refs: Sequence[Mapping[str, Any]] = (),
    tool_run_refs: Sequence[Mapping[str, Any]] = (),
    missing_information: Sequence[Mapping[str, Any]] = (),
    generated_at: str,
    previous_report_id: str | None = None,
    producer_version: str = STRATEGY_REPORT_PRODUCER_VERSION,
) -> dict[str, Any]:
    """Build a deterministic immutable report revision.

    ``generated_at`` is explicit source data rather than a wall-clock call.
    Rebuilding with the same normalized inputs therefore yields the same id and
    content hash.
    """

    body = _normalize_bundle_body(
        {
            "schema_version": STRATEGY_REPORT_BUNDLE_SCHEMA_VERSION,
            "report_revision": report_revision,
            "task_id": task_id,
            "strategy_id": strategy_id,
            "strategy_version": strategy_version,
            "strategy_type": strategy_type,
            "title": title,
            "status": status,
            "effect_stages": _effect_stages_from_sections(sections),
            "sections": list(sections),
            "dataset_refs": list(dataset_refs),
            "strategy_artifact_refs": list(strategy_artifact_refs),
            "tool_run_refs": list(tool_run_refs),
            "missing_information": list(missing_information),
            "completeness_summary": _completeness_from_inputs(
                sections,
                missing_information,
            ),
            "generated_at": generated_at,
            "producer_version": producer_version,
            "data_classification": STRATEGY_REPORT_DATA_CLASSIFICATION,
            "previous_report_id": previous_report_id,
        }
    )
    report_id = "strategy-report-" + _sha256(_canonical_json(body))[:24]
    without_hash = {**body, "report_id": report_id}
    return {
        **without_hash,
        "content_sha256": _sha256(_canonical_json(without_hash)),
    }


def validate_strategy_report_bundle(value: object) -> dict[str, Any]:
    obj = _object(value, "strategy report bundle")
    _exact_fields(obj, _TOP_LEVEL_FIELDS, "strategy report bundle")
    body = _normalize_bundle_body(
        {
            key: obj[key]
            for key in obj
            if key not in {"report_id", "content_sha256"}
        }
    )
    report_id = obj["report_id"]
    expected_id = "strategy-report-" + _sha256(_canonical_json(body))[:24]
    if (
        not isinstance(report_id, str)
        or _REPORT_ID_RE.fullmatch(report_id) is None
        or not hmac.compare_digest(report_id, expected_id)
    ):
        raise StrategyReportBundleError(
            "strategy report_id does not match canonical content"
        )
    content_hash = _hash(obj["content_sha256"], "content_sha256")
    without_hash = {**body, "report_id": report_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(content_hash, expected_hash):
        raise StrategyReportBundleError(
            "strategy report content_sha256 does not match canonical content"
        )
    return {**without_hash, "content_sha256": content_hash}


def canonical_strategy_report_bundle_json(value: object) -> str:
    return _canonical_json(validate_strategy_report_bundle(value))


def strategy_report_bundle_from_json(
    raw: str | bytes | bytearray,
) -> dict[str, Any]:
    if not isinstance(raw, (str, bytes, bytearray)):
        raise StrategyReportBundleError("strategy report JSON must be text or bytes")
    try:
        byte_count = (
            len(raw)
            if isinstance(raw, (bytes, bytearray))
            else len(raw.encode("utf-8"))
        )
    except UnicodeEncodeError as exc:
        raise StrategyReportBundleError(
            "strategy report JSON must contain valid UTF-8 text"
        ) from exc
    if byte_count > MAX_REPORT_BUNDLE_JSON_BYTES:
        raise StrategyReportBundleError("strategy report JSON exceeds byte budget")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise StrategyReportBundleError(
            "strategy report is not valid JSON"
        ) from exc
    return validate_strategy_report_bundle(value)


def _normalize_bundle_body(value: object) -> dict[str, Any]:
    _preflight_json_tree(value, name="strategy report bundle body")
    obj = _object(value, "strategy report bundle body")
    expected = _TOP_LEVEL_FIELDS - {"report_id", "content_sha256"}
    _exact_fields(obj, expected, "strategy report bundle body")
    if obj["schema_version"] != STRATEGY_REPORT_BUNDLE_SCHEMA_VERSION:
        raise StrategyReportBundleError("strategy report schema_version is invalid")
    if obj["producer_version"] != STRATEGY_REPORT_PRODUCER_VERSION:
        raise StrategyReportBundleError("strategy report producer_version is invalid")
    if obj["data_classification"] != STRATEGY_REPORT_DATA_CLASSIFICATION:
        raise StrategyReportBundleError(
            "strategy report data_classification is invalid"
        )
    task_id = _text(obj["task_id"], "task_id")
    revision = _positive_int(obj["report_revision"], "report_revision")
    previous = _optional_text(obj["previous_report_id"], "previous_report_id")
    if revision == 1 and previous is not None:
        raise StrategyReportBundleError(
            "first strategy report revision cannot have previous_report_id"
        )
    if revision > 1 and (
        previous is None or _REPORT_ID_RE.fullmatch(previous) is None
    ):
        raise StrategyReportBundleError(
            "later strategy report revision requires valid previous_report_id"
        )
    strategy_id = _optional_text(obj["strategy_id"], "strategy_id")
    strategy_version = _optional_text(
        obj["strategy_version"], "strategy_version"
    )
    strategy_type = (
        None
        if obj["strategy_type"] is None
        else _enum(obj["strategy_type"], STRATEGY_TYPES, "strategy_type")
    )
    if (strategy_id is None) != (strategy_version is None):
        raise StrategyReportBundleError(
            "strategy_id and strategy_version must be present or absent together"
        )
    if strategy_id is not None and strategy_type is None:
        raise StrategyReportBundleError(
            "persisted strategy identity requires strategy_type"
        )
    title = _report_field(obj["title"], "title")
    if title["availability"] != "present" or not isinstance(title["value"], str):
        raise StrategyReportBundleError("strategy report title must be present text")
    if title["blocking"] != "none":
        raise StrategyReportBundleError("strategy report title cannot carry a blocker")
    status = _enum(obj["status"], REPORT_STATUSES, "status")
    sections = [
        _report_section(item, f"sections[{index}]")
        for index, item in enumerate(_array(obj["sections"], "sections"))
    ]
    if [item["key"] for item in sections] != list(REPORT_SECTION_KEYS):
        raise StrategyReportBundleError(
            "strategy report must contain the seven canonical sections in order"
        )
    effect_stages = [
        _enum(item, EFFECT_STAGES, f"effect_stages[{index}]")
        for index, item in enumerate(_array(obj["effect_stages"], "effect_stages"))
    ]
    if len(effect_stages) != len(set(effect_stages)):
        raise StrategyReportBundleError("effect_stages contains duplicates")
    expected_stages = _effect_stages_from_sections(sections)
    if effect_stages != expected_stages:
        raise StrategyReportBundleError(
            "effect_stages must be derived from section evidence"
        )
    dataset_refs = _source_refs(obj["dataset_refs"], "dataset_refs")
    strategy_refs = _source_refs(
        obj["strategy_artifact_refs"], "strategy_artifact_refs"
    )
    tool_refs = _source_refs(obj["tool_run_refs"], "tool_run_refs")
    missing = [
        _missing_record(item, task_id=task_id, name=f"missing_information[{index}]")
        for index, item in enumerate(
            _array(obj["missing_information"], "missing_information")
        )
    ]
    _reject_duplicates(
        [item["missing_information_id"] for item in missing],
        "missing_information",
    )
    _enforce_resource_budgets(
        title=title,
        sections=sections,
        top_level_refs=(*dataset_refs, *strategy_refs, *tool_refs),
        missing_information=missing,
    )
    _enforce_global_source_identity(
        title=title,
        sections=sections,
        top_level_refs=(*dataset_refs, *strategy_refs, *tool_refs),
        missing_information=missing,
    )
    completeness = _completeness(obj["completeness_summary"])
    expected_completeness = _completeness_from_inputs(sections, missing)
    if completeness != expected_completeness:
        raise StrategyReportBundleError(
            "completeness_summary must be derived from sections and missing information"
        )
    if status == "final" and completeness["has_strategy_blocker"]:
        raise StrategyReportBundleError(
            "final strategy report cannot retain a strategy blocker"
        )
    if strategy_type is None and status == "final":
        raise StrategyReportBundleError(
            "final strategy report requires a strategy_type"
        )
    if status == "final":
        unavailable_required = sorted(
            section["key"]
            for section in sections
            if section["key"] in _REQUIRED_FINAL_SECTION_KEYS
            and section["availability"] != "present"
        )
        if unavailable_required:
            raise StrategyReportBundleError(
                "final strategy report requires present sections: "
                + ", ".join(unavailable_required)
            )
    if completeness["has_validation_blocker"] and "oot_validated" in effect_stages:
        raise StrategyReportBundleError(
            "strategy report with a validation blocker cannot claim OOT evidence"
        )
    generated_at = _timestamp(obj["generated_at"], "generated_at")
    return {
        "schema_version": STRATEGY_REPORT_BUNDLE_SCHEMA_VERSION,
        "report_revision": revision,
        "task_id": task_id,
        "strategy_id": strategy_id,
        "strategy_version": strategy_version,
        "strategy_type": strategy_type,
        "title": title,
        "status": status,
        "effect_stages": effect_stages,
        "sections": sections,
        "dataset_refs": dataset_refs,
        "strategy_artifact_refs": strategy_refs,
        "tool_run_refs": tool_refs,
        "missing_information": missing,
        "completeness_summary": completeness,
        "generated_at": generated_at,
        "producer_version": STRATEGY_REPORT_PRODUCER_VERSION,
        "data_classification": STRATEGY_REPORT_DATA_CLASSIFICATION,
        "previous_report_id": previous,
    }


def _report_section(value: object, name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _SECTION_FIELDS, name)
    if obj["schema_version"] != STRATEGY_REPORT_SECTION_SCHEMA_VERSION:
        raise StrategyReportBundleError(f"{name} schema_version is invalid")
    key = _enum(obj["key"], frozenset(REPORT_SECTION_KEYS), f"{name}.key")
    title = _text(obj["title"], f"{name}.title")
    availability = _enum(
        obj["availability"],
        REPORT_FIELD_AVAILABILITIES,
        f"{name}.availability",
    )
    fields = [
        _named_field(item, f"{name}.summary_fields[{index}]")
        for index, item in enumerate(
            _array(obj["summary_fields"], f"{name}.summary_fields")
        )
    ]
    _reject_duplicates([item["field_id"] for item in fields], f"{name}.summary_fields")
    tables = [
        _report_table(item, f"{name}.tables[{index}]")
        for index, item in enumerate(_array(obj["tables"], f"{name}.tables"))
    ]
    if len(tables) > MAX_REPORT_TABLES:
        raise StrategyReportBundleError(f"{name}.tables exceeds budget")
    _reject_duplicates([item["table_id"] for item in tables], f"{name}.tables")
    allowed_sheets = _SECTION_TABLE_SHEETS[key]
    for table in tables:
        sheet_key = table["sheet_key"]
        if not (
            sheet_key in allowed_sheets
            or _APPENDIX_SHEET_KEY_RE.fullmatch(sheet_key) is not None
        ):
            raise StrategyReportBundleError(
                f"{name}.tables sheet_key {sheet_key!r} is not valid for section {key}"
            )
    stage_evidence = [
        _stage_evidence(item, f"{name}.stage_evidence[{index}]")
        for index, item in enumerate(
            _array(obj["stage_evidence"], f"{name}.stage_evidence")
        )
    ]
    _reject_duplicates(
        [
            (
                item["effect_stage"],
                item["population"],
                item["partition"],
                item["binding"],
            )
            for item in stage_evidence
        ],
        f"{name}.stage_evidence",
    )
    red_flags = [
        _red_flag(item, f"{name}.red_flags[{index}]")
        for index, item in enumerate(
            _array(obj["red_flags"], f"{name}.red_flags")
        )
    ]
    if len(red_flags) > MAX_REPORT_RED_FLAGS:
        raise StrategyReportBundleError(f"{name}.red_flags exceeds budget")
    refs = _source_refs(obj["source_refs"], f"{name}.source_refs")
    if availability == "present":
        if not refs:
            raise StrategyReportBundleError(
                f"{name} present section requires source_refs"
            )
    elif fields or tables or stage_evidence:
        raise StrategyReportBundleError(
            f"{name} non-present section cannot contain report facts"
        )
    table_stages = {
        item["effect_stage"]
        for item in tables
        if item["effect_stage"] is not None
    }
    evidence_stages = {item["effect_stage"] for item in stage_evidence}
    if not table_stages <= evidence_stages:
        raise StrategyReportBundleError(
            f"{name} table effect_stage lacks matching stage evidence"
        )
    for table in tables:
        if table["effect_stage"] is None:
            continue
        stage_sources = {
            (
                _stage_result_ref(item)["kind"],
                _stage_result_ref(item)["ref_id"],
                _stage_result_ref(item)["content_hash"],
            )
            for item in stage_evidence
            if item["effect_stage"] == table["effect_stage"]
        }
        table_sources = {
            (item["kind"], item["ref_id"], item["content_hash"])
            for item in table["source_refs"]
        }
        if not stage_sources & table_sources:
            raise StrategyReportBundleError(
                f"{name} table effect_stage is not bound to its stage evidence"
            )
    return {
        "schema_version": STRATEGY_REPORT_SECTION_SCHEMA_VERSION,
        "key": key,
        "title": title,
        "availability": availability,
        "summary_fields": fields,
        "tables": tables,
        "stage_evidence": stage_evidence,
        "red_flags": red_flags,
        "source_refs": refs,
    }


def _named_field(value: object, name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _NAMED_FIELD_FIELDS, name)
    return {
        "field_id": _text(obj["field_id"], f"{name}.field_id"),
        "label": _text(obj["label"], f"{name}.label"),
        "field": _report_field(obj["field"], f"{name}.field"),
    }


def _report_table(value: object, name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _TABLE_FIELDS, name)
    if obj["schema_version"] != STRATEGY_REPORT_TABLE_SCHEMA_VERSION:
        raise StrategyReportBundleError(f"{name} schema_version is invalid")
    stage = (
        None
        if obj["effect_stage"] is None
        else _enum(obj["effect_stage"], EFFECT_STAGES, f"{name}.effect_stage")
    )
    columns = [
        _table_column(item, f"{name}.columns[{index}]")
        for index, item in enumerate(_array(obj["columns"], f"{name}.columns"))
    ]
    if not columns or len(columns) > MAX_REPORT_COLUMNS:
        raise StrategyReportBundleError(
            f"{name}.columns must be non-empty within budget"
        )
    keys = [item["key"] for item in columns]
    _reject_duplicates(keys, f"{name}.columns")
    rows = [
        _table_row(item, columns=columns, name=f"{name}.rows[{index}]")
        for index, item in enumerate(_array(obj["rows"], f"{name}.rows"))
    ]
    if len(rows) > MAX_REPORT_ROWS:
        raise StrategyReportBundleError(f"{name}.rows exceeds budget")
    _reject_duplicates([item["row_id"] for item in rows], f"{name}.rows")
    refs = _source_refs(obj["source_refs"], f"{name}.source_refs")
    if rows and not refs:
        raise StrategyReportBundleError(f"{name} with rows requires source_refs")
    return {
        "schema_version": STRATEGY_REPORT_TABLE_SCHEMA_VERSION,
        "table_id": _text(obj["table_id"], f"{name}.table_id"),
        "title": _text(obj["title"], f"{name}.title"),
        "sheet_key": _sheet_key(obj["sheet_key"], f"{name}.sheet_key"),
        "granularity": _enum(
            obj["granularity"],
            REPORT_TABLE_GRANULARITIES,
            f"{name}.granularity",
        ),
        "content_class": _enum(
            obj["content_class"],
            REPORT_TABLE_CONTENT_CLASSES,
            f"{name}.content_class",
        ),
        "effect_stage": stage,
        "columns": columns,
        "rows": rows,
        "source_refs": refs,
    }


def _table_column(value: object, name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _COLUMN_FIELDS, name)
    precision = obj["precision"]
    if precision is not None:
        precision = _non_negative_int(precision, f"{name}.precision")
        if precision > 12:
            raise StrategyReportBundleError(f"{name}.precision exceeds 12")
    return {
        "key": _text(obj["key"], f"{name}.key"),
        "label": _text(obj["label"], f"{name}.label"),
        "unit": _optional_text(obj["unit"], f"{name}.unit"),
        "precision": precision,
    }


def _table_row(
    value: object,
    *,
    columns: Sequence[Mapping[str, Any]],
    name: str,
) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _ROW_FIELDS, name)
    cells = _object(obj["cells"], f"{name}.cells")
    column_keys = [str(column["key"]) for column in columns]
    if set(cells) != set(column_keys):
        raise StrategyReportBundleError(
            f"{name}.cells must match declared table columns exactly"
        )
    normalized_cells = {
        key: _report_field(cells[key], f"{name}.cells.{key}")
        for key in column_keys
    }
    for column in columns:
        field = normalized_cells[str(column["key"])]
        if column["unit"] == "%" and field["availability"] == "present":
            observed = field["value"]
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(observed)
                or not 0 <= observed <= 1
            ):
                raise StrategyReportBundleError(
                    f"{name}.cells.{column['key']} percentage value "
                    "must be a finite ratio between 0 and 1"
                )
    return {
        "row_id": _text(obj["row_id"], f"{name}.row_id"),
        "cells": normalized_cells,
    }


def _stage_evidence(value: object, name: str) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _STAGE_EVIDENCE_FIELDS, name)
    stage = _enum(obj["effect_stage"], EFFECT_STAGES, f"{name}.effect_stage")
    population = (
        None
        if obj["population"] is None
        else _enum(obj["population"], SAMPLE_POPULATIONS, f"{name}.population")
    )
    partition = (
        None
        if obj["partition"] is None
        else _enum(obj["partition"], SAMPLE_PARTITIONS, f"{name}.partition")
    )
    if population is None and partition is not None:
        raise StrategyReportBundleError(f"{name}.partition requires population")
    if stage == "estimated":
        if partition is not None:
            raise StrategyReportBundleError(
                f"{name} estimated evidence cannot claim a sample partition"
            )
    elif stage == "backtested" and (
        population is None or partition != "development"
    ):
        raise StrategyReportBundleError(
            f"{name} backtested evidence requires a development population"
        )
    elif stage == "oot_validated" and (
        population is None or partition not in {"validation", "oot"}
    ):
        raise StrategyReportBundleError(
            f"{name} oot_validated evidence requires a validation or oot population"
        )
    elif stage == "post_launch_observed" and partition is not None:
        raise StrategyReportBundleError(
            f"{name} post-launch evidence cannot claim a development partition"
        )
    return {
        "effect_stage": stage,
        "population": population,
        "partition": partition,
        "binding": _stage_binding(
            obj["binding"],
            stage=stage,
            name=f"{name}.binding",
        ),
    }


def _stage_binding(
    value: object,
    *,
    stage: str,
    name: str,
) -> dict[str, Any]:
    obj = _object(value, name)
    if stage == "estimated":
        _exact_fields(obj, _ESTIMATED_STAGE_BINDING_FIELDS, name)
        if obj["kind"] != "estimated_scenario":
            raise StrategyReportBundleError(
                f"{name}.kind must be estimated_scenario"
            )
        return {
            "kind": "estimated_scenario",
            "scenario_ref": _source_ref_of_kinds(
                obj["scenario_ref"],
                f"{name}.scenario_ref",
                _STAGE_REF_KINDS["scenario_ref"],
            ),
            "result_ref": _source_ref_of_kinds(
                obj["result_ref"],
                f"{name}.result_ref",
                _STAGE_REF_KINDS["estimated_result_ref"],
            ),
        }
    if stage in {"backtested", "oot_validated"}:
        _exact_fields(obj, _SAMPLE_STAGE_BINDING_FIELDS, name)
        expected_kind = (
            "development_backtest"
            if stage == "backtested"
            else "independent_validation"
        )
        if obj["kind"] != expected_kind:
            raise StrategyReportBundleError(
                f"{name}.kind must be {expected_kind}"
            )
        return {
            "kind": expected_kind,
            "dataset_ref": _source_ref_of_kinds(
                obj["dataset_ref"],
                f"{name}.dataset_ref",
                _STAGE_REF_KINDS["dataset_ref"],
            ),
            "frozen_artifact_ref": _source_ref_of_kinds(
                obj["frozen_artifact_ref"],
                f"{name}.frozen_artifact_ref",
                _STAGE_REF_KINDS["frozen_artifact_ref"],
            ),
            "result_ref": _source_ref_of_kinds(
                obj["result_ref"],
                f"{name}.result_ref",
                _STAGE_REF_KINDS[
                    "backtest_result_ref"
                    if stage == "backtested"
                    else "validation_result_ref"
                ],
            ),
        }
    _exact_fields(obj, _POST_LAUNCH_STAGE_BINDING_FIELDS, name)
    if obj["kind"] != "post_launch_monitoring":
        raise StrategyReportBundleError(
            f"{name}.kind must be post_launch_monitoring"
        )
    return {
        "kind": "post_launch_monitoring",
        "deployment_ref": _source_ref_of_kinds(
            obj["deployment_ref"],
            f"{name}.deployment_ref",
            _STAGE_REF_KINDS["deployment_ref"],
        ),
        "environment_ref": _source_ref_of_kinds(
            obj["environment_ref"],
            f"{name}.environment_ref",
            _STAGE_REF_KINDS["environment_ref"],
        ),
        "effective_period": _effective_period(
            obj["effective_period"],
            f"{name}.effective_period",
        ),
        "monitoring_ref": _source_ref_of_kinds(
            obj["monitoring_ref"],
            f"{name}.monitoring_ref",
            _STAGE_REF_KINDS["monitoring_ref"],
        ),
    }


def _stage_result_ref(value: Mapping[str, Any]) -> Mapping[str, str]:
    binding = value["binding"]
    if binding["kind"] == "post_launch_monitoring":
        return binding["monitoring_ref"]
    return binding["result_ref"]


def _stage_binding_refs(value: Mapping[str, Any]) -> list[Mapping[str, str]]:
    binding = value["binding"]
    if binding["kind"] == "estimated_scenario":
        return [binding["scenario_ref"], binding["result_ref"]]
    if binding["kind"] in {"development_backtest", "independent_validation"}:
        return [
            binding["dataset_ref"],
            binding["frozen_artifact_ref"],
            binding["result_ref"],
        ]
    return [
        binding["deployment_ref"],
        binding["environment_ref"],
        binding["monitoring_ref"],
    ]


def _effective_period(value: object, name: str) -> dict[str, str]:
    obj = _object(value, name)
    _exact_fields(obj, _EFFECTIVE_PERIOD_FIELDS, name)
    start = _date_or_timestamp(obj["start"], f"{name}.start")
    end = _date_or_timestamp(obj["end"], f"{name}.end")
    try:
        normalized_start = datetime.fromisoformat(
            start.replace("Z", "+00:00")
        )
        normalized_end = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover - guarded by helper
        raise StrategyReportBundleError(f"{name} is invalid") from exc
    if (normalized_start.tzinfo is None) != (normalized_end.tzinfo is None):
        raise StrategyReportBundleError(
            f"{name} boundaries must use the same timezone convention"
        )
    if normalized_end < normalized_start:
        raise StrategyReportBundleError(f"{name}.end cannot precede start")
    return {"start": start, "end": end}


def _effect_stages_from_sections(
    sections: Sequence[Mapping[str, Any]],
) -> list[str]:
    found: set[str] = set()
    for raw in sections:
        if not isinstance(raw, Mapping):
            continue
        for item in raw.get("stage_evidence", ()):
            if isinstance(item, Mapping) and item.get("effect_stage") in EFFECT_STAGES:
                found.add(str(item["effect_stage"]))
        for item in raw.get("tables", ()):
            if isinstance(item, Mapping) and item.get("effect_stage") in EFFECT_STAGES:
                found.add(str(item["effect_stage"]))
    order = ("estimated", "backtested", "oot_validated", "post_launch_observed")
    return [item for item in order if item in found]


def _completeness_from_inputs(
    sections: Sequence[Mapping[str, Any]],
    missing_information: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    availability_counts = {key: 0 for key in _AVAILABILITY_ORDER}
    section_counts = {key: 0 for key in _AVAILABILITY_ORDER}
    blocking_counts = {key: 0 for key in _BLOCKING_LEVELS}
    field_count = 0
    for raw_section in sections:
        if not isinstance(raw_section, Mapping):
            continue
        availability = raw_section.get("availability")
        if availability in section_counts:
            section_counts[str(availability)] += 1
        for raw_named in raw_section.get("summary_fields", ()):
            if not isinstance(raw_named, Mapping):
                continue
            raw_field = raw_named.get("field")
            if isinstance(raw_field, Mapping):
                observed = raw_field.get("availability")
                if observed in availability_counts:
                    availability_counts[str(observed)] += 1
                    field_count += 1
                blocking = raw_field.get("blocking")
                if blocking in {"strategy", "impact", "validation"}:
                    blocking_counts[str(blocking)] += 1
        for raw_table in raw_section.get("tables", ()):
            if not isinstance(raw_table, Mapping):
                continue
            for raw_row in raw_table.get("rows", ()):
                if not isinstance(raw_row, Mapping):
                    continue
                cells = raw_row.get("cells")
                if not isinstance(cells, Mapping):
                    continue
                for raw_field in cells.values():
                    if isinstance(raw_field, Mapping):
                        observed = raw_field.get("availability")
                        if observed in availability_counts:
                            availability_counts[str(observed)] += 1
                            field_count += 1
                        blocking = raw_field.get("blocking")
                        if blocking in {"strategy", "impact", "validation"}:
                            blocking_counts[str(blocking)] += 1
    availability_counts["total"] = field_count
    missing_counts = {key: 0 for key in _MISSING_STATUSES}
    for raw in missing_information:
        if not isinstance(raw, Mapping):
            continue
        status = raw.get("status")
        blocking = raw.get("blocking")
        if status in missing_counts:
            missing_counts[str(status)] += 1
        if status != "provided" and blocking in blocking_counts:
            blocking_counts[str(blocking)] += 1
    return {
        "field_counts": availability_counts,
        "section_counts": section_counts,
        "missing_information_counts": missing_counts,
        "blocking_counts": blocking_counts,
        "has_strategy_blocker": blocking_counts["strategy"] > 0,
        "has_impact_blocker": blocking_counts["impact"] > 0,
        "has_validation_blocker": blocking_counts["validation"] > 0,
    }


def _completeness(value: object) -> dict[str, Any]:
    obj = _object(value, "completeness_summary")
    _exact_fields(obj, _COMPLETENESS_FIELDS, "completeness_summary")
    field_counts = _count_map(
        obj["field_counts"],
        (*_AVAILABILITY_ORDER, "total"),
        "completeness_summary.field_counts",
    )
    if field_counts["total"] != sum(
        field_counts[key] for key in _AVAILABILITY_ORDER
    ):
        raise StrategyReportBundleError(
            "completeness field total does not reconcile"
        )
    section_counts = _count_map(
        obj["section_counts"],
        _AVAILABILITY_ORDER,
        "completeness_summary.section_counts",
    )
    if sum(section_counts.values()) != len(REPORT_SECTION_KEYS):
        raise StrategyReportBundleError(
            "completeness section counts do not reconcile"
        )
    missing_counts = _count_map(
        obj["missing_information_counts"],
        _MISSING_STATUSES,
        "completeness_summary.missing_information_counts",
    )
    blocking_counts = _count_map(
        obj["blocking_counts"],
        _BLOCKING_LEVELS,
        "completeness_summary.blocking_counts",
    )
    booleans = {
        "has_strategy_blocker": _boolean(
            obj["has_strategy_blocker"], "has_strategy_blocker"
        ),
        "has_impact_blocker": _boolean(
            obj["has_impact_blocker"], "has_impact_blocker"
        ),
        "has_validation_blocker": _boolean(
            obj["has_validation_blocker"], "has_validation_blocker"
        ),
    }
    if booleans != {
        "has_strategy_blocker": blocking_counts["strategy"] > 0,
        "has_impact_blocker": blocking_counts["impact"] > 0,
        "has_validation_blocker": blocking_counts["validation"] > 0,
    }:
        raise StrategyReportBundleError(
            "completeness blocker booleans do not match counts"
        )
    return {
        "field_counts": field_counts,
        "section_counts": section_counts,
        "missing_information_counts": missing_counts,
        "blocking_counts": blocking_counts,
        **booleans,
    }


def _count_map(value: object, fields: Sequence[str], name: str) -> dict[str, int]:
    obj = _object(value, name)
    _exact_fields(obj, frozenset(fields), name)
    return {field: _non_negative_int(obj[field], f"{name}.{field}") for field in fields}


def _report_field(value: object, name: str) -> dict[str, Any]:
    try:
        field = validate_report_field(value)
    except StrategyError as exc:
        raise StrategyReportBundleError(f"{name} is invalid: {exc}") from exc
    return field


def _missing_record(
    value: object,
    *,
    task_id: str,
    name: str,
) -> dict[str, Any]:
    try:
        record = validate_missing_information_record(value)
    except StrategyError as exc:
        raise StrategyReportBundleError(f"{name} is invalid: {exc}") from exc
    if record["task_id"] != task_id:
        raise StrategyReportBundleError(f"{name} belongs to another task")
    return record


def _red_flag(value: object, name: str) -> dict[str, Any]:
    try:
        return validate_red_flag(value)
    except StrategyError as exc:
        raise StrategyReportBundleError(f"{name} is invalid: {exc}") from exc


def _source_refs(value: object, name: str) -> list[dict[str, str]]:
    refs = [
        _source_ref(item, f"{name}[{index}]")
        for index, item in enumerate(_array(value, name))
    ]
    _reject_duplicates(
        [(item["kind"], item["ref_id"]) for item in refs],
        name,
    )
    return sorted(refs, key=lambda item: (item["kind"], item["ref_id"]))


def _source_ref(value: object, name: str) -> dict[str, str]:
    try:
        return validate_source_ref(value)
    except StrategyError as exc:
        raise StrategyReportBundleError(f"{name} is invalid: {exc}") from exc


def _source_ref_of_kinds(
    value: object,
    name: str,
    allowed_kinds: frozenset[str],
) -> dict[str, str]:
    ref = _source_ref(value, name)
    if ref["kind"] not in allowed_kinds:
        raise StrategyReportBundleError(
            f"{name}.kind must be one of {', '.join(sorted(allowed_kinds))}"
        )
    return ref


def _enforce_global_source_identity(
    *,
    title: Mapping[str, Any],
    sections: Sequence[Mapping[str, Any]],
    top_level_refs: Sequence[Mapping[str, Any]],
    missing_information: Sequence[Mapping[str, Any]],
) -> None:
    refs: list[Mapping[str, str]] = [
        *top_level_refs,
        *title["source_refs"],
    ]
    for section in sections:
        refs.extend(section["source_refs"])
        refs.extend(
            ref
            for stage in section["stage_evidence"]
            for ref in _stage_binding_refs(stage)
        )
        refs.extend(
            ref
            for field in section["summary_fields"]
            for ref in field["field"]["source_refs"]
        )
        refs.extend(
            ref
            for red_flag in section["red_flags"]
            for ref in red_flag["source_refs"]
        )
        for table in section["tables"]:
            refs.extend(table["source_refs"])
            refs.extend(
                ref
                for row in table["rows"]
                for field in row["cells"].values()
                for ref in field["source_refs"]
            )
    refs.extend(
        item["answer_source_ref"]
        for item in missing_information
        if item["answer_source_ref"] is not None
    )

    seen: dict[tuple[str, str], str] = {}
    for ref in refs:
        identity = (ref["kind"], ref["ref_id"])
        existing = seen.get(identity)
        if existing is not None and not hmac.compare_digest(
            existing,
            ref["content_hash"],
        ):
            raise StrategyReportBundleError(
                "strategy report contains global source_ref identity drift"
            )
        seen[identity] = ref["content_hash"]


def _enforce_resource_budgets(
    *,
    title: Mapping[str, Any],
    sections: Sequence[Mapping[str, Any]],
    top_level_refs: Sequence[Mapping[str, Any]],
    missing_information: Sequence[Mapping[str, Any]],
) -> None:
    field_count = 1
    table_count = 0
    row_count = 0
    red_flag_count = 0
    ref_count = len(top_level_refs) + len(title["source_refs"])
    table_ids: list[str] = []
    appendix_sheet_keys: set[str] = set()

    for section in sections:
        summary_fields = section["summary_fields"]
        tables = section["tables"]
        stage_evidence = section["stage_evidence"]
        red_flags = section["red_flags"]
        field_count += len(summary_fields)
        table_count += len(tables)
        table_ids.extend(item["table_id"] for item in tables)
        appendix_sheet_keys.update(
            item["sheet_key"]
            for item in tables
            if item["sheet_key"] not in REPORT_CORE_SHEET_KEYS
        )
        red_flag_count += len(red_flags)
        ref_count += len(section["source_refs"])
        ref_count += sum(
            len(_stage_binding_refs(item)) for item in stage_evidence
        )
        ref_count += sum(
            len(item["field"]["source_refs"]) for item in summary_fields
        )
        ref_count += sum(len(item["source_refs"]) for item in red_flags)
        for table in tables:
            rows = table["rows"]
            row_count += len(rows)
            ref_count += len(table["source_refs"])
            for row in rows:
                field_count += len(row["cells"])
                ref_count += sum(
                    len(field["source_refs"]) for field in row["cells"].values()
                )

    ref_count += sum(
        1
        for item in missing_information
        if item["answer_source_ref"] is not None
    )
    if field_count > MAX_REPORT_FIELDS:
        raise StrategyReportBundleError("strategy report fields exceed budget")
    if table_count > MAX_REPORT_TABLES:
        raise StrategyReportBundleError("strategy report tables exceed budget")
    _reject_duplicates(table_ids, "strategy report table ids")
    if len(REPORT_CORE_SHEET_KEYS) + len(appendix_sheet_keys) > MAX_REPORT_SHEETS:
        raise StrategyReportBundleError("strategy report sheets exceed budget")
    if row_count > MAX_REPORT_ROWS:
        raise StrategyReportBundleError("strategy report rows exceed budget")
    if red_flag_count > MAX_REPORT_RED_FLAGS:
        raise StrategyReportBundleError("strategy report red flags exceed budget")
    if ref_count > MAX_REPORT_REFS:
        raise StrategyReportBundleError("strategy report references exceed budget")


def _preflight_json_tree(value: object, *, name: str) -> None:
    stack: list[tuple[object, int, bool]] = [(value, 0, False)]
    active: set[int] = set()
    nodes = 0
    while stack:
        current, depth, exiting = stack.pop()
        identity = id(current)
        if exiting:
            active.discard(identity)
            continue
        nodes += 1
        if nodes > MAX_REPORT_BUNDLE_JSON_NODES:
            raise StrategyReportBundleError(f"{name} exceeds node budget")
        if depth > MAX_REPORT_BUNDLE_JSON_DEPTH:
            raise StrategyReportBundleError(f"{name} exceeds depth budget")
        if current is None or isinstance(current, (str, bool, int)):
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise StrategyReportBundleError(
                    f"{name} contains non-finite number"
                )
            continue
        if isinstance(current, Mapping):
            if identity in active:
                raise StrategyReportBundleError(f"{name} contains a cycle")
            if not all(isinstance(key, str) for key in current):
                raise StrategyReportBundleError(f"{name} keys must be strings")
            active.add(identity)
            stack.append((current, depth, True))
            stack.extend(
                (item, depth + 1, False)
                for item in reversed(list(current.values()))
            )
            continue
        if isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            if identity in active:
                raise StrategyReportBundleError(f"{name} contains a cycle")
            active.add(identity)
            stack.append((current, depth, True))
            stack.extend(
                (item, depth + 1, False) for item in reversed(list(current))
            )
            continue
        raise StrategyReportBundleError(
            f"{name} contains unsupported {type(current).__name__}"
        )


def _canonical_json(value: object) -> str:
    _preflight_json_tree(value, name="strategy report")
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise StrategyReportBundleError(
            "strategy report is not finite canonical JSON"
        ) from exc
    try:
        encoded = raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise StrategyReportBundleError(
            "strategy report must contain valid UTF-8 text"
        ) from exc
    if len(encoded) > MAX_REPORT_BUNDLE_JSON_BYTES:
        raise StrategyReportBundleError("strategy report exceeds byte budget")
    return raw


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrategyReportBundleError(
                f"strategy report JSON has duplicate key: {key}"
            )
        result[key] = value
    return result


def _reject_non_finite_json_constant(value: str) -> None:
    raise StrategyReportBundleError(
        f"strategy report JSON contains non-finite constant {value}"
    )


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyReportBundleError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise StrategyReportBundleError(f"{name} must be an array")
    return list(value)


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) == expected:
        return
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    detail = []
    if missing:
        detail.append("missing " + ", ".join(missing))
    if unexpected:
        detail.append("unsupported " + ", ".join(unexpected))
    raise StrategyReportBundleError(f"{name} fields are invalid ({'; '.join(detail)})")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyReportBundleError(f"{name} must be non-empty text")
    return value.strip()


def _optional_text(value: object, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _enum(value: object, allowed: frozenset[str], name: str) -> str:
    normalized = _text(value, name)
    if normalized not in allowed:
        raise StrategyReportBundleError(
            f"{name} must be one of {', '.join(sorted(allowed))}"
        )
    return normalized


def _sheet_key(value: object, name: str) -> str:
    normalized = _text(value, name)
    if (
        normalized not in REPORT_CORE_SHEET_KEYS
        and _APPENDIX_SHEET_KEY_RE.fullmatch(normalized) is None
    ):
        raise StrategyReportBundleError(
            f"{name} must be a canonical core or appendix sheet key"
        )
    if len(normalized) > 31:
        raise StrategyReportBundleError(f"{name} exceeds Excel's sheet-name limit")
    return normalized


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyReportBundleError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyReportBundleError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    observed = _non_negative_int(value, name)
    if observed == 0:
        raise StrategyReportBundleError(f"{name} must be positive")
    return observed


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise StrategyReportBundleError(f"{name} must be boolean")
    return value


def _timestamp(value: object, name: str) -> str:
    observed = _text(value, name)
    try:
        parsed = datetime.fromisoformat(observed.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StrategyReportBundleError(
            f"{name} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise StrategyReportBundleError(f"{name} must include a timezone")
    return observed


def _date_or_timestamp(value: object, name: str) -> str:
    observed = _text(value, name)
    try:
        datetime.fromisoformat(observed.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StrategyReportBundleError(
            f"{name} must be an ISO-8601 date or timestamp"
        ) from exc
    return observed


def _reject_duplicates(values: Sequence[object], name: str) -> None:
    seen: set[str] = set()
    for item in values:
        key = _canonical_json(item)
        if key in seen:
            raise StrategyReportBundleError(f"{name} contains duplicates")
        seen.add(key)


def _sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "EFFECT_STAGES",
    "MAX_REPORT_BUNDLE_JSON_BYTES",
    "REPORT_CORE_SHEET_KEYS",
    "REPORT_SECTION_KEYS",
    "REPORT_STATUSES",
    "STRATEGY_REPORT_BUNDLE_SCHEMA_VERSION",
    "STRATEGY_REPORT_DATA_CLASSIFICATION",
    "STRATEGY_REPORT_PRODUCER_VERSION",
    "STRATEGY_REPORT_SECTION_SCHEMA_VERSION",
    "STRATEGY_REPORT_TABLE_SCHEMA_VERSION",
    "StrategyReportBundleError",
    "build_named_report_field",
    "build_strategy_report_bundle",
    "build_strategy_report_section",
    "build_strategy_report_table",
    "canonical_strategy_report_bundle_json",
    "strategy_report_bundle_from_json",
    "validate_strategy_report_bundle",
]
