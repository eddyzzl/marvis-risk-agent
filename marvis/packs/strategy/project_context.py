"""Pure, deterministic contracts for task-scoped strategy project context.

This module owns JSON shape, semantic validation, canonicalization, and content
addressing only.  It deliberately owns no database, filesystem, Agent, or Tool
runtime behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
import hashlib
import hmac
import json
import math
import re
from typing import Any

from marvis.packs.strategy.errors import StrategyError


REPORT_FIELD_AVAILABILITIES = frozenset(
    {"present", "unavailable", "not_applicable", "not_matured"}
)
REPORT_FIELD_ORIGINS = frozenset(
    {"tool_output", "repository", "uploaded_file", "user"}
)
REPORT_FIELD_BLOCKING_LEVELS = frozenset(
    {"none", "strategy", "impact", "validation"}
)
MISSING_INFORMATION_BLOCKING_LEVELS = frozenset(
    {"strategy", "impact", "validation", "report_optional"}
)
MISSING_INFORMATION_STATUSES = frozenset({"pending", "provided", "unavailable"})

MISSING_INFORMATION_SCHEMA_VERSION = "strategy.missing-information.v1"
CURRENT_PROJECT_SNAPSHOT_SCHEMA_VERSION = "strategy.current-project-snapshot.v1"
HISTORICAL_STRATEGY_REVIEW_SCHEMA_VERSION = "strategy.historical-strategy-review.v1"
STRATEGY_PROJECT_CONTEXT_STATE_SCHEMA_VERSION = "strategy.project-context-state.v1"
STRATEGY_PROJECT_CONTEXT_REVISION_SCHEMA_VERSION = "strategy.project-context-revision.v1"
STRATEGY_PROJECT_CONTEXT_OPERATION_SCHEMA_VERSION = "strategy.project-context-operation.v1"
STRATEGY_PROJECT_CONTEXT_PRODUCER_VERSION = "marvis.strategy.project-context/1"

CURRENT_STATUS_FIELDS = frozenset({"volume", "approval", "risk", "economics"})
RED_FLAG_LEVELS = frozenset({"info", "amber", "red"})
EFFECT_STAGES = frozenset(
    {"estimated", "backtested", "oot_validated", "post_launch_observed"}
)

MAX_PROJECT_CONTEXT_JSON_BYTES = 16 * 1024 * 1024
MAX_PROJECT_CONTEXT_JSON_DEPTH = 32
MAX_PROJECT_CONTEXT_JSON_NODES = 100_000

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MISSING_INFORMATION_ID_RE = re.compile(
    r"^strategy-missing-information-[0-9a-f]{24}$"
)
_CURRENT_PROJECT_SNAPSHOT_ID_RE = re.compile(
    r"^current-project-snapshot-[0-9a-f]{24}$"
)
_HISTORICAL_STRATEGY_REVIEW_ID_RE = re.compile(
    r"^historical-strategy-review-[0-9a-f]{24}$"
)
_PROJECT_CONTEXT_ID_RE = re.compile(r"^strategy-project-context-[0-9a-f]{24}$")
_PROJECT_CONTEXT_REVISION_ID_RE = re.compile(
    r"^strategy-project-context-revision-[0-9a-f]{24}$"
)
_SOURCE_REF_FIELDS = frozenset({"kind", "ref_id", "content_hash"})
_REPORT_FIELD_FIELDS = frozenset(
    {
        "value",
        "availability",
        "origin",
        "source_refs",
        "as_of",
        "blocking",
        "note",
    }
)
_MISSING_INFORMATION_FIELDS = frozenset(
    {
        "schema_version",
        "missing_information_id",
        "task_id",
        "field_path",
        "reason",
        "blocking",
        "question",
        "status",
        "asked_count",
        "asked_at",
        "answered_at",
        "answer_source_ref",
        "dependency_hash",
        "content_hash",
    }
)
_RED_FLAG_FIELDS = frozenset({"code", "level", "message", "source_refs"})
_CONTEXT_FIELD_FIELDS = frozenset({"field_path", "field"})
_CURRENT_PROJECT_SNAPSHOT_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "snapshot_id",
        "task_id",
        "as_of",
        "scope",
        "dataset_refs",
        "workspace_ref",
        "champion_strategy_ref",
        "status_fields",
        "metric_definition_refs",
        "metric_observation_refs",
        "monthly_observation_refs",
        "segment_observation_refs",
        "maturity_summary",
        "user_context_fields",
        "red_flags",
        "tool_run_refs",
        "content_hash",
    }
)
_RULE_FIELDS = frozenset({"rule_id", "priority", "condition", "action"})
_RULE_REF_FIELDS = frozenset({"rule_id", "content_hash"})
_MODIFIED_RULE_REF_FIELDS = frozenset(
    {"rule_id", "before_content_hash", "after_content_hash"}
)
_CHANGE_SET_FIELDS = frozenset(
    {"added_rule_refs", "modified_rule_refs", "removed_rule_refs"}
)
_PERIOD_FIELDS = frozenset({"start", "end"})
_EFFECT_OBSERVATION_REF_FIELDS = frozenset(
    {"observation_ref", "deployment_ref", "environment_ref", "effective_period"}
)
_HISTORICAL_STRATEGY_REVIEW_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "review_id",
        "task_id",
        "strategy_ref",
        "version",
        "effective_period",
        "asset_status",
        "scope",
        "traffic_allocation",
        "change_set",
        "observation_refs_by_effect_stage",
        "external_source_refs",
        "decision_context_fields",
        "availability",
        "red_flags",
        "tool_run_refs",
        "content_hash",
    }
)
_PROJECT_CONTEXT_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "context_id",
        "task_id",
        "as_of",
        "current_project_snapshot",
        "historical_strategy_reviews",
        "missing_information_records",
        "source_refs",
        "red_flags",
        "content_hash",
    }
)
_PROJECT_CONTEXT_REVISION_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "revision_id",
        "task_id",
        "revision",
        "parent_revision_id",
        "parent_state_hash",
        "operation_kind",
        "operation_hash",
        "state",
        "state_hash",
        "content_hash",
    }
)


class StrategyProjectContextError(StrategyError):
    """A project-context value violates the immutable V1 contract."""


def build_source_ref(*, kind: str, ref_id: str, content_hash: str) -> dict[str, str]:
    """Build one exact, content-bound evidence pointer."""

    return validate_source_ref(
        {"kind": kind, "ref_id": ref_id, "content_hash": content_hash}
    )


def validate_source_ref(value: object) -> dict[str, str]:
    obj = _object(value, "source_ref")
    _require_exact_fields(obj, _SOURCE_REF_FIELDS, name="source_ref")
    return {
        "kind": _text(obj["kind"], "source_ref.kind"),
        "ref_id": _text(obj["ref_id"], "source_ref.ref_id"),
        "content_hash": _hash(obj["content_hash"], "source_ref.content_hash"),
    }


def build_report_field(
    *,
    value: Any,
    availability: str,
    origin: str,
    source_refs: Sequence[Mapping[str, Any]] = (),
    as_of: str | None = None,
    blocking: str = "none",
    note: str | None = None,
) -> dict[str, Any]:
    """Build a typed report field without collapsing absence into a value."""

    return validate_report_field(
        {
            "value": value,
            "availability": availability,
            "origin": origin,
            "source_refs": list(source_refs),
            "as_of": as_of,
            "blocking": blocking,
            "note": note,
        }
    )


def validate_report_field(value: object) -> dict[str, Any]:
    obj = _object(value, "report_field")
    _require_exact_fields(obj, _REPORT_FIELD_FIELDS, name="report_field")
    availability = _enum(
        obj["availability"], REPORT_FIELD_AVAILABILITIES, "report_field.availability"
    )
    normalized_value = _json_value(obj["value"], "report_field.value")
    source_refs = _source_refs(obj["source_refs"], name="report_field.source_refs")
    if availability == "present":
        if normalized_value is None:
            raise StrategyProjectContextError(
                "present report_field must have a non-null value"
            )
        if not source_refs:
            raise StrategyProjectContextError(
                "present report_field must have at least one trusted source_ref"
            )
    elif normalized_value is not None:
        raise StrategyProjectContextError(
            "non-present report_field must have null value"
        )
    return {
        "value": normalized_value,
        "availability": availability,
        "origin": _enum(
            obj["origin"], REPORT_FIELD_ORIGINS, "report_field.origin"
        ),
        "source_refs": source_refs,
        "as_of": _optional_as_of(obj["as_of"], "report_field.as_of"),
        "blocking": _enum(
            obj["blocking"], REPORT_FIELD_BLOCKING_LEVELS, "report_field.blocking"
        ),
        "note": _optional_text(obj["note"], "report_field.note"),
    }


def build_red_flag(
    *,
    code: str,
    level: str,
    message: str,
    source_refs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return validate_red_flag(
        {
            "code": code,
            "level": level,
            "message": message,
            "source_refs": list(source_refs),
        }
    )


def validate_red_flag(value: object) -> dict[str, Any]:
    obj = _object(value, "red_flag")
    _require_exact_fields(obj, _RED_FLAG_FIELDS, name="red_flag")
    return {
        "code": _text(obj["code"], "red_flag.code"),
        "level": _enum(obj["level"], RED_FLAG_LEVELS, "red_flag.level"),
        "message": _text(obj["message"], "red_flag.message"),
        "source_refs": _source_refs(
            obj["source_refs"], name="red_flag.source_refs"
        ),
    }


def build_context_field(
    *, field_path: str, field: Mapping[str, Any]
) -> dict[str, Any]:
    return _validate_context_field({"field_path": field_path, "field": field})


def build_current_project_snapshot(
    *,
    task_id: str,
    as_of: str,
    scope: Mapping[str, Any],
    dataset_refs: Sequence[Mapping[str, Any]],
    workspace_ref: Mapping[str, Any] | None,
    champion_strategy_ref: Mapping[str, Any] | None,
    status_fields: Mapping[str, Mapping[str, Any]],
    metric_definition_refs: Sequence[Mapping[str, Any]],
    metric_observation_refs: Sequence[Mapping[str, Any]],
    monthly_observation_refs: Sequence[Mapping[str, Any]],
    segment_observation_refs: Sequence[Mapping[str, Any]],
    maturity_summary: Mapping[str, Any],
    user_context_fields: Sequence[Mapping[str, Any]],
    red_flags: Sequence[Mapping[str, Any]],
    tool_run_refs: Sequence[Mapping[str, Any]],
    producer_version: str = STRATEGY_PROJECT_CONTEXT_PRODUCER_VERSION,
) -> dict[str, Any]:
    """Build the exact current project snapshot from already-resolved evidence."""

    body = _normalize_current_project_snapshot_body(
        {
            "schema_version": CURRENT_PROJECT_SNAPSHOT_SCHEMA_VERSION,
            "producer_version": producer_version,
            "task_id": task_id,
            "as_of": as_of,
            "scope": scope,
            "dataset_refs": list(dataset_refs),
            "workspace_ref": workspace_ref,
            "champion_strategy_ref": champion_strategy_ref,
            "status_fields": dict(status_fields),
            "metric_definition_refs": list(metric_definition_refs),
            "metric_observation_refs": list(metric_observation_refs),
            "monthly_observation_refs": list(monthly_observation_refs),
            "segment_observation_refs": list(segment_observation_refs),
            "maturity_summary": maturity_summary,
            "user_context_fields": list(user_context_fields),
            "red_flags": list(red_flags),
            "tool_run_refs": list(tool_run_refs),
        }
    )
    return _address_object(
        body,
        id_field="snapshot_id",
        id_prefix="current-project-snapshot-",
    )


def validate_current_project_snapshot(value: object) -> dict[str, Any]:
    obj = _object(value, "current_project_snapshot")
    _require_exact_fields(
        obj,
        _CURRENT_PROJECT_SNAPSHOT_FIELDS,
        name="current_project_snapshot",
    )
    body = _normalize_current_project_snapshot_body(obj)
    return _validate_addressed_object(
        obj,
        normalized_body=body,
        id_field="snapshot_id",
        id_pattern=_CURRENT_PROJECT_SNAPSHOT_ID_RE,
        id_prefix="current-project-snapshot-",
        name="current_project_snapshot",
    )


def _normalize_current_project_snapshot_body(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if value["schema_version"] != CURRENT_PROJECT_SNAPSHOT_SCHEMA_VERSION:
        raise StrategyProjectContextError(
            "current_project_snapshot schema_version is invalid"
        )
    producer = _producer_version(value["producer_version"])
    as_of = _optional_as_of(value["as_of"], "current_project_snapshot.as_of")
    if as_of is None:
        raise StrategyProjectContextError(
            "current_project_snapshot.as_of must not be null"
        )
    status_obj = _object(
        value["status_fields"], "current_project_snapshot.status_fields"
    )
    _require_exact_fields(
        status_obj,
        CURRENT_STATUS_FIELDS,
        name="current_project_snapshot.status_fields",
    )
    status_fields = {
        key: validate_report_field(status_obj[key])
        for key in sorted(CURRENT_STATUS_FIELDS)
    }
    return {
        "schema_version": CURRENT_PROJECT_SNAPSHOT_SCHEMA_VERSION,
        "producer_version": producer,
        "task_id": _text(value["task_id"], "current_project_snapshot.task_id"),
        "as_of": as_of,
        "scope": validate_report_field(value["scope"]),
        "dataset_refs": _refs_of_kind(
            value["dataset_refs"], {"dataset"}, "current_project_snapshot.dataset_refs"
        ),
        "workspace_ref": _optional_ref_of_kind(
            value["workspace_ref"],
            {"workspace"},
            "current_project_snapshot.workspace_ref",
        ),
        "champion_strategy_ref": _optional_ref_of_kind(
            value["champion_strategy_ref"],
            {"strategy"},
            "current_project_snapshot.champion_strategy_ref",
        ),
        "status_fields": status_fields,
        "metric_definition_refs": _refs_of_kind(
            value["metric_definition_refs"],
            {"metric_definition"},
            "current_project_snapshot.metric_definition_refs",
        ),
        "metric_observation_refs": _refs_of_kind(
            value["metric_observation_refs"],
            {"metric_observation"},
            "current_project_snapshot.metric_observation_refs",
        ),
        "monthly_observation_refs": _refs_of_kind(
            value["monthly_observation_refs"],
            {"metric_observation"},
            "current_project_snapshot.monthly_observation_refs",
        ),
        "segment_observation_refs": _refs_of_kind(
            value["segment_observation_refs"],
            {"metric_observation"},
            "current_project_snapshot.segment_observation_refs",
        ),
        "maturity_summary": validate_report_field(value["maturity_summary"]),
        "user_context_fields": _context_fields(
            value["user_context_fields"],
            name="current_project_snapshot.user_context_fields",
        ),
        "red_flags": _red_flags(
            value["red_flags"], name="current_project_snapshot.red_flags"
        ),
        "tool_run_refs": _refs_of_kind(
            value["tool_run_refs"],
            {"tool_run"},
            "current_project_snapshot.tool_run_refs",
        ),
    }


def _validate_context_field(value: object) -> dict[str, Any]:
    obj = _object(value, "context_field")
    _require_exact_fields(obj, _CONTEXT_FIELD_FIELDS, name="context_field")
    return {
        "field_path": _text(obj["field_path"], "context_field.field_path"),
        "field": validate_report_field(obj["field"]),
    }


def _context_fields(value: object, *, name: str) -> list[dict[str, Any]]:
    fields = [_validate_context_field(item) for item in _array(value, name)]
    paths = [item["field_path"] for item in fields]
    if len(paths) != len(set(paths)):
        raise StrategyProjectContextError(f"{name} contains duplicate field_path values")
    fields.sort(key=lambda item: item["field_path"])
    return fields


def _red_flags(value: object, *, name: str) -> list[dict[str, Any]]:
    flags = [validate_red_flag(item) for item in _array(value, name)]
    codes = [item["code"] for item in flags]
    if len(codes) != len(set(codes)):
        raise StrategyProjectContextError(f"{name} contains duplicate codes")
    flags.sort(key=lambda item: item["code"])
    return flags


def diff_strategy_rules(
    previous_rules: Sequence[Mapping[str, Any]] | None,
    current_rules: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """Return a deterministic add/modify/remove diff keyed only by ``rule_id``."""

    previous = _normalize_rules(previous_rules or (), name="previous_rules")
    current = _normalize_rules(current_rules, name="current_rules")
    previous_by_id = {item["rule_id"]: item for item in previous}
    current_by_id = {item["rule_id"]: item for item in current}
    added = [
        _rule_ref(current_by_id[rule_id])
        for rule_id in sorted(current_by_id.keys() - previous_by_id.keys())
    ]
    removed = [
        _rule_ref(previous_by_id[rule_id])
        for rule_id in sorted(previous_by_id.keys() - current_by_id.keys())
    ]
    modified = []
    for rule_id in sorted(previous_by_id.keys() & current_by_id.keys()):
        before_hash = _rule_content_hash(previous_by_id[rule_id])
        after_hash = _rule_content_hash(current_by_id[rule_id])
        if not hmac.compare_digest(before_hash, after_hash):
            modified.append(
                {
                    "rule_id": rule_id,
                    "before_content_hash": before_hash,
                    "after_content_hash": after_hash,
                }
            )
    return {
        "added_rule_refs": added,
        "modified_rule_refs": modified,
        "removed_rule_refs": removed,
    }


def validate_rule_change_set(value: object) -> dict[str, list[dict[str, str]]]:
    obj = _object(value, "change_set")
    _require_exact_fields(obj, _CHANGE_SET_FIELDS, name="change_set")
    added = _rule_refs(obj["added_rule_refs"], name="change_set.added_rule_refs")
    removed = _rule_refs(
        obj["removed_rule_refs"], name="change_set.removed_rule_refs"
    )
    modified = _modified_rule_refs(
        obj["modified_rule_refs"], name="change_set.modified_rule_refs"
    )
    changed_ids = [
        *(item["rule_id"] for item in added),
        *(item["rule_id"] for item in removed),
        *(item["rule_id"] for item in modified),
    ]
    if len(changed_ids) != len(set(changed_ids)):
        raise StrategyProjectContextError(
            "change_set rule_id values must be disjoint"
        )
    return {
        "added_rule_refs": added,
        "modified_rule_refs": modified,
        "removed_rule_refs": removed,
    }


def _normalize_rules(
    value: object, *, name: str
) -> list[dict[str, Any]]:
    rules = []
    for item in _array(value, name):
        obj = _object(item, "strategy_rule")
        _require_exact_fields(obj, _RULE_FIELDS, name="strategy_rule")
        priority = _non_negative_int(obj["priority"], "strategy_rule.priority")
        condition = _json_value(obj["condition"], "strategy_rule.condition")
        action = _json_value(obj["action"], "strategy_rule.action")
        if not isinstance(condition, dict):
            raise StrategyProjectContextError(
                "strategy_rule.condition must be an object"
            )
        if not isinstance(action, dict):
            raise StrategyProjectContextError("strategy_rule.action must be an object")
        rules.append(
            {
                "rule_id": _text(obj["rule_id"], "strategy_rule.rule_id"),
                "priority": priority,
                "condition": condition,
                "action": action,
            }
        )
    rule_ids = [item["rule_id"] for item in rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise StrategyProjectContextError(f"{name} contains duplicate rule_id values")
    priorities = [item["priority"] for item in rules]
    if len(priorities) != len(set(priorities)):
        raise StrategyProjectContextError(f"{name} contains duplicate priorities")
    rules.sort(key=lambda item: item["rule_id"])
    return rules


def _rule_content_hash(rule: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json(rule))


def _rule_ref(rule: Mapping[str, Any]) -> dict[str, str]:
    return {
        "rule_id": str(rule["rule_id"]),
        "content_hash": _rule_content_hash(rule),
    }


def _rule_refs(value: object, *, name: str) -> list[dict[str, str]]:
    refs = []
    for item in _array(value, name):
        obj = _object(item, "rule_ref")
        _require_exact_fields(obj, _RULE_REF_FIELDS, name="rule_ref")
        refs.append(
            {
                "rule_id": _text(obj["rule_id"], "rule_ref.rule_id"),
                "content_hash": _hash(
                    obj["content_hash"], "rule_ref.content_hash"
                ),
            }
        )
    ids = [item["rule_id"] for item in refs]
    if len(ids) != len(set(ids)):
        raise StrategyProjectContextError(f"{name} contains duplicate rule_id values")
    refs.sort(key=lambda item: item["rule_id"])
    return refs


def _modified_rule_refs(value: object, *, name: str) -> list[dict[str, str]]:
    refs = []
    for item in _array(value, name):
        obj = _object(item, "modified_rule_ref")
        _require_exact_fields(
            obj, _MODIFIED_RULE_REF_FIELDS, name="modified_rule_ref"
        )
        before = _hash(
            obj["before_content_hash"], "modified_rule_ref.before_content_hash"
        )
        after = _hash(
            obj["after_content_hash"], "modified_rule_ref.after_content_hash"
        )
        if hmac.compare_digest(before, after):
            raise StrategyProjectContextError(
                "modified_rule_ref hashes must be different"
            )
        refs.append(
            {
                "rule_id": _text(
                    obj["rule_id"], "modified_rule_ref.rule_id"
                ),
                "before_content_hash": before,
                "after_content_hash": after,
            }
        )
    ids = [item["rule_id"] for item in refs]
    if len(ids) != len(set(ids)):
        raise StrategyProjectContextError(f"{name} contains duplicate rule_id values")
    refs.sort(key=lambda item: item["rule_id"])
    return refs


def build_effect_observation_ref(
    *,
    observation_ref: Mapping[str, Any],
    deployment_ref: Mapping[str, Any] | None = None,
    environment_ref: Mapping[str, Any] | None = None,
    effective_period: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an effect pointer; stage-specific claims are checked by its review."""

    return _normalize_effect_observation_ref(
        {
            "observation_ref": observation_ref,
            "deployment_ref": deployment_ref,
            "environment_ref": environment_ref,
            "effective_period": effective_period,
        },
        stage=None,
    )


def build_historical_strategy_review(
    *,
    task_id: str,
    strategy_ref: Mapping[str, Any] | None,
    version: int | None,
    effective_period: Mapping[str, Any],
    asset_status: Mapping[str, Any],
    scope: Mapping[str, Any],
    traffic_allocation: Mapping[str, Any],
    change_set: Mapping[str, Any],
    observation_refs_by_effect_stage: Mapping[str, Any],
    external_source_refs: Sequence[Mapping[str, Any]],
    decision_context_fields: Sequence[Mapping[str, Any]],
    availability: str,
    red_flags: Sequence[Mapping[str, Any]],
    tool_run_refs: Sequence[Mapping[str, Any]],
    producer_version: str = STRATEGY_PROJECT_CONTEXT_PRODUCER_VERSION,
) -> dict[str, Any]:
    body = _normalize_historical_strategy_review_body(
        {
            "schema_version": HISTORICAL_STRATEGY_REVIEW_SCHEMA_VERSION,
            "producer_version": producer_version,
            "task_id": task_id,
            "strategy_ref": strategy_ref,
            "version": version,
            "effective_period": effective_period,
            "asset_status": asset_status,
            "scope": scope,
            "traffic_allocation": traffic_allocation,
            "change_set": change_set,
            "observation_refs_by_effect_stage": dict(
                observation_refs_by_effect_stage
            ),
            "external_source_refs": list(external_source_refs),
            "decision_context_fields": list(decision_context_fields),
            "availability": availability,
            "red_flags": list(red_flags),
            "tool_run_refs": list(tool_run_refs),
        }
    )
    return _address_object(
        body,
        id_field="review_id",
        id_prefix="historical-strategy-review-",
    )


def validate_historical_strategy_review(value: object) -> dict[str, Any]:
    obj = _object(value, "historical_strategy_review")
    _require_exact_fields(
        obj,
        _HISTORICAL_STRATEGY_REVIEW_FIELDS,
        name="historical_strategy_review",
    )
    body = _normalize_historical_strategy_review_body(obj)
    return _validate_addressed_object(
        obj,
        normalized_body=body,
        id_field="review_id",
        id_pattern=_HISTORICAL_STRATEGY_REVIEW_ID_RE,
        id_prefix="historical-strategy-review-",
        name="historical_strategy_review",
    )


def _normalize_historical_strategy_review_body(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if value["schema_version"] != HISTORICAL_STRATEGY_REVIEW_SCHEMA_VERSION:
        raise StrategyProjectContextError(
            "historical_strategy_review schema_version is invalid"
        )
    producer = _producer_version(value["producer_version"])
    strategy_ref = _optional_ref_of_kind(
        value["strategy_ref"], {"strategy"}, "historical_strategy_review.strategy_ref"
    )
    version = value["version"]
    if version is not None:
        version = _positive_int(version, "historical_strategy_review.version")
    if (strategy_ref is None) != (version is None):
        raise StrategyProjectContextError(
            "historical_strategy_review strategy_ref and version must be provided together"
        )
    effect_stages = _effect_stage_refs(value["observation_refs_by_effect_stage"])
    external_refs = _refs_of_kind(
        value["external_source_refs"],
        {"external_report", "task_artifact"},
        "historical_strategy_review.external_source_refs",
    )
    decision_fields = _context_fields(
        value["decision_context_fields"],
        name="historical_strategy_review.decision_context_fields",
    )
    flags = _red_flags(
        value["red_flags"], name="historical_strategy_review.red_flags"
    )
    tool_refs = _refs_of_kind(
        value["tool_run_refs"],
        {"tool_run", "backtest", "pool_impact", "monitoring_plan", "monitoring_run"},
        "historical_strategy_review.tool_run_refs",
    )
    effective_period = _period_report_field(value["effective_period"])
    asset_status = validate_report_field(value["asset_status"])
    scope = validate_report_field(value["scope"])
    traffic_allocation = validate_report_field(value["traffic_allocation"])
    availability = _enum(
        value["availability"],
        REPORT_FIELD_AVAILABILITIES,
        "historical_strategy_review.availability",
    )
    if availability == "present":
        source_count = int(strategy_ref is not None) + len(external_refs) + len(tool_refs)
        source_count += sum(len(items) for items in effect_stages.values())
        source_count += sum(
            len(field["source_refs"])
            for field in (effective_period, asset_status, scope, traffic_allocation)
        )
        source_count += sum(
            len(item["field"]["source_refs"]) for item in decision_fields
        )
        if source_count == 0:
            raise StrategyProjectContextError(
                "present historical_strategy_review requires trusted evidence"
            )
    return {
        "schema_version": HISTORICAL_STRATEGY_REVIEW_SCHEMA_VERSION,
        "producer_version": producer,
        "task_id": _text(value["task_id"], "historical_strategy_review.task_id"),
        "strategy_ref": strategy_ref,
        "version": version,
        "effective_period": effective_period,
        "asset_status": asset_status,
        "scope": scope,
        "traffic_allocation": traffic_allocation,
        "change_set": validate_rule_change_set(value["change_set"]),
        "observation_refs_by_effect_stage": effect_stages,
        "external_source_refs": external_refs,
        "decision_context_fields": decision_fields,
        "availability": availability,
        "red_flags": flags,
        "tool_run_refs": tool_refs,
    }


def _period_report_field(value: object) -> dict[str, Any]:
    field = validate_report_field(value)
    if field["availability"] == "present":
        field = {**field, "value": _period(field["value"], "effective_period.value")}
    return field


def _period(value: object, name: str) -> dict[str, str | None]:
    obj = _object(value, name)
    _require_exact_fields(obj, _PERIOD_FIELDS, name=name)
    start = _iso_date(obj["start"], f"{name}.start")
    end = None if obj["end"] is None else _iso_date(obj["end"], f"{name}.end")
    if end is not None and end < start:
        raise StrategyProjectContextError(f"{name}.end cannot precede start")
    return {"start": start.isoformat(), "end": None if end is None else end.isoformat()}


def _effect_stage_refs(value: object) -> dict[str, list[dict[str, Any]]]:
    obj = _object(value, "observation_refs_by_effect_stage")
    _require_exact_fields(
        obj, EFFECT_STAGES, name="observation_refs_by_effect_stage"
    )
    return {
        stage: _effect_refs(obj[stage], stage=stage)
        for stage in sorted(EFFECT_STAGES)
    }


def _effect_refs(value: object, *, stage: str) -> list[dict[str, Any]]:
    refs = [
        _normalize_effect_observation_ref(item, stage=stage)
        for item in _array(value, f"observation_refs_by_effect_stage.{stage}")
    ]
    identities = [
        (
            item["observation_ref"]["kind"],
            item["observation_ref"]["ref_id"],
            item["observation_ref"]["content_hash"],
        )
        for item in refs
    ]
    if len(identities) != len(set(identities)):
        raise StrategyProjectContextError(
            f"observation_refs_by_effect_stage.{stage} contains duplicates"
        )
    refs.sort(
        key=lambda item: (
            item["observation_ref"]["kind"],
            item["observation_ref"]["ref_id"],
            item["observation_ref"]["content_hash"],
        )
    )
    return refs


def _normalize_effect_observation_ref(
    value: object, *, stage: str | None
) -> dict[str, Any]:
    obj = _object(value, "effect_observation_ref")
    _require_exact_fields(
        obj, _EFFECT_OBSERVATION_REF_FIELDS, name="effect_observation_ref"
    )
    observation = validate_source_ref(obj["observation_ref"])
    deployment = _optional_ref_of_kind(
        obj["deployment_ref"], {"deployment"}, "effect_observation_ref.deployment_ref"
    )
    environment = _optional_ref_of_kind(
        obj["environment_ref"], {"environment"}, "effect_observation_ref.environment_ref"
    )
    period = (
        None
        if obj["effective_period"] is None
        else _period(obj["effective_period"], "effect_observation_ref.effective_period")
    )
    if stage == "post_launch_observed":
        if deployment is None:
            raise StrategyProjectContextError(
                "post_launch_observed requires deployment_ref"
            )
        if environment is None:
            raise StrategyProjectContextError(
                "post_launch_observed requires environment_ref"
            )
        if period is None:
            raise StrategyProjectContextError(
                "post_launch_observed requires effective_period"
            )
        if observation["kind"] not in {
            "monitoring_run",
            "metric_observation",
            "external_report",
        }:
            raise StrategyProjectContextError(
                "post_launch_observed requires monitoring or observed evidence"
            )
    elif stage is not None and any(
        item is not None for item in (deployment, environment, period)
    ):
        raise StrategyProjectContextError(
            "only post_launch_observed may bind deployment evidence"
        )
    return {
        "observation_ref": observation,
        "deployment_ref": deployment,
        "environment_ref": environment,
        "effective_period": period,
    }


def build_strategy_project_context_state(
    *,
    task_id: str,
    as_of: str,
    current_project_snapshot: Mapping[str, Any],
    historical_strategy_reviews: Sequence[Mapping[str, Any]] = (),
    missing_information_records: Sequence[Mapping[str, Any]] = (),
    source_refs: Sequence[Mapping[str, Any]] = (),
    red_flags: Sequence[Mapping[str, Any]] = (),
    producer_version: str = STRATEGY_PROJECT_CONTEXT_PRODUCER_VERSION,
) -> dict[str, Any]:
    """Build the complete semantic state and its canonical evidence inventory."""

    nested_sources = _collect_source_refs(
        [
            current_project_snapshot,
            list(historical_strategy_reviews),
            list(missing_information_records),
            list(red_flags),
        ]
    )
    combined_sources = _merge_source_refs([*source_refs, *nested_sources])
    body = _normalize_strategy_project_context_state_body(
        {
            "schema_version": STRATEGY_PROJECT_CONTEXT_STATE_SCHEMA_VERSION,
            "producer_version": producer_version,
            "task_id": task_id,
            "as_of": as_of,
            "current_project_snapshot": current_project_snapshot,
            "historical_strategy_reviews": list(historical_strategy_reviews),
            "missing_information_records": list(missing_information_records),
            "source_refs": combined_sources,
            "red_flags": list(red_flags),
        }
    )
    return _address_object(
        body,
        id_field="context_id",
        id_prefix="strategy-project-context-",
    )


def validate_strategy_project_context_state(value: object) -> dict[str, Any]:
    obj = _object(value, "strategy_project_context_state")
    _preflight_json_tree(obj, name="strategy_project_context_state")
    _require_exact_fields(
        obj, _PROJECT_CONTEXT_STATE_FIELDS, name="strategy_project_context_state"
    )
    body = _normalize_strategy_project_context_state_body(obj)
    normalized = _validate_addressed_object(
        obj,
        normalized_body=body,
        id_field="context_id",
        id_pattern=_PROJECT_CONTEXT_ID_RE,
        id_prefix="strategy-project-context-",
        name="strategy_project_context_state",
    )
    if len(_canonical_json(normalized).encode("utf-8")) > MAX_PROJECT_CONTEXT_JSON_BYTES:
        raise StrategyProjectContextError(
            "strategy_project_context_state exceeds byte budget"
        )
    return normalized


def _normalize_strategy_project_context_state_body(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if value["schema_version"] != STRATEGY_PROJECT_CONTEXT_STATE_SCHEMA_VERSION:
        raise StrategyProjectContextError(
            "strategy_project_context_state schema_version is invalid"
        )
    producer = _producer_version(value["producer_version"])
    task_id = _text(value["task_id"], "strategy_project_context_state.task_id")
    as_of = _optional_as_of(value["as_of"], "strategy_project_context_state.as_of")
    if as_of is None:
        raise StrategyProjectContextError(
            "strategy_project_context_state.as_of must not be null"
        )
    snapshot = validate_current_project_snapshot(value["current_project_snapshot"])
    if snapshot["task_id"] != task_id:
        raise StrategyProjectContextError(
            "current_project_snapshot task_id does not match context task_id"
        )
    if snapshot["producer_version"] != producer:
        raise StrategyProjectContextError(
            "current_project_snapshot producer_version does not match context"
        )
    if snapshot["as_of"] != as_of:
        raise StrategyProjectContextError(
            "current_project_snapshot as_of does not match context"
        )
    histories = [
        validate_historical_strategy_review(item)
        for item in _array(
            value["historical_strategy_reviews"],
            "strategy_project_context_state.historical_strategy_reviews",
        )
    ]
    history_ids = [item["review_id"] for item in histories]
    if len(history_ids) != len(set(history_ids)):
        raise StrategyProjectContextError(
            "historical_strategy_reviews contain duplicate review_id values"
        )
    for review in histories:
        if review["task_id"] != task_id or review["producer_version"] != producer:
            raise StrategyProjectContextError(
                "historical_strategy_review task or producer does not match context"
            )
    histories.sort(
        key=lambda item: (
            item["version"] is None,
            item["version"] if item["version"] is not None else 0,
            item["review_id"],
        )
    )
    missing = [
        validate_missing_information_record(item)
        for item in _array(
            value["missing_information_records"],
            "strategy_project_context_state.missing_information_records",
        )
    ]
    missing_ids = [item["missing_information_id"] for item in missing]
    if len(missing_ids) != len(set(missing_ids)):
        raise StrategyProjectContextError(
            "missing_information_records contain duplicate ids"
        )
    for record in missing:
        if record["task_id"] != task_id:
            raise StrategyProjectContextError(
                "missing_information_record task_id does not match context"
            )
    missing.sort(
        key=lambda item: (
            item["field_path"],
            item["dependency_hash"],
            item["missing_information_id"],
        )
    )
    flags = _red_flags(
        value["red_flags"], name="strategy_project_context_state.red_flags"
    )
    sources = _source_refs(
        value["source_refs"], name="strategy_project_context_state.source_refs"
    )
    nested_sources = _collect_source_refs([snapshot, histories, missing, flags])
    source_identities = {_source_identity(item) for item in sources}
    missing_sources = sorted(
        _source_identity(item)
        for item in nested_sources
        if _source_identity(item) not in source_identities
    )
    if missing_sources:
        raise StrategyProjectContextError(
            "strategy_project_context_state source_refs omit nested evidence"
        )
    return {
        "schema_version": STRATEGY_PROJECT_CONTEXT_STATE_SCHEMA_VERSION,
        "producer_version": producer,
        "task_id": task_id,
        "as_of": as_of,
        "current_project_snapshot": snapshot,
        "historical_strategy_reviews": histories,
        "missing_information_records": missing,
        "source_refs": sources,
        "red_flags": flags,
    }


def canonical_strategy_project_context_state_json(
    value: Mapping[str, Any],
) -> str:
    return _canonical_json(validate_strategy_project_context_state(value))


def strategy_project_context_state_hash(value: Mapping[str, Any]) -> str:
    return validate_strategy_project_context_state(value)["content_hash"]


def strategy_project_context_state_from_json(
    raw: str | bytes | bytearray,
) -> dict[str, Any]:
    if not isinstance(raw, (str, bytes, bytearray)):
        raise StrategyProjectContextError(
            "strategy project-context state JSON must be text or bytes"
        )
    raw_size = len(raw.encode("utf-8")) if isinstance(raw, str) else len(raw)
    if raw_size > MAX_PROJECT_CONTEXT_JSON_BYTES:
        raise StrategyProjectContextError(
            "strategy project-context state JSON exceeds byte budget"
        )
    try:
        payload = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except StrategyProjectContextError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise StrategyProjectContextError(
            "strategy project-context state is not valid bounded JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise StrategyProjectContextError(
            "strategy project-context state JSON must contain an object"
        )
    return validate_strategy_project_context_state(payload)


def build_strategy_project_context_revision(
    *,
    state: Mapping[str, Any],
    revision: int,
    parent_revision_id: str | None,
    parent_state_hash: str | None,
    operation_kind: str,
    producer_version: str = STRATEGY_PROJECT_CONTEXT_PRODUCER_VERSION,
) -> dict[str, Any]:
    normalized_state = validate_strategy_project_context_state(state)
    producer = _producer_version(producer_version)
    revision_number = _positive_int(revision, "strategy_project_context_revision.revision")
    parent_id, parent_hash = _revision_parent(
        revision_number,
        parent_revision_id=parent_revision_id,
        parent_state_hash=parent_state_hash,
    )
    operation = _text(
        operation_kind, "strategy_project_context_revision.operation_kind"
    )
    operation_hash = strategy_project_context_operation_hash(
        task_id=normalized_state["task_id"],
        revision=revision_number,
        parent_revision_id=parent_id,
        parent_state_hash=parent_hash,
        operation_kind=operation,
        state_hash=normalized_state["content_hash"],
    )
    body = _normalize_strategy_project_context_revision_body(
        {
            "schema_version": STRATEGY_PROJECT_CONTEXT_REVISION_SCHEMA_VERSION,
            "producer_version": producer,
            "task_id": normalized_state["task_id"],
            "revision": revision_number,
            "parent_revision_id": parent_id,
            "parent_state_hash": parent_hash,
            "operation_kind": operation,
            "operation_hash": operation_hash,
            "state": normalized_state,
            "state_hash": normalized_state["content_hash"],
        }
    )
    return _address_object(
        body,
        id_field="revision_id",
        id_prefix="strategy-project-context-revision-",
    )


def validate_strategy_project_context_revision(value: object) -> dict[str, Any]:
    obj = _object(value, "strategy_project_context_revision")
    _preflight_json_tree(obj, name="strategy_project_context_revision")
    _require_exact_fields(
        obj,
        _PROJECT_CONTEXT_REVISION_FIELDS,
        name="strategy_project_context_revision",
    )
    body = _normalize_strategy_project_context_revision_body(obj)
    normalized = _validate_addressed_object(
        obj,
        normalized_body=body,
        id_field="revision_id",
        id_pattern=_PROJECT_CONTEXT_REVISION_ID_RE,
        id_prefix="strategy-project-context-revision-",
        name="strategy_project_context_revision",
    )
    if len(_canonical_json(normalized).encode("utf-8")) > MAX_PROJECT_CONTEXT_JSON_BYTES:
        raise StrategyProjectContextError(
            "strategy_project_context_revision exceeds byte budget"
        )
    return normalized


def _normalize_strategy_project_context_revision_body(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if value["schema_version"] != STRATEGY_PROJECT_CONTEXT_REVISION_SCHEMA_VERSION:
        raise StrategyProjectContextError(
            "strategy_project_context_revision schema_version is invalid"
        )
    producer = _producer_version(value["producer_version"])
    task_id = _text(value["task_id"], "strategy_project_context_revision.task_id")
    revision = _positive_int(
        value["revision"], "strategy_project_context_revision.revision"
    )
    parent_id, parent_hash = _revision_parent(
        revision,
        parent_revision_id=value["parent_revision_id"],
        parent_state_hash=value["parent_state_hash"],
    )
    state = validate_strategy_project_context_state(value["state"])
    if state["task_id"] != task_id:
        raise StrategyProjectContextError(
            "strategy_project_context_revision state task_id is inconsistent"
        )
    if state["producer_version"] != producer:
        raise StrategyProjectContextError(
            "strategy_project_context_revision producer_version is inconsistent"
        )
    state_hash = _hash(
        value["state_hash"], "strategy_project_context_revision.state_hash"
    )
    if not hmac.compare_digest(state_hash, state["content_hash"]):
        raise StrategyProjectContextError(
            "strategy_project_context_revision state_hash does not match state"
        )
    operation_kind = _text(
        value["operation_kind"], "strategy_project_context_revision.operation_kind"
    )
    operation_hash = _hash(
        value["operation_hash"], "strategy_project_context_revision.operation_hash"
    )
    expected_operation_hash = strategy_project_context_operation_hash(
        task_id=task_id,
        revision=revision,
        parent_revision_id=parent_id,
        parent_state_hash=parent_hash,
        operation_kind=operation_kind,
        state_hash=state_hash,
    )
    if not hmac.compare_digest(operation_hash, expected_operation_hash):
        raise StrategyProjectContextError(
            "strategy_project_context_revision operation_hash does not match operation"
        )
    return {
        "schema_version": STRATEGY_PROJECT_CONTEXT_REVISION_SCHEMA_VERSION,
        "producer_version": producer,
        "task_id": task_id,
        "revision": revision,
        "parent_revision_id": parent_id,
        "parent_state_hash": parent_hash,
        "operation_kind": operation_kind,
        "operation_hash": operation_hash,
        "state": state,
        "state_hash": state_hash,
    }


def strategy_project_context_operation_hash(
    *,
    task_id: str,
    revision: int,
    parent_revision_id: str | None,
    parent_state_hash: str | None,
    operation_kind: str,
    state_hash: str,
) -> str:
    revision_number = _positive_int(revision, "project_context_operation.revision")
    parent_id, parent_hash = _revision_parent(
        revision_number,
        parent_revision_id=parent_revision_id,
        parent_state_hash=parent_state_hash,
    )
    payload = {
        "schema_version": STRATEGY_PROJECT_CONTEXT_OPERATION_SCHEMA_VERSION,
        "task_id": _text(task_id, "project_context_operation.task_id"),
        "revision": revision_number,
        "parent_revision_id": parent_id,
        "parent_state_hash": parent_hash,
        "operation_kind": _text(
            operation_kind, "project_context_operation.operation_kind"
        ),
        "state_hash": _hash(state_hash, "project_context_operation.state_hash"),
    }
    return _sha256(_canonical_json(payload))


def _revision_parent(
    revision: int,
    *,
    parent_revision_id: object,
    parent_state_hash: object,
) -> tuple[str | None, str | None]:
    if revision == 1:
        if parent_revision_id is not None or parent_state_hash is not None:
            raise StrategyProjectContextError(
                "initial revision requires null parent revision and state hash"
            )
        return None, None
    if parent_revision_id is None or parent_state_hash is None:
        raise StrategyProjectContextError(
            "non-initial revision requires parent revision and state hash"
        )
    if (
        not isinstance(parent_revision_id, str)
        or _PROJECT_CONTEXT_REVISION_ID_RE.fullmatch(parent_revision_id) is None
    ):
        raise StrategyProjectContextError(
            "parent_revision_id must identify a project-context revision"
        )
    return parent_revision_id, _hash(
        parent_state_hash, "strategy_project_context_revision.parent_state_hash"
    )


def canonical_strategy_project_context_revision_json(
    value: Mapping[str, Any],
) -> str:
    return _canonical_json(validate_strategy_project_context_revision(value))


def strategy_project_context_revision_from_json(
    raw: str | bytes | bytearray,
) -> dict[str, Any]:
    if not isinstance(raw, (str, bytes, bytearray)):
        raise StrategyProjectContextError(
            "strategy project-context revision JSON must be text or bytes"
        )
    raw_size = len(raw.encode("utf-8")) if isinstance(raw, str) else len(raw)
    if raw_size > MAX_PROJECT_CONTEXT_JSON_BYTES:
        raise StrategyProjectContextError(
            "strategy project-context revision JSON exceeds byte budget"
        )
    try:
        payload = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except StrategyProjectContextError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise StrategyProjectContextError(
            "strategy project-context revision is not valid bounded JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise StrategyProjectContextError(
            "strategy project-context revision JSON must contain an object"
        )
    return validate_strategy_project_context_revision(payload)


def _collect_source_refs(value: object) -> list[dict[str, str]]:
    refs: dict[tuple[str, str, str], dict[str, str]] = {}
    stack: list[tuple[object, int]] = [(value, 1)]
    seen_containers: set[int] = set()
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_PROJECT_CONTEXT_JSON_NODES:
            raise StrategyProjectContextError(
                "project-context evidence exceeds JSON node budget"
            )
        if depth > MAX_PROJECT_CONTEXT_JSON_DEPTH:
            raise StrategyProjectContextError(
                "project-context evidence exceeds JSON depth budget"
            )
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            if set(current) == _SOURCE_REF_FIELDS:
                ref = validate_source_ref(current)
                refs[_source_identity(ref)] = ref
                continue
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            identity = id(current)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            stack.extend((child, depth + 1) for child in current)
    return [refs[key] for key in sorted(refs)]


def _merge_source_refs(
    refs: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for value in refs:
        ref = validate_source_ref(value)
        identity = (ref["kind"], ref["ref_id"])
        existing = merged.get(identity)
        if existing is not None and not hmac.compare_digest(
            existing["content_hash"], ref["content_hash"]
        ):
            raise StrategyProjectContextError(
                "source_refs contain source_ref identity drift"
            )
        merged[identity] = ref
    return [merged[key] for key in sorted(merged)]


def _source_identity(value: Mapping[str, str]) -> tuple[str, str, str]:
    return (value["kind"], value["ref_id"], value["content_hash"])


def build_missing_information_record(
    *,
    task_id: str,
    field_path: str,
    reason: str,
    blocking: str,
    question: str,
    status: str,
    asked_count: int,
    asked_at: str | None,
    answered_at: str | None,
    answer_source_ref: Mapping[str, Any] | None,
    dependency_hash: str,
) -> dict[str, Any]:
    """Build one revision-local state of a dependency-stable missing record."""

    normalized_body = _normalize_missing_information_body(
        {
            "schema_version": MISSING_INFORMATION_SCHEMA_VERSION,
            "task_id": task_id,
            "field_path": field_path,
            "reason": reason,
            "blocking": blocking,
            "question": question,
            "status": status,
            "asked_count": asked_count,
            "asked_at": asked_at,
            "answered_at": answered_at,
            "answer_source_ref": answer_source_ref,
            "dependency_hash": dependency_hash,
        }
    )
    identity = _missing_information_identity(normalized_body)
    record_id = "strategy-missing-information-" + _sha256(
        _canonical_json(identity)
    )[:24]
    without_hash = {
        **normalized_body,
        "missing_information_id": record_id,
    }
    return {
        **without_hash,
        "content_hash": _sha256(_canonical_json(without_hash)),
    }


def validate_missing_information_record(value: object) -> dict[str, Any]:
    obj = _object(value, "missing_information")
    _require_exact_fields(
        obj, _MISSING_INFORMATION_FIELDS, name="missing_information"
    )
    body = _normalize_missing_information_body(obj)
    identity = _missing_information_identity(body)
    expected_id = "strategy-missing-information-" + _sha256(
        _canonical_json(identity)
    )[:24]
    record_id = obj["missing_information_id"]
    if (
        not isinstance(record_id, str)
        or _MISSING_INFORMATION_ID_RE.fullmatch(record_id) is None
        or not hmac.compare_digest(record_id, expected_id)
    ):
        raise StrategyProjectContextError(
            "missing_information missing_information_id does not match identity"
        )
    content_hash = _hash(
        obj["content_hash"], "missing_information.content_hash"
    )
    without_hash = {**body, "missing_information_id": record_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(content_hash, expected_hash):
        raise StrategyProjectContextError(
            "missing_information content_hash does not match content"
        )
    return {**without_hash, "content_hash": content_hash}


def _normalize_missing_information_body(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if value["schema_version"] != MISSING_INFORMATION_SCHEMA_VERSION:
        raise StrategyProjectContextError(
            "missing_information schema_version is invalid"
        )
    blocking = _enum(
        value["blocking"],
        MISSING_INFORMATION_BLOCKING_LEVELS,
        "missing_information.blocking",
    )
    status = _enum(
        value["status"], MISSING_INFORMATION_STATUSES, "missing_information.status"
    )
    asked_count = _non_negative_int(
        value["asked_count"], "missing_information.asked_count"
    )
    asked_at = _optional_timestamp(value["asked_at"], "missing_information.asked_at")
    answered_at = _optional_timestamp(
        value["answered_at"], "missing_information.answered_at"
    )
    answer_ref = (
        None
        if value["answer_source_ref"] is None
        else validate_source_ref(value["answer_source_ref"])
    )
    if asked_count == 0 and asked_at is not None:
        raise StrategyProjectContextError(
            "missing_information asked_at requires asked_count greater than zero"
        )
    if asked_count > 0 and asked_at is None:
        raise StrategyProjectContextError(
            "missing_information asked_count requires asked_at"
        )
    if blocking == "report_optional" and asked_count > 1:
        raise StrategyProjectContextError(
            "report_optional missing information may be asked at most once"
        )
    if status == "pending":
        if answered_at is not None or answer_ref is not None:
            raise StrategyProjectContextError(
                "pending missing_information answer fields must be null"
            )
    elif answered_at is None or answer_ref is None:
        raise StrategyProjectContextError(
            "resolved missing_information requires answer fields"
        )
    if asked_at is not None and answered_at is not None:
        if datetime.fromisoformat(answered_at) < datetime.fromisoformat(asked_at):
            raise StrategyProjectContextError(
                "missing_information answered_at cannot precede asked_at"
            )
    return {
        "schema_version": MISSING_INFORMATION_SCHEMA_VERSION,
        "task_id": _text(value["task_id"], "missing_information.task_id"),
        "field_path": _text(
            value["field_path"], "missing_information.field_path"
        ),
        "reason": _text(value["reason"], "missing_information.reason"),
        "blocking": blocking,
        "question": _text(value["question"], "missing_information.question"),
        "status": status,
        "asked_count": asked_count,
        "asked_at": asked_at,
        "answered_at": answered_at,
        "answer_source_ref": answer_ref,
        "dependency_hash": _hash(
            value["dependency_hash"], "missing_information.dependency_hash"
        ),
    }


def _missing_information_identity(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "task_id": value["task_id"],
        "field_path": value["field_path"],
        "dependency_hash": value["dependency_hash"],
    }


def _source_refs(value: object, *, name: str) -> list[dict[str, str]]:
    refs = [validate_source_ref(item) for item in _array(value, name)]
    seen: dict[tuple[str, str], str] = {}
    for item in refs:
        identity = (item["kind"], item["ref_id"])
        existing_hash = seen.get(identity)
        if existing_hash is not None:
            if not hmac.compare_digest(existing_hash, item["content_hash"]):
                raise StrategyProjectContextError(
                    f"{name} contains source_ref identity drift"
                )
            raise StrategyProjectContextError(
                f"{name} contains duplicate evidence pointers"
            )
        seen[identity] = item["content_hash"]
    refs.sort(key=lambda item: (item["kind"], item["ref_id"], item["content_hash"]))
    return refs


def _refs_of_kind(
    value: object, allowed: set[str], name: str
) -> list[dict[str, str]]:
    refs = _source_refs(value, name=name)
    invalid = sorted({item["kind"] for item in refs} - allowed)
    if invalid:
        raise StrategyProjectContextError(
            f"{name} contains invalid source kinds: {', '.join(invalid)}"
        )
    return refs


def _optional_ref_of_kind(
    value: object, allowed: set[str], name: str
) -> dict[str, str] | None:
    if value is None:
        return None
    ref = validate_source_ref(value)
    if ref["kind"] not in allowed:
        raise StrategyProjectContextError(
            f"{name} source kind must be one of {', '.join(sorted(allowed))}"
        )
    return ref


def _address_object(
    body: Mapping[str, Any], *, id_field: str, id_prefix: str
) -> dict[str, Any]:
    _preflight_json_tree(body, name=id_field)
    object_id = id_prefix + _sha256(_canonical_json(body))[:24]
    without_hash = {**body, id_field: object_id}
    return {**without_hash, "content_hash": _sha256(_canonical_json(without_hash))}


def _validate_addressed_object(
    original: Mapping[str, Any],
    *,
    normalized_body: Mapping[str, Any],
    id_field: str,
    id_pattern: re.Pattern[str],
    id_prefix: str,
    name: str,
) -> dict[str, Any]:
    object_id = original[id_field]
    if not isinstance(object_id, str) or id_pattern.fullmatch(object_id) is None:
        raise StrategyProjectContextError(f"{name} {id_field} is invalid")
    expected_id = id_prefix + _sha256(_canonical_json(normalized_body))[:24]
    if not hmac.compare_digest(object_id, expected_id):
        raise StrategyProjectContextError(f"{name} {id_field} does not match content")
    content_hash = _hash(original["content_hash"], f"{name}.content_hash")
    without_hash = {**normalized_body, id_field: object_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(content_hash, expected_hash):
        raise StrategyProjectContextError(f"{name} content_hash does not match content")
    return {**without_hash, "content_hash": content_hash}


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyProjectContextError(f"{name} must be an object")
    return value


def _array(value: object, name: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise StrategyProjectContextError(f"{name} must be an array")
    return list(value)


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, name: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise StrategyProjectContextError(f"{name} keys must be strings")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise StrategyProjectContextError(
            f"{name} fields are invalid ({'; '.join(details)})"
        )


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise StrategyProjectContextError(f"{name} must be non-empty canonical text")
    return value


def _optional_text(value: object, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _producer_version(value: object) -> str:
    return _text(value, "producer_version")


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategyProjectContextError(
            f"{name} must be a lowercase SHA-256 hash"
        )
    return value


def _enum(value: object, allowed: frozenset[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise StrategyProjectContextError(
            f"{name} must be one of {', '.join(sorted(allowed))}"
        )
    return value


def _optional_as_of(value: object, name: str) -> str | None:
    if value is None:
        return None
    text = _text(value, name)
    try:
        if "T" not in text:
            if date.fromisoformat(text).isoformat() != text:
                raise ValueError
        else:
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None or parsed.isoformat() != text:
                raise ValueError
    except ValueError as exc:
        raise StrategyProjectContextError(
            f"{name} must be a canonical ISO date or timezone-aware datetime"
        ) from exc
    return text


def _optional_timestamp(value: object, name: str) -> str | None:
    if value is None:
        return None
    text = _text(value, name)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise StrategyProjectContextError(
            f"{name} must be a canonical timezone-aware datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.isoformat() != text:
        raise StrategyProjectContextError(
            f"{name} must be a canonical timezone-aware datetime"
        )
    return text


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyProjectContextError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _non_negative_int(value, name)
    if result == 0:
        raise StrategyProjectContextError(f"{name} must be a positive integer")
    return result


def _iso_date(value: object, name: str) -> date:
    text = _text(value, name)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise StrategyProjectContextError(f"{name} must be a canonical ISO date") from exc
    if parsed.isoformat() != text:
        raise StrategyProjectContextError(f"{name} must be a canonical ISO date")
    return parsed


def _json_value(value: object, name: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except StrategyProjectContextError:
        raise
    except (TypeError, ValueError, RecursionError, MemoryError) as exc:
        raise StrategyProjectContextError(f"{name} must be bounded JSON") from exc


def _preflight_json_tree(value: object, *, name: str) -> None:
    stack: list[tuple[object, int]] = [(value, 1)]
    seen_containers: set[int] = set()
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > MAX_PROJECT_CONTEXT_JSON_NODES:
            raise StrategyProjectContextError(f"{name} exceeds JSON node budget")
        if depth > MAX_PROJECT_CONTEXT_JSON_DEPTH:
            raise StrategyProjectContextError(f"{name} exceeds JSON depth budget")
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in seen_containers:
                raise StrategyProjectContextError(
                    f"{name} contains a repeated or cyclic container"
                )
            seen_containers.add(identity)
            if any(not isinstance(key, str) for key in current):
                raise StrategyProjectContextError(
                    f"{name} contains a non-string JSON key"
                )
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            identity = id(current)
            if identity in seen_containers:
                raise StrategyProjectContextError(
                    f"{name} contains a repeated or cyclic container"
                )
            seen_containers.add(identity)
            stack.extend((child, depth + 1) for child in current)
        elif current is None or isinstance(current, (str, bool, int, float)):
            if isinstance(current, float) and not math.isfinite(current):
                raise StrategyProjectContextError(
                    f"{name} contains a non-finite number"
                )
        else:
            raise StrategyProjectContextError(
                f"{name} contains unsupported JSON value {type(current).__name__}"
            )


def _canonical_json(value: object) -> str:
    _preflight_json_tree(value, name="canonical JSON")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise StrategyProjectContextError("value is not finite canonical JSON") from exc


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrategyProjectContextError(
                f"strategy project-context JSON has duplicate key: {key}"
            )
        result[key] = value
    return result


def _sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "CURRENT_PROJECT_SNAPSHOT_SCHEMA_VERSION",
    "CURRENT_STATUS_FIELDS",
    "EFFECT_STAGES",
    "HISTORICAL_STRATEGY_REVIEW_SCHEMA_VERSION",
    "MAX_PROJECT_CONTEXT_JSON_BYTES",
    "MAX_PROJECT_CONTEXT_JSON_DEPTH",
    "MAX_PROJECT_CONTEXT_JSON_NODES",
    "MISSING_INFORMATION_BLOCKING_LEVELS",
    "MISSING_INFORMATION_SCHEMA_VERSION",
    "MISSING_INFORMATION_STATUSES",
    "REPORT_FIELD_AVAILABILITIES",
    "REPORT_FIELD_BLOCKING_LEVELS",
    "REPORT_FIELD_ORIGINS",
    "RED_FLAG_LEVELS",
    "STRATEGY_PROJECT_CONTEXT_PRODUCER_VERSION",
    "STRATEGY_PROJECT_CONTEXT_OPERATION_SCHEMA_VERSION",
    "STRATEGY_PROJECT_CONTEXT_REVISION_SCHEMA_VERSION",
    "STRATEGY_PROJECT_CONTEXT_STATE_SCHEMA_VERSION",
    "StrategyProjectContextError",
    "build_context_field",
    "build_current_project_snapshot",
    "build_effect_observation_ref",
    "build_historical_strategy_review",
    "build_missing_information_record",
    "build_red_flag",
    "build_report_field",
    "build_source_ref",
    "build_strategy_project_context_revision",
    "build_strategy_project_context_state",
    "canonical_strategy_project_context_revision_json",
    "canonical_strategy_project_context_state_json",
    "diff_strategy_rules",
    "strategy_project_context_operation_hash",
    "strategy_project_context_revision_from_json",
    "strategy_project_context_state_hash",
    "strategy_project_context_state_from_json",
    "validate_current_project_snapshot",
    "validate_historical_strategy_review",
    "validate_report_field",
    "validate_missing_information_record",
    "validate_red_flag",
    "validate_source_ref",
    "validate_strategy_project_context_revision",
    "validate_strategy_project_context_state",
    "validate_rule_change_set",
]
