"""Pure, deterministic Strategy Pool core.

The pool is a task-scoped draft assembly of immutable candidate artifacts.  It
owns ordering and typed actions, but it does not own candidate metrics,
validation, adoption, deployment, persistence, or dataset execution.  Every
mutation returns a new canonical snapshot; the tool adapter persists that
snapshot under an optimistic revision/hash compare-and-swap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
from typing import Any

from marvis.packs.strategy.candidate_asset import validate_candidate_asset
from marvis.packs.strategy.dsl import (
    StrategyAction,
    StrategyRuleSpec,
    StrategySpec,
    canonicalize_expression,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.repositories.strategy_pool import (
    ABSENT_POOL_REVISION,
    ABSENT_POOL_SNAPSHOT_HASH,
    strategy_pool_id,
    strategy_pool_operation_hash,
    strategy_pool_revision_id,
    strategy_pool_snapshot_hash as _repository_snapshot_hash,
)


POOL_SCHEMA_VERSION = "strategy.candidate-pool.v1"
POOL_PRODUCER_VERSION = "strategy.candidate-pool/1"
SELECTED_STRATEGY_DESIGN_SCHEMA_VERSION = "strategy.selected-strategy-design.v1"

_STATUS = "draft"
_VALIDATION_STATUS = "unvalidated"
_CANDIDATE_KIND = "univariate_refinement"
_CANDIDATE_ARTIFACT_KIND = "strategy_candidate_asset_json"
_MUTATION_KINDS = frozenset(
    {
        "add_candidate",
        "remove_entry",
        "set_entry_action",
        "reorder_entries",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "pool_id",
        "task_id",
        "strategy_type",
        "revision",
        "revision_id",
        "parent_revision_id",
        "snapshot_hash",
        "operation",
        "default_action",
        "entries",
        "status",
        "validation_status",
    }
)
_OPERATION_FIELDS = frozenset({"kind", "operation_hash", "reason"})
_ENTRY_FIELDS = frozenset(
    {
        "entry_id",
        "rule_id",
        "position",
        "source",
        "execution",
        "action",
        "enabled",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "content_hash",
        "asset_id",
        "asset_hash",
        "candidate_kind",
        "fragment_id",
        "effect_id",
        "effect_stage",
        "validation_status",
        "parent_candidate_id",
        "parent_evidence_hash",
        "evidence_identity",
    }
)
_EVIDENCE_IDENTITY_FIELDS = frozenset(
    {
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    }
)
_EXECUTION_FIELDS = frozenset({"condition", "requirements"})


class CandidatePoolError(StrategyError):
    """A Strategy Pool snapshot or mutation violates its strict contract."""


def add_candidate(
    pool: Mapping[str, Any] | None,
    *,
    task_id: str,
    strategy_type: str,
    default_action: Mapping[str, Any],
    candidate_asset: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    action: Mapping[str, Any],
    reason: str | None = None,
) -> dict[str, Any]:
    """Append one verified candidate fragment to a new immutable pool revision."""

    task = _text(task_id, "task_id")
    strategy_kind = _text(strategy_type, "strategy_type")
    canonical_default = _action(default_action, "default_action")
    if pool is None:
        pool_id = strategy_pool_id(task, strategy_kind)
        revision = ABSENT_POOL_REVISION + 1
        parent_revision_id = None
        entries: list[dict[str, Any]] = []
    else:
        current = validate_strategy_pool(pool)
        if current["task_id"] != task:
            raise CandidatePoolError("task_id does not match the existing pool")
        if current["strategy_type"] != strategy_kind:
            raise CandidatePoolError("strategy_type does not match the existing pool")
        if current["default_action"] != canonical_default:
            raise CandidatePoolError("default_action does not match the existing pool")
        pool_id = current["pool_id"]
        revision = int(current["revision"]) + 1
        parent_revision_id = current["revision_id"]
        entries = [_json_object(item, "pool entry") for item in current["entries"]]

    asset = validate_candidate_asset(candidate_asset)
    source = _candidate_source(source_binding, asset=asset)
    rule = asset["rule"]
    condition = canonicalize_expression(rule["condition"])
    rule_id = _text(rule["rule_id"], "candidate rule_id")
    canonical_action = _action(action, "action")
    entry_id = _stable_id(
        "pool-entry",
        {
            "pool_id": pool_id,
            "artifact_id": source["artifact_id"],
            "asset_id": source["asset_id"],
            "rule_id": rule_id,
            "fragment_id": source["fragment_id"],
        },
    )
    if any(item["source"]["asset_id"] == source["asset_id"] for item in entries):
        raise CandidatePoolError(f"duplicate asset in strategy pool: {source['asset_id']}")
    if any(item["rule_id"] == rule_id for item in entries):
        raise CandidatePoolError(f"duplicate rule in strategy pool: {rule_id}")
    if any(item["entry_id"] == entry_id for item in entries):
        raise CandidatePoolError(f"duplicate entry in strategy pool: {entry_id}")
    if entries and any(
        item["source"]["evidence_identity"] != source["evidence_identity"]
        for item in entries
    ):
        raise CandidatePoolError(
            "candidate evidence identity does not match the existing strategy pool"
        )
    entries.append(
        {
            "entry_id": entry_id,
            "rule_id": rule_id,
            "position": len(entries),
            "source": source,
            "execution": {"condition": condition, "requirements": []},
            "action": canonical_action,
            "enabled": True,
        }
    )
    _assert_strategy_actions(strategy_kind, canonical_default, entries)
    return _snapshot(
        pool_id=pool_id,
        task_id=task,
        strategy_type=strategy_kind,
        revision=revision,
        parent_revision_id=parent_revision_id,
        operation_kind="add_candidate",
        reason=reason,
        default_action=canonical_default,
        entries=entries,
    )


def remove_pool_entry(
    pool: Mapping[str, Any], entry_id: str, *, reason: str | None = None
) -> dict[str, Any]:
    """Remove one known membership and return the next canonical revision."""

    current = validate_strategy_pool(pool)
    target = _text(entry_id, "entry_id")
    if all(item["entry_id"] != target for item in current["entries"]):
        raise CandidatePoolError(f"unknown pool entry: {target}")
    entries = [
        _json_object(item, "pool entry")
        for item in current["entries"]
        if item["entry_id"] != target
    ]
    entries = _with_positions(entries)
    return _next_snapshot(current, "remove_entry", entries, reason=reason)


def set_pool_entry_action(
    pool: Mapping[str, Any],
    entry_id: str,
    action: Mapping[str, Any],
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """Replace the Pool-owned typed action for one known membership."""

    current = validate_strategy_pool(pool)
    target = _text(entry_id, "entry_id")
    canonical_action = _action(action, "action")
    found = False
    entries: list[dict[str, Any]] = []
    for item in current["entries"]:
        entry = _json_object(item, "pool entry")
        if entry["entry_id"] == target:
            entry["action"] = canonical_action
            found = True
        entries.append(entry)
    if not found:
        raise CandidatePoolError(f"unknown pool entry: {target}")
    _assert_strategy_actions(
        current["strategy_type"], current["default_action"], entries
    )
    return _next_snapshot(current, "set_entry_action", entries, reason=reason)


def reorder_strategy_pool(
    pool: Mapping[str, Any],
    ordered_entry_ids: Sequence[str],
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """Apply one complete, duplicate-free permutation of all Pool memberships."""

    current = validate_strategy_pool(pool)
    if isinstance(ordered_entry_ids, str | bytes | bytearray) or not isinstance(
        ordered_entry_ids, Sequence
    ):
        raise CandidatePoolError("ordered_entry_ids must be a complete list")
    ordered = [_text(value, "ordered_entry_ids item") for value in ordered_entry_ids]
    if len(set(ordered)) != len(ordered):
        raise CandidatePoolError("ordered_entry_ids must not contain duplicate ids")
    by_id = {item["entry_id"]: item for item in current["entries"]}
    unknown = sorted(set(ordered) - set(by_id))
    if unknown:
        raise CandidatePoolError(
            "ordered_entry_ids contains unknown ids: " + ", ".join(unknown)
        )
    missing = sorted(set(by_id) - set(ordered))
    if missing or len(ordered) != len(by_id):
        raise CandidatePoolError(
            "ordered_entry_ids must be a complete ordering of pool entries"
        )
    entries = _with_positions(
        [_json_object(by_id[entry_id], "pool entry") for entry_id in ordered]
    )
    return _next_snapshot(current, "reorder_entries", entries, reason=reason)


def compile_strategy_pool(pool: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one non-empty Pool snapshot to a canonical SelectedStrategyDesign."""

    current = validate_strategy_pool(pool)
    entries = current["entries"]
    if not entries:
        raise CandidatePoolError("cannot compile an empty strategy pool")
    rules = tuple(
        StrategyRuleSpec(
            rule_id=item["rule_id"],
            priority=(index + 1) * 10,
            condition=item["execution"]["condition"],
            action=StrategyAction.from_dict(item["action"]),
        )
        for index, item in enumerate(entries)
    )
    snapshot_hash = current["snapshot_hash"]
    source_entry_refs = [
        {
            "entry_id": item["entry_id"],
            "rule_id": item["rule_id"],
            "source": _json_object(item["source"], "entry source"),
        }
        for item in entries
    ]
    pool_ref = {
        "pool_id": current["pool_id"],
        "task_id": current["task_id"],
        "strategy_type": current["strategy_type"],
        "revision": current["revision"],
        "revision_id": current["revision_id"],
        "snapshot_hash": snapshot_hash,
    }
    requirements: list[dict[str, Any]] = []
    spec = StrategySpec(
        strategy_type=current["strategy_type"],
        default_action=StrategyAction.from_dict(current["default_action"]),
        rules=rules,
        metadata={
            "lineage": {
                "source": "strategy_pool",
                "pool_ref": pool_ref,
                "source_entry_refs": source_entry_refs,
            }
        },
    ).to_dict()
    body = {
        "schema_version": SELECTED_STRATEGY_DESIGN_SCHEMA_VERSION,
        "pool_ref": pool_ref,
        "requirements": requirements,
        "strategy_spec": spec,
        "source_entry_refs": source_entry_refs,
    }
    return {**body, "design_hash": _sha256(_canonical_json(body))}


def validate_strategy_pool(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an exact snapshot and return its detached canonical form."""

    if not isinstance(payload, Mapping):
        raise CandidatePoolError("strategy pool must be an object")
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "strategy pool")
    if payload["schema_version"] != POOL_SCHEMA_VERSION:
        raise CandidatePoolError(
            f"schema_version must be {POOL_SCHEMA_VERSION}"
        )
    task_id = _text(payload["task_id"], "task_id")
    strategy_type = _text(payload["strategy_type"], "strategy_type")
    pool_id = _text(payload["pool_id"], "pool_id")
    if pool_id != strategy_pool_id(task_id, strategy_type):
        raise CandidatePoolError("pool_id does not match task_id and strategy_type")
    revision = _integer(payload["revision"], "revision", 1)
    parent_revision_id = payload["parent_revision_id"]
    if revision == 1:
        if parent_revision_id is not None:
            raise CandidatePoolError("initial pool revision cannot have a parent")
    else:
        parent_revision_id = _text(parent_revision_id, "parent_revision_id")
    default_action = _action(payload["default_action"], "default_action")
    entries_value = payload["entries"]
    if isinstance(entries_value, str | bytes | bytearray) or not isinstance(
        entries_value, Sequence
    ):
        raise CandidatePoolError("entries must be a list")
    entries = [
        _normalize_entry(item, pool_id=pool_id, expected_position=index)
        for index, item in enumerate(entries_value)
    ]
    entry_ids = [item["entry_id"] for item in entries]
    rule_ids = [item["rule_id"] for item in entries]
    asset_ids = [item["source"]["asset_id"] for item in entries]
    _assert_unique(entry_ids, "entry_id")
    _assert_unique(rule_ids, "rule_id")
    _assert_unique(asset_ids, "source asset_id")
    evidence_identities = {
        _canonical_json(item["source"]["evidence_identity"]) for item in entries
    }
    if len(evidence_identities) > 1:
        raise CandidatePoolError(
            "strategy pool entries must share one candidate evidence identity"
        )
    if payload["status"] != _STATUS:
        raise CandidatePoolError("strategy pool status must remain draft")
    if payload["validation_status"] != _VALIDATION_STATUS:
        raise CandidatePoolError(
            "strategy pool validation_status must remain unvalidated"
        )
    operation = _normalize_operation(payload["operation"])
    expected_operation_hash = strategy_pool_operation_hash(
        pool_id=pool_id,
        parent_revision_id=parent_revision_id,
        kind=operation["kind"],
        reason=operation["reason"],
        default_action=default_action,
        entries=entries,
        status=_STATUS,
        validation_status=_VALIDATION_STATUS,
    )
    if not hmac.compare_digest(operation["operation_hash"], expected_operation_hash):
        raise CandidatePoolError("operation_hash does not match pool mutation")
    revision_id = _text(payload["revision_id"], "revision_id")
    expected_revision_id = strategy_pool_revision_id(
        pool_id, parent_revision_id, expected_operation_hash
    )
    if revision_id != expected_revision_id:
        raise CandidatePoolError("revision_id does not match pool mutation")
    _assert_strategy_actions(strategy_type, default_action, entries)
    normalized_body = {
        "schema_version": POOL_SCHEMA_VERSION,
        "pool_id": pool_id,
        "task_id": task_id,
        "strategy_type": strategy_type,
        "revision": revision,
        "revision_id": revision_id,
        "parent_revision_id": parent_revision_id,
        "operation": operation,
        "default_action": default_action,
        "entries": entries,
        "status": _STATUS,
        "validation_status": _VALIDATION_STATUS,
    }
    snapshot_hash = _hash(payload["snapshot_hash"], "snapshot_hash")
    expected_snapshot_hash = _repository_snapshot_hash(normalized_body)
    if not hmac.compare_digest(snapshot_hash, expected_snapshot_hash):
        raise CandidatePoolError("snapshot_hash does not match canonical pool snapshot")
    return {**normalized_body, "snapshot_hash": snapshot_hash}


def canonical_strategy_pool_json(payload: Mapping[str, Any]) -> str:
    return _canonical_json(validate_strategy_pool(payload))


def strategy_pool_snapshot_hash(payload: Mapping[str, Any]) -> str:
    return validate_strategy_pool(payload)["snapshot_hash"]


def _next_snapshot(
    current: Mapping[str, Any],
    operation_kind: str,
    entries: list[dict[str, Any]],
    *,
    reason: str | None,
) -> dict[str, Any]:
    return _snapshot(
        pool_id=current["pool_id"],
        task_id=current["task_id"],
        strategy_type=current["strategy_type"],
        revision=int(current["revision"]) + 1,
        parent_revision_id=current["revision_id"],
        operation_kind=operation_kind,
        reason=reason,
        default_action=current["default_action"],
        entries=entries,
    )


def _snapshot(
    *,
    pool_id: str,
    task_id: str,
    strategy_type: str,
    revision: int,
    parent_revision_id: str | None,
    operation_kind: str,
    reason: str | None,
    default_action: Mapping[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_reason = _optional_text(reason, "operation reason")
    operation_hash = strategy_pool_operation_hash(
        pool_id=pool_id,
        parent_revision_id=parent_revision_id,
        kind=operation_kind,
        reason=normalized_reason,
        default_action=default_action,
        entries=entries,
        status=_STATUS,
        validation_status=_VALIDATION_STATUS,
    )
    body = {
        "schema_version": POOL_SCHEMA_VERSION,
        "pool_id": pool_id,
        "task_id": task_id,
        "strategy_type": strategy_type,
        "revision": revision,
        "revision_id": strategy_pool_revision_id(
            pool_id, parent_revision_id, operation_hash
        ),
        "parent_revision_id": parent_revision_id,
        "operation": {
            "kind": operation_kind,
            "operation_hash": operation_hash,
            "reason": normalized_reason,
        },
        "default_action": _json_object(default_action, "default_action"),
        "entries": entries,
        "status": _STATUS,
        "validation_status": _VALIDATION_STATUS,
    }
    return validate_strategy_pool(
        {**body, "snapshot_hash": _repository_snapshot_hash(body)}
    )


def _candidate_source(
    value: Mapping[str, Any], *, asset: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidatePoolError("source_binding must be an object")
    _exact_fields(value, _SOURCE_FIELDS, "source_binding")
    source = {
        "artifact_id": _text(value["artifact_id"], "source artifact_id"),
        "kind": _text(value["kind"], "source kind"),
        "content_hash": _hash(value["content_hash"], "source content_hash"),
        "asset_id": _text(value["asset_id"], "source asset_id"),
        "asset_hash": _hash(value["asset_hash"], "source asset_hash"),
        "candidate_kind": _text(value["candidate_kind"], "candidate_kind"),
        "fragment_id": _text(value["fragment_id"], "fragment_id"),
        "effect_id": _text(value["effect_id"], "effect_id"),
        "effect_stage": _text(value["effect_stage"], "effect_stage"),
        "validation_status": _text(
            value["validation_status"], "source validation_status"
        ),
        "parent_candidate_id": _text(
            value["parent_candidate_id"], "parent_candidate_id"
        ),
        "parent_evidence_hash": _hash(
            value["parent_evidence_hash"], "parent_evidence_hash"
        ),
        "evidence_identity": _evidence_identity(value["evidence_identity"]),
    }
    if source["kind"] != _CANDIDATE_ARTIFACT_KIND:
        raise CandidatePoolError(
            "source kind must be strategy_candidate_asset_json"
        )
    if source["candidate_kind"] != _CANDIDATE_KIND:
        raise CandidatePoolError("candidate_kind must be univariate_refinement")
    comparisons = {
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "fragment_id": asset["rule"]["rule_id"],
        "effect_id": asset["effect"]["effect_id"],
        "effect_stage": asset["effect_stage"],
        "validation_status": asset["validation_status"],
        "parent_candidate_id": asset["parent"]["candidate_id"],
        "parent_evidence_hash": asset["parent"]["evidence_hash"],
    }
    for field, expected in comparisons.items():
        if source[field] != expected:
            raise CandidatePoolError(
                f"source {field} does not match the candidate asset"
            )
    if (
        source["effect_stage"] != "development"
        or source["validation_status"] != "unvalidated"
    ):
        raise CandidatePoolError(
            "candidate asset must remain development and unvalidated"
        )
    return source


def _normalize_entry(
    value: object, *, pool_id: str, expected_position: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidatePoolError("pool entry must be an object")
    _exact_fields(value, _ENTRY_FIELDS, "pool entry")
    rule_id = _text(value["rule_id"], "entry rule_id")
    position = _integer(value["position"], "entry position", 0)
    if position != expected_position:
        raise CandidatePoolError("entry positions must be contiguous from zero")
    source = _normalize_source(value["source"])
    execution = _normalize_execution(value["execution"])
    action = _action(value["action"], "entry action")
    if value["enabled"] is not True:
        raise CandidatePoolError("pool entries must remain enabled in v1")
    entry_id = _text(value["entry_id"], "entry_id")
    expected_id = _stable_id(
        "pool-entry",
        {
            "pool_id": pool_id,
            "artifact_id": source["artifact_id"],
            "asset_id": source["asset_id"],
            "rule_id": rule_id,
            "fragment_id": source["fragment_id"],
        },
    )
    if entry_id != expected_id:
        raise CandidatePoolError("entry_id does not match candidate membership")
    if rule_id != source["fragment_id"]:
        raise CandidatePoolError("entry rule_id must match source fragment_id")
    return {
        "entry_id": entry_id,
        "rule_id": rule_id,
        "position": position,
        "source": source,
        "execution": execution,
        "action": action,
        "enabled": True,
    }


def _normalize_source(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidatePoolError("entry source must be an object")
    _exact_fields(value, _SOURCE_FIELDS, "entry source")
    source = {
        "artifact_id": _text(value["artifact_id"], "source artifact_id"),
        "kind": _text(value["kind"], "source kind"),
        "content_hash": _hash(value["content_hash"], "source content_hash"),
        "asset_id": _text(value["asset_id"], "source asset_id"),
        "asset_hash": _hash(value["asset_hash"], "source asset_hash"),
        "candidate_kind": _text(value["candidate_kind"], "candidate_kind"),
        "fragment_id": _text(value["fragment_id"], "fragment_id"),
        "effect_id": _text(value["effect_id"], "effect_id"),
        "effect_stage": _text(value["effect_stage"], "effect_stage"),
        "validation_status": _text(
            value["validation_status"], "source validation_status"
        ),
        "parent_candidate_id": _text(
            value["parent_candidate_id"], "parent_candidate_id"
        ),
        "parent_evidence_hash": _hash(
            value["parent_evidence_hash"], "parent_evidence_hash"
        ),
        "evidence_identity": _evidence_identity(value["evidence_identity"]),
    }
    if source["kind"] != _CANDIDATE_ARTIFACT_KIND:
        raise CandidatePoolError(
            "entry source kind must be strategy_candidate_asset_json"
        )
    if source["candidate_kind"] != _CANDIDATE_KIND:
        raise CandidatePoolError("entry candidate_kind must be univariate_refinement")
    if (
        source["effect_stage"] != "development"
        or source["validation_status"] != "unvalidated"
    ):
        raise CandidatePoolError(
            "entry source must remain development and unvalidated"
        )
    return source


def _evidence_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidatePoolError("evidence_identity must be an object")
    _exact_fields(value, _EVIDENCE_IDENTITY_FIELDS, "evidence_identity")
    return {
        "dataset_id": _text(value["dataset_id"], "evidence dataset_id"),
        "dataset_content_hash": _hash(
            value["dataset_content_hash"], "evidence dataset_content_hash"
        ),
        "workspace_revision": _integer(
            value["workspace_revision"], "evidence workspace_revision", 0
        ),
        "workspace_generation": _integer(
            value["workspace_generation"], "evidence workspace_generation", 0
        ),
        "semantic_mapping_hash": _hash(
            value["semantic_mapping_hash"], "evidence semantic_mapping_hash"
        ),
    }


def _normalize_execution(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidatePoolError("entry execution must be an object")
    _exact_fields(value, _EXECUTION_FIELDS, "entry execution")
    requirements = value["requirements"]
    if requirements != []:
        raise CandidatePoolError("candidate pool v1 requirements must be empty")
    condition = value["condition"]
    if not isinstance(condition, Mapping):
        raise CandidatePoolError("entry condition must be an object")
    return {
        "condition": canonicalize_expression(condition),
        "requirements": [],
    }


def _normalize_operation(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidatePoolError("operation must be an object")
    _exact_fields(value, _OPERATION_FIELDS, "operation")
    kind = _text(value["kind"], "operation kind")
    if kind not in _MUTATION_KINDS:
        raise CandidatePoolError(f"unsupported pool operation: {kind}")
    return {
        "kind": kind,
        "operation_hash": _hash(value["operation_hash"], "operation_hash"),
        "reason": _optional_text(value["reason"], "operation reason"),
    }


def _assert_strategy_actions(
    strategy_type: str,
    default_action: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> None:
    try:
        StrategySpec(
            strategy_type=strategy_type,
            default_action=StrategyAction.from_dict(default_action),
            rules=tuple(
                StrategyRuleSpec(
                    rule_id=item["rule_id"],
                    priority=(index + 1) * 10,
                    condition=item["execution"]["condition"],
                    action=StrategyAction.from_dict(item["action"]),
                )
                for index, item in enumerate(entries)
            ),
        )
    except StrategyError as exc:
        raise CandidatePoolError(f"pool action is incompatible: {exc}") from exc


def _action(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidatePoolError(f"{name} must be an object")
    try:
        return StrategyAction.from_dict(value).to_dict()
    except StrategyError as exc:
        raise CandidatePoolError(f"{name} is invalid: {exc}") from exc


def _with_positions(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(entries):
        item = _json_object(raw, "pool entry")
        item["position"] = index
        result.append(item)
    return result


def _assert_unique(values: Sequence[str], name: str) -> None:
    if len(set(values)) != len(values):
        raise CandidatePoolError(f"strategy pool contains duplicate {name}")


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}-{_sha256(_canonical_json(value))[:32]}"


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CandidatePoolError("strategy pool must be finite canonical JSON") from exc


def _json_object(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    try:
        raw = _canonical_json(value)
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CandidatePoolError(f"{name} must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise CandidatePoolError(f"{name} must be a JSON object")
    return parsed


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CandidatePoolError(f"{name} must be a lowercase SHA-256 hash")
    if any(character not in "0123456789abcdef" for character in value):
        raise CandidatePoolError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CandidatePoolError(f"{name} must be non-empty canonical text")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CandidatePoolError(f"{name} must be an integer >= {minimum}")
    return value


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported fields: " + ", ".join(unexpected))
        raise CandidatePoolError(f"invalid {name} ({'; '.join(details)})")


__all__ = [
    "ABSENT_POOL_REVISION",
    "ABSENT_POOL_SNAPSHOT_HASH",
    "CandidatePoolError",
    "POOL_PRODUCER_VERSION",
    "POOL_SCHEMA_VERSION",
    "SELECTED_STRATEGY_DESIGN_SCHEMA_VERSION",
    "add_candidate",
    "canonical_strategy_pool_json",
    "compile_strategy_pool",
    "remove_pool_entry",
    "reorder_strategy_pool",
    "set_pool_entry_action",
    "strategy_pool_snapshot_hash",
    "validate_strategy_pool",
]
