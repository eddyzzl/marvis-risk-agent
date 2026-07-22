"""Canonical immutable candidates for deterministic ``n_of_k`` voting rules.

The domain seam consumes one exact Strategy Pool revision and a set of enabled
Pool memberships.  It canonicalizes that set by Pool position, freezes only
the executable conditions and immutable lineage, and binds a deterministic
labelled-sample measurement.  It deliberately carries no Pool-owned action and
has no adoption, deployment, persistence, or registry authority.

The calling Tool remains responsible for loading the live Pool artifact and
the exact dataset bytes.  ``verify_voting_candidate_asset_against_pool`` then
replays all Pool-derived references against that independently loaded snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
import re
from typing import Any

from marvis.packs.strategy.dsl import (
    canonicalize_expression,
    semantic_expression_key,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import CandidatePoolError, validate_strategy_pool
from marvis.packs.strategy.sample_design_binding import StrategySampleDesignRef


VOTING_CANDIDATE_ASSET_SCHEMA_VERSION_V1 = "strategy.voting-candidate-asset.v1"
VOTING_CANDIDATE_ASSET_SCHEMA_VERSION = "strategy.voting-candidate-asset.v2"
VOTING_CANDIDATE_ASSET_TYPE = "voting_n_of_k"
VOTING_CANDIDATE_ASSET_PRODUCER_VERSION_V1 = "strategy.voting-candidate-asset/1"
VOTING_CANDIDATE_ASSET_PRODUCER_VERSION = "strategy.voting-candidate-asset/2"
VOTING_EFFECT_SCHEMA_VERSION = "strategy.voting-effect.v1"
VOTING_METRICS_SCHEMA_VERSION = "strategy.voting-metrics.v1"

_CANDIDATE_STAGE = "development"
_OBSERVATION_STAGE = "backtested"
_VALIDATION_STATUS = "unvalidated"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_RULE_ID_RE = re.compile(r"^candidate-rule-[0-9a-f]{32}$")
_FRAGMENT_ID_RE = re.compile(r"^candidate-fragment-[0-9a-f]{32}$")
_EFFECT_ID_RE = re.compile(r"^candidate-effect-[0-9a-f]{32}$")
_CANDIDATE_ID_RE = re.compile(r"^candidate-[0-9a-f]{32}$")

_TOP_LEVEL_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "asset_type",
        "lifecycle",
        "pool_ref",
        "evidence_identity",
        "measurement_context",
        "selected_entries",
        "voting",
        "rule",
        "fragment",
        "effect",
        "metrics",
        "candidate_evidence",
        "producer_version",
        "asset_id",
        "asset_hash",
    }
)
_TOP_LEVEL_FIELDS = _TOP_LEVEL_FIELDS_V1 | {"sample_design_ref"}
_BODY_FIELDS_V1 = _TOP_LEVEL_FIELDS_V1 - {"asset_id", "asset_hash"}
_BODY_FIELDS = _TOP_LEVEL_FIELDS - {"asset_id", "asset_hash"}
_LIFECYCLE_FIELDS = frozenset(
    {"candidate_stage", "observation_stage", "validation_status"}
)
_POOL_REF_FIELDS = frozenset(
    {
        "pool_id",
        "task_id",
        "strategy_type",
        "revision",
        "revision_id",
        "snapshot_hash",
    }
)
_EVIDENCE_IDENTITY_FIELDS = frozenset(
    {
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_context_hash",
    }
)
_MEASUREMENT_CONTEXT_FIELDS = frozenset({"target_col", "sample_context_hash"})
_SELECTED_ENTRY_FIELDS = frozenset(
    {
        "selection_index",
        "pool_position",
        "entry_id",
        "entry_hash",
        "rule_id",
        "rule_hash",
        "condition",
        "requirements",
        "artifact_id",
        "artifact_kind",
        "artifact_schema_version",
        "artifact_content_hash",
        "origin_tool",
        "asset_schema_version",
        "source_asset_id",
        "source_asset_hash",
        "source_asset_type",
        "source_fragment_id",
        "source_fragment_hash",
        "source_fragment_type",
        "source_effect_id",
        "source_evidence_id",
        "source_evidence_hash",
        "candidate_stage",
        "observation_stage",
        "validation_status",
        "source_hash",
    }
)
_VOTING_FIELDS = frozenset({"n", "k"})
_RULE_FIELDS = frozenset({"rule_id", "rule_hash", "condition"})
_FRAGMENT_FIELDS = frozenset(
    {
        "fragment_id",
        "fragment_hash",
        "fragment_type",
        "rule_id",
        "condition",
        "requirements",
        "effect_id",
    }
)
_EFFECT_BODY_FIELDS = frozenset(
    {
        "population_count",
        "labeled_count",
        "matched_count",
        "matched_rate",
        "matched_bad_count",
        "matched_bad_rate",
        "unmatched_count",
        "unmatched_bad_count",
        "unmatched_bad_rate",
        "bad_capture_rate",
        "lift",
    }
)
_EFFECT_FIELDS = _EFFECT_BODY_FIELDS | frozenset(
    {"schema_version", "effect_id", "effect_hash"}
)
_METRICS_BODY_FIELDS = frozenset(
    {
        "population_count",
        "labeled_count",
        "matched_count",
        "matched_rate",
        "matched_bad_count",
        "matched_bad_rate",
        "unmatched_count",
        "unmatched_bad_count",
        "unmatched_bad_rate",
        "bad_capture_rate",
        "lift",
    }
)
_METRICS_FIELDS = _METRICS_BODY_FIELDS | frozenset(
    {"schema_version", "metrics_hash"}
)
_CANDIDATE_EVIDENCE_FIELDS_V1 = frozenset({"candidate_id", "evidence_hash"})
_CANDIDATE_EVIDENCE_FIELDS = _CANDIDATE_EVIDENCE_FIELDS_V1 | {
    "sample_design_ref"
}


class VotingCandidateAssetError(StrategyError):
    """A Voting candidate violated its exact immutable domain contract."""


def build_voting_candidate_asset(
    pool: Mapping[str, Any],
    *,
    selected_entry_ids: Sequence[str],
    n: int,
    target_col: str,
    sample_design_ref: Mapping[str, Any],
    effect: Mapping[str, Any],
    producer_version: str = VOTING_CANDIDATE_ASSET_PRODUCER_VERSION,
) -> dict[str, Any]:
    """Build one canonical Voting candidate from an exact Pool snapshot.

    ``selected_entry_ids`` has set semantics.  Duplicate ids are rejected and
    the accepted memberships are always ordered by their current Pool position.
    ``effect`` is the strict deterministic measurement body for the exact
    labelled sample named by the Pool evidence identity and ``target_col``.
    """

    return _build_voting_candidate_asset(
        pool,
        selected_entry_ids=selected_entry_ids,
        n=n,
        target_col=target_col,
        sample_design_ref=sample_design_ref,
        effect=effect,
        producer_version=producer_version,
        schema_version=VOTING_CANDIDATE_ASSET_SCHEMA_VERSION,
    )


def _build_voting_candidate_asset(
    pool: Mapping[str, Any],
    *,
    selected_entry_ids: Sequence[str],
    n: int,
    target_col: str,
    sample_design_ref: Mapping[str, Any] | None,
    effect: Mapping[str, Any],
    producer_version: str,
    schema_version: str,
) -> dict[str, Any]:
    current = _validated_pool(pool)
    if schema_version == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION:
        normalized_sample_design_ref = _sample_design_ref(sample_design_ref)
    elif schema_version == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION_V1:
        if sample_design_ref is not None:
            raise VotingCandidateAssetError(
                "legacy Voting candidate cannot carry sample_design_ref"
            )
        normalized_sample_design_ref = None
    else:
        raise VotingCandidateAssetError("Voting candidate schema_version is invalid")
    selected_ids = _selected_entry_id_set(selected_entry_ids)
    by_id = {entry["entry_id"]: entry for entry in current["entries"]}
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise VotingCandidateAssetError(
            "selected_entry_ids contains unknown Pool entries: " + ", ".join(unknown)
        )
    selected_pool_entries = [by_id[entry_id] for entry_id in selected_ids]
    if any(entry.get("enabled") is not True for entry in selected_pool_entries):
        raise VotingCandidateAssetError("selected Pool entries must be enabled")
    selected_pool_entries.sort(key=lambda entry: int(entry["position"]))

    required = _voting_n(n, len(selected_pool_entries))
    pool_ref = _pool_reference(current)
    evidence_identities = [
        _evidence_identity(entry["source"]["evidence_identity"])
        for entry in selected_pool_entries
    ]
    evidence_identity = evidence_identities[0]
    if any(identity != evidence_identity for identity in evidence_identities[1:]):
        raise VotingCandidateAssetError(
            "selected Pool entries must share one exact evidence identity"
        )
    measurement_context = _measurement_context(
        {
            "target_col": target_col,
            "sample_context_hash": evidence_identity["sample_context_hash"],
        },
        evidence_identity=evidence_identity,
    )
    selected_entries = [
        _selected_entry_reference(entry, selection_index=index)
        for index, entry in enumerate(selected_pool_entries)
    ]
    condition = _voting_condition(selected_entries, n=required)
    rule = _derive_rule(
        pool_ref=pool_ref,
        selected_entries=selected_entries,
        condition=condition,
    )
    effect_body = _normalize_effect_body(effect)
    metrics = _derive_metrics(effect_body)
    normalized_effect = _derive_effect(
        effect_body,
        rule=rule,
        measurement_context=measurement_context,
        metrics=metrics,
    )
    fragment = _derive_fragment(
        pool_ref=pool_ref,
        selected_entries=selected_entries,
        rule=rule,
        effect=normalized_effect,
    )
    candidate_evidence = _derive_candidate_evidence(
        schema_version=schema_version,
        sample_design_ref=normalized_sample_design_ref,
        pool_ref=pool_ref,
        evidence_identity=evidence_identity,
        measurement_context=measurement_context,
        selected_entries=selected_entries,
        rule=rule,
        fragment=fragment,
        effect=normalized_effect,
        metrics=metrics,
    )
    body = {
        "schema_version": schema_version,
        "asset_type": VOTING_CANDIDATE_ASSET_TYPE,
        "lifecycle": _fixed_lifecycle(),
        "pool_ref": pool_ref,
        "evidence_identity": evidence_identity,
        "measurement_context": measurement_context,
        "selected_entries": selected_entries,
        "voting": {"n": required, "k": len(selected_entries)},
        "rule": rule,
        "fragment": fragment,
        "effect": normalized_effect,
        "metrics": metrics,
        "candidate_evidence": candidate_evidence,
        "producer_version": _producer_version(
            producer_version,
            schema_version=schema_version,
        ),
    }
    if normalized_sample_design_ref is not None:
        body["sample_design_ref"] = normalized_sample_design_ref
    normalized_body = _normalize_body(body)
    asset_id = _stable_id("candidate-asset", normalized_body)
    without_hash = {**normalized_body, "asset_id": asset_id}
    asset_hash = _sha256(_canonical_json(without_hash))
    return validate_voting_candidate_asset({**without_hash, "asset_hash": asset_hash})


def validate_voting_candidate_asset(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every field, derivation, identity, and integrity hash."""

    if not isinstance(payload, Mapping):
        raise VotingCandidateAssetError("Voting candidate asset must be an object")
    schema_version = _schema_version(payload.get("schema_version"))
    expected_fields = (
        _TOP_LEVEL_FIELDS
        if schema_version == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION
        else _TOP_LEVEL_FIELDS_V1
    )
    _exact_fields(payload, expected_fields, "Voting candidate asset")
    asset_id = _text(payload["asset_id"], "asset_id")
    if _ASSET_ID_RE.fullmatch(asset_id) is None:
        raise VotingCandidateAssetError("asset_id has an invalid format")
    asset_hash = _hash(payload["asset_hash"], "asset_hash")
    body = _normalize_body(
        {
            key: payload[key]
            for key in payload
            if key not in {"asset_id", "asset_hash"}
        }
    )
    expected_id = _stable_id("candidate-asset", body)
    if not hmac.compare_digest(asset_id, expected_id):
        raise VotingCandidateAssetError(
            "asset_id does not match canonical Voting candidate identity"
        )
    without_hash = {**body, "asset_id": asset_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(asset_hash, expected_hash):
        raise VotingCandidateAssetError(
            "asset_hash does not match canonical Voting candidate"
        )
    return {**without_hash, "asset_hash": asset_hash}


def canonical_voting_candidate_asset_json(payload: Mapping[str, Any]) -> str:
    """Return the sole canonical JSON serialization of a verified asset."""

    return _canonical_json(validate_voting_candidate_asset(payload))


def parse_voting_candidate_asset_json(raw: str | bytes | bytearray) -> dict[str, Any]:
    """Parse JSON with duplicate-key rejection through the strict validator."""

    if not isinstance(raw, str | bytes | bytearray):
        raise VotingCandidateAssetError("Voting candidate JSON must be text or bytes")
    try:
        payload = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except VotingCandidateAssetError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise VotingCandidateAssetError(
            f"Voting candidate is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise VotingCandidateAssetError(
            "Voting candidate JSON must contain an object"
        )
    return validate_voting_candidate_asset(payload)


def verify_voting_candidate_asset_against_pool(
    payload: Mapping[str, Any],
    pool: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay one valid asset against one independently loaded exact Pool."""

    asset = validate_voting_candidate_asset(payload)
    current = _validated_pool(pool)
    if asset["pool_ref"] != _pool_reference(current):
        raise VotingCandidateAssetError(
            "Voting candidate does not bind the exact pool revision_id/snapshot_hash"
        )
    selected_ids = [entry["entry_id"] for entry in asset["selected_entries"]]
    effect_body = {
        field: asset["effect"][field] for field in _EFFECT_BODY_FIELDS
    }
    if asset["schema_version"] == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION:
        rebuilt = build_voting_candidate_asset(
            current,
            selected_entry_ids=selected_ids,
            n=asset["voting"]["n"],
            target_col=asset["measurement_context"]["target_col"],
            sample_design_ref=asset["sample_design_ref"],
            effect=effect_body,
            producer_version=asset["producer_version"],
        )
    else:
        rebuilt = _build_voting_candidate_asset(
            current,
            selected_entry_ids=selected_ids,
            n=asset["voting"]["n"],
            target_col=asset["measurement_context"]["target_col"],
            sample_design_ref=None,
            effect=effect_body,
            producer_version=asset["producer_version"],
            schema_version=VOTING_CANDIDATE_ASSET_SCHEMA_VERSION_V1,
        )
    if rebuilt != asset:
        raise VotingCandidateAssetError(
            "Voting candidate entry/source/fragment or evidence lineage changed"
        )
    return asset


def _normalize_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = _schema_version(payload.get("schema_version"))
    expected_fields = (
        _BODY_FIELDS
        if schema_version == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION
        else _BODY_FIELDS_V1
    )
    _exact_fields(payload, expected_fields, "Voting candidate body")
    if payload["asset_type"] != VOTING_CANDIDATE_ASSET_TYPE:
        raise VotingCandidateAssetError(
            "asset_type must be " + VOTING_CANDIDATE_ASSET_TYPE
        )
    lifecycle = _lifecycle(payload["lifecycle"])
    pool_ref = _normalize_pool_reference(payload["pool_ref"])
    evidence_identity = _evidence_identity(payload["evidence_identity"])
    measurement_context = _measurement_context(
        payload["measurement_context"],
        evidence_identity=evidence_identity,
    )
    selected_entries = _normalize_selected_entries(payload["selected_entries"])
    voting = _normalize_voting(payload["voting"], k=len(selected_entries))
    expected_condition = _voting_condition(selected_entries, n=voting["n"])
    rule = _normalize_rule(
        payload["rule"],
        pool_ref=pool_ref,
        selected_entries=selected_entries,
        expected_condition=expected_condition,
    )
    metrics = _normalize_metrics(payload["metrics"])
    effect = _normalize_effect(
        payload["effect"],
        rule=rule,
        measurement_context=measurement_context,
        metrics=metrics,
    )
    fragment = _normalize_fragment(
        payload["fragment"],
        pool_ref=pool_ref,
        selected_entries=selected_entries,
        rule=rule,
        effect=effect,
    )
    sample_design_ref = (
        _sample_design_ref(payload["sample_design_ref"])
        if schema_version == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION
        else None
    )
    candidate_evidence = _normalize_candidate_evidence(
        payload["candidate_evidence"],
        schema_version=schema_version,
        sample_design_ref=sample_design_ref,
        pool_ref=pool_ref,
        evidence_identity=evidence_identity,
        measurement_context=measurement_context,
        selected_entries=selected_entries,
        rule=rule,
        fragment=fragment,
        effect=effect,
        metrics=metrics,
    )
    body = {
        "schema_version": schema_version,
        "asset_type": VOTING_CANDIDATE_ASSET_TYPE,
        "lifecycle": lifecycle,
        "pool_ref": pool_ref,
        "evidence_identity": evidence_identity,
        "measurement_context": measurement_context,
        "selected_entries": selected_entries,
        "voting": voting,
        "rule": rule,
        "fragment": fragment,
        "effect": effect,
        "metrics": metrics,
        "candidate_evidence": candidate_evidence,
        "producer_version": _producer_version(
            payload["producer_version"],
            schema_version=schema_version,
        ),
    }
    if sample_design_ref is not None:
        body["sample_design_ref"] = sample_design_ref
    return body


def _validated_pool(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError("strategy pool must be an object")
    try:
        return validate_strategy_pool(value)
    except CandidatePoolError as exc:
        raise VotingCandidateAssetError(
            f"strategy pool failed strict validation: {exc}"
        ) from exc


def _selected_entry_id_set(value: object) -> list[str]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise VotingCandidateAssetError("selected_entry_ids must be a list")
    selected = [_text(item, "selected_entry_ids item") for item in value]
    if not 2 <= len(selected) <= 50:
        raise VotingCandidateAssetError(
            "selected_entry_ids must contain between 2 and 50 entries"
        )
    if len(set(selected)) != len(selected):
        raise VotingCandidateAssetError(
            "selected_entry_ids must not contain duplicate entries"
        )
    return selected


def _pool_reference(pool: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize_pool_reference(
        {
            "pool_id": pool["pool_id"],
            "task_id": pool["task_id"],
            "strategy_type": pool["strategy_type"],
            "revision": pool["revision"],
            "revision_id": pool["revision_id"],
            "snapshot_hash": pool["snapshot_hash"],
        }
    )


def _normalize_pool_reference(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError("pool_ref must be an object")
    _exact_fields(value, _POOL_REF_FIELDS, "pool_ref")
    return {
        "pool_id": _text(value["pool_id"], "pool_ref.pool_id"),
        "task_id": _text(value["task_id"], "pool_ref.task_id"),
        "strategy_type": _text(
            value["strategy_type"], "pool_ref.strategy_type"
        ),
        "revision": _positive_int(value["revision"], "pool_ref.revision"),
        "revision_id": _text(value["revision_id"], "pool_ref.revision_id"),
        "snapshot_hash": _hash(
            value["snapshot_hash"], "pool_ref.snapshot_hash"
        ),
    }


def _evidence_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError("evidence_identity must be an object")
    _exact_fields(value, _EVIDENCE_IDENTITY_FIELDS, "evidence_identity")
    return {
        "dataset_id": _text(value["dataset_id"], "evidence_identity.dataset_id"),
        "dataset_content_hash": _hash(
            value["dataset_content_hash"],
            "evidence_identity.dataset_content_hash",
        ),
        "workspace_revision": _non_negative_int(
            value["workspace_revision"],
            "evidence_identity.workspace_revision",
        ),
        "workspace_generation": _non_negative_int(
            value["workspace_generation"],
            "evidence_identity.workspace_generation",
        ),
        "semantic_mapping_hash": _hash(
            value["semantic_mapping_hash"],
            "evidence_identity.semantic_mapping_hash",
        ),
        "sample_context_hash": _hash(
            value["sample_context_hash"],
            "evidence_identity.sample_context_hash",
        ),
    }


def _measurement_context(
    value: object,
    *,
    evidence_identity: Mapping[str, Any],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError("measurement_context must be an object")
    _exact_fields(value, _MEASUREMENT_CONTEXT_FIELDS, "measurement_context")
    sample_hash = _hash(
        value["sample_context_hash"],
        "measurement_context.sample_context_hash",
    )
    if not hmac.compare_digest(
        sample_hash, str(evidence_identity["sample_context_hash"])
    ):
        raise VotingCandidateAssetError(
            "measurement_context sample_context_hash does not match evidence identity"
        )
    return {
        "target_col": _text(value["target_col"], "measurement_context.target_col"),
        "sample_context_hash": sample_hash,
    }


def _selected_entry_reference(
    entry: Mapping[str, Any],
    *,
    selection_index: int,
) -> dict[str, Any]:
    if entry.get("enabled") is not True:
        raise VotingCandidateAssetError("selected Pool entries must be enabled")
    source = entry["source"]
    execution = entry["execution"]
    condition = _canonical_condition(execution["condition"], "entry condition")
    requirements = _json_array(execution["requirements"], "entry requirements")
    rule_hash = _sha256(
        _canonical_json(
            {
                "rule_id": entry["rule_id"],
                "condition": condition,
                "requirements": requirements,
            }
        )
    )
    source_content = {
        "artifact_id": source["artifact_id"],
        "artifact_kind": source["artifact_kind"],
        "artifact_schema_version": source["artifact_schema_version"],
        "artifact_content_hash": source["artifact_content_hash"],
        "origin_tool": source["origin_tool"],
        "asset_schema_version": source["asset_schema_version"],
        "source_asset_id": source["asset_id"],
        "source_asset_hash": source["asset_hash"],
        "source_asset_type": source["asset_type"],
        "source_fragment_id": source["fragment_id"],
        "source_fragment_hash": source["fragment_hash"],
        "source_fragment_type": source["fragment_type"],
        "source_effect_id": source["effect_id"],
        "source_evidence_id": source["evidence_id"],
        "source_evidence_hash": source["evidence_hash"],
        "candidate_stage": source["candidate_stage"],
        "observation_stage": source["observation_stage"],
        "validation_status": source["validation_status"],
    }
    source_hash = _sha256(_canonical_json(source_content))
    entry_content = {
        "pool_position": entry["position"],
        "entry_id": entry["entry_id"],
        "rule_id": entry["rule_id"],
        "rule_hash": rule_hash,
        "condition": condition,
        "requirements": requirements,
        **source_content,
        "source_hash": source_hash,
    }
    return _normalize_selected_entry(
        {
            "selection_index": selection_index,
            **entry_content,
            "entry_hash": _sha256(_canonical_json(entry_content)),
        }
    )


def _normalize_selected_entries(value: object) -> list[dict[str, Any]]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise VotingCandidateAssetError("selected_entries must be a list")
    if not 2 <= len(value) <= 50:
        raise VotingCandidateAssetError(
            "selected_entries must contain between 2 and 50 entries"
        )
    entries = [_normalize_selected_entry(item) for item in value]
    entry_ids = [entry["entry_id"] for entry in entries]
    if len(set(entry_ids)) != len(entry_ids):
        raise VotingCandidateAssetError("selected_entries contains duplicate entry_id")
    rule_ids = [entry["rule_id"] for entry in entries]
    if len(set(rule_ids)) != len(rule_ids):
        raise VotingCandidateAssetError("selected_entries contains duplicate rule_id")
    condition_keys = [semantic_expression_key(entry["condition"]) for entry in entries]
    if len(set(condition_keys)) != len(condition_keys):
        raise VotingCandidateAssetError(
            "selected_entries must represent distinct executable conditions"
        )
    for index, entry in enumerate(entries):
        if entry["selection_index"] != index:
            raise VotingCandidateAssetError(
                "selected_entries selection_index must be contiguous from zero"
            )
    positions = [entry["pool_position"] for entry in entries]
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise VotingCandidateAssetError(
            "selected_entries must use unique ascending Pool positions"
        )
    return entries


def _normalize_selected_entry(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError("selected entry reference must be an object")
    _exact_fields(value, _SELECTED_ENTRY_FIELDS, "selected entry reference")
    condition = _canonical_condition(value["condition"], "selected entry condition")
    requirements = _json_array(
        value["requirements"], "selected entry requirements"
    )
    normalized = {
        "selection_index": _non_negative_int(
            value["selection_index"], "selected entry selection_index"
        ),
        "pool_position": _non_negative_int(
            value["pool_position"], "selected entry pool_position"
        ),
        "entry_id": _text(value["entry_id"], "selected entry entry_id"),
        "entry_hash": _hash(value["entry_hash"], "selected entry entry_hash"),
        "rule_id": _text(value["rule_id"], "selected entry rule_id"),
        "rule_hash": _hash(value["rule_hash"], "selected entry rule_hash"),
        "condition": condition,
        "requirements": requirements,
        "artifact_id": _text(value["artifact_id"], "selected entry artifact_id"),
        "artifact_kind": _text(
            value["artifact_kind"], "selected entry artifact_kind"
        ),
        "artifact_schema_version": _text(
            value["artifact_schema_version"],
            "selected entry artifact_schema_version",
        ),
        "artifact_content_hash": _hash(
            value["artifact_content_hash"],
            "selected entry artifact_content_hash",
        ),
        "origin_tool": _text(value["origin_tool"], "selected entry origin_tool"),
        "asset_schema_version": _text(
            value["asset_schema_version"],
            "selected entry asset_schema_version",
        ),
        "source_asset_id": _text(
            value["source_asset_id"], "selected entry source_asset_id"
        ),
        "source_asset_hash": _hash(
            value["source_asset_hash"], "selected entry source_asset_hash"
        ),
        "source_asset_type": _text(
            value["source_asset_type"], "selected entry source_asset_type"
        ),
        "source_fragment_id": _text(
            value["source_fragment_id"], "selected entry source_fragment_id"
        ),
        "source_fragment_hash": _hash(
            value["source_fragment_hash"], "selected entry source_fragment_hash"
        ),
        "source_fragment_type": _text(
            value["source_fragment_type"], "selected entry source_fragment_type"
        ),
        "source_effect_id": _text(
            value["source_effect_id"], "selected entry source_effect_id"
        ),
        "source_evidence_id": _text(
            value["source_evidence_id"], "selected entry source_evidence_id"
        ),
        "source_evidence_hash": _hash(
            value["source_evidence_hash"], "selected entry source_evidence_hash"
        ),
        "candidate_stage": _text(
            value["candidate_stage"], "selected entry candidate_stage"
        ),
        "observation_stage": _text(
            value["observation_stage"], "selected entry observation_stage"
        ),
        "validation_status": _text(
            value["validation_status"], "selected entry validation_status"
        ),
        "source_hash": _hash(value["source_hash"], "selected entry source_hash"),
    }
    if (
        normalized["candidate_stage"] != _CANDIDATE_STAGE
        or normalized["observation_stage"] != _OBSERVATION_STAGE
        or normalized["validation_status"] != _VALIDATION_STATUS
    ):
        raise VotingCandidateAssetError(
            "selected entry source must remain development/backtested/unvalidated"
        )
    expected_rule_hash = _sha256(
        _canonical_json(
            {
                "rule_id": normalized["rule_id"],
                "condition": condition,
                "requirements": requirements,
            }
        )
    )
    if not hmac.compare_digest(normalized["rule_hash"], expected_rule_hash):
        raise VotingCandidateAssetError(
            "selected entry rule_hash does not match rule content"
        )
    source_content = {
        field: normalized[field]
        for field in (
            "artifact_id",
            "artifact_kind",
            "artifact_schema_version",
            "artifact_content_hash",
            "origin_tool",
            "asset_schema_version",
            "source_asset_id",
            "source_asset_hash",
            "source_asset_type",
            "source_fragment_id",
            "source_fragment_hash",
            "source_fragment_type",
            "source_effect_id",
            "source_evidence_id",
            "source_evidence_hash",
            "candidate_stage",
            "observation_stage",
            "validation_status",
        )
    }
    expected_source_hash = _sha256(_canonical_json(source_content))
    if not hmac.compare_digest(normalized["source_hash"], expected_source_hash):
        raise VotingCandidateAssetError(
            "selected entry source_hash does not match source content"
        )
    entry_content = {
        field: normalized[field]
        for field in normalized
        if field not in {"selection_index", "entry_hash"}
    }
    expected_entry_hash = _sha256(_canonical_json(entry_content))
    if not hmac.compare_digest(normalized["entry_hash"], expected_entry_hash):
        raise VotingCandidateAssetError(
            "selected entry entry_hash does not match immutable entry content"
        )
    return normalized


def _voting_n(value: object, k: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise VotingCandidateAssetError(
            f"n must be an integer between 1 and {k}"
        )
    normalized = int(value)
    if normalized < 1 or normalized > k:
        raise VotingCandidateAssetError(
            f"n must be an integer between 1 and {k}"
        )
    return normalized


def _normalize_voting(value: object, *, k: int) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError("voting must be an object")
    _exact_fields(value, _VOTING_FIELDS, "voting")
    supplied_k = _positive_int(value["k"], "voting.k")
    if supplied_k != k:
        raise VotingCandidateAssetError("voting.k must equal selected entry count")
    return {"n": _voting_n(value["n"], k), "k": k}


def _voting_condition(
    selected_entries: Sequence[Mapping[str, Any]],
    *,
    n: int,
) -> dict[str, Any]:
    try:
        return canonicalize_expression(
            {
                "op": "n_of_k",
                "n": n,
                "args": [entry["condition"] for entry in selected_entries],
            }
        )
    except StrategyError as exc:
        raise VotingCandidateAssetError(
            f"Voting condition failed canonical Strategy DSL validation: {exc}"
        ) from exc


def _derive_rule(
    *,
    pool_ref: Mapping[str, Any],
    selected_entries: Sequence[Mapping[str, Any]],
    condition: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": "strategy.voting-rule.v1",
        "pool_ref": pool_ref,
        "selected_entry_hashes": [entry["entry_hash"] for entry in selected_entries],
        "condition": condition,
    }
    rule_id = _stable_id("candidate-rule", identity)
    without_hash = {"rule_id": rule_id, "condition": _json_object(condition, "rule")}
    return {
        **without_hash,
        "rule_hash": _sha256(_canonical_json(without_hash)),
    }


def _normalize_rule(
    value: object,
    *,
    pool_ref: Mapping[str, Any],
    selected_entries: Sequence[Mapping[str, Any]],
    expected_condition: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError("rule must be an object")
    _exact_fields(value, _RULE_FIELDS, "rule")
    condition = _canonical_condition(value["condition"], "rule.condition")
    if condition != expected_condition:
        raise VotingCandidateAssetError(
            "rule.condition must be the canonical Pool-ordered n_of_k condition"
        )
    expected = _derive_rule(
        pool_ref=pool_ref,
        selected_entries=selected_entries,
        condition=condition,
    )
    actual_id = _text(value["rule_id"], "rule.rule_id")
    if _RULE_ID_RE.fullmatch(actual_id) is None or not hmac.compare_digest(
        actual_id, expected["rule_id"]
    ):
        raise VotingCandidateAssetError("rule_id does not match canonical Voting rule")
    actual_hash = _hash(value["rule_hash"], "rule.rule_hash")
    if not hmac.compare_digest(actual_hash, expected["rule_hash"]):
        raise VotingCandidateAssetError("rule_hash does not match canonical Voting rule")
    return expected


def _normalize_effect_body(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError("effect must be an object")
    _exact_fields(value, _EFFECT_BODY_FIELDS, "effect measurement")
    population_count = _positive_int(
        value["population_count"], "effect.population_count"
    )
    labeled_count = _bounded_count(
        value["labeled_count"],
        "effect.labeled_count",
        upper=population_count,
    )
    matched_count = _bounded_count(
        value["matched_count"],
        "effect.matched_count",
        upper=labeled_count,
    )
    unmatched_count = _bounded_count(
        value["unmatched_count"],
        "effect.unmatched_count",
        upper=labeled_count,
    )
    if unmatched_count != labeled_count - matched_count:
        raise VotingCandidateAssetError(
            "effect.unmatched_count must equal labeled_count minus matched_count"
        )
    matched_bad_count = _bounded_count(
        value["matched_bad_count"],
        "effect.matched_bad_count",
        upper=matched_count,
    )
    unmatched_bad_count = _bounded_count(
        value["unmatched_bad_count"],
        "effect.unmatched_bad_count",
        upper=unmatched_count,
    )
    matched_rate = _derived_rate(
        value["matched_rate"],
        numerator=matched_count,
        denominator=labeled_count,
        name="effect.matched_rate",
    )
    matched_bad_rate = _derived_rate(
        value["matched_bad_rate"],
        numerator=matched_bad_count,
        denominator=matched_count,
        name="effect.matched_bad_rate",
    )
    unmatched_bad_rate = _derived_rate(
        value["unmatched_bad_rate"],
        numerator=unmatched_bad_count,
        denominator=unmatched_count,
        name="effect.unmatched_bad_rate",
    )
    total_bad = matched_bad_count + unmatched_bad_count
    bad_capture_rate = _derived_rate(
        value["bad_capture_rate"],
        numerator=matched_bad_count,
        denominator=total_bad,
        name="effect.bad_capture_rate",
    )
    overall_bad_rate = None if labeled_count == 0 else total_bad / labeled_count
    expected_lift = (
        None
        if matched_bad_rate is None or overall_bad_rate in {None, 0.0}
        else matched_bad_rate / overall_bad_rate
    )
    lift = _derived_nullable_number(value["lift"], expected_lift, "effect.lift")
    return {
        "population_count": population_count,
        "labeled_count": labeled_count,
        "matched_count": matched_count,
        "matched_rate": matched_rate,
        "matched_bad_count": matched_bad_count,
        "matched_bad_rate": matched_bad_rate,
        "unmatched_count": unmatched_count,
        "unmatched_bad_count": unmatched_bad_count,
        "unmatched_bad_rate": unmatched_bad_rate,
        "bad_capture_rate": bad_capture_rate,
        "lift": lift,
    }


def _derive_metrics(effect_body: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": VOTING_METRICS_SCHEMA_VERSION,
        **{field: effect_body[field] for field in _METRICS_BODY_FIELDS},
    }
    return {**body, "metrics_hash": _sha256(_canonical_json(body))}


def _normalize_metrics(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError("metrics must be an object")
    _exact_fields(value, _METRICS_FIELDS, "metrics")
    if value["schema_version"] != VOTING_METRICS_SCHEMA_VERSION:
        raise VotingCandidateAssetError(
            "metrics.schema_version must be " + VOTING_METRICS_SCHEMA_VERSION
        )
    body = {
        "population_count": _positive_int(
            value["population_count"], "metrics.population_count"
        ),
        "labeled_count": _non_negative_int(
            value["labeled_count"], "metrics.labeled_count"
        ),
        "matched_count": _non_negative_int(
            value["matched_count"], "metrics.matched_count"
        ),
        "matched_rate": _nullable_rate_value(
            value["matched_rate"], "metrics.matched_rate"
        ),
        "matched_bad_count": _non_negative_int(
            value["matched_bad_count"], "metrics.matched_bad_count"
        ),
        "matched_bad_rate": _nullable_rate_value(
            value["matched_bad_rate"], "metrics.matched_bad_rate"
        ),
        "unmatched_count": _non_negative_int(
            value["unmatched_count"], "metrics.unmatched_count"
        ),
        "unmatched_bad_count": _non_negative_int(
            value["unmatched_bad_count"], "metrics.unmatched_bad_count"
        ),
        "unmatched_bad_rate": _nullable_rate_value(
            value["unmatched_bad_rate"], "metrics.unmatched_bad_rate"
        ),
        "bad_capture_rate": _nullable_rate_value(
            value["bad_capture_rate"], "metrics.bad_capture_rate"
        ),
        "lift": _nullable_non_negative_number(value["lift"], "metrics.lift"),
    }
    expected = _derive_metrics(body)
    supplied_hash = _hash(value["metrics_hash"], "metrics.metrics_hash")
    if not hmac.compare_digest(supplied_hash, expected["metrics_hash"]):
        raise VotingCandidateAssetError(
            "metrics_hash does not match deterministic Voting metrics"
        )
    return expected


def _derive_effect(
    effect_body: Mapping[str, Any],
    *,
    rule: Mapping[str, Any],
    measurement_context: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": VOTING_EFFECT_SCHEMA_VERSION,
        "rule_id": rule["rule_id"],
        "rule_hash": rule["rule_hash"],
        "measurement_context": measurement_context,
        "metrics_hash": metrics["metrics_hash"],
        **effect_body,
    }
    effect_id = _stable_id("candidate-effect", identity)
    without_hash = {
        "schema_version": VOTING_EFFECT_SCHEMA_VERSION,
        "effect_id": effect_id,
        **effect_body,
    }
    effect_hash = _sha256(
        _canonical_json(
            {
                **without_hash,
                "rule_hash": rule["rule_hash"],
                "measurement_context": measurement_context,
                "metrics_hash": metrics["metrics_hash"],
            }
        )
    )
    return {**without_hash, "effect_hash": effect_hash}


def _normalize_effect(
    value: object,
    *,
    rule: Mapping[str, Any],
    measurement_context: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError("effect must be an object")
    _exact_fields(value, _EFFECT_FIELDS, "effect")
    if value["schema_version"] != VOTING_EFFECT_SCHEMA_VERSION:
        raise VotingCandidateAssetError(
            "effect.schema_version must be " + VOTING_EFFECT_SCHEMA_VERSION
        )
    body = _normalize_effect_body(
        {field: value[field] for field in _EFFECT_BODY_FIELDS}
    )
    if _derive_metrics(body) != metrics:
        raise VotingCandidateAssetError(
            "metrics do not match deterministic effect measurement"
        )
    expected = _derive_effect(
        body,
        rule=rule,
        measurement_context=measurement_context,
        metrics=metrics,
    )
    effect_id = _text(value["effect_id"], "effect.effect_id")
    if _EFFECT_ID_RE.fullmatch(effect_id) is None or not hmac.compare_digest(
        effect_id, expected["effect_id"]
    ):
        raise VotingCandidateAssetError(
            "effect_id does not match canonical Voting effect"
        )
    effect_hash = _hash(value["effect_hash"], "effect.effect_hash")
    if not hmac.compare_digest(effect_hash, expected["effect_hash"]):
        raise VotingCandidateAssetError(
            "effect_hash does not match deterministic Voting effect"
        )
    return expected


def _fragment_requirements(
    selected_entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "entry_id": entry["entry_id"],
            "rule_id": entry["rule_id"],
            "fragment_id": entry["source_fragment_id"],
            "requirement": _json_value(requirement, "fragment requirement"),
        }
        for entry in selected_entries
        for requirement in entry["requirements"]
    ]


def _derive_fragment(
    *,
    pool_ref: Mapping[str, Any],
    selected_entries: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    effect: Mapping[str, Any],
) -> dict[str, Any]:
    requirements = _fragment_requirements(selected_entries)
    identity = {
        "schema_version": "strategy.voting-fragment.v1",
        "pool_ref": pool_ref,
        "rule_id": rule["rule_id"],
        "rule_hash": rule["rule_hash"],
        "effect_id": effect["effect_id"],
        "effect_hash": effect["effect_hash"],
        "requirements": requirements,
    }
    fragment_id = _stable_id("candidate-fragment", identity)
    without_hash = {
        "fragment_id": fragment_id,
        "fragment_type": "strategy_rule",
        "rule_id": rule["rule_id"],
        "condition": _json_object(rule["condition"], "fragment condition"),
        "requirements": requirements,
        "effect_id": effect["effect_id"],
    }
    return {
        **without_hash,
        "fragment_hash": _sha256(_canonical_json(without_hash)),
    }


def _normalize_fragment(
    value: object,
    *,
    pool_ref: Mapping[str, Any],
    selected_entries: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    effect: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError("fragment must be an object")
    _exact_fields(value, _FRAGMENT_FIELDS, "fragment")
    expected = _derive_fragment(
        pool_ref=pool_ref,
        selected_entries=selected_entries,
        rule=rule,
        effect=effect,
    )
    fragment_id = _text(value["fragment_id"], "fragment.fragment_id")
    if _FRAGMENT_ID_RE.fullmatch(fragment_id) is None or not hmac.compare_digest(
        fragment_id, expected["fragment_id"]
    ):
        raise VotingCandidateAssetError(
            "fragment_id does not match canonical Voting fragment"
        )
    fragment_hash = _hash(value["fragment_hash"], "fragment.fragment_hash")
    if not hmac.compare_digest(fragment_hash, expected["fragment_hash"]):
        raise VotingCandidateAssetError(
            "fragment_hash does not match canonical Voting fragment"
        )
    if _canonical_condition(value["condition"], "fragment.condition") != rule[
        "condition"
    ]:
        raise VotingCandidateAssetError("fragment condition does not match rule")
    if _json_array(value["requirements"], "fragment requirements") != expected[
        "requirements"
    ]:
        raise VotingCandidateAssetError(
            "fragment requirements do not match selected entry lineage"
        )
    for field in ("fragment_type", "rule_id", "effect_id"):
        if value[field] != expected[field]:
            raise VotingCandidateAssetError(f"fragment {field} does not match")
    return expected


def _candidate_evidence_content(
    *,
    schema_version: str,
    sample_design_ref: Mapping[str, Any] | None,
    pool_ref: Mapping[str, Any],
    evidence_identity: Mapping[str, Any],
    measurement_context: Mapping[str, Any],
    selected_entries: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    fragment: Mapping[str, Any],
    effect: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    content = {
        "schema_version": (
            "strategy.voting-candidate-evidence.v2"
            if schema_version == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION
            else "strategy.voting-candidate-evidence.v1"
        ),
        "candidate_type": VOTING_CANDIDATE_ASSET_TYPE,
        "pool_ref": pool_ref,
        "evidence_identity": evidence_identity,
        "measurement_context": measurement_context,
        "selected_entry_refs": [
            {
                "entry_id": entry["entry_id"],
                "entry_hash": entry["entry_hash"],
                "rule_id": entry["rule_id"],
                "rule_hash": entry["rule_hash"],
                "source_fragment_id": entry["source_fragment_id"],
                "source_fragment_hash": entry["source_fragment_hash"],
                "source_hash": entry["source_hash"],
                "source_evidence_id": entry["source_evidence_id"],
                "source_evidence_hash": entry["source_evidence_hash"],
            }
            for entry in selected_entries
        ],
        "rule_hash": rule["rule_hash"],
        "fragment_hash": fragment["fragment_hash"],
        "effect_hash": effect["effect_hash"],
        "metrics_hash": metrics["metrics_hash"],
    }
    if sample_design_ref is not None:
        content["sample_design_ref"] = _sample_design_ref(sample_design_ref)
    return content


def _derive_candidate_evidence(
    *,
    schema_version: str,
    sample_design_ref: Mapping[str, Any] | None,
    pool_ref: Mapping[str, Any],
    evidence_identity: Mapping[str, Any],
    measurement_context: Mapping[str, Any],
    selected_entries: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    fragment: Mapping[str, Any],
    effect: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, str]:
    content = _candidate_evidence_content(
        schema_version=schema_version,
        sample_design_ref=sample_design_ref,
        pool_ref=pool_ref,
        evidence_identity=evidence_identity,
        measurement_context=measurement_context,
        selected_entries=selected_entries,
        rule=rule,
        fragment=fragment,
        effect=effect,
        metrics=metrics,
    )
    candidate_id = _stable_id("candidate", content)
    evidence_hash = _sha256(
        _canonical_json({**content, "candidate_id": candidate_id})
    )
    result = {"candidate_id": candidate_id, "evidence_hash": evidence_hash}
    if sample_design_ref is not None:
        result["sample_design_ref"] = _sample_design_ref(sample_design_ref)
    return result


def _normalize_candidate_evidence(
    value: object,
    *,
    schema_version: str,
    sample_design_ref: Mapping[str, Any] | None,
    pool_ref: Mapping[str, Any],
    evidence_identity: Mapping[str, Any],
    measurement_context: Mapping[str, Any],
    selected_entries: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    fragment: Mapping[str, Any],
    effect: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError("candidate_evidence must be an object")
    expected_fields = (
        _CANDIDATE_EVIDENCE_FIELDS
        if schema_version == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION
        else _CANDIDATE_EVIDENCE_FIELDS_V1
    )
    _exact_fields(value, expected_fields, "candidate_evidence")
    expected = _derive_candidate_evidence(
        schema_version=schema_version,
        sample_design_ref=sample_design_ref,
        pool_ref=pool_ref,
        evidence_identity=evidence_identity,
        measurement_context=measurement_context,
        selected_entries=selected_entries,
        rule=rule,
        fragment=fragment,
        effect=effect,
        metrics=metrics,
    )
    candidate_id = _text(value["candidate_id"], "candidate_evidence.candidate_id")
    if _CANDIDATE_ID_RE.fullmatch(candidate_id) is None or not hmac.compare_digest(
        candidate_id, expected["candidate_id"]
    ):
        raise VotingCandidateAssetError(
            "candidate_id does not match canonical Voting evidence"
        )
    evidence_hash = _hash(
        value["evidence_hash"], "candidate_evidence.evidence_hash"
    )
    if not hmac.compare_digest(evidence_hash, expected["evidence_hash"]):
        raise VotingCandidateAssetError(
            "evidence_hash does not authenticate Voting candidate evidence"
        )
    if sample_design_ref is not None:
        supplied_ref = _sample_design_ref(value["sample_design_ref"])
        if supplied_ref != expected["sample_design_ref"]:
            raise VotingCandidateAssetError(
                "candidate_evidence sample_design_ref does not match asset"
            )
    return expected


def _fixed_lifecycle() -> dict[str, str]:
    return {
        "candidate_stage": _CANDIDATE_STAGE,
        "observation_stage": _OBSERVATION_STAGE,
        "validation_status": _VALIDATION_STATUS,
    }


def _lifecycle(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError("lifecycle must be an object")
    _exact_fields(value, _LIFECYCLE_FIELDS, "lifecycle")
    expected = _fixed_lifecycle()
    if value != expected:
        raise VotingCandidateAssetError(
            "Voting lifecycle must remain development/backtested/unvalidated"
        )
    return expected


def _producer_version(value: object, *, schema_version: str) -> str:
    normalized = _text(value, "producer_version")
    expected = (
        VOTING_CANDIDATE_ASSET_PRODUCER_VERSION
        if schema_version == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION
        else VOTING_CANDIDATE_ASSET_PRODUCER_VERSION_V1
    )
    if normalized != expected:
        raise VotingCandidateAssetError(
            "producer_version must be " + expected
        )
    return normalized


def _schema_version(value: object) -> str:
    normalized = _text(value, "schema_version")
    if normalized not in {
        VOTING_CANDIDATE_ASSET_SCHEMA_VERSION_V1,
        VOTING_CANDIDATE_ASSET_SCHEMA_VERSION,
    }:
        raise VotingCandidateAssetError("Voting candidate schema_version is invalid")
    return normalized


def _sample_design_ref(value: object) -> dict[str, str]:
    try:
        return StrategySampleDesignRef.from_value(value).to_ref_dict()
    except StrategyError as exc:
        raise VotingCandidateAssetError(
            "sample_design_ref must be one exact governed development reference"
        ) from exc


def _canonical_condition(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError(f"{name} must be an object")
    try:
        condition = canonicalize_expression(value)
    except StrategyError as exc:
        raise VotingCandidateAssetError(f"{name} is invalid: {exc}") from exc
    if _canonical_json(condition) != _canonical_json(value):
        raise VotingCandidateAssetError(f"{name} must be canonical Strategy DSL")
    return condition


def _derived_rate(
    value: object,
    *,
    numerator: int,
    denominator: int,
    name: str,
) -> float | None:
    expected = None if denominator == 0 else numerator / denominator
    return _derived_nullable_number(value, expected, name)


def _derived_nullable_number(
    value: object,
    expected: float | None,
    name: str,
) -> float | None:
    if expected is None:
        if value is not None:
            raise VotingCandidateAssetError(
                f"{name} must be null when its denominator is zero"
            )
        return None
    actual = _finite_number(value, name)
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise VotingCandidateAssetError(f"{name} is inconsistent with effect counts")
    return float(expected)


def _bounded_count(value: object, name: str, *, upper: int) -> int:
    normalized = _non_negative_int(value, name)
    if normalized > upper:
        raise VotingCandidateAssetError(f"{name} must be at most {upper}")
    return normalized


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise VotingCandidateAssetError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise VotingCandidateAssetError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise VotingCandidateAssetError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise VotingCandidateAssetError(f"{name} must be a finite number")
    return normalized


def _nullable_rate_value(value: object, name: str) -> float | None:
    if value is None:
        return None
    normalized = _finite_number(value, name)
    if normalized < 0.0 or normalized > 1.0:
        raise VotingCandidateAssetError(f"{name} must be between 0 and 1")
    return normalized


def _nullable_non_negative_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    normalized = _finite_number(value, name)
    if normalized < 0.0:
        raise VotingCandidateAssetError(f"{name} must be non-negative")
    return normalized


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise VotingCandidateAssetError(f"{name} must be non-empty canonical text")
    if "\x00" in value:
        raise VotingCandidateAssetError(f"{name} must not contain NUL")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise VotingCandidateAssetError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _json_array(value: object, name: str) -> list[Any]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise VotingCandidateAssetError(f"{name} must be an array")
    normalized = _json_value(value, name)
    if not isinstance(normalized, list):
        raise VotingCandidateAssetError(f"{name} must be an array")
    return normalized


def _json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VotingCandidateAssetError(f"{name} must be an object")
    normalized = _json_value(value, name)
    if not isinstance(normalized, dict):
        raise VotingCandidateAssetError(f"{name} must be an object")
    return normalized


def _json_value(value: object, name: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise VotingCandidateAssetError(f"{name} must contain finite JSON")
        return normalized
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise VotingCandidateAssetError(f"{name} keys must be strings")
        return {
            key: _json_value(child, f"{name}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _json_value(child, f"{name}[{index}]")
            for index, child in enumerate(value)
        ]
    raise VotingCandidateAssetError(f"{name} must contain canonical JSON values")


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise VotingCandidateAssetError(f"{name} keys must be strings")
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported fields: " + ", ".join(unexpected))
        raise VotingCandidateAssetError(f"invalid {name} ({'; '.join(details)})")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VotingCandidateAssetError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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
    except (TypeError, ValueError, RecursionError) as exc:
        raise VotingCandidateAssetError(
            "Voting candidate must be finite canonical JSON"
        ) from exc


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "VOTING_CANDIDATE_ASSET_PRODUCER_VERSION",
    "VOTING_CANDIDATE_ASSET_PRODUCER_VERSION_V1",
    "VOTING_CANDIDATE_ASSET_SCHEMA_VERSION",
    "VOTING_CANDIDATE_ASSET_SCHEMA_VERSION_V1",
    "VOTING_CANDIDATE_ASSET_TYPE",
    "VOTING_EFFECT_SCHEMA_VERSION",
    "VOTING_METRICS_SCHEMA_VERSION",
    "VotingCandidateAssetError",
    "build_voting_candidate_asset",
    "canonical_voting_candidate_asset_json",
    "parse_voting_candidate_asset_json",
    "validate_voting_candidate_asset",
    "verify_voting_candidate_asset_against_pool",
]
