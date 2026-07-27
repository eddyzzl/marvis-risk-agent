"""Materialize one exact current Strategy Pool as a canonical draft Strategy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import hmac
import json
from typing import Any

from marvis.packs.strategy.dsl import (
    canonical_strategy_json,
    strategy_spec_hash,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool_requirement_resolver import (
    normalize_pool_requirements,
)
from marvis.packs.strategy.pool_tools import (
    load_current_strategy_candidate_pool_artifact,
    require_strategy_candidate_pool_artifact_binding_on_connection,
)
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.repositories.strategy import (
    POOL_MATERIALIZATION_AUDIT_KIND,
    POOL_MATERIALIZATION_LEDGER_SCHEMA_VERSION,
    POOL_MATERIALIZATION_PRODUCER_VERSION,
)


MATERIALIZATION_TOOL_SCHEMA_VERSION = "strategy.pool-materialization-tool.v1"
MATERIALIZATION_AUDIT_KIND = POOL_MATERIALIZATION_AUDIT_KIND
_INPUT_FIELDS = frozenset(
    {
        "strategy_type",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "expected_pool_artifact_id",
        "expected_pool_artifact_content_hash",
        "expected_design_hash",
    }
)
_STRATEGY_TYPES = frozenset(
    {"approval", "reject", "limit", "pricing", "segmentation"}
)


def run_materialize_strategy_from_pool(inputs, ctx, runtime) -> dict[str, Any]:
    """Persist the exact compiled Pool design as one root draft Strategy.

    The loader authenticates the current Pool and all candidate lineages before
    the writer transaction. The same binding is authenticated again under
    ``BEGIN IMMEDIATE`` before Strategy, audit, and materialization ledger rows
    are committed together.
    """

    normalized = _normalize_inputs(inputs)
    task_id = _text(ctx.task_id, "task_id")
    binding = load_current_strategy_candidate_pool_artifact(
        runtime,
        task_id=task_id,
        strategy_type=normalized["strategy_type"],
        expected_pool_revision=normalized["expected_pool_revision"],
        expected_pool_snapshot_hash=normalized["expected_pool_snapshot_hash"],
        expected_artifact_id=normalized["expected_pool_artifact_id"],
        expected_artifact_content_hash=normalized[
            "expected_pool_artifact_content_hash"
        ],
    )
    design = binding.compiled_design
    if not hmac.compare_digest(
        design["design_hash"],
        normalized["expected_design_hash"],
    ):
        raise StrategyError("selected Strategy Pool design hash changed")

    strategy_spec = design["strategy_spec"]
    requirements = list(design["requirements"])
    requirements_json = _canonical_json(requirements)
    requirements_hash = hashlib.sha256(requirements_json.encode("utf-8")).hexdigest()
    identity = _materialization_identity(
        task_id=task_id,
        binding=binding,
        design_hash=design["design_hash"],
        requirements_hash=requirements_hash,
    )
    strategy = replace(
        build_strategy_from_spec(strategy_spec),
        id=identity["strategy_id"],
    )
    if strategy.spec is None or strategy.spec.to_dict() != strategy_spec:
        raise StrategyError(
            "compiled Strategy Pool design did not produce the exact canonical DSL"
        )
    effect_hash = strategy_spec_hash(strategy.spec)
    dsl_content_hash = hashlib.sha256(
        canonical_strategy_json(strategy.spec).encode("utf-8")
    ).hexdigest()
    materialization = {
        "id": identity["materialization_id"],
        "task_id": task_id,
        "strategy_type": binding.strategy_type,
        "strategy_id": identity["strategy_id"],
        "pool_id": binding.pool["pool_id"],
        "pool_revision_id": binding.pool["revision_id"],
        "pool_revision": binding.pool["revision"],
        "pool_snapshot_hash": binding.pool["snapshot_hash"],
        "pool_artifact_id": binding.artifact_id,
        "pool_artifact_content_hash": binding.artifact_content_hash,
        "selected_design_hash": design["design_hash"],
        "requirements": requirements,
        "requirements_hash": requirements_hash,
        "strategy_spec_hash": effect_hash,
        "strategy_dsl_content_hash": dsl_content_hash,
        "audit_id": identity["audit_id"],
    }

    repository = runtime.strategies
    with repository.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_strategy_candidate_pool_artifact_binding_on_connection(
            conn,
            binding,
        )
        if not hmac.compare_digest(
            binding.compiled_design["design_hash"],
            normalized["expected_design_hash"],
        ):
            raise StrategyError("selected Strategy Pool design hash changed")
        persisted = (
            repository.materialize_pool_strategy_draft_with_audit_on_connection(
                conn,
                strategy=strategy,
                materialization=materialization,
            )
        )
        conn.commit()
    return _tool_output(persisted)


def _materialization_identity(
    *,
    task_id: str,
    binding,
    design_hash: str,
    requirements_hash: str,
) -> dict[str, str]:
    body = {
        "schema_version": POOL_MATERIALIZATION_LEDGER_SCHEMA_VERSION,
        "producer_version": POOL_MATERIALIZATION_PRODUCER_VERSION,
        "task_id": task_id,
        "strategy_type": binding.strategy_type,
        "pool_id": binding.pool["pool_id"],
        "pool_revision_id": binding.pool["revision_id"],
        "pool_revision": binding.pool["revision"],
        "pool_snapshot_hash": binding.pool["snapshot_hash"],
        "pool_artifact_id": binding.artifact_id,
        "pool_artifact_content_hash": binding.artifact_content_hash,
        "selected_design_hash": design_hash,
        "requirements_hash": requirements_hash,
    }
    digest = hashlib.sha256(
        (
            "marvis.strategy.pool-materialization.identity.v1:"
            + _canonical_json(body)
        ).encode("utf-8")
    ).hexdigest()
    strategy_digest = hashlib.sha256(
        (
            "marvis.strategy.pool-materialized-strategy.identity.v1:"
            + digest
        ).encode("utf-8")
    ).hexdigest()
    audit_digest = hashlib.sha256(
        (
            "marvis.strategy.pool-materialization-audit.identity.v1:"
            + digest
        ).encode("utf-8")
    ).hexdigest()
    return {
        "materialization_id": f"strategy-pool-materialization-{digest[:24]}",
        "strategy_id": f"strategy-pool-{strategy_digest[:24]}",
        "audit_id": f"strategy-pool-materialization-audit-{audit_digest[:24]}",
    }


def _tool_output(persisted: Mapping[str, Any]) -> dict[str, Any]:
    materialization = persisted["materialization"]
    metadata = persisted["metadata"]
    normalized_requirements = normalize_pool_requirements(
        materialization["requirements"]
    )
    virtual_fields: list[str] = []
    for item in normalized_requirements:
        field = str(item["requirement"]["virtual_field"])
        if field not in virtual_fields:
            virtual_fields.append(field)
    return {
        "schema_version": MATERIALIZATION_TOOL_SCHEMA_VERSION,
        "materialization_id": materialization["id"],
        "strategy_ref": {
            "strategy_id": materialization["strategy_id"],
            "strategy_type": materialization["strategy_type"],
            "version": metadata["version"],
            "strategy_spec_hash": materialization["strategy_spec_hash"],
            "strategy_dsl_content_hash": materialization[
                "strategy_dsl_content_hash"
            ],
        },
        "pool_ref": {
            "pool_id": materialization["pool_id"],
            "revision_id": materialization["pool_revision_id"],
            "revision": materialization["pool_revision"],
            "snapshot_hash": materialization["pool_snapshot_hash"],
            "artifact_id": materialization["pool_artifact_id"],
            "artifact_content_hash": materialization[
                "pool_artifact_content_hash"
            ],
        },
        "design_hash": materialization["selected_design_hash"],
        "requirements": {
            "requirements_hash": materialization["requirements_hash"],
            "requirement_count": len(normalized_requirements),
            "virtual_fields": virtual_fields,
            "runtime_requirements_supported": True,
            "blocker_code": None,
        },
        "lifecycle": {
            "created_status": "draft",
            "created_asset_status": "draft",
            "current_status": metadata["status"],
            "current_asset_status": metadata["asset_status"],
            "adopted_by_this_tool": False,
            "deployed_by_this_tool": False,
        },
    }


def _normalize_inputs(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError("materialize_strategy_from_pool inputs must be an object")
    if any(not isinstance(key, str) for key in value):
        raise StrategyError(
            "materialize_strategy_from_pool input keys must be strings"
        )
    missing = sorted(_INPUT_FIELDS - set(value))
    unexpected = sorted(set(value) - _INPUT_FIELDS)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise StrategyError(
            "invalid materialize_strategy_from_pool inputs ("
            + "; ".join(details)
            + ")"
        )
    strategy_type = _text(value["strategy_type"], "strategy_type")
    if strategy_type not in _STRATEGY_TYPES:
        raise StrategyError("unsupported strategy_type")
    revision = value["expected_pool_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise StrategyError("expected_pool_revision must be a positive integer")
    return {
        "strategy_type": strategy_type,
        "expected_pool_revision": revision,
        "expected_pool_snapshot_hash": _hash(
            value["expected_pool_snapshot_hash"],
            "expected_pool_snapshot_hash",
        ),
        "expected_pool_artifact_id": _hash(
            value["expected_pool_artifact_id"],
            "expected_pool_artifact_id",
        ),
        "expected_pool_artifact_content_hash": _hash(
            value["expected_pool_artifact_content_hash"],
            "expected_pool_artifact_content_hash",
        ),
        "expected_design_hash": _hash(
            value["expected_design_hash"],
            "expected_design_hash",
        ),
    }


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StrategyError(f"{field} must be non-empty canonical text")
    return value


def _hash(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StrategyError(f"{field} must be a lowercase SHA-256")
    return value


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
        raise StrategyError(
            "Strategy Pool materialization must be finite canonical JSON"
        ) from exc


__all__ = [
    "MATERIALIZATION_AUDIT_KIND",
    "MATERIALIZATION_TOOL_SCHEMA_VERSION",
    "run_materialize_strategy_from_pool",
]
