"""Runtime score bindings for Strategies materialized from immutable Pools.

The materialization ledger is the sole bridge from a persisted Strategy back
to the exact historical Pool, sample design, and model-score artifacts that
made its virtual fields executable.  Score vectors are hydrated only in
memory; callers keep the governed source dataset unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
from typing import Any

import pandas as pd

from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool_requirement_resolver import (
    ResolvedPoolRequirements,
    hydrate_requirement_fields,
    normalize_pool_requirements,
    pool_requirement_bindings_provenance,
    require_historical_resolved_pool_requirements_on_connection,
    resolve_historical_pool_requirements,
    validate_pool_requirement_bindings_provenance,
)
from marvis.packs.strategy.pool_tools import (
    StrategyPoolDevelopmentExecutionBinding,
    bind_strategy_pool_revision_development_execution,
    load_strategy_candidate_pool_revision_artifact,
    require_strategy_pool_revision_development_execution_binding_on_connection,
)
from marvis.packs.strategy.voting_candidate_tools import (
    resolve_pool_requirement_sample_design_v2,
)


RUNTIME_REQUIREMENTS_PROVENANCE_SCHEMA_VERSION = (
    "strategy.materialized-runtime-requirements.v1"
)
_PROVENANCE_FIELDS = frozenset({"schema_version", "candidate", "baseline"})
_BINDING_FIELDS = frozenset(
    {
        "materialization_id",
        "strategy_id",
        "strategy_type",
        "strategy_version",
        "strategy_spec_hash",
        "pool_id",
        "pool_revision_id",
        "pool_revision",
        "pool_snapshot_hash",
        "pool_artifact_id",
        "pool_artifact_content_hash",
        "selected_design_hash",
        "source_dataset_ref",
        "sample_design_ref",
        "requirement_bindings",
    }
)
_SOURCE_DATASET_REF_FIELDS = frozenset({"dataset_id", "content_hash"})
_SAMPLE_DESIGN_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "partition",
    }
)
_STRATEGY_TYPES = frozenset(
    {"approval", "reject", "limit", "pricing", "segmentation"}
)
_MATERIALIZATION_LEDGER_INPUT_FIELDS = frozenset(
    {
        "id",
        "task_id",
        "strategy_type",
        "strategy_id",
        "pool_id",
        "pool_revision_id",
        "pool_revision",
        "pool_snapshot_hash",
        "pool_artifact_id",
        "pool_artifact_content_hash",
        "selected_design_hash",
        "requirements",
        "requirements_hash",
        "strategy_spec_hash",
        "strategy_dsl_content_hash",
        "audit_id",
    }
)


@dataclass(frozen=True, slots=True)
class MaterializedStrategyRuntimeRequirements:
    """Authenticated executable requirements for one persisted Strategy."""

    task_id: str
    strategy_id: str
    materialization: dict[str, Any]
    development: StrategyPoolDevelopmentExecutionBinding
    resolved: ResolvedPoolRequirements


def load_materialized_strategy_runtime_requirements(
    runtime,
    *,
    task_id: str,
    strategy_id: str,
    dataset_id: str,
    dataset_content_hash: str,
) -> MaterializedStrategyRuntimeRequirements | None:
    """Load exact historical score requirements for a materialized Strategy.

    Non-materialized Strategies and materializations with no executable
    requirements deliberately retain their existing runtime behavior.
    """

    task = _text(task_id, "task_id")
    selected_strategy = _text(strategy_id, "strategy_id")
    selected_dataset = _text(dataset_id, "dataset_id")
    selected_dataset_hash = _hash(
        dataset_content_hash,
        "dataset_content_hash",
    )
    materialization = (
        runtime.strategies.get_pool_materialization_for_strategy(
            selected_strategy
        )
    )
    if materialization is None:
        return None
    requirements = normalize_pool_requirements(
        materialization["requirements"]
    )
    if not requirements:
        return None
    if (
        materialization["task_id"] != task
        or materialization["strategy_id"] != selected_strategy
    ):
        raise StrategyError(
            "materialized Strategy runtime ownership changed"
        )

    pool = load_strategy_candidate_pool_revision_artifact(
        runtime,
        task_id=task,
        strategy_type=materialization["strategy_type"],
        revision_id=materialization["pool_revision_id"],
        artifact_id=materialization["pool_artifact_id"],
        expected_artifact_content_hash=materialization[
            "pool_artifact_content_hash"
        ],
    )
    development = bind_strategy_pool_revision_development_execution(
        runtime,
        pool,
    )
    _require_materialization_matches_pool(
        materialization,
        development=development,
        requirements=requirements,
    )
    if (
        development.dataset.dataset_id != selected_dataset
        or not hmac.compare_digest(
            development.dataset.content_hash,
            selected_dataset_hash,
        )
    ):
        raise StrategyError(
            "materialized Strategy requirements bind a different source dataset"
        )
    physical_sample = resolve_pool_requirement_sample_design_v2(
        runtime,
        development=development,
        require_current=False,
    )
    resolved = resolve_historical_pool_requirements(
        runtime,
        task_id=task,
        compiled_design={"requirements": list(requirements)},
        sample_design=physical_sample,
    )
    if (
        resolved.requirements != requirements
        or not hmac.compare_digest(
            resolved.requirements_hash,
            materialization["requirements_hash"],
        )
    ):
        raise StrategyError(
            "materialized Strategy runtime requirements changed"
        )
    binding = MaterializedStrategyRuntimeRequirements(
        task_id=task,
        strategy_id=selected_strategy,
        materialization=dict(materialization),
        development=development,
        resolved=resolved,
    )
    _validate_binding(binding)
    return binding


def require_materialized_strategy_runtime_requirements_on_connection(
    conn,
    runtime,
    binding: MaterializedStrategyRuntimeRequirements,
) -> None:
    """Re-authenticate the complete immutable runtime chain under writer lock."""

    _validate_binding(binding)
    runtime.strategies.require_pool_materialization_on_connection(
        conn,
        {
            field: binding.materialization[field]
            for field in _MATERIALIZATION_LEDGER_INPUT_FIELDS
        },
    )
    require_strategy_pool_revision_development_execution_binding_on_connection(
        conn,
        binding.development,
    )
    require_historical_resolved_pool_requirements_on_connection(
        conn,
        binding.resolved,
    )


def hydrate_materialized_strategy_runtime_requirements(
    frame: pd.DataFrame,
    bindings: Sequence[
        MaterializedStrategyRuntimeRequirements | None
    ],
) -> pd.DataFrame:
    """Hydrate the de-duplicated candidate/baseline score fields in memory."""

    active = tuple(binding for binding in bindings if binding is not None)
    if not active:
        return frame
    for binding in active:
        _validate_binding(binding)
    task_ids = {binding.task_id for binding in active}
    dataset_refs = {
        (
            binding.development.dataset.dataset_id,
            binding.development.dataset.content_hash,
            binding.development.dataset.row_count,
        )
        for binding in active
    }
    if len(task_ids) != 1 or len(dataset_refs) != 1:
        raise StrategyError(
            "candidate and baseline runtime requirements bind different datasets"
        )

    requirements: list[dict[str, Any]] = []
    field_bindings = []
    by_field: dict[str, tuple[str, str, str, str]] = {}
    for binding in active:
        requirements.extend(binding.resolved.requirements)
        for field, score in binding.resolved.field_bindings:
            identity = (
                str(score.evidence_record["id"]),
                str(score.evidence_record["content_hash"]),
                str(score.vector_record["id"]),
                str(score.vector_record["content_hash"]),
            )
            previous = by_field.get(field)
            if previous is not None:
                if previous != identity:
                    raise StrategyError(
                        "one virtual score field cannot bind different score evidence"
                    )
                continue
            by_field[field] = identity
            field_bindings.append((field, score))
    canonical_requirements = normalize_pool_requirements(requirements)
    combined = ResolvedPoolRequirements(
        task_id=active[0].task_id,
        requirements_hash=hashlib.sha256(
            _canonical_json(list(canonical_requirements)).encode("utf-8")
        ).hexdigest(),
        requirements=canonical_requirements,
        field_bindings=tuple(field_bindings),
    )
    return hydrate_requirement_fields(frame, resolved=combined)


def materialized_runtime_requirements_provenance(
    *,
    candidate: MaterializedStrategyRuntimeRequirements | None,
    baseline: MaterializedStrategyRuntimeRequirements | None,
) -> dict[str, Any] | None:
    """Project deterministic aggregate-only lineage for public Tool evidence."""

    if candidate is None and baseline is None:
        return None
    value = {
        "schema_version": RUNTIME_REQUIREMENTS_PROVENANCE_SCHEMA_VERSION,
        "candidate": (
            None if candidate is None else _binding_provenance(candidate)
        ),
        "baseline": (
            None if baseline is None else _binding_provenance(baseline)
        ),
    }
    return validate_materialized_runtime_requirements_provenance(value)


def validate_materialized_runtime_requirements_provenance(
    value: object,
) -> dict[str, Any]:
    """Validate detached runtime lineage without reading mutable state."""

    obj = _object(value, "runtime requirements provenance")
    _exact_fields(obj, _PROVENANCE_FIELDS, "runtime requirements provenance")
    if obj["schema_version"] != RUNTIME_REQUIREMENTS_PROVENANCE_SCHEMA_VERSION:
        raise StrategyError(
            "runtime requirements provenance schema is unsupported"
        )
    candidate = _optional_binding_provenance(
        obj["candidate"],
        name="runtime requirements candidate",
    )
    baseline = _optional_binding_provenance(
        obj["baseline"],
        name="runtime requirements baseline",
    )
    if candidate is None and baseline is None:
        raise StrategyError(
            "runtime requirements provenance must contain a binding"
        )
    return {
        "schema_version": RUNTIME_REQUIREMENTS_PROVENANCE_SCHEMA_VERSION,
        "candidate": candidate,
        "baseline": baseline,
    }


def load_historical_materialized_runtime_requirements_from_provenance(
    runtime,
    *,
    task_id: str,
    candidate_strategy_id: str,
    dataset_id: str,
    dataset_content_hash: str,
    provenance: object,
) -> tuple[
    MaterializedStrategyRuntimeRequirements | None,
    MaterializedStrategyRuntimeRequirements | None,
]:
    """Reload and compare every exact runtime binding recorded by a backtest."""

    normalized = validate_materialized_runtime_requirements_provenance(
        provenance
    )
    candidate_expected = normalized["candidate"]
    if (
        candidate_expected is not None
        and candidate_expected["strategy_id"] != candidate_strategy_id
    ):
        raise StrategyError(
            "backtest runtime requirements belong to another candidate Strategy"
        )
    candidate = load_materialized_strategy_runtime_requirements(
        runtime,
        task_id=task_id,
        strategy_id=candidate_strategy_id,
        dataset_id=dataset_id,
        dataset_content_hash=dataset_content_hash,
    )
    if (candidate is None) != (candidate_expected is None):
        raise StrategyError(
            "candidate runtime requirements changed from backtest evidence"
        )
    if candidate is not None and _binding_provenance(candidate) != candidate_expected:
        raise StrategyError(
            "candidate runtime requirements changed from backtest evidence"
        )

    baseline_expected = normalized["baseline"]
    baseline = None
    if baseline_expected is not None:
        baseline = load_materialized_strategy_runtime_requirements(
            runtime,
            task_id=task_id,
            strategy_id=baseline_expected["strategy_id"],
            dataset_id=dataset_id,
            dataset_content_hash=dataset_content_hash,
        )
        if baseline is None or _binding_provenance(baseline) != baseline_expected:
            raise StrategyError(
                "baseline runtime requirements changed from backtest evidence"
            )
    return candidate, baseline


def _binding_provenance(
    binding: MaterializedStrategyRuntimeRequirements,
) -> dict[str, Any]:
    _validate_binding(binding)
    materialization = binding.materialization
    development = binding.development
    value = {
        "materialization_id": materialization["id"],
        "strategy_id": materialization["strategy_id"],
        "strategy_type": materialization["strategy_type"],
        "strategy_version": materialization["strategy_version"],
        "strategy_spec_hash": materialization["strategy_spec_hash"],
        "pool_id": materialization["pool_id"],
        "pool_revision_id": materialization["pool_revision_id"],
        "pool_revision": materialization["pool_revision"],
        "pool_snapshot_hash": materialization["pool_snapshot_hash"],
        "pool_artifact_id": materialization["pool_artifact_id"],
        "pool_artifact_content_hash": materialization[
            "pool_artifact_content_hash"
        ],
        "selected_design_hash": materialization["selected_design_hash"],
        "source_dataset_ref": {
            "dataset_id": development.dataset.dataset_id,
            "content_hash": development.dataset.content_hash,
        },
        "sample_design_ref": development.sample_design.to_ref_dict(),
        "requirement_bindings": pool_requirement_bindings_provenance(
            binding.resolved
        ),
    }
    return _binding_provenance_value(value, name="runtime requirements binding")


def _optional_binding_provenance(
    value: object,
    *,
    name: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    return _binding_provenance_value(value, name=name)


def _binding_provenance_value(
    value: object,
    *,
    name: str,
) -> dict[str, Any]:
    obj = _object(value, name)
    _exact_fields(obj, _BINDING_FIELDS, name)
    strategy_type = _text(obj["strategy_type"], f"{name}.strategy_type")
    if strategy_type not in _STRATEGY_TYPES:
        raise StrategyError(f"{name}.strategy_type is unsupported")
    strategy_version = _positive_int(
        obj["strategy_version"],
        f"{name}.strategy_version",
    )
    pool_revision = _positive_int(
        obj["pool_revision"],
        f"{name}.pool_revision",
    )
    source = _object(obj["source_dataset_ref"], f"{name}.source_dataset_ref")
    _exact_fields(
        source,
        _SOURCE_DATASET_REF_FIELDS,
        f"{name}.source_dataset_ref",
    )
    sample = _object(obj["sample_design_ref"], f"{name}.sample_design_ref")
    _exact_fields(
        sample,
        _SAMPLE_DESIGN_REF_FIELDS,
        f"{name}.sample_design_ref",
    )
    partition = _text(
        sample["partition"],
        f"{name}.sample_design_ref.partition",
    )
    if partition not in {"development", "risk/development"}:
        raise StrategyError(
            f"{name}.sample_design_ref.partition is unsupported"
        )
    return {
        "materialization_id": _text(
            obj["materialization_id"],
            f"{name}.materialization_id",
        ),
        "strategy_id": _text(obj["strategy_id"], f"{name}.strategy_id"),
        "strategy_type": strategy_type,
        "strategy_version": strategy_version,
        "strategy_spec_hash": _hash(
            obj["strategy_spec_hash"],
            f"{name}.strategy_spec_hash",
        ),
        "pool_id": _text(obj["pool_id"], f"{name}.pool_id"),
        "pool_revision_id": _text(
            obj["pool_revision_id"],
            f"{name}.pool_revision_id",
        ),
        "pool_revision": pool_revision,
        "pool_snapshot_hash": _hash(
            obj["pool_snapshot_hash"],
            f"{name}.pool_snapshot_hash",
        ),
        "pool_artifact_id": _hash(
            obj["pool_artifact_id"],
            f"{name}.pool_artifact_id",
        ),
        "pool_artifact_content_hash": _hash(
            obj["pool_artifact_content_hash"],
            f"{name}.pool_artifact_content_hash",
        ),
        "selected_design_hash": _hash(
            obj["selected_design_hash"],
            f"{name}.selected_design_hash",
        ),
        "source_dataset_ref": {
            "dataset_id": _text(
                source["dataset_id"],
                f"{name}.source_dataset_ref.dataset_id",
            ),
            "content_hash": _hash(
                source["content_hash"],
                f"{name}.source_dataset_ref.content_hash",
            ),
        },
        "sample_design_ref": {
            "artifact_id": _hash(
                sample["artifact_id"],
                f"{name}.sample_design_ref.artifact_id",
            ),
            "artifact_content_hash": _hash(
                sample["artifact_content_hash"],
                f"{name}.sample_design_ref.artifact_content_hash",
            ),
            "sample_design_id": _text(
                sample["sample_design_id"],
                f"{name}.sample_design_ref.sample_design_id",
            ),
            "sample_design_content_hash": _hash(
                sample["sample_design_content_hash"],
                f"{name}.sample_design_ref.sample_design_content_hash",
            ),
            "partition": partition,
        },
        "requirement_bindings": (
            validate_pool_requirement_bindings_provenance(
                obj["requirement_bindings"]
            )
        ),
    }


def _require_materialization_matches_pool(
    materialization: Mapping[str, Any],
    *,
    development: StrategyPoolDevelopmentExecutionBinding,
    requirements: tuple[dict[str, Any], ...],
) -> None:
    pool = development.pool
    design = pool.compiled_design
    if (
        pool.task_id != materialization["task_id"]
        or pool.strategy_type != materialization["strategy_type"]
        or pool.pool["pool_id"] != materialization["pool_id"]
        or pool.pool["revision_id"] != materialization["pool_revision_id"]
        or pool.pool["revision"] != materialization["pool_revision"]
        or not hmac.compare_digest(
            pool.pool["snapshot_hash"],
            materialization["pool_snapshot_hash"],
        )
        or pool.artifact_id != materialization["pool_artifact_id"]
        or not hmac.compare_digest(
            pool.artifact_content_hash,
            materialization["pool_artifact_content_hash"],
        )
        or not hmac.compare_digest(
            design["design_hash"],
            materialization["selected_design_hash"],
        )
        or normalize_pool_requirements(design["requirements"])
        != requirements
        or not hmac.compare_digest(
            strategy_spec_hash(design["strategy_spec"]),
            materialization["strategy_spec_hash"],
        )
    ):
        raise StrategyError(
            "materialized Strategy runtime Pool binding changed"
        )


def _validate_binding(value: object) -> None:
    if not isinstance(value, MaterializedStrategyRuntimeRequirements):
        raise StrategyError(
            "materialized Strategy runtime requirements binding is invalid"
        )
    if (
        value.task_id != value.materialization.get("task_id")
        or value.strategy_id != value.materialization.get("strategy_id")
        or value.development.task_id != value.task_id
        or value.resolved.task_id != value.task_id
        or not value.resolved.requirements
    ):
        raise StrategyError(
            "materialized Strategy runtime requirements binding changed"
        )
    requirements = normalize_pool_requirements(
        value.materialization.get("requirements")
    )
    _require_materialization_matches_pool(
        value.materialization,
        development=value.development,
        requirements=requirements,
    )
    if (
        value.resolved.requirements != requirements
        or not hmac.compare_digest(
            value.resolved.requirements_hash,
            value.materialization["requirements_hash"],
        )
    ):
        raise StrategyError(
            "materialized Strategy runtime requirements binding changed"
        )
    pool_requirement_bindings_provenance(value.resolved)


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise StrategyError(f"{name} must be an object")
    return dict(value)


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise StrategyError(f"{name} fields are invalid")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StrategyError(f"{name} must be non-empty canonical text")
    return value


def _hash(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StrategyError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StrategyError(f"{name} must be a positive integer")
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
            "runtime requirements provenance must be canonical JSON"
        ) from exc


__all__ = [
    "MaterializedStrategyRuntimeRequirements",
    "RUNTIME_REQUIREMENTS_PROVENANCE_SCHEMA_VERSION",
    "hydrate_materialized_strategy_runtime_requirements",
    "load_historical_materialized_runtime_requirements_from_provenance",
    "load_materialized_strategy_runtime_requirements",
    "materialized_runtime_requirements_provenance",
    "require_materialized_strategy_runtime_requirements_on_connection",
    "validate_materialized_runtime_requirements_provenance",
]
