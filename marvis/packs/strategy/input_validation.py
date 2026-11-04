"""Shared structural checks for governed strategy inputs.

No tool, repository or artifact hashing is owned here. Each caller retains its
family-specific evidence validation and error context.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .errors import StrategyError

if TYPE_CHECKING:
    from .sample_design_v2_tools import StrategySampleDesignV2ArtifactBinding
    from .sample_design_v2_native_tools import StrategySampleDesignV2NativeArtifactBinding
    _StrategySampleDesignV2Binding = StrategySampleDesignV2ArtifactBinding | StrategySampleDesignV2NativeArtifactBinding


def dataset_evidence_binding(
    sample: _StrategySampleDesignV2Binding,
) -> dict[str, Any]:
    source = sample.source_binding
    return {
        "task_id": source.task_id,
        "dataset_id": source.dataset_id,
        "dataset_content_hash": source.dataset_content_hash,
        "dataset_source_path": source.dataset_source_path,
        "dataset_registry_metadata_hash": (
            source.dataset_registry_metadata_hash
        ),
        "workspace_revision": source.workspace_revision,
        "workspace_generation": source.workspace_generation,
        "semantic_mapping_hash": source.semantic_mapping_hash,
    }


def sample_design_v2_identity(
    binding: _StrategySampleDesignV2Binding,
) -> tuple[str, ...]:
    return (
        binding.task_id,
        binding.membership_artifact_id,
        binding.membership_artifact_content_hash,
        binding.bundle_artifact_id,
        binding.bundle_artifact_content_hash,
        str(binding.bundle["bundle_id"]),
        str(binding.bundle["sample_design"]["sample_design_id"]),
        str(binding.bundle["sample_design"]["content_hash"]),
    )


def require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(str(field) for field in actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unsupported " + ", ".join(unexpected))
        raise StrategyError(f"{name} has " + "; ".join(details))


def expression_fields(value: object) -> set[str]:
    fields: set[str] = set()
    if isinstance(value, Mapping):
        field = value.get("field")
        if isinstance(field, str):
            fields.add(field)
        for item in value.values():
            fields.update(expression_fields(item))
    elif isinstance(value, list | tuple):
        for item in value:
            fields.update(expression_fields(item))
    return fields


def require_binding_connection(conn, *, db_path: Path, name: str) -> None:
    if not conn.in_transaction:
        raise StrategyError(f"{name} binding requires a caller-owned transaction")
    database = conn.execute(
        "SELECT file FROM pragma_database_list WHERE name = 'main'"
    ).fetchone()
    if (
        database is None
        or not str(database["file"])
        or Path(str(database["file"])).absolute() != db_path
    ):
        raise StrategyError(f"{name} binding database changed")
