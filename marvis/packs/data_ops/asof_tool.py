"""Governed Plugin entrypoint for explicitly declared point-in-time joins."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from marvis.data.asof_join import AsOfJoinEngine
from marvis.data.time_contracts import (
    AsOfJoinSpec, Assurance, DatasetTimeContract, Sha256, TimeColumn,
)
from marvis.plugins.sdk import PackRuntime
from marvis.repositories.task_artifacts import TaskArtifactRepository


class DecisionInput(DatasetTimeContract):
    role: Literal["decision"]


class FeatureInput(DatasetTimeContract):
    role: Literal["feature_snapshot"]
    # Omission and an intentional unknown are different user decisions.
    available_at: TimeColumn | None


class SelectionInput(AsOfJoinSpec):
    mode: Literal["verified", "exploration"]


class AsOfToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    decision_contract: DecisionInput
    feature_contract: FeatureInput
    spec: SelectionInput


class ParentIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    dataset_id: str
    content_hash: Sha256
    contract_sha256: Sha256
    role: Literal["decision", "feature_snapshot"]


class EvidenceIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    artifact_id: str
    content_hash: Sha256
    kind: Literal["dataset_point_in_time"] = "dataset_point_in_time"


class AsOfToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["dataset-asof-tool-result.v1"] = "dataset-asof-tool-result.v1"
    result_dataset_id: str
    result_content_hash: Sha256
    row_count: int = Field(ge=1)
    mode: Literal["verified", "exploration"]
    assurance: Assurance
    assurance_reasons: list[str]
    verification_scope: Literal["recorded_temporal_constraints"] = "recorded_temporal_constraints"
    external_source_attestation: Literal["not_independently_verified"] = "not_independently_verified"
    evidence: EvidenceIdentity
    membership_sha256: Sha256
    parents: list[ParentIdentity] = Field(min_length=2, max_length=2)


def tool_asof_join(inputs: dict, ctx) -> dict:
    """Called only through the existing manifest-governed ToolRunner boundary.

    The task comes from the trusted execution context. Neither inputs nor a
    temporal claim can override dataset ownership or supply an output path.
    This Tool creates a new immutable child; selecting it as active training
    data remains an explicit downstream action.
    """
    request = AsOfToolInput.model_validate_json(json.dumps(inputs, allow_nan=False))
    runtime = PackRuntime(ctx)
    artifacts = TaskArtifactRepository(runtime.settings.db_path)
    engine = AsOfJoinEngine(runtime.registry, artifacts, workspace_root=runtime.settings.workspace)
    result = engine.execute(
        task_id=ctx.task_id,
        decision_contract=request.decision_contract,
        feature_contract=request.feature_contract,
        spec=request.spec,
    )
    record = artifacts.get_for_task(ctx.task_id, result.status.artifact_id)
    if record is None:
        raise ValueError("as-of evidence artifact is missing after registration")
    return AsOfToolOutput(
        result_dataset_id=result.dataset.id,
        result_content_hash=result.dataset.content_hash,
        row_count=result.dataset.row_count,
        mode=request.spec.mode,
        assurance=result.status.assurance,
        assurance_reasons=list(result.status.reasons),
        evidence=EvidenceIdentity(artifact_id=record["id"], content_hash=record["content_hash"]),
        membership_sha256=result.membership_sha256,
        parents=[ParentIdentity.model_validate(parent) for parent in record["provenance"]["parents"]],
    ).model_dump(mode="json")
