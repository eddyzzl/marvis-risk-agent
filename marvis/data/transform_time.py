"""Authenticate the existing immutable cleaning producer for temporal readers."""

import hashlib
from pathlib import Path

from marvis.feature.errors import FeatureError
from marvis.repositories.data_transform import (
    DATA_TRANSFORM_ARTIFACT_KIND,
    DATA_TRANSFORM_ORIGIN_TOOL,
    DataTransformIdentity,
    DataTransformRepository,
    data_transform_artifact_provenance,
)


def transform_time_parent(registry, artifacts, dataset):
    records = [
        record
        for record in artifacts.list_for_task(dataset.task_id)
        if record["kind"] == DATA_TRANSFORM_ARTIFACT_KIND
        and record["provenance"].get("result_dataset_id") == dataset.id
    ]
    if not records:
        return None
    if len(records) != 1:
        raise FeatureError("ambiguous temporal transform evidence")
    artifact = records[0]
    record = DataTransformRepository(registry._repo.db_path).get_for_task(
        dataset.task_id,
        artifact["provenance"]["run_id"],
    )
    if (
        record is None
        or record.result_dataset_id != dataset.id
        or record.result_content_hash != dataset.content_hash
        or record.result_artifact_id != artifact["id"]
    ):
        raise FeatureError("temporal transform producer binding changed")
    identity = DataTransformIdentity(
        task_id=record.task_id,
        source_dataset_id=record.source_dataset_id,
        source_content_hash=record.source_content_hash,
        workspace_revision=record.workspace_revision,
        analysis_generation=record.analysis_generation,
        semantic_mapping_hash=record.semantic_mapping_hash,
        operations=record.operations,
        producer_version=record.producer_version,
    )
    expected = data_transform_artifact_provenance(
        identity,
        result_dataset_id=dataset.id,
        result_content_hash=dataset.content_hash,
    )
    if (
        artifact["origin_tool"] != DATA_TRANSFORM_ORIGIN_TOOL
        or artifact["content_hash"] != record.result_hash
        or artifact["provenance"] != expected
    ):
        raise FeatureError("temporal transform artifact binding changed")
    path = Path(artifact["path"])
    root = (registry.datasets_root.parent / "tasks" / dataset.task_id).resolve(
        strict=True
    )
    resolved = path.resolve(strict=True)
    if path != resolved or not resolved.is_relative_to(root):
        raise FeatureError("temporal transform evidence path is unsafe")
    with resolved.open("rb") as stream:
        raw = stream.read(20_000_001)
    if (
        len(raw) > 20_000_000
        or hashlib.sha256(raw).hexdigest() != record.result_hash
        or raw != record.result_json.encode("utf-8")
    ):
        raise FeatureError("temporal transform evidence content changed")
    source = registry.get(record.source_dataset_id)
    if (
        source.task_id != dataset.task_id
        or source.content_hash != record.source_content_hash
    ):
        raise FeatureError("temporal transform source binding changed")
    registry.resolve_verified_path(source.id)
    # A run stays valid when another dataset later becomes active. The receipt
    # proves its own execution and is not tied to today's workspace selection.
    return record
