"""Direct-tool StrategySampleDesign fixture for focused strategy tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.packs.strategy.tools import tool_materialize_sample_design
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository


def materialize_strategy_tool_sample_design(
    settings,
    task,
    dataset,
    *,
    target_col: str = "bad",
    target_bad_value: int = 1,
    drop_nan_labels: bool = False,
    split_col: str | None = None,
    development_values: Sequence[object] = (),
    validation_values: Sequence[object] = (),
    oot_values: Sequence[object] = (),
    field_roles: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Activate, map, and freeze one mature development sample for a tool test."""

    repository = DataWorkspaceRepository(settings.db_path)
    workspace = repository.get_or_default(task.id)
    if workspace.active_dataset_id is None:
        workspace = repository.save(
            task.id,
            DataWorkspaceDraft(
                active_dataset_id=dataset.id,
                active_dataset_content_hash=dataset.content_hash,
            ),
            expected_revision=workspace.revision,
        )
    else:
        assert workspace.active_dataset_id == dataset.id
        assert workspace.active_dataset_content_hash == dataset.content_hash

    dataset_columns = {profile.name for profile in dataset.columns}
    roles = {
        column: role
        for column, role in workspace.semantic_mapping.field_roles.items()
        if column in dataset_columns
    }
    roles.update(
        {
            column: role
            for column, role in dict(field_roles or {}).items()
            if column in dataset_columns
        }
    )
    roles[target_col] = "target"
    if split_col is not None:
        roles.setdefault(split_col, "segment")
    mapping = DataSemanticMapping(
        target_col=target_col,
        field_roles=roles,
        business_names=workspace.semantic_mapping.business_names,
    )
    if workspace.semantic_mapping != mapping:
        workspace = repository.save(
            task.id,
            DataWorkspaceDraft(
                active_dataset_id=dataset.id,
                active_dataset_content_hash=dataset.content_hash,
                page=workspace.page,
                selected_field=workspace.selected_field,
                semantic_mapping=mapping,
            ),
            expected_revision=workspace.revision,
        )

    ctx = ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    request: dict[str, object] = {
        "dataset_id": dataset.id,
        "expected_dataset_content_hash": dataset.content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "target_col": target_col,
        "target_bad_value": target_bad_value,
        "performance_window_status": "provided",
        "performance_window_days": 90,
        "observation_window_status": "provided",
        "observation_window_start": "2026-01-01",
        "observation_window_end": "2026-06-30",
        "maturity_status": "confirmed_matured",
        "drop_nan_labels": drop_nan_labels,
    }
    if split_col is not None:
        request.update(
            {
                "split_col": split_col,
                "development_values": list(development_values),
                "validation_values": list(validation_values),
                "oot_values": list(oot_values),
            }
        )
    output = tool_materialize_sample_design(request, ctx)
    return {
        "artifact_id": output["artifact"]["artifact_id"],
        "artifact_content_hash": output["artifact"]["content_hash"],
        "sample_design_id": output["sample_design_id"],
        "sample_design_content_hash": output["content_hash"],
        "partition": "development",
    }
