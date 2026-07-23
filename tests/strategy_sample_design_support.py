"""Reusable governed sample prerequisite for strategy-development E2E tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marvis.agent.plan_driver import PlanDriver
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, TaskRepository
from marvis.repositories.data_workspace import DataWorkspaceRepository


def _ensure_active_data_workspace(client: TestClient, task_id: str) -> str:
    settings = client.app.state.settings
    workspace_repo = DataWorkspaceRepository(settings.db_path)
    workspace = workspace_repo.get_or_default(task_id)
    task = TaskRepository(settings.db_path).get_task(task_id)
    dataset_repo = DatasetRepository(settings.db_path)
    registry = DatasetRegistry(
        dataset_repo,
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    if workspace.active_dataset_id is None:
        supported_suffixes = {".csv", ".parquet", ".pq", ".xlsx", ".xls", ".xlsb"}
        source_files = sorted(
            path
            for path in Path(task.source_dir).iterdir()
            if path.is_file() and path.suffix.lower() in supported_suffixes
        )
        assert len(source_files) == 1, (
            "sample-design E2E prerequisite expects one source data file, got "
            f"{source_files}"
        )
        dataset = registry.register_from_upload(
            task_id,
            source_files[0],
            role="strategy_sample",
        )
        assert dataset.content_hash is not None
        workspace = workspace_repo.save(
            task_id,
            DataWorkspaceDraft(
                active_dataset_id=dataset.id,
                active_dataset_content_hash=dataset.content_hash,
            ),
            expected_revision=workspace.revision,
        )
    else:
        dataset = registry.get(workspace.active_dataset_id)

    column_names = {profile.name for profile in dataset.columns}
    assert task.target_col in column_names
    roles = dict(workspace.semantic_mapping.field_roles)
    roles = {
        column: role
        for column, role in roles.items()
        if role != "target" or column == task.target_col
    }
    roles[task.target_col] = "target"
    conventional_roles = {
        "score": "score",
        "month": "month",
        "weight": "weight",
        "loan_amount": "loan_amount",
        "overdue_amount": "overdue_amount",
    }
    for column, role in conventional_roles.items():
        if column in column_names:
            roles[column] = role
    mapping = DataSemanticMapping(
        target_col=task.target_col,
        field_roles=roles,
        business_names=workspace.semantic_mapping.business_names,
    )
    if workspace.semantic_mapping != mapping:
        workspace = workspace_repo.save(
            task_id,
            DataWorkspaceDraft(
                active_dataset_id=dataset.id,
                active_dataset_content_hash=dataset.content_hash,
                page=workspace.page,
                selected_field=workspace.selected_field,
                semantic_mapping=mapping,
            ),
            expected_revision=workspace.revision,
        )
    assert workspace.active_dataset_id == dataset.id
    return dataset.id


def materialize_mature_strategy_sample_design(
    client: TestClient,
    task_id: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    drop_nan_labels: bool = False,
    optional_columns: Mapping[str, str | None] | None = None,
) -> dict[str, str]:
    """Materialize one mature legacy anchor through its governed template.

    Fresh natural-language requests now use ``strategy_sample_design_v2`` and
    may not downgrade to the legacy single-population workflow.  Older
    downstream E2E cases still need the compatibility anchor they were written
    against, so this fixture starts that trusted template directly instead of
    teaching a fake LLM to emit a fresh, now-retired request.  Conventional
    month/weight/amount columns are included when present. Tests may override
    or disable each optional binding through ``optional_columns``.
    """

    del monkeypatch  # retained in the fixture signature for existing call sites
    expected_dataset_id = _ensure_active_data_workspace(client, task_id)
    workspace_response = client.get(f"/api/tasks/{task_id}/data-workspace")
    assert workspace_response.status_code == 200, workspace_response.text
    workspace = workspace_response.json()
    dataset_id = workspace["active_dataset_id"]
    assert dataset_id == expected_dataset_id

    preview_response = client.get(
        f"/api/tasks/{task_id}/datasets/{dataset_id}/preview?rows=1"
    )
    assert preview_response.status_code == 200, preview_response.text
    columns = set(preview_response.json()["columns"])
    conventional = {
        "month_col": "month",
        "weight_col": "weight",
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
    }
    selected_optional: dict[str, str | None] = {
        field: column if column in columns else None
        for field, column in conventional.items()
    }
    if optional_columns is not None:
        unexpected = set(optional_columns) - set(conventional)
        assert not unexpected, f"unsupported sample-design optional fields: {unexpected}"
        selected_optional.update(optional_columns)
    for field, column in selected_optional.items():
        assert column is None or column in columns, (
            f"sample-design {field}={column!r} is not in the active dataset"
        )

    workflow_inputs: dict[str, object] = {
        "target_bad_value": 1,
        "performance_window_status": "provided",
        "performance_window_days": 90,
        "observation_window_status": "provided",
        "observation_start": "2025-01-01",
        "observation_end": "2025-12-31",
        "maturity_status": "confirmed_matured",
        "drop_nan_labels": drop_nan_labels,
    }
    for field, column in selected_optional.items():
        if column is not None:
            workflow_inputs[field] = column
    workspace_record = DataWorkspaceRepository(
        client.app.state.settings.db_path
    ).get_or_default(task_id)
    task = TaskRepository(client.app.state.settings.db_path).get_task(task_id)
    slots = {
        "dataset_id": dataset_id,
        "expected_dataset_content_hash": workspace_record.active_dataset_content_hash,
        "workspace_revision": workspace_record.revision,
        "workspace_generation": workspace_record.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(
            workspace_record.semantic_mapping
        ),
        "target_col": task.target_col,
        **workflow_inputs,
    }
    driver = PlanDriver(
        client.app.state.plan_repo,
        client.app.state.plan_executor,
        planner=client.app.state.planner,
        validator=client.app.state.plan_validator,
        governance_service=getattr(
            client.app.state,
            "governance_service",
            None,
        ),
    )
    before_ids = {
        plan["id"]
        for plan in client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    }
    started = driver.start(
        task_id=task_id,
        template_id="strategy_sample_design",
        slots=slots,
    )
    driver.resume(plan_id=started.plan_id, user_text="开始")
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    created = [plan for plan in plans if plan["id"] not in before_ids]
    assert len(created) == 1
    plan = created[0]
    assert plan["template_id"] == "strategy_sample_design"
    assert plan["status"] == "done"

    stored = client.app.state.plan_repo.load_plan(plan["id"])
    assert len(stored.steps) == 1
    assert stored.steps[0].inputs["drop_nan_labels"] is drop_nan_labels
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["bundle"]["sample_design"]["maturity"] == "confirmed_matured"
    assert output["bundle"]["sample_design"]["target_definition"][
        "drop_nan_labels"
    ] is drop_nan_labels
    return {
        "artifact_id": output["artifact"]["artifact_id"],
        "artifact_content_hash": output["artifact"]["content_hash"],
        "sample_design_id": output["sample_design_id"],
        "sample_design_content_hash": output["content_hash"],
        "partition": "development",
    }
