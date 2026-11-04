"""A repeat single-table join confirmation tolerates later field directions.

Confirmed field directions are user facts added after the join binding; they
must neither break the replay nor be dropped. Target/roles/names stay strict.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from marvis.agent.join_setup import JoinSetupError
from marvis.agent.turn_handlers import join
from marvis.data.backend import DataBackend
from marvis.data.field_directions import confirm_field_directions
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import data_semantic_mapping_hash
from marvis.db import init_db
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.datasets import DatasetRepository
from marvis.settings import build_settings
from tests.test_data_workspace_api import (
    _app,
    _create_task,
    _put_workspace,
    _upload_csv,
    _workspace_draft,
)


def _joined_workspace(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    client = TestClient(_app(settings))
    task = _create_task(settings)
    dataset = _upload_csv(client, task.id)
    # A dataset change needs a reset payload before the mapping write.
    assert _put_workspace(client, task.id, _workspace_draft(dataset), revision=0).status_code == 200
    joined = _workspace_draft(
        dataset, target_col="bad_flag", field_roles={"bad_flag": "target"}
    )
    assert _put_workspace(client, task.id, joined, revision=1).status_code == 200
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    return client, settings, task, dataset, registry


def _confirm_direction(settings, task_id):
    snapshot = DataWorkspaceRepository(settings.db_path).get_or_default(task_id)
    confirm_field_directions(
        settings,
        task_id=task_id,
        actor_id="maker-1",
        expected_revision=snapshot.revision,
        expected_semantic_mapping_hash=data_semantic_mapping_hash(
            snapshot.semantic_mapping
        ),
        dataset_id=snapshot.active_dataset_id,
        dataset_content_hash=snapshot.active_dataset_content_hash,
        directions={"loan_amount": "higher_is_riskier"},
    )


def _repeat_join_confirmation(settings, task, dataset, registry, calls):
    join._bind_single_join_dataset(
        SimpleNamespace(settings=settings),
        task,
        registry=registry,
        dataset_id=dataset["id"],
        target_col="bad_flag",
        authenticated_binding=registry.authenticate_dataset_binding(
            dataset["id"],
            expected_task_id=task.id,
            expected_content_hash=dataset["content_hash"],
        ),
        on_connection=lambda conn: calls.append(conn),
    )


def test_repeat_join_confirmation_keeps_later_field_directions(tmp_path) -> None:
    _client, settings, task, dataset, registry = _joined_workspace(tmp_path)
    _confirm_direction(settings, task.id)
    before = DataWorkspaceRepository(settings.db_path).get_or_default(task.id)
    calls: list = []

    _repeat_join_confirmation(settings, task, dataset, registry, calls)

    assert len(calls) == 1
    after = DataWorkspaceRepository(settings.db_path).get_or_default(task.id)
    assert after.revision == before.revision
    assert after.semantic_mapping == before.semantic_mapping
    assert after.semantic_mapping.risk_directions["loan_amount"].direction == (
        "higher_is_riskier"
    )


def test_repeat_join_confirmation_still_rejects_a_role_change(tmp_path) -> None:
    client, settings, task, dataset, registry = _joined_workspace(tmp_path)
    _confirm_direction(settings, task.id)
    snapshot = DataWorkspaceRepository(settings.db_path).get_or_default(task.id)
    changed = _workspace_draft(
        dataset,
        target_col="bad_flag",
        field_roles={"bad_flag": "target", "mobile": "phone"},
    )
    assert (
        _put_workspace(client, task.id, changed, revision=snapshot.revision).status_code
        == 200
    )
    calls: list = []

    with pytest.raises(JoinSetupError, match="语义映射已变化"):
        _repeat_join_confirmation(settings, task, dataset, registry, calls)

    assert calls == []
