from copy import deepcopy
import json

import pandas as pd
import pytest

from marvis.agent.auto_drive import decide_gate
from marvis.agent.join_setup import build_join_proposal
from marvis.agent.turn_handlers.c1_state import _c1_state_from_proposal
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db_schema import init_db
from marvis.domain import TaskCreate
from marvis.llm_outbound import OutboundContextError
from marvis.llm_outbound_role_assignment import (
    ApprovedRoleAssignment,
    approved_role_assignment,
)
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.tasks import TaskRepository
from marvis.settings import build_settings
from tests.test_llm_outbound import Wire


def _fixture(tmp_path, *, wide=False):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    tasks = TaskRepository(settings.db_path)
    task = tasks.create_task(
        TaskCreate(
            model_name="roles",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path),
            task_type="data_join",
        )
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    sample_path = tmp_path / "PRIVATE_FILENAME_sample.parquet"
    frame = {"id": range(8), "y": [0, 1] * 4}
    if wide:
        frame.update({f"field_{i}": range(8) for i in range(305)})
    pd.DataFrame(frame).to_parquet(sample_path, index=False)
    registry.register_from_upload(task.id, sample_path, role="sample")
    features_path = tmp_path / "PRIVATE_FILENAME_features.parquet"
    pd.DataFrame(
        {"id": range(8), "balance": range(8), "record": ["PRIVATE_RECORD"] * 8}
    ).to_parquet(features_path, index=False)
    registry.register_from_upload(task.id, features_path, role="feature")
    c1 = _c1_state_from_proposal(build_join_proposal(registry, task.id, None))
    metadata = {"join_c1": c1, "tables": [{"rows": [["PRIVATE_RECORD"]]}]}
    tasks.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content="PRIVATE_DISPLAY",
        metadata=metadata,
    )
    return (
        settings,
        tasks,
        task,
        registry,
        {"content": "PRIVATE_DISPLAY", "metadata": metadata},
    )


@pytest.mark.parametrize("wide", [False, True])
def test_registered_role_gate_sends_exact_ids_and_complete_schema_without_data(
    monkeypatch, tmp_path, wide
):
    settings, tasks, task, registry, gate = _fixture(tmp_path, wide=wide)
    capability = approved_role_assignment(
        tasks, settings, task_id=task.id, gate_metadata=gate["metadata"]
    )
    wire = Wire(monkeypatch)
    wire.reply = {
        "action": "confirm",
        "reason": "接受已绑定的注册文件角色",
        "confidence": 0.9,
    }
    assert (
        decide_gate(wire.client, gate=gate, role_assignment=capability)["action"]
        == "confirm"
    )
    sent = wire.last_context()["role_assignment"]
    assert sent["purpose"] == "accept_registered_file_roles"
    assert sent["anchor_id"] == gate["metadata"]["join_c1"]["anchor_id"]
    assert sent["feature_ids"] == gate["metadata"]["join_c1"]["feature_ids"]
    assert sent["target_col"] == "y"
    assert "PRIVATE_" not in json.dumps(wire.requests)
    for row in sent["registered_files"]:
        registered = registry.get(row["dataset_id"])
        assert row["content_hash"] == registered.content_hash
        assert row["columns"] == [column.name for column in registered.columns]
    if wide:
        assert max(row["column_count"] for row in sent["registered_files"]) == 307
        assert (
            max(len(row["columns"]) for row in gate["metadata"]["join_c1"]["files"])
            == 300
        )


def test_unbound_changed_card_registry_or_bytes_refuses_before_protocol(
    monkeypatch, tmp_path
):
    settings, tasks, task, registry, gate = _fixture(tmp_path)
    wire = Wire(monkeypatch)
    bad = deepcopy(gate["metadata"])
    bad["join_c1"]["files"][0]["row_count"] += 1
    with pytest.raises(OutboundContextError):
        approved_role_assignment(tasks, settings, task_id=task.id, gate_metadata=bad)
    tasks.add_agent_message(
        task.id, role="assistant", stage="chat", content="changed", metadata=bad
    )
    with pytest.raises(OutboundContextError):
        approved_role_assignment(tasks, settings, task_id=task.id, gate_metadata=bad)
    for value in [
        {"registered_files": []},
        ApprovedRoleAssignment('{"private":"PRIVATE_BODY"}', object()),
    ]:
        with pytest.raises(OutboundContextError):
            decide_gate(wire.client, gate=gate, role_assignment=value)
    tasks.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content="original",
        metadata=gate["metadata"],
    )
    path = registry.resolve_path(gate["metadata"]["join_c1"]["anchor_id"])
    path.write_bytes(b"changed-synthetic-source")
    with pytest.raises(OutboundContextError, match="内容已变化"):
        approved_role_assignment(
            tasks, settings, task_id=task.id, gate_metadata=gate["metadata"]
        )
    assert wire.requests == []
