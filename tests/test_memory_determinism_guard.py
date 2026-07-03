from dataclasses import asdict

from fastapi.testclient import TestClient
import pandas as pd

from marvis.agent_memory.distillation import new_distillation
from marvis.agent_memory.models import MemoryCandidate
from marvis.agent_memory.store import AgentMemoryStore
from marvis.app import create_app
from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate, TaskStatus
from marvis.validation.config import ValidationConfig
from marvis.validation.effectiveness import run_effectiveness


def test_agent_memory_distillations_do_not_change_deterministic_effectiveness(tmp_path):
    sample = pd.DataFrame(
        [
            {"split": "train", "month": "202601", "score": 0.10, "target": 0, "x": 1.0},
            {"split": "train", "month": "202601", "score": 0.20, "target": 0, "x": 1.0},
            {"split": "train", "month": "202601", "score": 0.80, "target": 1, "x": 2.0},
            {"split": "train", "month": "202601", "score": 0.90, "target": 1, "x": 2.0},
            {"split": "test", "month": "202602", "score": 0.15, "target": 0, "x": 1.0},
            {"split": "test", "month": "202602", "score": 0.25, "target": 0, "x": 1.0},
            {"split": "test", "month": "202602", "score": 0.75, "target": 1, "x": 2.0},
            {"split": "test", "month": "202602", "score": 0.85, "target": 1, "x": 2.0},
            {"split": "oot", "month": "202603", "score": 0.12, "target": 0, "x": 1.0},
            {"split": "oot", "month": "202603", "score": 0.22, "target": 0, "x": 1.0},
            {"split": "oot", "month": "202603", "score": 0.78, "target": 1, "x": 2.0},
            {"split": "oot", "month": "202603", "score": 0.88, "target": 1, "x": 2.0},
        ]
    )
    config = ValidationConfig(
        target_col="target",
        score_col="score",
        split_col="split",
        time_col="month",
        feature_columns=["x"],
        bin_count=2,
    )
    before = asdict(run_effectiveness(sample=sample, config=config))

    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    memory = store.create(
        MemoryCandidate(
            memory_type="field_convention",
            summary="目标字段常见取值包括 target。",
            payload={"target_col": "target"},
            confidence="high",
        )
    )
    store.create_distillation(
        new_distillation(
            category="field_convention",
            scope_key="field_convention:target_col",
            distilled_summary="目标字段常见取值包括 target。",
            structured={"fields": {"target_col": ["target"]}},
            source_memory_ids=(memory.id,),
            support_count=4,
        )
    )

    after = asdict(run_effectiveness(sample=sample, config=config))

    assert after == before


def test_metrics_endpoint_does_not_pass_agent_memory_context_to_validation_stage(
    tmp_path,
    monkeypatch,
):
    client = TestClient(create_app(tmp_path))
    db_path = tmp_path / "marvis.sqlite"
    store = AgentMemoryStore(db_path)
    store.create_distillation(
        new_distillation(
            category="field_convention",
            scope_key="field_convention:target_col",
            distilled_summary="A卡坏样本字段常见取值包括 bad_flag。",
            structured={"fields": {"target_col": ["bad_flag"]}},
            source_memory_ids=("mem-a", "mem-b", "mem-c", "mem-d"),
            support_count=4,
        )
    )
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="A卡模型",
            model_version="V2026",
            validator="qa",
            source_dir=str(tmp_path),
            algorithm="lgb",
            target_col="bad_flag",
            score_col="score",
            split_col="split",
            time_col="apply_month",
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned")
    repo.update_status(task.id, TaskStatus.RUNNING, "running")
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed")
    calls = []

    def fake_run_metrics_stage(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "marvis.routers.validation_stages.run_metrics_stage",
        fake_run_metrics_stage,
    )

    response = client.post(f"/api/tasks/{task.id}/metrics")

    assert response.status_code == 202, response.text
    assert len(calls) == 1
    assert set(calls[0]) == {"task_id", "settings", "stage_claimed"}
    assert calls[0]["task_id"] == task.id
    assert calls[0]["stage_claimed"] is True
    assert "memory_context" not in calls[0]
