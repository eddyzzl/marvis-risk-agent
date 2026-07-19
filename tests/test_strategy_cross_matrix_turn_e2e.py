"""Natural language -> exact evidence -> complete 2D Cross Matrix vertical."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.repositories.task_artifacts import TaskArtifactRepository


class _CrossMatrixLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "cross_matrix_analysis",
                "workflow_inputs": {
                    "x_feature": "age",
                    "x_method": "equal_width",
                    "y_feature": "score",
                    "y_method": "equal_width",
                    "bin_count": 3,
                    "min_bin_pct": 0.02,
                    "loan_amount_col": "loan_amount",
                    "overdue_amount_col": "overdue_amount",
                    "sentinel_values": [],
                },
            },
            ensure_ascii=False,
        )


def _task(client: TestClient, tmp_path: Path, *, one_nan_label: bool) -> str:
    source = tmp_path / ("cross-nan-source" if one_nan_label else "cross-source")
    source.mkdir(parents=True)
    labels: list[float | int] = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]
    if one_nan_label:
        labels[3] = float("nan")
    pd.DataFrame(
        {
            "age": [20, 22, 24, 26, 40, 42, 44, 46, 60, 62, 64, 66],
            "score": [100, 110, 300, 310, 120, 130, 320, 330, 140, 150, 340, 350],
            "loan_amount": [
                100,
                120,
                140,
                160,
                180,
                200,
                220,
                240,
                260,
                280,
                300,
                320,
            ],
            "overdue_amount": [0, 0, 3, 5, 0, 10, 0, 15, 20, 25, 30, 40],
            "bad": labels,
        }
    ).to_csv(source / "sample.csv", index=False)
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "自然语言二维 Cross Matrix",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _utterance() -> str:
    return (
        "构建 age 等距 3 箱 × score 等距 3 箱的二维交叉矩阵，"
        "放款金额列 loan_amount，逾期金额列 overdue_amount"
    )


def _install_llm(monkeypatch: pytest.MonkeyPatch, llm: _CrossMatrixLLM) -> None:
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_cross_matrix_consumes_exact_first_step_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path, one_nan_label=False)
    llm = _CrossMatrixLLM()
    _install_llm(monkeypatch, llm)

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _utterance()},
    )

    assert opened.status_code == 202, opened.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_cross_matrix_analysis"
    ]
    assert plans[0]["status"] == "done"
    assert [step["status"] for step in plans[0]["steps"]] == ["done", "done"]
    assert all(step["needs_confirmation"] is False for step in plans[0]["steps"])

    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    first = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    cross = client.app.state.plan_repo.load_step_output(stored.steps[1].id)
    source_json = next(
        artifact
        for artifact in first["artifacts"]
        if artifact["kind"] == "strategy_candidate_json"
    )
    assert cross["parent_candidate_id"] == first["candidate_id"]
    assert cross["parent_evidence_hash"] == first["evidence_hash"]
    assert cross["candidate_id"] == cross["cross_matrix_candidate"][
        "candidate_evidence"
    ]["candidate_id"]
    assert cross["evidence_hash"] == cross["cross_matrix_candidate"][
        "candidate_evidence"
    ]["evidence_hash"]
    assert cross["dataset_id"] == first["candidate_evidence"]["identity"]["dataset_id"]
    assert cross["row_axis"] == {
        "feature": "age",
        "method": "equal_width",
        "bin_count": 3,
    }
    assert cross["column_axis"] == {
        "feature": "score",
        "method": "equal_width",
        "bin_count": 3,
    }
    assert cross["cell_count"] == 9
    assert cross["not_selected"] is True
    assert cross["not_admitted"] is True
    assert cross["not_applied"] is True
    assert cross["not_adopted"] is True
    assert cross["not_deployed"] is True

    cross_artifact = cross["artifacts"][0]
    record = TaskArtifactRepository(client.app.state.settings.db_path).get_for_task(
        task_id,
        cross_artifact["artifact_id"],
    )
    assert record is not None
    assert record["provenance"]["source_artifact_id"] == source_json["artifact_id"]
    assert record["provenance"]["source_artifact_content_hash"] == source_json[
        "content_hash"
    ]
    assert record["provenance"]["parent_candidate_id"] == first["candidate_id"]
    assert record["provenance"]["parent_evidence_hash"] == first["evidence_hash"]
    assert record["provenance"]["candidate_id"] == cross["candidate_id"]
    assert record["provenance"]["evidence_hash"] == cross["evidence_hash"]

    assistant_text = "\n".join(
        message.get("content", "")
        for message in opened.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert "二维 Cross Matrix 候选构建完成" in assistant_text
    assert "绑定样本观测（未独立验证）" in assistant_text
    assert "未选择格子、未入池、未应用写回、未采纳、未部署" in assistant_text
    assert len(llm.calls) == 1


@pytest.mark.slow
@pytest.mark.e2e
def test_cross_matrix_nan_label_consent_resumes_same_two_step_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path, one_nan_label=True)
    llm = _CrossMatrixLLM()
    _install_llm(monkeypatch, llm)

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _utterance()},
    )

    assert opened.status_code == 202, opened.text
    assert opened.json()["status"] == "clarification_required"
    assert opened.json()["code"] == "strategy_drop_nan_labels_confirmation_required"
    assert opened.json()["label_quality"] == {
        "target_col": "bad",
        "n_total": 12,
        "n_nan": 1,
    }
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []

    resumed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认丢弃空标签并继续"},
    )

    assert resumed.status_code == 202, resumed.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_cross_matrix_analysis"
    ]
    assert plans[0]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    first = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    cross = client.app.state.plan_repo.load_step_output(stored.steps[1].id)
    assert first["nan_labels_dropped"] == 1
    assert cross["population_count"] == 12
    assert cross["labeled_count"] == 11
    assert cross["drop_nan_labels"] is True
    assert cross["nan_labels_dropped"] == 1
    assert len(llm.calls) == 1
