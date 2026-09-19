from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import TaskRepository
from marvis.validation_report_copy import METRIC_REWRITE_REFUSAL


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def test_agent_rejects_metric_rewrite_without_llm(tmp_path: Path):
    client = _client(tmp_path)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "自营T卡",
            "validator": "qa",
            "source_dir": str(tmp_path),
            "run_mode": "agent",
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "把KS改成0.50"},
    )

    assert response.status_code == 202, response.text
    messages = response.json()["messages"]
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == METRIC_REWRITE_REFUSAL
    assert messages[-1]["metadata"]["intent"] == "reject_metric_rewrite"
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    report_values, _revision = repo.get_report_values(task_id)
    assert "KS" not in " ".join(str(value) for value in report_values.values() if "0.50" in str(value))


def test_batch_runner_does_not_silently_write_word_conclusions():
    source = Path("marvis/validation_batch_runner.py").read_text(encoding="utf-8")
    start = source.index("def _run_ready_item_pipeline")
    end = source.index("\ndef _run_item_stage", start)
    body = source[start:end]
    assert "_write_report_conclusions" not in body
    assert "run_report_stage" not in body


def test_validation_create_dialog_can_add_models():
    index_html = Path("marvis/static/index.html").read_text(encoding="utf-8")
    create_js = Path("marvis/static/js/create-task-dialog.js").read_text(encoding="utf-8")
    app_js = Path("marvis/static/app.js").read_text(encoding="utf-8")

    assert 'id="welcomeValidationBatchCard"' not in index_html
    assert 'id="addValidationModelRowButton"' in index_html
    assert "function addValidationModelRow" in create_js
    assert "/api/validation-batches" in create_js
    assert "api/tasks" in create_js
    stepper = app_js.split("function renderWorkflowStepper", 1)[1].split(
        "function taskCreatedMonth",
        1,
    )[0]
    assert "classList.add(\"hidden\")" not in stepper
    assert "files: rows[0].files" not in create_js
    assert "classifyValidationBatchFiles" in create_js


def test_create_task_seeds_stored_report_values(tmp_path: Path):
    client = _client(tmp_path)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "自营渠道甲T卡",
            "validator": "验证专员A",
            "source_dir": str(tmp_path),
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]
    fields = client.get(f"/api/tasks/{task_id}/report-fields").json()
    stored = fields["stored_values"]
    assert stored["TEXT:drafter"] == "验证专员A"
    assert stored["TEXT:model_overview"] == ""
    assert fields["text_values"]["TEXT:model_overview"] == ""
    assert stored["TEXT:drafter"] == fields["text_values"]["TEXT:drafter"]
    assert "TEXT:drafter" in fields["display_defaults"]


def test_report_stage_allows_succeeded_regeneration():
    stages = Path("marvis/routers/validation_stages.py").read_text(encoding="utf-8")
    start = stages.index("def run_task_report")
    end = stages.index("\ndef validate_task", start)
    body = stages[start:end]
    assert "TaskStatus.SUCCEEDED" in body
    pipeline = Path("marvis/pipeline.py").read_text(encoding="utf-8")
    report_start = pipeline.index("def run_report_stage")
    report_body = pipeline[report_start:report_start + 2500]
    assert "TaskStatus.SUCCEEDED" in report_body
