from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from marvis.app import create_app
from marvis.db import TaskRepository, connect
from marvis.domain import TaskStatus
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


def test_validation_create_dialog_can_add_models():
    index_html = Path("marvis/static/index.html").read_text(encoding="utf-8")
    assert 'id="welcomeValidationBatchCard"' not in index_html
    assert 'id="addValidationModelRowButton"' in index_html


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


@pytest.mark.parametrize("workflow_version", [1, 2])
def test_report_stage_allows_succeeded_regeneration(
    pipeline_settings, ready_validation_task, workflow_version,
):
    """A completed manual task can render edited text through the real API/job/writers."""
    repo = TaskRepository(pipeline_settings.db_path)
    task_id = ready_validation_task.id
    if workflow_version == 1:
        with connect(pipeline_settings.db_path) as conn:
            conn.execute("UPDATE tasks SET validation_workflow_version = 1 WHERE id = ?", (task_id,))
    for status in (
        TaskStatus.RUNNING, TaskStatus.EXECUTED, TaskStatus.COMPUTING_METRICS,
        TaskStatus.WRITING_ARTIFACTS, TaskStatus.SUCCEEDED,
    ):
        repo.update_status(task_id, status, "fixture completed before regeneration")
    outputs = pipeline_settings.workspace / "tasks" / task_id / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    fixture = Path(__file__).parent / "fixtures" / "legacy_validation_results.json"
    (outputs / "validation_results.json").write_bytes(fixture.read_bytes())

    # Always use this fixture's workspace template, never the bundled fallback.
    template = Document()
    template.add_paragraph("{{TEXT:model_overview}}")
    template.save(pipeline_settings.report_template_path)
    client = TestClient(create_app(pipeline_settings.workspace))
    for revision, text in enumerate(("First saved overview", "Updated saved overview")):
        repo.update_report_values(task_id, {"TEXT:model_overview": text}, revision)
        response = client.post(f"/api/tasks/{task_id}/report")

        assert response.status_code == 202, response.text
        assert repo.get_task(task_id).status is TaskStatus.SUCCEEDED
        with connect(pipeline_settings.db_path) as conn:
            jobs = conn.execute("SELECT kind, status FROM jobs WHERE task_id = ?", (task_id,)).fetchall()
        assert len(jobs) == revision + 1
        assert all(job["kind"] == "report" and job["status"] == "succeeded" for job in jobs)
        paragraphs = [paragraph.text for paragraph in Document(outputs / "validation_report.docx").paragraphs]
        assert text in paragraphs
        if revision:
            assert "First saved overview" not in paragraphs
        workbook = load_workbook(outputs / "validation.xlsx", read_only=True)
        try:
            assert workbook.sheetnames
        finally:
            workbook.close()
