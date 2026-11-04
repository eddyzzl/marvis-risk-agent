"""Execution completion is separate from a validation's risk conclusions."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from docx import Document
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook

from marvis import pipeline
from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate, TaskStatus
from marvis.routers.tasks import router
from marvis.settings import build_settings


def _ready_task(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    repo = TaskRepository(settings.db_path)
    task = repo.create_task(TaskCreate(
        model_name="completion fixture", model_version="", validator="pytest",
        source_dir=str(tmp_path), run_mode="agent",
    ))
    for status in (
        TaskStatus.SCANNED, TaskStatus.RUNNING, TaskStatus.EXECUTED,
        TaskStatus.COMPUTING_METRICS, TaskStatus.WRITING_ARTIFACTS,
    ):
        repo.update_status(task.id, status, message="fixture stage complete")
    return settings, repo, repo.get_task(task.id)


def _write_outputs(outputs):
    outputs.mkdir(parents=True, exist_ok=True)
    document = Document()
    document.add_paragraph("{{TEXT:model_overview}}")
    document.save(outputs / "validation_report.docx")
    Workbook().save(outputs / "validation.xlsx")
    results_fixture = Path(__file__).parent / "fixtures/legacy_validation_results.json"
    (outputs / "validation_results.json").write_text(
        results_fixture.read_text(encoding="utf-8"), encoding="utf-8",
    )


@pytest.mark.parametrize("finding", ["pass", "review", "fail"])
@pytest.mark.parametrize("unfilled_metadata", [False, True])
def test_finalized_word_and_excel_complete_task_despite_findings_or_metadata(
    tmp_path, monkeypatch, finding, unfilled_metadata,
):
    settings, repo, task = _ready_task(tmp_path)
    monkeypatch.setattr(pipeline, "_load_validation_results", lambda _: SimpleNamespace(
        reproducibility=SimpleNamespace(summary=SimpleNamespace(status=finding)),
    ))

    def word_writer(_results, *, output_path, image_output_dir, **_kwargs):
        image_output_dir.mkdir(parents=True, exist_ok=True)
        document = Document()
        document.add_paragraph("{{TEXT:model_overview}}" if unfilled_metadata else "Done")
        document.save(output_path)
        return SimpleNamespace(unresolved_placeholders=(
            ["{{TEXT:model_overview}}"] if unfilled_metadata else []
        ))

    monkeypatch.setattr(pipeline, "write_validation_word", word_writer)
    def excel_writer(_results, path, *, image_output_dir, **_kwargs):
        image_output_dir.mkdir(parents=True, exist_ok=True)
        Workbook().save(path)

    monkeypatch.setattr(pipeline, "write_validation_excel", excel_writer)
    pipeline.run_report_stage(task_id=task.id, settings=pipeline.PipelineSettings(
        workspace=settings.workspace, db_path=settings.db_path,
        report_template_path=tmp_path / "unused.docx",
    ))

    assert repo.get_task(task.id).status is TaskStatus.SUCCEEDED
    outputs = settings.tasks_dir / task.id / "outputs"
    assert (outputs / "validation_report.docx").is_file()
    assert (outputs / "validation.xlsx").is_file()


@pytest.mark.parametrize("failed_artifact", ["word", "excel"])
def test_failed_artifact_does_not_complete_task(tmp_path, monkeypatch, failed_artifact):
    settings, repo, task = _ready_task(tmp_path)
    monkeypatch.setattr(pipeline, "_load_validation_results", lambda _: SimpleNamespace())

    def fail(*_args, **_kwargs):
        raise RuntimeError("fixture output failure")

    def word_writer(_results, *, output_path, **_kwargs):
        Document().save(output_path)
        return SimpleNamespace(unresolved_placeholders=[])

    monkeypatch.setattr(pipeline, "write_validation_word", fail if failed_artifact == "word" else word_writer)
    monkeypatch.setattr(pipeline, "write_validation_excel", fail)
    with pytest.raises(RuntimeError, match="fixture output failure"):
        pipeline.run_report_stage(task_id=task.id, settings=pipeline.PipelineSettings(
            workspace=settings.workspace, db_path=settings.db_path,
            report_template_path=tmp_path / "unused.docx",
        ))
    assert repo.get_task(task.id).status is TaskStatus.FAILED
    assert not (settings.tasks_dir / task.id / "outputs/validation_report.docx").exists()


@pytest.mark.parametrize("legacy_message", [
    "报告已生成，需人工复核", "验证已完成，需人工复核报告",
])
def test_legacy_finished_validation_is_projected_complete_without_rewriting_store(
    tmp_path, legacy_message,
):
    settings, repo, task = _ready_task(tmp_path)
    repo.update_status(task.id, TaskStatus.REVIEW_REQUIRED, message=legacy_message)
    values, revision = repo.get_report_values(task.id)
    values.update({
        "TEXT:pressure_test_summary": "fixture pressure summary",
        "TEXT:pressure_impact_recommendation": "fixture recommendation",
        "TEXT:final_validation_conclusion": "Review model findings before business use.",
    })
    repo.update_agent_report_conclusions(task.id, {key: values[key] for key in (
        "TEXT:pressure_test_summary", "TEXT:pressure_impact_recommendation",
        "TEXT:final_validation_conclusion",
    )}, revision)
    outputs = settings.tasks_dir / task.id / "outputs"
    _write_outputs(outputs)
    app = FastAPI()
    app.state.settings = settings
    app.include_router(router)
    client = TestClient(app)

    for payload in (client.get("/api/tasks").json()[0], client.get(f"/api/tasks/{task.id}").json()):
        assert payload["status"] == "succeeded"
        assert payload["status_message"] == "pipeline succeeded"
    assert repo.get_task(task.id).status is TaskStatus.REVIEW_REQUIRED


@pytest.mark.parametrize("incomplete", [
    "missing_word", "missing_excel", "corrupt_word", "corrupt_excel", "cancelled",
    "invalid_word_xml", "invalid_excel_xml", "xml_entities", "unconfirmed_conclusions", "pending_draft",
    "pending_generated", "pending_confirmed", "running", "pending_workflow",
])
def test_legacy_projection_preserves_unfinished_work(tmp_path, incomplete):
    from marvis.api_task_payloads import task_payload

    settings, repo, task = _ready_task(tmp_path)
    repo.update_status(task.id, TaskStatus.REVIEW_REQUIRED, message="报告已生成，需人工复核")
    task = repo.get_task(task.id)
    conclusions = {key: "fixture narrative" for key in (
        "TEXT:pressure_test_summary", "TEXT:pressure_impact_recommendation",
        "TEXT:final_validation_conclusion",
    )}
    if incomplete != "unconfirmed_conclusions":
        repo.update_agent_report_conclusions(task.id, conclusions, 0)
    outputs = settings.tasks_dir / task.id / "outputs"
    _write_outputs(outputs)
    if incomplete.startswith("missing_"):
        (outputs / ("validation_report.docx" if incomplete == "missing_word" else "validation.xlsx")).unlink()
    elif incomplete.startswith("corrupt_"):
        (outputs / ("validation_report.docx" if incomplete == "corrupt_word" else "validation.xlsx")).write_bytes(b"partial")
    elif incomplete.startswith("invalid_"):
        word = incomplete == "invalid_word_xml"
        with ZipFile(outputs / ("validation_report.docx" if word else "validation.xlsx"), "w") as archive:
            archive.writestr("word/document.xml" if word else "xl/workbook.xml", "not XML")
    elif incomplete == "xml_entities":
        with ZipFile(outputs / "validation_report.docx", "w") as archive:
            archive.writestr("word/document.xml", '<!DOCTYPE document [<!ENTITY value "untrusted">]>'
                '<document xmlns="http://schemas.openxmlformats.org/wordprocessingml/2006/main">&value;</document>')
            archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
    elif incomplete == "cancelled":
        task = replace(task, status_reason_code="user_cancelled")
    elif incomplete in {"pending_draft", "pending_generated", "pending_confirmed"}:
        stage = "word_conclusion_" + incomplete.removeprefix("pending_")
        repo.add_agent_message(task.id, role="assistant", stage=stage, content="fixture", metadata={"awaiting_confirmation": True})
    payload = task_payload(repo, task, settings.tasks_dir,
        active_job_kind="report" if incomplete == "running" else None,
        workflow_status="awaiting_confirm" if incomplete == "pending_workflow" else None,
    )
    assert payload["status"] == "review_required"
