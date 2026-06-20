from fastapi.testclient import TestClient
from docx import Document

from marvis.app import create_app


def test_artifact_api_serves_workspace_task_artifact_by_relative_path(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    artifact = tmp_path / "tasks" / "task-1" / "outputs" / "validation_results.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"ks":0.31}', encoding="utf-8")

    response = client.get(
        "/api/artifacts/tasks%2Ftask-1%2Foutputs%2Fvalidation_results.json"
    )

    assert response.status_code == 200
    assert response.text == '{"ks":0.31}'


def test_artifact_api_rejects_paths_outside_task_artifacts(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    secret = tmp_path / "secret.txt"
    secret.write_text("do-not-serve", encoding="utf-8")

    response = client.get("/api/artifacts/..%2Fsecret.txt")

    assert response.status_code == 404


def test_artifact_api_previews_docx_task_artifact(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    artifact = tmp_path / "tasks" / "task-1" / "outputs" / "validation_report.docx"
    artifact.parent.mkdir(parents=True)
    document = Document()
    document.add_paragraph("Validation summary")
    document.save(artifact)

    response = client.get(
        "/api/artifacts/tasks%2Ftask-1%2Foutputs%2Fvalidation_report.docx/preview"
    )

    assert response.status_code == 200
    assert "Validation summary" in response.text
