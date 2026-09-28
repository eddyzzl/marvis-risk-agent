"""Generic file downloads must not downgrade source-scoped evidence permissions."""

import json
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd

from marvis.repositories.task_artifacts import TaskArtifactRepository
from test_reference_event_binding import build, event_runtime
import test_risk_source_runtime as sources


def generic_urls(settings, task_id, record):
    return (
        f"/api/tasks/{task_id}/task-artifacts/{record['id']}/download?expected_content_hash={record['content_hash']}",
        "/api/artifacts/"
        + Path(record["path"]).relative_to(settings.workspace).as_posix(),
    )


def assert_restricted(clients, settings, task_id, record, domain):
    urls = generic_urls(settings, task_id, record)
    for client in clients:
        for url in (*urls, urls[1] + "/preview"):
            response = client.get(url)
            assert response.status_code == 403, (url, response.status_code)
            assert (
                response.json()["detail"]["code"]
                == "source_scoped_artifact_requires_domain_route"
            )
            assert response.json()["detail"]["evidence_url"] == domain
        listed = client.get(f"/api/tasks/{task_id}/task-artifacts").json()["artifacts"]
        row = next(r for r in listed if r["id"] == record["id"])
        assert row["available"] is False and row["download_url"] is None


def test_real_event_receipt_requires_authorized_domain_for_all_download_aliases(
    tmp_path,
):
    rt = event_runtime.__wrapped__(tmp_path)
    _, reference, _, evidence = build(rt)
    repo = TaskArtifactRepository(rt.settings.db_path)
    record = repo.get_for_task(rt.task.id, evidence["artifact_id"])
    domain = (
        f"/api/tasks/{rt.task.id}/risk-events/requests/{reference.request_id}/evidence"
    )
    read = domain + "?grant_id=" + reference.grant_id
    anonymous = TestClient(rt.app)
    assert rt.maker.get(read).status_code == 200
    for client in (anonymous, rt.other):
        assert client.get(read).status_code == 403
    assert_restricted(
        [anonymous, rt.other, rt.maker], rt.settings, rt.task.id, record, domain
    )
    # A non-sensitive registry alias must not weaken the file's existing source scope.
    alias = repo.register(
        task_id=rt.task.id,
        kind="ordinary_report",
        path=record["path"],
        content_hash=record["content_hash"],
        origin_tool="test.alias",
        provenance={},
    )
    assert_restricted(
        [anonymous, rt.other, rt.maker], rt.settings, rt.task.id, alias, domain
    )
    assert (
        rt.admin.post(
            f"/api/tasks/{rt.task.id}/risk-events/grants/{reference.grant_id}/revoke"
        ).status_code
        == 200
    )
    assert rt.maker.get(read).status_code == 403
    assert_restricted([rt.maker], rt.settings, rt.task.id, record, domain)
    assert_restricted([rt.maker], rt.settings, rt.task.id, alias, domain)


def test_real_source_receipt_cannot_bypass_current_grant_through_files(tmp_path):
    rt = sources.runtime.__wrapped__(tmp_path)
    app, _, registry, task, maker, admin, other, *_ = rt
    grant = sources.authorize(rt)
    path = tmp_path / "historical-source.parquet"
    pd.DataFrame(
        [{"id": "one", "response": json.dumps(sources.source_record())}]
    ).to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="source_history")
    sources.prepare(
        rt,
        grant,
        historical={
            "dataset_id": dataset.id,
            "expected_content_hash": dataset.content_hash,
            "row_id_column": "id",
            "row_id": "one",
            "response_column": "response",
        },
    )
    result = sources.execute(rt)
    domain = result["evidence_url"]
    record = TaskArtifactRepository(app.state.settings.db_path).get_for_task(
        task.id, result["artifact_id"]
    )
    anonymous = TestClient(app)
    assert maker.get(domain).status_code == 200
    for client in (anonymous, other):
        assert client.get(domain).status_code == 403
    assert_restricted(
        [anonymous, other, maker], app.state.settings, task.id, record, domain
    )
    assert admin.post(f"/api/risk-sources/grants/{grant}/revoke").status_code == 200
    audit = maker.get(domain)
    assert audit.status_code == 200
    assert audit.json()["summary"]["status"] == "unauthorized"
    assert audit.json()["normalized_source"] is None
    assert audit.json()["assessment"] is None
    assert_restricted([maker], app.state.settings, task.id, record, domain)
