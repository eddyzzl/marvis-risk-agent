from pathlib import Path

from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import PluginRepository, TaskRepository


def _client(tmp_path: Path) -> TestClient:
    app = create_app(tmp_path)
    return TestClient(app)


def _create_task(client: TestClient, tmp_path: Path, *, model_name: str = "A卡") -> str:
    response = client.post(
        "/api/tasks",
        json={"model_name": model_name, "validator": "qa", "source_dir": str(tmp_path)},
    )
    assert response.status_code == 200
    return response.json()["id"]


def _write_audit(app, **kwargs) -> None:
    PluginRepository(app.state.settings.db_path).write_audit(**kwargs)


def test_list_audit_returns_empty_page_with_pagination_metadata(tmp_path: Path):
    client = _client(tmp_path)
    response = client.get("/api/audit")
    assert response.status_code == 200
    body = response.json()
    assert body == {"items": [], "total": 0, "limit": 100, "offset": 0, "has_more": False}


def test_list_audit_filters_by_exact_kind(tmp_path: Path):
    client = _client(tmp_path)
    app = client.app
    _write_audit(app, kind="plan.create", target_ref="plan-1", outcome="succeeded", detail={})
    _write_audit(app, kind="plan.status", target_ref="plan-1", outcome="succeeded", detail={})

    response = client.get("/api/audit", params={"kind": "plan.create"})
    assert response.status_code == 200
    items = response.json()["items"]
    assert [row["kind"] for row in items] == ["plan.create"]


def test_list_audit_filters_by_kind_prefix(tmp_path: Path):
    client = _client(tmp_path)
    app = client.app
    _write_audit(app, kind="plan.create", target_ref="plan-1", outcome="succeeded", detail={})
    _write_audit(app, kind="plan.status", target_ref="plan-1", outcome="succeeded", detail={})
    _write_audit(app, kind="job.heartbeat_lost", target_ref="job-1", outcome="failed", detail={})

    response = client.get("/api/audit", params={"kind_prefix": "plan."})
    assert response.status_code == 200
    kinds = sorted(row["kind"] for row in response.json()["items"])
    assert kinds == ["plan.create", "plan.status"]


def test_list_audit_kind_prefix_escapes_sql_wildcards(tmp_path: Path):
    client = _client(tmp_path)
    app = client.app
    _write_audit(app, kind="plan.create", target_ref="plan-1", outcome="succeeded", detail={})
    _write_audit(app, kind="planXcreate", target_ref="plan-1", outcome="succeeded", detail={})

    response = client.get("/api/audit", params={"kind_prefix": "plan."})
    kinds = [row["kind"] for row in response.json()["items"]]
    # "." in the LIKE pattern must match a literal dot, not the SQL "any char"
    # wildcard -- otherwise "planXcreate" would incorrectly match too.
    assert kinds == ["plan.create"]


def test_list_audit_filters_by_target_ref(tmp_path: Path):
    client = _client(tmp_path)
    app = client.app
    _write_audit(app, kind="plan.create", target_ref="plan-1", outcome="succeeded", detail={})
    _write_audit(app, kind="plan.create", target_ref="plan-2", outcome="succeeded", detail={})

    response = client.get("/api/audit", params={"target_ref": "plan-1"})
    items = response.json()["items"]
    assert [row["target_ref"] for row in items] == ["plan-1"]


def test_list_audit_filters_by_target_ref_prefix(tmp_path: Path):
    client = _client(tmp_path)
    app = client.app
    _write_audit(app, kind="plan.step.confirm", target_ref="step-1", outcome="succeeded", detail={})
    _write_audit(app, kind="plan.step.confirm", target_ref="step-10", outcome="succeeded", detail={})
    _write_audit(app, kind="plan.step.confirm", target_ref="other", outcome="succeeded", detail={})

    response = client.get("/api/audit", params={"target_ref_prefix": "step-1"})
    target_refs = sorted(row["target_ref"] for row in response.json()["items"])
    assert target_refs == ["step-1", "step-10"]


def test_list_audit_filters_by_time_range(tmp_path: Path):
    client = _client(tmp_path)
    app = client.app
    repo = PluginRepository(app.state.settings.db_path)
    from marvis.db_schema import connect

    with connect(app.state.settings.db_path) as conn:
        from marvis.repositories.audit import _write_audit_row

        _write_audit_row(
            conn, kind="plan.create", target_ref="early", outcome="succeeded", detail={}
        )
    with connect(app.state.settings.db_path) as conn:
        conn.execute("UPDATE audit SET at = ? WHERE target_ref = ?", ("2020-01-01T00:00:00", "early"))
    repo.write_audit(kind="plan.create", target_ref="late", outcome="succeeded", detail={})
    with connect(app.state.settings.db_path) as conn:
        conn.execute("UPDATE audit SET at = ? WHERE target_ref = ?", ("2030-01-01T00:00:00", "late"))

    after_response = client.get("/api/audit", params={"after": "2025-01-01T00:00:00"})
    assert [row["target_ref"] for row in after_response.json()["items"]] == ["late"]

    before_response = client.get("/api/audit", params={"before": "2025-01-01T00:00:00"})
    assert [row["target_ref"] for row in before_response.json()["items"]] == ["early"]


def test_list_audit_supports_limit_offset_and_reports_has_more(tmp_path: Path):
    client = _client(tmp_path)
    app = client.app
    for index in range(3):
        _write_audit(
            app,
            kind="plan.create",
            target_ref=f"plan-{index}",
            outcome="succeeded",
            detail={},
        )

    first_page = client.get("/api/audit", params={"limit": 2, "offset": 0})
    body = first_page.json()
    assert len(body["items"]) == 2
    assert body["total"] == 3
    assert body["has_more"] is True

    second_page = client.get("/api/audit", params={"limit": 2, "offset": 2})
    body2 = second_page.json()
    assert len(body2["items"]) == 1
    assert body2["total"] == 3
    assert body2["has_more"] is False


def test_task_audit_endpoint_matches_target_ref_and_detail_task_id(tmp_path: Path):
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    app = client.app
    # target_ref IS the task_id (e.g. task.create / report.agent_conclusions.confirm).
    _write_audit(app, kind="task.create", target_ref=task_id, outcome="succeeded", detail={})
    # target_ref is some other entity id, but detail_json embeds task_id (e.g. job.*,
    # dataset.*, strategy.* kinds).
    _write_audit(
        app,
        kind="job.heartbeat_lost",
        target_ref="job-xyz",
        outcome="failed",
        detail={"task_id": task_id},
    )
    # unrelated audit row for a different task must not leak in.
    _write_audit(
        app,
        kind="plan.create",
        target_ref="plan-other",
        outcome="succeeded",
        detail={"task_id": "some-other-task"},
    )

    response = client.get(f"/api/tasks/{task_id}/audit")
    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == task_id
    assert body["total"] == 2
    kinds = sorted(row["kind"] for row in body["items"])
    assert kinds == ["job.heartbeat_lost", "task.create"]


def test_task_audit_endpoint_supports_kind_filter(tmp_path: Path):
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    app = client.app
    _write_audit(app, kind="task.create", target_ref=task_id, outcome="succeeded", detail={})
    _write_audit(
        app,
        kind="job.heartbeat_lost",
        target_ref="job-xyz",
        outcome="failed",
        detail={"task_id": task_id},
    )

    response = client.get(f"/api/tasks/{task_id}/audit", params={"kind": "task.create"})
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["kind"] == "task.create"


def test_task_audit_endpoint_404s_for_unknown_task(tmp_path: Path):
    client = _client(tmp_path)
    response = client.get("/api/tasks/does-not-exist/audit")
    assert response.status_code == 404


def test_audit_export_streams_csv_with_all_fields(tmp_path: Path):
    client = _client(tmp_path)
    app = client.app
    _write_audit(
        app,
        kind="plan.create",
        target_ref="plan-1",
        actor="tester",
        inputs_hash="abc123",
        outcome="succeeded",
        detail={"task_id": "task-1", "step_count": 3},
    )

    response = client.get("/api/audit/export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment;" in response.headers["content-disposition"]

    lines = response.text.strip("\n").split("\r\n") if "\r\n" in response.text else response.text.strip("\n").split("\n")
    header = lines[0].split(",")
    assert header == [
        "id",
        "kind",
        "actor",
        "target_ref",
        "inputs_hash",
        "outcome",
        "detail",
        "at",
    ]
    assert "plan.create" in lines[1]
    assert "tester" in lines[1]
    assert "abc123" in lines[1]
    assert "task_id" in lines[1]


def test_audit_export_respects_filters(tmp_path: Path):
    client = _client(tmp_path)
    app = client.app
    _write_audit(app, kind="plan.create", target_ref="plan-1", outcome="succeeded", detail={})
    _write_audit(app, kind="job.heartbeat_lost", target_ref="job-1", outcome="failed", detail={})

    response = client.get("/api/audit/export", params={"kind": "plan.create"})
    assert response.status_code == 200
    assert "plan.create" in response.text
    assert "job.heartbeat_lost" not in response.text


def test_audit_export_filename_includes_task_id_when_scoped(tmp_path: Path):
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)

    response = client.get("/api/audit/export", params={"task_id": task_id})
    assert response.status_code == 200
    assert task_id in response.headers["content-disposition"]


def test_count_audit_matches_list_audit_total_across_filters(tmp_path: Path):
    db_path = tmp_path / "custom.sqlite"
    from marvis.db_schema import init_db

    init_db(db_path)
    repo = TaskRepository(db_path)
    for index in range(5):
        repo.list_audit()  # no-op sanity call to confirm the method exists on TaskRepository
    plugin_repo = PluginRepository(db_path)
    for index in range(5):
        plugin_repo.write_audit(
            kind="plan.create" if index % 2 == 0 else "plan.status",
            target_ref=f"plan-{index}",
            outcome="succeeded",
            detail={},
        )

    all_rows = repo.list_audit()
    assert repo.count_audit() == len(all_rows) == 5

    create_rows = repo.list_audit(kind="plan.create")
    assert repo.count_audit(kind="plan.create") == len(create_rows) == 3

    limited = repo.list_audit(kind="plan.create", limit=1, offset=1)
    assert len(limited) == 1
    assert repo.count_audit(kind="plan.create") == 3
