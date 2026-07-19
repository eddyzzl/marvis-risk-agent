from urllib.parse import quote

from docx import Document
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import StrategyRepository, TaskRepository, connect
from marvis.domain import TaskCreate
from marvis.files import sha256_file
from marvis.packs.strategy.contracts import Strategy, StrategyRule
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.routers.artifacts import router as artifacts_router


def test_artifact_routes_are_served_from_dedicated_router():
    routes = {
        (route.path, tuple(sorted(route.methods or []))): route.endpoint.__module__
        for route in artifacts_router.routes
    }

    assert routes[("/api/artifacts/{artifact_path:path}/preview", ("GET",))] == (
        "marvis.routers.artifacts"
    )
    assert routes[("/api/artifacts/{artifact_path:path}", ("GET",))] == (
        "marvis.routers.artifacts"
    )
    assert routes[("/api/tasks/{task_id}/strategy-artifacts", ("GET",))] == (
        "marvis.routers.artifacts"
    )
    assert routes[
        (
            "/api/tasks/{task_id}/strategy-artifacts/{artifact_id}/download",
            ("GET",),
        )
    ] == "marvis.routers.artifacts"
    assert routes[("/api/tasks/{task_id}/task-artifacts", ("GET",))] == (
        "marvis.routers.artifacts"
    )
    assert routes[
        (
            "/api/tasks/{task_id}/task-artifacts/{artifact_id}/download",
            ("GET",),
        )
    ] == "marvis.routers.artifacts"


def test_artifact_api_serves_workspace_task_artifact_by_relative_path(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="旧验证任务",
            model_version="v1",
            validator="owner",
            source_dir=str(tmp_path),
        )
    )
    artifact = tmp_path / "tasks" / task.id / "outputs" / "validation_results.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"ks":0.31}', encoding="utf-8")

    response = client.get(
        f"/api/artifacts/tasks%2F{task.id}%2Foutputs%2Fvalidation_results.json"
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


def test_artifact_api_rejects_cross_task_parent_traversal(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    first = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="路径来源任务",
            model_version="v1",
            validator="owner",
            source_dir=str(tmp_path),
        )
    )
    second = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="路径目标任务",
            model_version="v1",
            validator="owner",
            source_dir=str(tmp_path),
        )
    )
    artifact = app.state.settings.tasks_dir / second.id / "outputs" / "secret.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"secret":true}', encoding="utf-8")
    traversal = f"tasks/{first.id}/../{second.id}/outputs/secret.json"

    response = client.get(f"/api/artifacts/{quote(traversal, safe='')}")

    assert first.id != second.id
    assert response.status_code == 404


def test_artifact_api_previews_docx_task_artifact(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="旧报告任务",
            model_version="v1",
            validator="owner",
            source_dir=str(tmp_path),
        )
    )
    artifact = tmp_path / "tasks" / task.id / "outputs" / "validation_report.docx"
    artifact.parent.mkdir(parents=True)
    document = Document()
    document.add_paragraph("Validation summary")
    document.save(artifact)

    response = client.get(
        f"/api/artifacts/tasks%2F{task.id}%2Foutputs%2Fvalidation_report.docx/preview"
    )

    assert response.status_code == 200
    assert "Validation summary" in response.text


def _seed_strategy_artifact(
    app,
    *,
    task_id: str | None = None,
    artifact_id: str = "strategy-artifact-1",
    path=None,
    kind: str = "strategy_doc_md",
    verified: bool = False,
    registered_content_size: int | None = None,
):
    task_repo = TaskRepository(app.state.settings.db_path)
    if task_id is None:
        task = task_repo.create_task(
            TaskCreate(
                model_name="策略任务",
                model_version="v1",
                validator="owner",
                source_dir=str(app.state.settings.workspace),
                task_type="strategy",
            )
        )
        task_id = task.id
    strategy = Strategy(
        id=f"strategy-{artifact_id}",
        strategy_type="approval",
        rules=(
            StrategyRule(
                condition="score >= 700",
                decision="approve",
                value=None,
            ),
        ),
        score_col="score",
        default_decision="review",
        description="candidate",
    )
    strategy_repo = StrategyRepository(app.state.settings.db_path)
    strategy_repo.create_strategy(task_id, strategy)
    with connect(app.state.settings.db_path) as conn:
        conn.execute(
            """
            UPDATE strategies
               SET status = 'adopted', asset_status = 'adopted_local'
             WHERE id = ?
            """,
            (strategy.id,),
        )
    if path is None:
        path = (
            app.state.settings.tasks_dir
            / task_id
            / "outputs"
            / "strategy_summary.md"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# 策略摘要", encoding="utf-8")
    if verified:
        record = strategy_repo.register_verified_strategy_artifact(
            strategy.id,
            kind=kind,
            path=str(path),
            content_hash=sha256_file(path),
            content_size=(
                path.stat().st_size
                if registered_content_size is None
                else registered_content_size
            ),
            provenance={
                "schema_version": "strategy-artifact-provenance.v1",
                "producer_version": "test.v1",
                "task_id": task_id,
                "strategy_id": strategy.id,
            },
        )
        artifact_id = str(record["id"])
    else:
        strategy_repo.save_strategy_artifact(
            strategy.id,
            kind=kind,
            path=str(path),
            artifact_id=artifact_id,
        )
    return task_id, strategy.id, artifact_id, path


def test_strategy_artifact_list_is_path_free_and_downloads_by_owned_id(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id, strategy_id, artifact_id, artifact = _seed_strategy_artifact(app)

    response = client.get(f"/api/tasks/{task_id}/strategy-artifacts")

    assert response.status_code == 200
    assert response.json() == {
        "task_id": task_id,
        "artifacts": [
            {
                "id": artifact_id,
                "kind": "strategy_doc_md",
                "filename": "strategy_summary.md",
                "strategy_id": strategy_id,
                "strategy_type": "approval",
                "version": 1,
                "asset_status": "adopted_local",
                "created_at": response.json()["artifacts"][0]["created_at"],
                "available": True,
                "download_url": (
                    f"/api/tasks/{task_id}/strategy-artifacts/"
                    f"{artifact_id}/download"
                ),
            }
        ],
    }
    assert "path" not in response.json()["artifacts"][0]
    assert str(artifact) not in response.text

    download = client.get(
        f"/api/tasks/{task_id}/strategy-artifacts/{artifact_id}/download"
    )
    assert download.status_code == 200
    assert download.content.decode("utf-8") == "# 策略摘要"
    assert download.headers["content-type"].startswith("text/markdown")


def test_strategy_artifact_download_requires_task_ownership(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    owner_task_id, _, artifact_id, _ = _seed_strategy_artifact(app)
    other_task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="其他策略任务",
            model_version="v1",
            validator="owner",
            source_dir=str(tmp_path),
            task_type="strategy",
        )
    )

    response = client.get(
        f"/api/tasks/{other_task.id}/strategy-artifacts/{artifact_id}/download"
    )

    assert owner_task_id != other_task.id
    assert response.status_code == 404


def test_strategy_artifact_download_rejects_outside_missing_and_forbidden_files(
    tmp_path,
):
    app = create_app(tmp_path)
    client = TestClient(app)
    secret = tmp_path / "secret.json"
    secret.write_text('{"secret":true}', encoding="utf-8")
    task_id, _, outside_id, _ = _seed_strategy_artifact(
        app,
        path=secret,
        artifact_id="outside",
    )
    forbidden = app.state.settings.tasks_dir / task_id / "outputs" / "report.xlsx"
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_bytes(b"xlsx")
    _seed_strategy_artifact(
        app,
        task_id=task_id,
        path=forbidden,
        artifact_id="forbidden",
    )
    missing = app.state.settings.tasks_dir / task_id / "outputs" / "missing.csv"
    _seed_strategy_artifact(
        app,
        task_id=task_id,
        path=missing,
        artifact_id="missing",
    )

    listed = client.get(f"/api/tasks/{task_id}/strategy-artifacts")
    availability = {
        item["id"]: item["available"] for item in listed.json()["artifacts"]
    }
    assert availability == {"forbidden": False, "missing": False, "outside": False}
    assert str(secret) not in listed.text
    for artifact_id in availability:
        response = client.get(
            f"/api/tasks/{task_id}/strategy-artifacts/{artifact_id}/download"
        )
        assert response.status_code == 404


def test_strategy_artifact_download_rejects_symlink(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="策略任务",
            model_version="v1",
            validator="owner",
            source_dir=str(tmp_path),
            task_type="strategy",
        )
    )
    outputs = app.state.settings.tasks_dir / task.id / "outputs"
    outputs.mkdir(parents=True)
    target = outputs / "real.json"
    target.write_text('{"ok":true}', encoding="utf-8")
    link = outputs / "linked.json"
    link.symlink_to(target)
    task_id, _, artifact_id, _ = _seed_strategy_artifact(
        app,
        task_id=task.id,
        path=link,
        artifact_id="symlink",
    )

    response = client.get(
        f"/api/tasks/{task_id}/strategy-artifacts/{artifact_id}/download"
    )

    assert response.status_code == 404


def test_verified_strategy_artifact_rejects_same_size_content_drift_everywhere(
    tmp_path,
):
    app = create_app(tmp_path)
    client = TestClient(app)
    task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="完整性校验任务",
            model_version="v1",
            validator="owner",
            source_dir=str(tmp_path),
            task_type="strategy",
        )
    )
    artifact = tmp_path / "tasks" / task.id / "outputs" / "strategy.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"original-bytes")
    task_id, _, artifact_id, _ = _seed_strategy_artifact(
        app,
        task_id=task.id,
        path=artifact,
        verified=True,
    )
    relative_path = artifact.relative_to(app.state.settings.workspace).as_posix()

    before_list = client.get(f"/api/tasks/{task_id}/strategy-artifacts")
    before_owned_download = client.get(
        f"/api/tasks/{task_id}/strategy-artifacts/{artifact_id}/download"
    )
    before_generic_download = client.get(
        f"/api/artifacts/{quote(relative_path, safe='')}"
    )

    assert before_list.json()["artifacts"][0]["available"] is True
    assert before_owned_download.content == b"original-bytes"
    assert before_generic_download.content == b"original-bytes"

    artifact.write_bytes(b"tampered-bytes")

    listed = client.get(f"/api/tasks/{task_id}/strategy-artifacts")
    owned_download = client.get(
        f"/api/tasks/{task_id}/strategy-artifacts/{artifact_id}/download"
    )
    generic_download = client.get(
        f"/api/artifacts/{quote(relative_path, safe='')}"
    )

    assert artifact.stat().st_size == len(b"original-bytes")
    assert listed.status_code == 200
    assert listed.json()["artifacts"][0]["available"] is False
    assert listed.json()["artifacts"][0]["download_url"] is None
    assert owned_download.status_code == 409
    assert owned_download.json()["detail"] == "strategy artifact integrity check failed"
    assert generic_download.status_code == 409
    assert generic_download.json()["detail"] == "artifact integrity check failed"
    assert owned_download.content != artifact.read_bytes()
    assert generic_download.content != artifact.read_bytes()


def test_verified_strategy_artifact_rejects_registered_size_drift(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="长度校验任务",
            model_version="v1",
            validator="owner",
            source_dir=str(tmp_path),
            task_type="strategy",
        )
    )
    artifact = tmp_path / "tasks" / task.id / "outputs" / "strategy.md"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"registered-content")
    task_id, _, artifact_id, _ = _seed_strategy_artifact(
        app,
        task_id=task.id,
        path=artifact,
        verified=True,
        registered_content_size=artifact.stat().st_size + 1,
    )
    relative_path = artifact.relative_to(app.state.settings.workspace).as_posix()

    listed = client.get(f"/api/tasks/{task_id}/strategy-artifacts")
    owned_download = client.get(
        f"/api/tasks/{task_id}/strategy-artifacts/{artifact_id}/download"
    )
    generic_download = client.get(
        f"/api/artifacts/{quote(relative_path, safe='')}"
    )

    assert listed.json()["artifacts"][0]["available"] is False
    assert owned_download.status_code == 409
    assert generic_download.status_code == 409


def test_verified_strategy_artifact_missing_path_remains_not_found(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id, _, artifact_id, artifact = _seed_strategy_artifact(
        app,
        verified=True,
    )
    artifact.unlink()

    listed = client.get(f"/api/tasks/{task_id}/strategy-artifacts")
    downloaded = client.get(
        f"/api/tasks/{task_id}/strategy-artifacts/{artifact_id}/download"
    )

    assert listed.json()["artifacts"][0]["available"] is False
    assert downloaded.status_code == 404


def test_verified_strategy_integrity_governs_legacy_alias_of_same_path(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id, strategy_id, verified_id, artifact = _seed_strategy_artifact(
        app,
        verified=True,
    )
    StrategyRepository(app.state.settings.db_path).save_strategy_artifact(
        strategy_id,
        kind="strategy_doc_md",
        path=str(artifact),
        artifact_id="legacy-alias",
    )

    before = client.get(f"/api/tasks/{task_id}/strategy-artifacts").json()[
        "artifacts"
    ]
    assert {row["id"]: row["available"] for row in before} == {
        verified_id: True,
        "legacy-alias": True,
    }

    artifact.write_text("# tampered strategy", encoding="utf-8")
    after = client.get(f"/api/tasks/{task_id}/strategy-artifacts").json()[
        "artifacts"
    ]

    assert {row["id"]: row["available"] for row in after} == {
        verified_id: False,
        "legacy-alias": False,
    }
    for artifact_id in (verified_id, "legacy-alias"):
        response = client.get(
            f"/api/tasks/{task_id}/strategy-artifacts/{artifact_id}/download"
        )
        assert response.status_code == 409
        assert response.json()["detail"] == (
            "strategy artifact integrity check failed"
        )
        assert response.content != artifact.read_bytes()


def test_task_registry_integrity_governs_legacy_strategy_alias(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id, _, artifact_id, artifact = _seed_strategy_artifact(app)
    repository = TaskArtifactRepository(app.state.settings.db_path)
    repository.register(
        task_id=task_id,
        kind="strategy_doc_copy",
        path=str(artifact),
        content_hash=sha256_file(artifact),
        origin_tool="strategy.copy",
        provenance={
            "schema_version": "task-artifact-provenance.v1",
            "producer_version": "strategy.copy.v1",
        },
    )
    artifact.write_text("# tampered through legacy alias", encoding="utf-8")

    listed = client.get(f"/api/tasks/{task_id}/strategy-artifacts")
    downloaded = client.get(
        f"/api/tasks/{task_id}/strategy-artifacts/{artifact_id}/download"
    )

    assert listed.json()["artifacts"][0]["available"] is False
    assert downloaded.status_code == 409
    assert downloaded.json()["detail"] == "strategy artifact integrity check failed"


def test_generic_artifact_download_rejects_cross_task_verified_registration(
    tmp_path,
):
    app = create_app(tmp_path)
    client = TestClient(app)
    owner = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="登记任务",
            model_version="v1",
            validator="owner",
            source_dir=str(tmp_path),
            task_type="strategy",
        )
    )
    path_task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="路径任务",
            model_version="v1",
            validator="owner",
            source_dir=str(tmp_path),
            task_type="strategy",
        )
    )
    artifact = (
        app.state.settings.tasks_dir
        / path_task.id
        / "outputs"
        / "cross-owned.md"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("cross-owned", encoding="utf-8")
    _seed_strategy_artifact(
        app,
        task_id=owner.id,
        path=artifact,
        verified=True,
    )
    relative_path = artifact.relative_to(app.state.settings.workspace).as_posix()

    response = client.get(f"/api/artifacts/{quote(relative_path, safe='')}")

    assert owner.id != path_task.id
    assert response.status_code == 409
    assert response.json()["detail"] == "artifact registry ownership mismatch"


def test_generic_legacy_artifact_requires_an_existing_path_task(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    owned_task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="旧产物任务",
            model_version="v1",
            validator="owner",
            source_dir=str(tmp_path),
            task_type="strategy",
        )
    )
    owned = app.state.settings.tasks_dir / owned_task.id / "outputs" / "legacy.json"
    owned.parent.mkdir(parents=True)
    owned.write_text('{"legacy":true}', encoding="utf-8")
    orphan = app.state.settings.tasks_dir / "missing-task" / "outputs" / "legacy.json"
    orphan.parent.mkdir(parents=True)
    orphan.write_text('{"orphan":true}', encoding="utf-8")

    owned_response = client.get(
        f"/api/artifacts/{quote(owned.relative_to(tmp_path).as_posix(), safe='')}"
    )
    orphan_response = client.get(
        f"/api/artifacts/{quote(orphan.relative_to(tmp_path).as_posix(), safe='')}"
    )

    assert owned_response.status_code == 200
    assert owned_response.json() == {"legacy": True}
    assert orphan_response.status_code == 404


def test_generic_registered_legacy_strategy_artifact_without_verified_alias_is_compatible(
    tmp_path,
):
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id, _, artifact_id, artifact = _seed_strategy_artifact(app)
    relative_path = artifact.relative_to(app.state.settings.workspace).as_posix()

    owned_download = client.get(
        f"/api/tasks/{task_id}/strategy-artifacts/{artifact_id}/download"
    )
    generic_download = client.get(
        f"/api/artifacts/{quote(relative_path, safe='')}"
    )

    assert owned_download.status_code == 200
    assert generic_download.status_code == 200
    assert generic_download.content == artifact.read_bytes()


def _seed_task_artifact(
    app,
    *,
    task_id: str | None = None,
    path=None,
    kind: str = "profit_csv",
):
    task_repo = TaskRepository(app.state.settings.db_path)
    if task_id is None:
        task = task_repo.create_task(
            TaskCreate(
                model_name="策略分析任务",
                model_version="v1",
                validator="owner",
                source_dir=str(app.state.settings.workspace),
                task_type="strategy",
            )
        )
        task_id = task.id
    if path is None:
        path = (
            app.state.settings.tasks_dir
            / task_id
            / "strategy_analysis"
            / "profit.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("segment,net_profit\nA,12.5\n", encoding="utf-8")
    content_hash = sha256_file(path) if path.is_file() else "0" * 64
    repository = TaskArtifactRepository(app.state.settings.db_path)
    with repository.transaction() as conn:
        record = repository.register_on_connection(
            conn,
            task_id=task_id,
            kind=kind,
            path=str(path),
            content_hash=content_hash,
            origin_tool="strategy.profit_calc",
            provenance={
                "schema_version": "task-artifact-provenance.v1",
                "producer_version": "strategy.profit_calc.v1",
                "source_dataset_content_hash": "1" * 64,
            },
        )
    return task_id, record, path


def test_task_artifact_list_is_path_free_and_downloads_by_owned_id(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id, record, artifact = _seed_task_artifact(app)

    listed = client.get(f"/api/tasks/{task_id}/task-artifacts")

    assert listed.status_code == 200
    assert listed.json() == {
        "task_id": task_id,
        "artifacts": [
            {
                "id": record["id"],
                "kind": "profit_csv",
                "filename": "profit.csv",
                "origin_tool": "strategy.profit_calc",
                "content_hash": record["content_hash"],
                "created_at": record["created_at"],
                "available": True,
                "download_url": (
                    f"/api/tasks/{task_id}/task-artifacts/{record['id']}/download"
                ),
            }
        ],
    }
    assert "path" not in listed.json()["artifacts"][0]
    assert "provenance" not in listed.json()["artifacts"][0]
    assert str(artifact) not in listed.text

    downloaded = client.get(
        f"/api/tasks/{task_id}/task-artifacts/{record['id']}/download"
    )
    relative_path = artifact.relative_to(app.state.settings.workspace).as_posix()
    generic_download = client.get(
        f"/api/artifacts/{quote(relative_path, safe='')}"
    )
    assert downloaded.status_code == 200
    assert downloaded.text == "segment,net_profit\nA,12.5\n"
    assert downloaded.headers["content-type"].startswith("text/csv")
    assert generic_download.status_code == 200
    assert generic_download.text == "segment,net_profit\nA,12.5\n"


def test_task_artifact_download_supports_xlsx_exports(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="数据导出任务",
            model_version="v1",
            validator="owner",
            source_dir=str(app.state.settings.workspace),
            task_type="strategy",
        )
    )
    artifact = (
        app.state.settings.tasks_dir
        / task.id
        / "data_exports"
        / "strategy_sample.xlsx"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"xlsx-export-evidence")
    task_id, record, _ = _seed_task_artifact(
        app,
        task_id=task.id,
        path=artifact,
        kind="dataset_export",
    )

    downloaded = client.get(
        f"/api/tasks/{task_id}/task-artifacts/{record['id']}/download"
    )

    assert downloaded.status_code == 200
    assert downloaded.content == b"xlsx-export-evidence"
    assert downloaded.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def test_task_artifact_download_rejects_content_drift(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task_id, record, artifact = _seed_task_artifact(app)
    artifact.write_text("segment,net_profit\nA,999\n", encoding="utf-8")

    listed = client.get(f"/api/tasks/{task_id}/task-artifacts")
    downloaded = client.get(
        f"/api/tasks/{task_id}/task-artifacts/{record['id']}/download"
    )
    relative_path = artifact.relative_to(app.state.settings.workspace).as_posix()
    generic_download = client.get(
        f"/api/artifacts/{quote(relative_path, safe='')}"
    )

    assert listed.status_code == 200
    assert listed.json()["artifacts"][0]["available"] is False
    assert listed.json()["artifacts"][0]["download_url"] is None
    assert downloaded.status_code == 409
    assert downloaded.json()["detail"] == "task artifact integrity check failed"
    assert generic_download.status_code == 409
    assert generic_download.json()["detail"] == "artifact integrity check failed"
    assert generic_download.content != artifact.read_bytes()


def test_task_artifact_download_is_task_owned_and_rejects_unsafe_file(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    owner_task_id, record, _ = _seed_task_artifact(app)
    other_task = TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="其他任务",
            model_version="v1",
            validator="owner",
            source_dir=str(tmp_path),
            task_type="strategy",
        )
    )
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret":true}', encoding="utf-8")
    _, outside_record, _ = _seed_task_artifact(
        app,
        task_id=owner_task_id,
        path=outside,
        kind="outside_json",
    )

    cross_task = client.get(
        f"/api/tasks/{other_task.id}/task-artifacts/{record['id']}/download"
    )
    unsafe = client.get(
        f"/api/tasks/{owner_task_id}/task-artifacts/"
        f"{outside_record['id']}/download"
    )

    assert cross_task.status_code == 404
    assert unsafe.status_code == 404
