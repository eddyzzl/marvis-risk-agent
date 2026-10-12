"""SQL Agent/manual HTTP vertical slice with real local source and registry."""
from copy import deepcopy
import json
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import duckdb
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.sql_ingest import SqlSourceSpec, SqlQueryTimeoutError, ingest_query
from marvis.data.sql_query import SqlQueryError, compile_select, parse_manual_select
from marvis.data.sql_sources import SqlSourceError, SqlSourceSnapshot, source_identity
from marvis.db import DatasetRepository, init_db
from marvis.db_schema import connect
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.data.workspace import DataWorkspaceDraft
from marvis.routers.plugins import ensure_plugin_admin_token
from marvis.routers.sql import router
from marvis.settings import build_settings
from tests.test_data_workspace_api import _create_task
from tests.test_llm_outbound import Wire


PRIVATE = "PRIVATE_SQL_SOURCE_RECORD_BODY_912"


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    ensure_plugin_admin_token(settings.plugin_admin_token_path)
    source = tmp_path / "private_source.sqlite"
    connection = sqlite3.connect(source)
    connection.execute("CREATE TABLE loans (channel TEXT, amount INTEGER, bad INTEGER, private_note TEXT)")
    connection.executemany("INSERT INTO loans VALUES (?,?,?,?)", [("app", 1000, 0, PRIVATE), ("app", 3000, 1, PRIVATE), ("branch", 2000, 0, PRIVATE)])
    connection.commit()
    connection.close()
    (settings.workspace / "sql_sources.json").write_text(json.dumps({"business source": {"backend": "sqlite", "path": str(source), "allowed_actor_ids": ["maker"]}}))
    task = _create_task(settings)
    app = FastAPI()
    app.state.settings = settings
    app.include_router(router)
    @app.middleware("http")
    async def test_actor(request, call_next):
        actor = request.headers.get("x-test-actor")
        request.state.private_client_authenticated = bool(actor)
        request.state.local_principal = SimpleNamespace(id=actor) if actor else None
        return await call_next(request)
    client = TestClient(app, headers={"x-test-actor": "maker"})
    return client, settings, task, source


def _proposal(client, task, **overrides):
    body = {"source_id": source_identity("business source"), "expected_revision": 0,
        "query": "SELECT channel, COUNT(*) AS n, AVG(amount) AS mean_amount FROM loans WHERE amount >= 2000 GROUP BY channel ORDER BY n DESC"}
    body.update(overrides)
    response = client.post(f"/api/tasks/{task.id}/sql/proposals", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _confirm(client, task, proposal):
    return client.post(f"/api/tasks/{task.id}/sql/confirm", json={"proposal_token": proposal["proposal_token"]})


def test_manual_sql_confirm_registers_hash_and_workspace_and_replays_without_query(monkeypatch, tmp_path):
    client, settings, task, _ = _runtime(tmp_path)
    sources = client.get(f"/api/tasks/{task.id}/sql/sources").json()
    assert sources["sources"][0]["status"] == "ready"
    assert "path" not in json.dumps(sources) and PRIVATE not in json.dumps(sources)
    before = DataWorkspaceRepository(settings.db_path).get_or_default(task.id)
    proposal = _proposal(client, task)
    assert proposal["execution_started"] is False
    assert DataWorkspaceRepository(settings.db_path).get_or_default(task.id) == before
    import marvis.agent.sql_import as sql_import
    original = sql_import._execute_query
    calls = []
    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(sql_import, "_execute_query", tracked)
    first = _confirm(client, task, proposal)
    assert first.status_code == 200, first.text
    payload = first.json()
    assert payload["status"] == "completed" and payload["workspace"]["revision"] == 1
    dataset = DatasetRepository(settings.db_path).get_dataset(payload["dataset"]["id"])
    assert dataset.task_id == task.id and dataset.content_hash == payload["evidence"]["content_hash"]
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(DatasetRepository(settings.db_path), backend, settings.datasets_dir)
    frame = backend.read_frame(registry.resolve_verified_path(dataset.id))
    assert sorted(frame.to_dict("records"), key=lambda row: row["channel"]) == [{"channel": "app", "n": 1, "mean_amount": 3000.0}, {"channel": "branch", "n": 1, "mean_amount": 2000.0}]
    repeated = _confirm(client, task, proposal)
    assert repeated.status_code == 200 and repeated.json()["replayed"] is True and len(calls) == 1


@pytest.mark.parametrize("transport", ["json_schema", "json_object", "none"])
def test_natural_language_sql_uses_exact_checked_question_and_source_schema_only(monkeypatch, tmp_path, transport):
    client, settings, task, source = _runtime(tmp_path)
    wire = Wire(monkeypatch)
    wire.client.profile["structured_output"] = transport
    wire.reply = {"spec": {"table": "loans", "select": [{"column": "channel"}, {"column": "amount", "aggregate": "avg", "alias": "mean_amount"}],
        "filters": [{"column": "amount", "op": ">=", "value": 2000}], "group_by": ["channel"]}}
    monkeypatch.setattr("marvis.routers.sql._llm", lambda *_: wire.client)
    question = "只统计 amount 至少 2000 的记录，按 channel 分组看 amount 均值；先给查询，不要执行。"
    proposal = _proposal(client, task, query=None, question=question)
    assert len(wire.requests) == 1 and proposal["execution_started"] is False
    encoded = json.dumps(wire.requests, ensure_ascii=False)
    assert PRIVATE not in encoded and str(source) not in encoded and "business source" not in encoded
    context = json.loads(wire.requests[0]["messages"][1]["content"])
    assert context["question"] == question and context["tables"][0]["table"] == "loans"
    assert context["tables"][0]["columns"][1] == {"name": "amount", "dtype": "integer"}
    assert DataWorkspaceRepository(settings.db_path).get_or_default(task.id).revision == 0
    assert _confirm(client, task, proposal).status_code == 200


@pytest.mark.parametrize("mutation", ["workspace", "schema", "source_config", "actor", "token"])
def test_stale_or_unowned_sql_confirmation_never_registers_data(tmp_path, mutation):
    client, settings, task, source = _runtime(tmp_path)
    proposal = _proposal(client, task)
    if mutation == "workspace":
        DataWorkspaceRepository(settings.db_path).save(task.id, DataWorkspaceDraft(page="fields"), 0)
    elif mutation == "schema":
        connection = sqlite3.connect(source)
        connection.execute("ALTER TABLE loans ADD COLUMN fresh TEXT")
        connection.close()
    elif mutation == "source_config":
        configuration = json.loads((settings.workspace / "sql_sources.json").read_text())
        configuration["business source"]["allowed_actor_ids"] = ["maker", "new-actor"]
        (settings.workspace / "sql_sources.json").write_text(json.dumps(configuration))
    elif mutation == "actor":
        client.headers["x-test-actor"] = "other-maker"
    else:
        proposal["proposal_token"] = proposal["proposal_token"][:-8] + "aaaaaaaa"
    response = _confirm(client, task, proposal)
    assert response.status_code in {409, 422}
    assert DatasetRepository(settings.db_path).list_datasets(task.id) == []


@pytest.mark.parametrize("query", ["SELECT * FROM read_csv('/Users/private/sample.csv')", "SELECT load_extension('x') FROM loans",
    "SELECT * FROM loans; DELETE FROM loans", "WITH x AS (SELECT * FROM loans) SELECT * FROM x", "SELECT * FROM loans JOIN loans b ON 1=1",
    "SELECT * FROM loans WHERE amount=1 OR amount=2", "SELECT private_missing FROM loans", "SELECT pg_read_file('/etc/passwd') FROM loans"])
def test_unsupported_or_external_sql_is_explicitly_refused_before_execution(tmp_path, query):
    client, settings, task, _ = _runtime(tmp_path)
    response = client.post(f"/api/tasks/{task.id}/sql/proposals", json={"source_id": source_identity("business source"), "expected_revision": 0, "query": query})
    assert response.status_code == 422 and "尚未执行" in response.text
    assert DatasetRepository(settings.db_path).list_datasets(task.id) == []


def test_unavailable_sources_remain_visible_and_credentials_stay_local(tmp_path):
    client, settings, task, _ = _runtime(tmp_path)
    config = json.loads((settings.workspace / "sql_sources.json").read_text())
    config["missing local source"] = {"backend": "sqlite", "path": "/private/missing/source.sqlite"}
    config["bad source"] = "password=private-secret"
    config["denied source"] = {"backend": "sqlite", "path": "/private/another.sqlite", "allowed_actor_ids": ["checker"]}
    (settings.workspace / "sql_sources.json").write_text(json.dumps(config))
    response = client.get(f"/api/tasks/{task.id}/sql/sources")
    assert response.status_code == 200
    assert len(response.json()["sources"]) == 4
    assert all(item["status"] == "unavailable" and item["next_action"] for item in response.json()["sources"][1:])
    assert "/private/" not in response.text and "private-secret" not in response.text
    client.headers.pop("x-test-actor")
    assert client.get(f"/api/tasks/{task.id}/sql/sources").status_code == 403


def test_sql_nl_sensitive_input_and_untrusted_capability_refuse_zero_requests(monkeypatch, tmp_path):
    client, _, task, _ = _runtime(tmp_path)
    wire = Wire(monkeypatch)
    monkeypatch.setattr("marvis.routers.sql._llm", lambda *_: wire.client)
    response = client.post(f"/api/tasks/{task.id}/sql/proposals", json={"source_id": source_identity("business source"),
        "expected_revision": 0, "question": "密码=private-secret 请统计金额"})
    assert response.status_code == 422 and wire.requests == []
    with pytest.raises(SqlSourceError):
        SqlSourceSnapshot("fake", "fake", SqlSourceSpec("sqlite", path="private"), [], "fake")


def test_query_subsets_preserve_quoted_literals_and_injection_is_only_data():
    tables = [{"table": "loans", "relation": "loans", "columns": [{"name": "channel"}, {"name": "amount"}]}]
    query = "SELECT channel FROM loans WHERE channel = 'x''; DELETE FROM loans; --' AND amount BETWEEN 1 AND 30 ORDER BY channel DESC LIMIT 8"
    spec = parse_manual_select(query, tables)
    assert spec["filters"][0]["value"] == "x'; DELETE FROM loans; --"
    sql = compile_select(spec, tables)
    assert "'x''; DELETE FROM loans; --'" in sql
    before = deepcopy(spec)
    compile_select(spec, tables)
    assert spec == before
    with pytest.raises(SqlQueryError):
        compile_select({"table": "loans", "select": [{"column": "amount", "aggregate": "dangerous"}]}, tables)


def test_grouped_queries_require_deterministic_projection_and_order():
    tables = [{"table": "loans", "relation": "loans", "columns": [{"name": "channel"}, {"name": "amount"}]}]
    assert compile_select(parse_manual_select("SELECT channel FROM loans GROUP BY channel ORDER BY channel", tables), tables) == 'SELECT "channel" FROM "loans" GROUP BY "channel" ORDER BY "channel" ASC'
    for query in ["SELECT channel, amount FROM loans GROUP BY channel", "SELECT * FROM loans GROUP BY channel", "SELECT channel FROM loans GROUP BY channel ORDER BY amount"]:
        with pytest.raises(SqlQueryError) as error:
            parse_manual_select(query, tables)
        assert error.value.code == "sql_grouping_required"


def test_sql_missing_model_gives_actionable_configuration_failure(tmp_path):
    client, settings, task, _ = _runtime(tmp_path)
    response = client.post(f"/api/tasks/{task.id}/sql/proposals", json={"source_id": source_identity("business source"), "expected_revision": 0, "question": "统计 loans 的数量"})
    assert response.status_code == 422 and response.json()["detail"]["code"] == "sql_model_configuration_required"
    assert "设置" in response.json()["detail"]["message"]
    assert DatasetRepository(settings.db_path).list_datasets(task.id) == []


def test_duckdb_core_blocks_external_file_functions_even_for_direct_call(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    path = tmp_path / "source.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE TABLE loans(x INTEGER)")
    connection.close()
    outside = tmp_path / "private.csv"
    outside.write_text("secret\nprivate-body\n")
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(DatasetRepository(settings.db_path), backend, settings.datasets_dir)
    with pytest.raises(duckdb.PermissionException):
        ingest_query(SqlSourceSpec("duckdb", path=str(path)), f"SELECT * FROM read_csv('{outside}')", registry=registry, task_id="sql-task")


def test_duckdb_source_uses_same_manual_proposal_and_atomic_import(tmp_path):
    client, settings, task, _ = _runtime(tmp_path)
    path = tmp_path / "native.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("CREATE TABLE loans(amount INTEGER)")
    connection.execute("INSERT INTO loans VALUES (15),(25)")
    connection.close()
    (settings.workspace / "sql_sources.json").write_text(json.dumps({"business source": {"backend": "duckdb", "path": str(path)}}))
    response = client.get(f"/api/tasks/{task.id}/sql/sources")
    assert response.status_code == 200 and response.json()["sources"][0]["tables"][0]["table"] == "main.loans"
    proposal = _proposal(client, task, query="SELECT amount FROM main.loans WHERE amount >= 20")
    confirmed = _confirm(client, task, proposal)
    assert confirmed.status_code == 200 and confirmed.json()["evidence"]["row_count"] == 1


def test_sql_import_failure_rolls_back_dataset_workspace_and_artifact(monkeypatch, tmp_path):
    client, settings, task, _ = _runtime(tmp_path)
    proposal = _proposal(client, task)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic persistence failure")
    monkeypatch.setattr(DataWorkspaceRepository, "save_on_connection", fail)
    response = _confirm(client, task, proposal)
    assert response.status_code == 422 and response.json()["detail"]["code"] == "sql_request_failed"
    assert DatasetRepository(settings.db_path).list_datasets(task.id) == []
    assert DataWorkspaceRepository(settings.db_path).get_or_default(task.id).revision == 0
    assert not list((settings.datasets_dir / task.id / "sql_imports").glob("*.parquet"))


def test_failed_execution_consumes_token_and_requires_fresh_proposal(monkeypatch, tmp_path):
    client, settings, task, _ = _runtime(tmp_path)
    proposal = _proposal(client, task)
    import marvis.agent.sql_import as sql_import
    execute, save = sql_import._execute_query, DataWorkspaceRepository.save_on_connection
    queries = []
    def tracked(*args, **kwargs):
        queries.append(True)
        return execute(*args, **kwargs)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic persistence failure")
    monkeypatch.setattr(sql_import, "_execute_query", tracked)
    monkeypatch.setattr(DataWorkspaceRepository, "save_on_connection", fail)
    first = _confirm(client, task, proposal)
    assert first.status_code == 422 and first.json()["detail"]["execution_consumed"] is True
    assert "重新生成" in first.json()["detail"]["next_action"]
    with connect(settings.db_path) as connection:
        claim = connection.execute("SELECT * FROM sql_import_executions WHERE proposal_id=?", (proposal["proposal_id"],)).fetchone()
        assert claim["state"] == "failed" and claim["error_code"] == "sql_request_failed"
        assert connection.execute("SELECT COUNT(*) FROM audit WHERE kind='sql.import.completed'").fetchone()[0] == 0
    monkeypatch.setattr(DataWorkspaceRepository, "save_on_connection", save)
    resolve = sql_import.resolve_sql_source
    monkeypatch.setattr(sql_import, "resolve_sql_source", lambda *_args, **_kwargs: pytest.fail("consumed token must not access source"))
    duplicate = _confirm(client, task, proposal)
    assert duplicate.status_code == 409 and duplicate.json()["detail"]["code"] == "sql_import_failed_consumed"
    assert queries == [True] and DatasetRepository(settings.db_path).list_datasets(task.id) == []
    monkeypatch.setattr(sql_import, "resolve_sql_source", resolve)
    fresh = _proposal(client, task)
    assert fresh["proposal_id"] != proposal["proposal_id"] and fresh["proposal_token"] != proposal["proposal_token"]
    assert _confirm(client, task, fresh).status_code == 200 and queries == [True, True]
    with connect(settings.db_path) as connection:
        assert [row[0] for row in connection.execute("SELECT state FROM sql_import_executions ORDER BY started_at")] == ["failed", "completed"]


def test_completed_execution_replays_without_source_access(monkeypatch, tmp_path):
    client, settings, task, _ = _runtime(tmp_path)
    proposal = _proposal(client, task)
    assert _confirm(client, task, proposal).status_code == 200
    import marvis.agent.sql_import as sql_import
    monkeypatch.setattr(sql_import, "resolve_sql_source", lambda *_args, **_kwargs: pytest.fail("completed token must not access source"))
    monkeypatch.setattr(sql_import, "_execute_query", lambda *_args, **_kwargs: pytest.fail("completed token must not query"))
    result = _confirm(client, task, proposal)
    assert result.status_code == 200 and result.json()["replayed"] is True
    with connect(settings.db_path) as connection:
        claim = connection.execute("SELECT * FROM sql_import_executions WHERE proposal_id=?", (proposal["proposal_id"],)).fetchone()
        assert claim["state"] == "completed" and claim["dataset_id"] == result.json()["dataset"]["id"]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE sql_import_executions SET state='running',dataset_id=NULL,ended_at=NULL")


def test_process_crash_after_query_leaves_nonreusable_claim(monkeypatch, tmp_path):
    client, settings, task, _ = _runtime(tmp_path)
    proposal = _proposal(client, task)
    marker = tmp_path / "query_completed"
    child = """
import os, sys
from pathlib import Path
from marvis.settings import build_settings
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository
import marvis.agent.sql_import as module
settings = build_settings(Path(sys.argv[1]))
registry = DatasetRegistry(DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir)
original = module._execute_query
def crash(*args, **kwargs):
    original(*args, **kwargs)
    Path(sys.argv[4]).write_text('completed')
    os._exit(23)
module._execute_query = crash
module.confirm_sql_import(settings, task_id=sys.argv[2], actor_id='maker', token=sys.argv[3], registry=registry)
"""
    process = subprocess.run([sys.executable, "-c", child, str(settings.workspace), task.id, proposal["proposal_token"], str(marker)], capture_output=True, timeout=15)
    assert process.returncode == 23 and marker.read_text() == "completed", process.stderr.decode()
    import marvis.agent.sql_import as sql_import
    monkeypatch.setattr(sql_import, "resolve_sql_source", lambda *_args, **_kwargs: pytest.fail("crashed claim must not access source"))
    monkeypatch.setattr(sql_import, "_execute_query", lambda *_args, **_kwargs: pytest.fail("crashed claim must not query"))
    response = _confirm(client, task, proposal)
    assert response.status_code == 409 and response.json()["detail"]["code"] == "sql_import_execution_uncertain"
    assert response.json()["detail"]["execution_consumed"] is True
    assert DatasetRepository(settings.db_path).list_datasets(task.id) == []
    assert DataWorkspaceRepository(settings.db_path).get_or_default(task.id).revision == 0
    with connect(settings.db_path) as connection:
        assert connection.execute("SELECT state FROM sql_import_executions WHERE proposal_id=?", (proposal["proposal_id"],)).fetchone()[0] == "running"


def test_two_processes_cannot_claim_same_proposal_twice(tmp_path):
    client, settings, task, _ = _runtime(tmp_path)
    proposal = _proposal(client, task)
    child = """
import sys
from pathlib import Path
from marvis.settings import build_settings
from marvis.agent.sql_import import _unseal, _claim_execution, SqlProposalError
settings = build_settings(Path(sys.argv[1]))
body = _unseal(settings, sys.argv[3], sys.argv[2], 'maker')
try:
    _claim_execution(settings, body)
    print('claimed')
except SqlProposalError as exc:
    print(exc.code)
"""
    processes = [subprocess.Popen([sys.executable, "-c", child, str(settings.workspace), task.id, proposal["proposal_token"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    results = [process.communicate(timeout=15) for process in processes]
    assert all(process.returncode == 0 for process in processes), results
    assert sorted(stdout.decode().strip() for stdout, _ in results) == ["claimed", "sql_import_execution_uncertain"]
    with connect(settings.db_path) as connection:
        claim = connection.execute("SELECT * FROM sql_import_executions").fetchone()
        assert claim["state"] == "running" and claim["task_id"] == task.id and claim["actor_id"] == "maker"
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE sql_import_executions SET owner_token='new'")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM sql_import_executions")


def test_sql_claim_migration_upgrades_v40_atomically_and_is_idempotent(monkeypatch, tmp_path):
    import marvis.db_schema as schema
    path = tmp_path / "upgrade.sqlite"
    with monkeypatch.context() as patch:
        patch.setattr(schema, "SCHEMA_VERSION", 40)
        patch.setattr(schema, "_MIGRATIONS", [(version, migration) for version, migration in schema._MIGRATIONS if version <= 40])
        schema.init_db(path)
    def interrupted_migration(connection):
        schema._migration_041_sql_execution_claims(connection)
        raise RuntimeError("synthetic interrupted migration")
    with monkeypatch.context() as patch:
        patch.setattr(schema, "_MIGRATIONS", [(version, interrupted_migration if version == 41 else migration) for version, migration in schema._MIGRATIONS])
        with pytest.raises(RuntimeError):
            schema.init_db(path)
    with connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 40
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sql_import_executions'").fetchone() is None
    schema.init_db(path)
    schema.init_db(path)
    with connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION
        assert connection.execute("SELECT COUNT(*) FROM sql_import_executions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND name LIKE 'sql_import_execution%'").fetchone()[0] == 3


def test_sql_core_timeout_is_typed_and_registers_no_partial_result(tmp_path):
    client, settings, task, source = _runtime(tmp_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(DatasetRepository(settings.db_path), backend, settings.datasets_dir)
    with pytest.raises(SqlQueryTimeoutError):
        ingest_query(SqlSourceSpec("sqlite", path=str(source)),
            "SELECT count(*) FROM (WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x < 100000000) SELECT x FROM cnt)",
            registry=registry, task_id=task.id, timeout_seconds=0.01)
    assert DatasetRepository(settings.db_path).list_datasets(task.id) == []
