"""Tests for governed, read-only SQL ingest (marvis/data/sql_ingest.py)."""

from __future__ import annotations

import re
import sqlite3
import sys
import types

import duckdb
import pytest

from marvis.data.backend import DataBackend
from marvis.data.errors import DatasetTooLargeError
from marvis.data.registry import DatasetRegistry
from marvis.data.sql_ingest import (
    NonSelectQueryError,
    SqlIngestError,
    SqlIngestResult,
    SqlSourceSpec,
    ingest_query,
    read_only_connect,
    redact_dsn,
    resolve_source_spec,
    validate_single_select,
)
from marvis.db import DatasetRepository, init_db
from marvis.settings import build_settings


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        backend,
        settings.datasets_dir,
    )
    return settings, registry


def _sqlite_file(tmp_path, *, name="sample.sqlite"):
    path = tmp_path / name
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE loans(id INTEGER, score REAL, bad INTEGER)")
        conn.executemany(
            "INSERT INTO loans(id, score, bad) VALUES (?, ?, ?)",
            [(1, 0.8, 0), (2, 0.6, 1), (3, 0.9, 0)],
        )
        conn.commit()
    finally:
        conn.close()
    return path


def _duckdb_file(tmp_path, *, name="sample.duckdb"):
    path = tmp_path / name
    conn = duckdb.connect(str(path))
    try:
        conn.execute("CREATE TABLE loans(id INTEGER, score DOUBLE, bad INTEGER)")
        conn.execute(
            "INSERT INTO loans VALUES (1, 0.8, 0), (2, 0.6, 1), (3, 0.9, 0)"
        )
    finally:
        conn.close()
    return path


@pytest.mark.parametrize("backend", ["sqlite", "duckdb"])
def test_ingest_query_registers_readable_fingerprinted_dataset(
    tmp_path, backend
):
    settings, registry = _runtime(tmp_path)
    source = (
        _sqlite_file(tmp_path) if backend == "sqlite" else _duckdb_file(tmp_path)
    )

    result = ingest_query(
        SqlSourceSpec(backend=backend, path=str(source)),
        "SELECT id, score, bad FROM loans ORDER BY id",
        registry=registry,
        task_id="task-1",
        role="sample",
    )

    assert isinstance(result, SqlIngestResult)
    assert result.frame.to_dict("records") == [
        {"id": 1, "score": 0.8, "bad": 0},
        {"id": 2, "score": 0.6, "bad": 1},
        {"id": 3, "score": 0.9, "bad": 0},
    ]
    assert result.dataset.task_id == "task-1"
    assert result.dataset.role == "sample"
    assert result.dataset.row_count == 3
    assert re.fullmatch(r"[0-9a-f]{64}", result.dataset.content_hash or "")

    # Registered dataset is readable through the registry and carries a
    # per-column fingerprint plus an immutable content hash.
    loaded = registry.get(result.dataset.id)
    assert loaded is not None
    assert loaded.content_hash == result.dataset.content_hash
    assert len(loaded.columns) == 3
    for profile in loaded.columns:
        assert profile.fingerprint.value_kind
        assert profile.fingerprint.value_kind != "unknown"

    # The result is task-owned on disk (registered under the task directory).
    assert loaded.source_path.startswith("task-1/")
    assert (settings.datasets_dir / loaded.source_path).is_file()


@pytest.mark.parametrize("backend", ["sqlite", "duckdb"])
def test_read_only_connection_rejects_writes(tmp_path, backend):
    source = (
        _sqlite_file(tmp_path) if backend == "sqlite" else _duckdb_file(tmp_path)
    )
    conn = read_only_connect(SqlSourceSpec(backend=backend, path=str(source)))
    try:
        expected_error = (
            sqlite3.OperationalError if backend == "sqlite" else duckdb.Error
        )
        with pytest.raises(expected_error):
            conn.execute("INSERT INTO loans(id, score, bad) VALUES (4, 0.1, 0)")
    finally:
        conn.close()


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM loans",
        "UPDATE loans SET bad = 1",
        "INSERT INTO loans VALUES (4, 0.1, 0)",
        "CREATE TABLE extra(id INTEGER)",
        "SELECT 1; DROP TABLE loans",
        "WITH cte AS (SELECT 1 AS x) SELECT x FROM cte",
        "",
        "   ",
    ],
)
def test_non_select_queries_are_rejected(query):
    with pytest.raises(NonSelectQueryError):
        validate_single_select(query)


def test_ingest_query_rejects_non_select_before_connecting(tmp_path):
    settings, registry = _runtime(tmp_path)
    source = _sqlite_file(tmp_path)

    with pytest.raises(NonSelectQueryError):
        ingest_query(
            SqlSourceSpec(backend="sqlite", path=str(source)),
            "DELETE FROM loans",
            registry=registry,
            task_id="task-1",
        )


def test_row_budget_exceeded_fails_closed(tmp_path):
    settings, registry = _runtime(tmp_path)
    source = _sqlite_file(tmp_path)

    with pytest.raises(DatasetTooLargeError):
        ingest_query(
            SqlSourceSpec(backend="sqlite", path=str(source)),
            "SELECT id, score, bad FROM loans",
            registry=registry,
            task_id="task-1",
            max_rows=2,
        )

    # Nothing was registered when the budget gate failed closed.
    assert registry.list_for_task("task-1") == []


def test_redact_dsn_masks_url_and_keyword_passwords():
    url = "postgresql://alice:s3cr3t-pass@db.example.com:5432/loans"
    assert "s3cr3t-pass" not in redact_dsn(url)
    assert "***" in redact_dsn(url)
    assert "alice" in redact_dsn(url)

    keyword = "host=db.example.com user=alice password=s3cr3t-pass dbname=loans"
    assert "s3cr3t-pass" not in redact_dsn(keyword)


def test_postgres_connection_error_redacts_dsn(monkeypatch):
    password = "s3cr3t-pass"
    spec = SqlSourceSpec(
        backend="postgresql",
        host="db.example.com",
        database="loans",
        user="alice",
        password=password,
    )

    fake_psycopg = types.ModuleType("psycopg")

    def failing_connect(dsn):
        raise RuntimeError(f"could not connect using {dsn}")

    fake_psycopg.connect = failing_connect
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    with pytest.raises(SqlIngestError) as exc_info:
        read_only_connect(spec)

    message = str(exc_info.value)
    assert password not in message
    assert "***" in message


def test_missing_postgres_credentials_raise_typed_error():
    with pytest.raises(SqlIngestError):
        read_only_connect(
            SqlSourceSpec(backend="postgresql", host="db.example.com")
        )
    with pytest.raises(SqlIngestError):
        read_only_connect(
            SqlSourceSpec(
                backend="postgresql",
                host="db.example.com",
                database="loans",
                user="alice",
            )
        )
    with pytest.raises(SqlIngestError):
        resolve_source_spec(SqlSourceSpec(backend="mongodb"))


def test_workspace_config_supplies_postgres_credentials(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sql_sources.json").write_text(
        '{"warehouse": {"host": "db.example.com", "database": "loans", '
        '"user": "alice", "password": "s3cr3t-pass"}}',
        encoding="utf-8",
    )

    resolved = resolve_source_spec(
        SqlSourceSpec(backend="postgresql", name="warehouse"),
        workspace=workspace,
    )

    assert resolved.host == "db.example.com"
    assert resolved.database == "loans"
    assert resolved.user == "alice"
    assert resolved.password == "s3cr3t-pass"

    # Explicit parameters win over the config entry.
    overridden = resolve_source_spec(
        SqlSourceSpec(backend="postgresql", name="warehouse", user="bob"),
        workspace=workspace,
    )
    assert overridden.user == "bob"
    assert overridden.password == "s3cr3t-pass"


def test_sqlite_read_only_connection_enforces_query_only(tmp_path):
    source = _sqlite_file(tmp_path)
    conn = read_only_connect(SqlSourceSpec(backend="sqlite", path=str(source)))
    try:
        assert conn.execute("SELECT count(*) FROM loans").fetchone()[0] == 3
    finally:
        conn.close()


def test_duckdb_read_only_connection_reads(tmp_path):
    source = _duckdb_file(tmp_path)
    conn = read_only_connect(SqlSourceSpec(backend="duckdb", path=str(source)))
    try:
        frame = conn.execute("SELECT count(*) AS n FROM loans").fetchdf()
        assert int(frame["n"].iloc[0]) == 3
    finally:
        conn.close()


def test_ingest_query_requires_positive_budgets(tmp_path):
    settings, registry = _runtime(tmp_path)
    source = _sqlite_file(tmp_path)

    with pytest.raises(SqlIngestError, match="max_rows"):
        ingest_query(
            SqlSourceSpec(backend="sqlite", path=str(source)),
            "SELECT * FROM loans",
            registry=registry,
            task_id="task-1",
            max_rows=0,
        )
    with pytest.raises(SqlIngestError, match="timeout_seconds"):
        ingest_query(
            SqlSourceSpec(backend="sqlite", path=str(source)),
            "SELECT * FROM loans",
            registry=registry,
            task_id="task-1",
            timeout_seconds=0,
        )


def test_missing_source_file_raises_typed_error(tmp_path):
    missing = tmp_path / "absent.sqlite"
    with pytest.raises(SqlIngestError, match="does not exist"):
        read_only_connect(SqlSourceSpec(backend="sqlite", path=str(missing)))
