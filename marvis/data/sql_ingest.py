"""Governed, read-only SQL data ingest (plan item B-7).

SQL query results enter the same task-owned, content-addressed dataset chain as
CSV/Excel uploads: ``ingest_query`` executes one bounded, read-only ``SELECT``,
materializes the result as Parquet, and registers it through
:class:`~marvis.data.registry.DatasetRegistry` so it receives the same SHA-256
``content_hash``, per-column :mod:`~marvis.data.fingerprint` profiling and
immutable source path as a file import.

Governance boundaries enforced here (fail-closed):

* **Read-only at the connection layer.** DuckDB opens ``read_only=True``; SQLite
  opens ``mode=ro`` plus ``PRAGMA query_only = ON``; PostgreSQL sets
  ``default_transaction_read_only = on``. A write statement therefore fails on
  the connection even if it were somehow passed through the SELECT gate.
* **Single SELECT only.** Anything whose first keyword is not ``SELECT``, or
  that carries a second statement, is rejected before it reaches the backend.
* **Bounded execution.** Every query runs under a row budget and a time budget;
  exceeding either fails closed (``DatasetTooLargeError`` / ``SqlQueryTimeoutError``).
* **No credential leakage.** Credentials never enter exception messages, logs,
  the registered :class:`~marvis.data.contracts.Dataset` metadata, or the
  returned :class:`SqlIngestResult`. Any DSN embedded in an error is redacted
  with :func:`redact_dsn` before the message is raised.

Credentials for PostgreSQL come from explicit :class:`SqlSourceSpec` fields or
from a workspace-local JSON file (``workspace/sql_sources.json``) keyed by
``spec.name``; explicit fields always win. The file is never read by the
DuckDB/SQLite paths, and its contents are never echoed.
"""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit, urlunsplit

import duckdb
import pandas as pd

from marvis.data.contracts import Dataset
from marvis.data.errors import (
    DataIngestError,
    DataSecurityError,
    DatasetTooLargeError,
)

if TYPE_CHECKING:
    from marvis.data.registry import DatasetRegistry

BACKEND_DUCKDB = "duckdb"
BACKEND_SQLITE = "sqlite"
BACKEND_POSTGRESQL = "postgresql"

SUPPORTED_BACKENDS = frozenset({BACKEND_DUCKDB, BACKEND_SQLITE, BACKEND_POSTGRESQL})

# Bounded by default: a governed ingest never runs an unbounded SELECT. Both
# ceilings are overridable per call but must be positive.
DEFAULT_MAX_ROWS = 1_000_000
DEFAULT_QUERY_TIMEOUT_SECONDS = 30.0

# Workspace-local credential file convention. Keyed by source name, mirroring
# the explicit SqlSourceSpec fields (``backend``/``host``/``port``/``database``/
# ``user``/``password``/``dsn``). Explicit parameters always take precedence.
SQL_SOURCE_CONFIG_FILENAME = "sql_sources.json"

_REDACTED = "***"
_BACKEND_ALIASES = {
    "duckdb": BACKEND_DUCKDB,
    "sqlite": BACKEND_SQLITE,
    "sqlite3": BACKEND_SQLITE,
    "postgres": BACKEND_POSTGRESQL,
    "postgresql": BACKEND_POSTGRESQL,
}
_SPEC_FIELDS = (
    "backend",
    "path",
    "dsn",
    "host",
    "port",
    "database",
    "user",
    "password",
    "name",
)
_SQL_KEYWORD_RE = re.compile(r"^([A-Za-z]+)")
_PASSWORD_KV_RE = re.compile(
    r"(?i)(password|passwd|pwd)\s*(?:=|:)\s*([^\s;,]+)"
)


class SqlIngestError(DataIngestError):
    """Base error for governed read-only SQL data access."""


class UnsupportedSqlBackendError(SqlIngestError):
    """The requested SQL backend is not one of duckdb / sqlite / postgresql."""


class MissingSqlCredentialsError(SqlIngestError):
    """A PostgreSQL source has no usable credentials (dsn/user/password)."""


class MissingSqlDriverError(SqlIngestError):
    """A required DB driver (e.g. psycopg for PostgreSQL) is not installed."""


class NonSelectQueryError(SqlIngestError, DataSecurityError):
    """The query is not a single, read-only SELECT statement."""


class SqlQueryTimeoutError(SqlIngestError):
    """A query exceeded its configured time budget and was aborted."""


class SqlReadOnlyViolationError(SqlIngestError):
    """A write was attempted against a read-only SQL connection."""


@dataclass(frozen=True)
class SqlSourceSpec:
    """Declarative description of a SQL source.

    ``backend`` selects the driver; ``path`` names the file for duckdb/sqlite;
    the remaining fields describe a PostgreSQL source. ``name`` optionally keys
    a workspace-local ``sql_sources.json`` entry that supplies missing fields.
    """

    backend: str
    path: str | None = None
    dsn: str | None = None
    host: str | None = None
    port: int | None = None
    database: str | None = None
    user: str | None = None
    password: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class SqlIngestResult:
    """The executed frame plus its registered dataset.

    ``source`` is a credential-free, redacted label for provenance; it is the
    only source descriptor this module ever exposes.
    """

    frame: pd.DataFrame
    dataset: Dataset
    source: str


def redact_dsn(dsn: str | None) -> str:
    """Return ``dsn`` with any embedded password masked; never the secret.

    Handles both URL-form DSNs (``postgresql://user:pass@host/db``) and
    ``key=value`` forms. Used on every DSN before it is interpolated into an
    error message or a provenance label.
    """
    if dsn is None:
        return ""
    text = str(dsn).strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
        if parts.scheme and parts.netloc and "@" in parts.netloc:
            userinfo, hostpart = parts.netloc.rsplit("@", 1)
            if ":" in userinfo:
                user, _password = userinfo.split(":", 1)
                userinfo = f"{user}:{_REDACTED}"
            netloc = f"{userinfo}@{hostpart}"
            return urlunsplit(
                (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
            )
    except ValueError:
        pass
    return _PASSWORD_KV_RE.sub(lambda match: f"{match.group(1)}={_REDACTED}", text)


def load_sql_source_config(workspace: str | Path) -> dict[str, dict]:
    """Read workspace-local SQL source definitions, or ``{}`` when absent.

    The file is ``workspace/sql_sources.json``; its values are the same fields
    accepted by :class:`SqlSourceSpec`. Only the DuckDB/SQLite-less PostgreSQL
    path ever consults this file.
    """
    config_path = Path(workspace) / SQL_SOURCE_CONFIG_FILENAME
    try:
        raw = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise SqlIngestError(f"cannot read SQL source config: {exc}") from exc
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise SqlIngestError(
            f"SQL source config {config_path.name} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SqlIngestError("SQL source config must be a JSON object")
    return {
        str(key): dict(value)
        for key, value in payload.items()
        if isinstance(value, dict)
    }


def resolve_source_spec(
    source_spec: SqlSourceSpec | Mapping[str, Any],
    *,
    workspace: str | Path | None = None,
) -> SqlSourceSpec:
    """Coerce ``source_spec`` and fill missing PostgreSQL fields from config.

    The backend is validated and normalized; explicit spec fields always take
    precedence over any workspace-local ``sql_sources.json`` entry named by
    ``spec.name``. Validation only; no connection is opened here.
    """
    spec = _coerce_spec(source_spec)
    backend = _normalize_backend(spec.backend)
    if backend in (BACKEND_DUCKDB, BACKEND_SQLITE):
        if not str(spec.path or "").strip():
            raise SqlIngestError(f"{backend} source requires a file path")
        return replace(spec, backend=backend, path=str(spec.path).strip())
    spec = replace(spec, backend=backend)
    if spec.name and workspace is not None:
        config_entry = load_sql_source_config(workspace).get(str(spec.name))
        if config_entry:
            spec = _merge_config(spec, config_entry)
    return spec


def read_only_connect(
    source_spec: SqlSourceSpec | Mapping[str, Any],
    *,
    workspace: str | Path | None = None,
) -> Any:
    """Open a read-only connection for ``source_spec``.

    Returns the backend-native connection object (DuckDB, sqlite3, or psycopg).
    The read-only guarantee is enforced at the connection layer, independent of
    the query gate in :func:`ingest_query`. The caller owns closing the
    returned connection.
    """
    spec = resolve_source_spec(source_spec, workspace=workspace)
    if spec.backend == BACKEND_DUCKDB:
        return _connect_duckdb_read_only(spec)
    if spec.backend == BACKEND_SQLITE:
        return _connect_sqlite_read_only(spec)
    if spec.backend == BACKEND_POSTGRESQL:
        return _connect_postgres_read_only(spec)
    raise UnsupportedSqlBackendError(spec.backend)


def ingest_query(
    source_spec: SqlSourceSpec | Mapping[str, Any],
    query: str,
    *,
    registry: "DatasetRegistry",
    task_id: str,
    role: str = "unknown",
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS,
    seed: int = 0,
    workspace: str | Path | None = None,
) -> SqlIngestResult:
    """Execute one read-only SELECT and register its result as a task dataset.

    ``query`` must be a single ``SELECT`` statement. The result is fetched under
    ``max_rows`` / ``timeout_seconds`` budgets (both fail closed when exceeded),
    written to Parquet, and registered through ``registry`` so it carries the
    same content hash + column fingerprints + task ownership as a file import.
    """
    if max_rows < 1:
        raise SqlIngestError("max_rows must be at least 1")
    if timeout_seconds <= 0:
        raise SqlIngestError("timeout_seconds must be positive")
    spec = resolve_source_spec(source_spec, workspace=workspace)
    validate_single_select(query)

    conn = read_only_connect(spec, workspace=workspace)
    try:
        frame = _execute_query(
            conn,
            spec.backend,
            query,
            max_rows=max_rows,
            timeout_seconds=timeout_seconds,
        )
    finally:
        _close_connection(conn)

    with tempfile.TemporaryDirectory(prefix="sql_ingest_") as temp_name:
        parquet_path = Path(temp_name) / "result.parquet"
        frame.to_parquet(parquet_path, index=False)
        dataset = registry.register_existing(
            parquet_path,
            task_id=str(task_id),
            role=role,
            seed=seed,
        )
    return SqlIngestResult(
        frame=frame,
        dataset=dataset,
        source=_safe_source_label(spec),
    )


def validate_single_select(query: str) -> str:
    """Return ``query`` when it is a single SELECT; otherwise raise.

    Strict by design: the leading keyword must be ``SELECT`` (so ``WITH`` /
    ``EXPLAIN`` / ``VALUES`` / any DML/DDL are rejected), and only one trailing
    statement terminator is tolerated. A second statement is always rejected.
    """
    if not isinstance(query, str):
        raise NonSelectQueryError("query must be a non-empty string")
    sql = query.strip()
    if not sql:
        raise NonSelectQueryError("query must be a non-empty string")
    keyword = _first_keyword(sql)
    if keyword != "SELECT":
        raise NonSelectQueryError(
            "only a single SELECT statement is allowed "
            f"(got {keyword or 'no keyword'})"
        )
    body = sql.rstrip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    if ";" in body:
        raise NonSelectQueryError(
            "only a single SELECT statement is allowed (multiple statements detected)"
        )
    return sql


def _coerce_spec(source_spec: SqlSourceSpec | Mapping[str, Any]) -> SqlSourceSpec:
    if isinstance(source_spec, SqlSourceSpec):
        return source_spec
    if isinstance(source_spec, Mapping):
        payload = dict(source_spec)
        unknown = sorted(set(payload) - set(_SPEC_FIELDS))
        if unknown:
            raise SqlIngestError(
                "unknown SQL source spec field(s): " + ", ".join(unknown)
            )
        if "backend" not in payload:
            raise SqlIngestError("SQL source spec requires a backend")
        return SqlSourceSpec(
            **{field: payload[field] for field in _SPEC_FIELDS if field in payload}
        )
    raise SqlIngestError("source_spec must be a SqlSourceSpec or a mapping")


def _normalize_backend(backend: object) -> str:
    value = str(backend or "").strip().lower()
    if value not in _BACKEND_ALIASES:
        raise UnsupportedSqlBackendError(value or "<empty>")
    return _BACKEND_ALIASES[value]


def _merge_config(spec: SqlSourceSpec, config: Mapping[str, Any]) -> SqlSourceSpec:
    overrides: dict[str, Any] = {}
    for field in _SPEC_FIELDS:
        explicit = getattr(spec, field)
        if explicit is not None:
            overrides[field] = explicit
            continue
        if field in config and config[field] is not None:
            overrides[field] = config[field]
    overrides["backend"] = spec.backend
    return SqlSourceSpec(**overrides)


def _connect_duckdb_read_only(spec: SqlSourceSpec) -> Any:
    path = Path(spec.path or "")
    if not path.is_file():
        raise SqlIngestError(f"DuckDB source file does not exist: {path}")
    try:
        return duckdb.connect(str(path), read_only=True)
    except Exception as exc:  # noqa: BLE001 - normalized into a typed error
        raise SqlIngestError(
            f"cannot open DuckDB source read-only ({path.name}): {exc}"
        ) from exc


def _connect_sqlite_read_only(spec: SqlSourceSpec) -> sqlite3.Connection:
    path = Path(spec.path or "")
    if not path.is_file():
        raise SqlIngestError(f"SQLite source file does not exist: {path}")
    uri = "file:" + _pathname_to_url(str(path)) + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise SqlIngestError(
            f"cannot open SQLite source read-only ({path.name}): {exc}"
        ) from exc
    try:
        conn.execute("PRAGMA query_only = ON")
    except sqlite3.Error as exc:
        conn.close()
        raise SqlIngestError(
            f"cannot enforce read-only on SQLite source ({path.name}): {exc}"
        ) from exc
    return conn


def _connect_postgres_read_only(spec: SqlSourceSpec) -> Any:
    dsn = _build_postgres_dsn(spec)
    try:
        import psycopg  # noqa: PLC0415 - optional driver, imported lazily
    except ImportError:
        try:
            import psycopg2 as psycopg  # noqa: PLC0415
        except ImportError as exc:
            raise MissingSqlDriverError(
                "PostgreSQL support requires psycopg (v3) or psycopg2; "
                "install one to query PostgreSQL sources"
            ) from exc
    try:
        conn = psycopg.connect(dsn)
    except Exception as exc:  # noqa: BLE001 - never echo the raw driver message
        raise SqlIngestError(
            f"cannot connect to PostgreSQL source {redact_dsn(dsn)}"
        ) from exc
    try:
        conn.autocommit = True
        with conn.cursor() as cursor:
            cursor.execute("SET default_transaction_read_only = on")
    except Exception:
        conn.close()
        raise
    return conn


def _build_postgres_dsn(spec: SqlSourceSpec) -> str:
    if spec.dsn:
        return str(spec.dsn)
    if not spec.host:
        raise MissingSqlCredentialsError(
            "PostgreSQL source requires host (or an explicit dsn)"
        )
    if not spec.database:
        raise MissingSqlCredentialsError(
            "PostgreSQL source requires database (or an explicit dsn)"
        )
    if not spec.user:
        raise MissingSqlCredentialsError(
            "PostgreSQL source requires user (or an explicit dsn)"
        )
    if not spec.password:
        raise MissingSqlCredentialsError(
            "PostgreSQL source requires password (or an explicit dsn)"
        )
    userinfo = quote(spec.user, safe="") + ":" + quote(spec.password, safe="")
    host = spec.host
    if spec.port:
        host = f"{host}:{int(spec.port)}"
    return f"postgresql://{userinfo}@{host}/{quote(spec.database, safe='')}"


def _execute_query(
    conn: Any,
    backend: str,
    query: str,
    *,
    max_rows: int,
    timeout_seconds: float,
) -> pd.DataFrame:
    if backend == BACKEND_DUCKDB:
        return _execute_and_fetch(
            conn,
            query,
            max_rows=max_rows,
            timeout_seconds=timeout_seconds,
            interrupt_fn=conn.interrupt,
            is_timeout=_is_duckdb_timeout,
        )
    if backend == BACKEND_SQLITE:
        return _execute_and_fetch(
            conn,
            query,
            max_rows=max_rows,
            timeout_seconds=timeout_seconds,
            interrupt_fn=conn.interrupt,
            is_timeout=_is_sqlite_timeout,
        )
    if backend == BACKEND_POSTGRESQL:
        return _execute_postgres(
            conn,
            query,
            max_rows=max_rows,
            timeout_seconds=timeout_seconds,
        )
    raise UnsupportedSqlBackendError(backend)


def _execute_and_fetch(
    conn: Any,
    query: str,
    *,
    max_rows: int,
    timeout_seconds: float,
    interrupt_fn: Callable[[], None],
    is_timeout: Callable[[BaseException], bool],
) -> pd.DataFrame:
    timer = threading.Timer(timeout_seconds, _safe_interrupt, args=(interrupt_fn,))
    timer.daemon = True
    timer.start()
    try:
        try:
            cursor = conn.execute(query)
            columns = [str(desc[0]) for desc in (cursor.description or ())]
            rows = _fetch_bounded(cursor, max_rows + 1)
        except Exception as exc:  # noqa: BLE001 - translate timeout, re-raise rest
            if is_timeout(exc):
                raise SqlQueryTimeoutError(timeout_seconds=timeout_seconds) from exc
            raise
    finally:
        timer.cancel()
    if len(rows) > max_rows:
        raise DatasetTooLargeError(
            reason="SQL 查询结果行数超过上限",
            limit=max_rows,
            actual=len(rows) - 1,
            unit="rows",
        )
    return pd.DataFrame(rows, columns=columns)


def _execute_postgres(
    conn: Any,
    query: str,
    *,
    max_rows: int,
    timeout_seconds: float,
) -> pd.DataFrame:
    timeout_ms = max(1, int(timeout_seconds * 1000))
    cursor = conn.cursor()
    try:
        cursor.execute(f"SET statement_timeout = {timeout_ms}")
        cursor.execute(query)
        columns = [str(desc[0]) for desc in (cursor.description or ())]
        rows = _fetch_bounded(cursor, max_rows + 1)
    except Exception as exc:  # noqa: BLE001 - translate server-side cancellation
        name = type(exc).__name__.lower()
        text = str(exc).lower()
        if "querycanceled" in name or "cancel" in text or "timeout" in text:
            raise SqlQueryTimeoutError(timeout_seconds=timeout_seconds) from exc
        raise
    finally:
        cursor.close()
    if len(rows) > max_rows:
        raise DatasetTooLargeError(
            reason="SQL 查询结果行数超过上限",
            limit=max_rows,
            actual=len(rows) - 1,
            unit="rows",
        )
    return pd.DataFrame(rows, columns=columns)


def _fetch_bounded(cursor: Any, limit: int, *, batch_size: int = 2000) -> list:
    rows: list = []
    remaining = int(limit)
    while remaining > 0:
        batch = cursor.fetchmany(min(batch_size, remaining))
        if not batch:
            break
        rows.extend(batch)
        remaining -= len(batch)
    return rows


def _safe_interrupt(interrupt_fn: Callable[[], None]) -> None:
    try:
        interrupt_fn()
    except Exception:  # noqa: BLE001 - the racing query reports the real error
        pass


def _is_duckdb_timeout(exc: BaseException) -> bool:
    return isinstance(exc, duckdb.InterruptException)


def _is_sqlite_timeout(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and "interrupt" in str(exc).lower()


def _close_connection(conn: Any) -> None:
    close = getattr(conn, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - best-effort close
            pass


def _safe_source_label(spec: SqlSourceSpec) -> str:
    if spec.backend == BACKEND_DUCKDB:
        return f"duckdb:{spec.path}"
    if spec.backend == BACKEND_SQLITE:
        return f"sqlite:{spec.path}"
    return f"postgresql:{redact_dsn(_postgres_display_dsn(spec))}"


def _postgres_display_dsn(spec: SqlSourceSpec) -> str:
    if spec.dsn:
        return str(spec.dsn)
    user = spec.user or ""
    host = f"{spec.host or ''}:{spec.port}" if spec.port else (spec.host or "")
    database = spec.database or ""
    return f"postgresql://{quote(user, safe='')}@{host}/{quote(database, safe='')}"


def _first_keyword(sql: str) -> str:
    text = sql
    while True:
        stripped = text.lstrip()
        if stripped.startswith("--"):
            newline = stripped.find("\n")
            text = stripped[newline + 1 :] if newline != -1 else ""
            continue
        if stripped.startswith("/*"):
            end = stripped.find("*/", 2)
            if end == -1:
                return ""
            text = stripped[end + 2 :]
            continue
        break
    match = _SQL_KEYWORD_RE.match(stripped)
    return match.group(1).upper() if match else ""


def _pathname_to_url(path: str) -> str:
    from urllib.request import pathname2url

    return pathname2url(path)


__all__ = [
    "BACKEND_DUCKDB",
    "BACKEND_POSTGRESQL",
    "BACKEND_SQLITE",
    "DEFAULT_MAX_ROWS",
    "DEFAULT_QUERY_TIMEOUT_SECONDS",
    "SQL_SOURCE_CONFIG_FILENAME",
    "SUPPORTED_BACKENDS",
    "MissingSqlCredentialsError",
    "MissingSqlDriverError",
    "NonSelectQueryError",
    "SqlIngestError",
    "SqlIngestResult",
    "SqlQueryTimeoutError",
    "SqlReadOnlyViolationError",
    "SqlSourceSpec",
    "UnsupportedSqlBackendError",
    "ingest_query",
    "load_sql_source_config",
    "read_only_connect",
    "redact_dsn",
    "resolve_source_spec",
    "validate_single_select",
]
