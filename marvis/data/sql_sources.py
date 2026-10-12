"""Local configured SQL aliases and authenticated, schema-only capabilities."""
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path

from marvis.data.sql_ingest import SqlSourceSpec, read_only_connect, resolve_source_spec
from marvis.llm_outbound import OutboundContextError
from marvis.llm_outbound_instructions import column_identifiers


_SOURCE_TOKEN = object()
MAX_TABLES = 128
MAX_COLUMNS = 2048


class SqlSourceError(ValueError):
    def __init__(self, code="sql_source_unavailable"):
        self.code = code
        super().__init__("SQL 数据源尚不可用，请在本地核对已配置来源和连接权限。")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _sign(secret, value):
    return hmac.new(secret, canonical(value).encode(), hashlib.sha256).hexdigest()


@dataclass(frozen=True, init=False)
class SqlSourceSnapshot:
    source_id: str
    display_name: str
    backend: str
    spec: SqlSourceSpec
    schema_json: str
    schema_hash: str
    source_fingerprint: str

    def __init__(self, source_id, name, spec, schema, source_fingerprint, *, _token=None):
        if _token is not _SOURCE_TOKEN:
            raise SqlSourceError("sql_source_not_authenticated")
        for key, value in {"source_id": source_id, "display_name": name, "backend": spec.backend, "spec": spec,
            "schema_json": canonical(schema), "schema_hash": _digest(schema), "source_fingerprint": source_fingerprint}.items():
            object.__setattr__(self, key, value)

    def schema(self):
        return json.loads(self.schema_json)

    def public(self):
        return {"source_id": self.source_id, "name": self.display_name, "backend": self.backend,
            "schema_hash": self.schema_hash, "tables": self.schema(), "status": "ready"}


def _configured_sources(workspace):
    path = Path(workspace) / "sql_sources.json"
    try:
        if not path.exists():
            return {}
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            raise SqlSourceError("sql_source_config_invalid")
        config = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        raise SqlSourceError("sql_source_config_invalid") from None
    if not isinstance(config, dict) or len(config) > 64:
        raise SqlSourceError("sql_source_config_invalid")
    return config


def source_identity(name):
    return "sql_" + _digest(str(name))[:24]


def _spec(entry, name, actor_id):
    if not isinstance(entry, dict) or not isinstance(name, str) or not name.strip():
        raise SqlSourceError("sql_source_config_invalid")
    actors = entry.get("allowed_actor_ids")
    if actors is not None and (not isinstance(actors, list) or actor_id not in actors):
        raise SqlSourceError("sql_source_forbidden")
    fields = {"backend", "path", "dsn", "host", "port", "database", "user", "password"}
    if set(entry) - fields - {"allowed_actor_ids", "name"}:
        raise SqlSourceError("sql_source_config_invalid")
    try:
        return resolve_source_spec(SqlSourceSpec(**{key: value for key, value in entry.items() if key in fields}, name=name))
    except Exception:
        raise SqlSourceError("sql_source_config_invalid") from None


def _dtype(value):
    raw = str(value).upper()
    if any(key in raw for key in ("INT", "SERIAL")):
        return "integer"
    if any(key in raw for key in ("DECIMAL", "NUMERIC", "REAL", "FLOAT", "DOUBLE")):
        return "number"
    if "BOOL" in raw:
        return "boolean"
    if any(key in raw for key in ("DATE", "TIME")):
        return "datetime"
    if any(key in raw for key in ("CHAR", "TEXT", "STRING")):
        return "text"
    if any(key in raw for key in ("BLOB", "BYTE", "BINARY")):
        return "binary"
    return "unknown"


def read_source_schema(spec):
    try:
        connection = read_only_connect(spec)
    except Exception:
        raise SqlSourceError("sql_source_connection_failed") from None
    try:
        if spec.backend == "sqlite":
            names = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name LIMIT 129").fetchall()
            if len(names) > MAX_TABLES:
                raise SqlSourceError("sql_source_schema_limit")
            tables = []
            for (name,) in names:
                escaped = str(name).replace('"', '""')
                columns = connection.execute(f'PRAGMA table_info("{escaped}")').fetchall()
                tables.append({"table": str(name), "relation": str(name), "columns": [{"name": str(row[1]), "dtype": _dtype(row[2])} for row in columns]})
        else:
            query = "SELECT c.table_schema,c.table_name,c.column_name,c.data_type FROM information_schema.columns c JOIN information_schema.tables t ON t.table_schema=c.table_schema AND t.table_name=c.table_name WHERE t.table_type='BASE TABLE' AND c.table_schema NOT IN ('information_schema','pg_catalog') ORDER BY c.table_schema,c.table_name,c.ordinal_position LIMIT 2049"
            cursor = connection.cursor() if spec.backend == "postgresql" else connection
            try:
                cursor.execute(query)
                rows = cursor.fetchall()
            finally:
                if spec.backend == "postgresql":
                    cursor.close()
            by_table = {}
            for schema, table, column, dtype in rows:
                by_table.setdefault((str(schema), str(table)), []).append({"name": str(column), "dtype": _dtype(dtype)})
            tables = [{"table": f"{schema}.{name}", "namespace": schema, "relation": name, "columns": columns} for (schema, name), columns in by_table.items()]
        if not tables:
            raise SqlSourceError("sql_source_empty_schema")
        if len(tables) > MAX_TABLES or sum(len(row["columns"]) for row in tables) > MAX_COLUMNS:
            raise SqlSourceError("sql_source_schema_limit")
        for table in tables:
            column_identifiers([table["table"], *[column["name"] for column in table["columns"]]])
        return tables
    except SqlSourceError:
        raise
    except OutboundContextError:
        raise SqlSourceError("sql_source_schema_not_admitted") from None
    except Exception:
        raise SqlSourceError("sql_source_connection_failed") from None
    finally:
        connection.close()


def resolve_sql_source(settings, source_id, actor_id, *, secret):
    matches = [(name, entry) for name, entry in _configured_sources(settings.workspace).items() if source_identity(name) == source_id]
    if len(matches) != 1:
        raise SqlSourceError("sql_source_not_found")
    name, entry = matches[0]
    spec = _spec(entry, name, actor_id)
    schema = read_source_schema(spec)
    binding = {"source_id": source_id, "config": entry}
    if spec.backend in {"sqlite", "duckdb"}:
        path = Path(spec.path)
        stat = path.stat()
        binding["local_identity"] = [str(path.resolve()), stat.st_dev, stat.st_ino]
    fingerprint = _sign(secret, binding)
    return SqlSourceSnapshot(source_id, name, spec, schema, fingerprint, _token=_SOURCE_TOKEN)


def list_sql_sources(settings, actor_id, *, secret):
    result = []
    for name in _configured_sources(settings.workspace):
        identity = source_identity(name)
        try:
            result.append(resolve_sql_source(settings, identity, actor_id, secret=secret).public())
        except SqlSourceError as exc:
            result.append({"source_id": identity, "name": str(name), "status": "unavailable", "code": exc.code,
                "next_action": "在本地修正数据源配置或连接权限，然后重新读取；未省略该来源。"})
    return result
