"""Governed SQL proposals and atomic task-owned imports."""
import base64
from datetime import UTC, datetime
import hashlib
import hmac
import json
import time
import uuid
from filelock import FileLock, Timeout

from marvis.agent.json_reply import load_json_object
from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.sql_ingest import _execute_query, read_only_connect
from marvis.data.sql_ingest import SqlQueryTimeoutError
from marvis.data.errors import DatasetTooLargeError
from marvis.data.sql_query import SqlQueryError, compile_select, parse_manual_select
from marvis.data.sql_sources import SqlSourceError, canonical, resolve_sql_source
from marvis.data.workspace import DataWorkspaceDraft
from marvis.db_schema import connect
from marvis.llm_outbound import application_schema
from marvis.llm_outbound_sql import sql_query_prompt
from marvis.llm_prompts import SQL_SELECT_COMPILER_SYS
from marvis.repositories.audit import _write_audit_row
from marvis.repositories.data_workspace import DataWorkspaceRepository, DataWorkspaceRevisionConflict
from marvis.files import sha256_file


MAX_ROWS = 100000
TIMEOUT_SECONDS = 30.0
PROPOSAL_TTL_SECONDS = 900
_PROPOSAL_VERSION = "sql-import-proposal.v1"
_RETRY_ACTION = "请重新生成待确认查询并核对后再确认；这次提案不会再次运行查询。"


class SqlProposalError(ValueError):
    def __init__(self, code, message="SQL 数据导入尚未完成，请重新生成并核对当前数据源和确认版本。", *, status=422, execution_consumed=False):
        self.code = code
        self.status = status
        self.execution_consumed = execution_consumed
        self.next_action = _RETRY_ACTION if execution_consumed else None
        super().__init__(message)


def _secret(settings):
    value = settings.plugin_admin_token_path.read_bytes().strip()
    if not value:
        raise SqlProposalError("sql_authority_unavailable")
    return value


def _hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _sign(settings, body):
    return hmac.new(_secret(settings), (_PROPOSAL_VERSION + canonical(body)).encode(), hashlib.sha256).hexdigest()


def _seal(settings, body):
    encoded = canonical({"body": body, "signature": _sign(settings, body)}).encode()
    return base64.urlsafe_b64encode(encoded).decode()


def _unseal(settings, token, task_id, actor_id):
    if not isinstance(token, str) or len(token) > 65536:
        raise SqlProposalError("sql_proposal_invalid")
    try:
        envelope = json.loads(base64.b64decode(token, altchars=b"-_", validate=True))
        body, signature = envelope["body"], envelope["signature"]
        valid = isinstance(signature, str) and hmac.compare_digest(signature, _sign(settings, body))
    except Exception:
        raise SqlProposalError("sql_proposal_invalid") from None
    if not valid or body.get("schema_version") != _PROPOSAL_VERSION or body.get("task_id") != task_id or body.get("actor_id") != actor_id:
        raise SqlProposalError("sql_proposal_identity_mismatch")
    return body


def propose_sql_import(settings, *, task_id, actor_id, source_id, expected_revision, query=None, question=None, client=None):
    snapshot = DataWorkspaceRepository(settings.db_path).get_or_default(task_id)
    if snapshot.revision != expected_revision:
        raise DataWorkspaceRevisionConflict("SQL 确认所依据的数据工作区版本已变化，请重新读取。")
    source = resolve_sql_source(settings, source_id, actor_id, secret=_secret(settings))
    if (query is None) == (question is None):
        raise SqlProposalError("sql_input_required", "请提供自然语言目标或手动 SELECT 查询，不能同时提供。")
    if question is not None:
        prompt = sql_query_prompt(question, source)
        if client is None:
            raise SqlProposalError("sql_model_unavailable", "当前未配置可用模型，请使用手动 SELECT，尚未执行查询。")
        raw = client.complete(system_prompt=SQL_SELECT_COMPILER_SYS.text, user_prompt=prompt,
            response_format={"type": "json_object"}, json_schema=application_schema("sql_select_compile", source=source),
            caller="sql_select_compile", prompt_name=SQL_SELECT_COMPILER_SYS.name, prompt_version=SQL_SELECT_COMPILER_SYS.version, stream=False)
        result, _ = load_json_object(raw)
        if not isinstance(result, dict) or set(result) != {"spec"}:
            raise SqlProposalError("sql_needs_clarification", "查询需要澄清；请指明单表、字段、筛选或分组口径，不支持的约束尚未执行。")
        spec = result["spec"]
    else:
        spec = parse_manual_select(query, source.schema(), max_rows=MAX_ROWS)
    sql = compile_select(spec, source.schema(), max_rows=MAX_ROWS)
    body = {"schema_version": _PROPOSAL_VERSION, "proposal_id": uuid.uuid4().hex, "task_id": task_id,
        "actor_id": actor_id, "source_id": source_id, "source_fingerprint": source.source_fingerprint,
        "schema_hash": source.schema_hash, "workspace_revision": expected_revision, "spec": spec, "query": sql,
        "max_rows": MAX_ROWS, "timeout_seconds": TIMEOUT_SECONDS, "expires_at": int(time.time()) + PROPOSAL_TTL_SECONDS}
    return {"status": "needs_confirmation", "proposal_id": body["proposal_id"], "proposal_token": _seal(settings, body),
        "query": sql, "source_id": source_id, "schema_hash": source.schema_hash, "workspace_revision": expected_revision,
        "max_rows": MAX_ROWS, "timeout_seconds": TIMEOUT_SECONDS, "expires_at": body["expires_at"],
        "supported_subset": "单表 SELECT、字段选择、COUNT/SUM/AVG/MIN/MAX、AND 筛选、分组、排序、LIMIT；不支持 JOIN/OR/子查询/CTE/UNION/窗口或其他函数。",
        "execution_started": False}


def _completed(settings, body):
    with connect(settings.db_path) as connection:
        row = connection.execute("SELECT detail_json FROM audit WHERE kind='sql.import.completed' AND target_ref=? AND actor=? AND inputs_hash=? AND outcome='succeeded' ORDER BY at DESC LIMIT 1",
            (body["proposal_id"], body["actor_id"], _hash(body))).fetchone()
    if row is None:
        return None
    try:
        envelope = json.loads(row[0])
        receipt = envelope["receipt"]
        if not hmac.compare_digest(envelope["signature"], _sign(settings, receipt)) or receipt["proposal_hash"] != _hash(body):
            raise ValueError
        return receipt
    except Exception:
        raise SqlProposalError("sql_completion_authentication_failed") from None


def _claim_execution(settings, body):
    owner_token = uuid.uuid4().hex
    with connect(settings.db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT * FROM sql_import_executions WHERE proposal_id=?", (body["proposal_id"],)).fetchone()
        if row is not None:
            if row["proposal_hash"] != _hash(body) or row["task_id"] != body["task_id"] or row["actor_id"] != body["actor_id"]:
                raise SqlProposalError("sql_execution_identity_mismatch")
            if row["state"] == "failed":
                raise SqlProposalError("sql_import_failed_consumed", "这次提案的执行流程已结束，数据未导入。" + _RETRY_ACTION, status=409, execution_consumed=True)
            if row["state"] == "completed":
                raise SqlProposalError("sql_completion_authentication_failed", "这次提案已有完成记录，但结果凭据不可验证。" + _RETRY_ACTION, status=409, execution_consumed=True)
            raise SqlProposalError("sql_import_execution_uncertain", "这次查询提案已进入执行流程，当前没有可验证的完成结果。" + _RETRY_ACTION, status=409, execution_consumed=True)
        connection.execute("""INSERT INTO sql_import_executions
          (proposal_id,task_id,actor_id,proposal_hash,source_id,source_fingerprint,schema_hash,query_hash,owner_token,state,started_at)
          VALUES (?,?,?,?,?,?,?,?,?,'running',?)""",
          (body["proposal_id"], body["task_id"], body["actor_id"], _hash(body), body["source_id"],
           body["source_fingerprint"], body["schema_hash"], _hash(body["query"]), owner_token, time.time()))
    # The execution budget is committed before reading the source. A process
    # crash leaves a durable running claim, which must never resume implicitly.
    return owner_token


def _fail_execution(settings, body, owner_token, code):
    with connect(settings.db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("""UPDATE sql_import_executions SET state='failed',ended_at=?,error_code=?
          WHERE proposal_id=? AND proposal_hash=? AND owner_token=? AND state='running'""",
          (time.time(), code, body["proposal_id"], _hash(body), owner_token))


def _consumed_error(exc):
    status = 422
    if isinstance(exc, (SqlProposalError, SqlSourceError, SqlQueryError)):
        code, message, status = exc.code, str(exc), getattr(exc, "status", 422)
        if code == "sql_source_changed":
            status = 409
    elif isinstance(exc, DataWorkspaceRevisionConflict):
        code, message, status = "sql_workspace_revision_conflict", str(exc), 409
    elif isinstance(exc, SqlQueryTimeoutError):
        code, message, status = "sql_query_timeout", str(exc), 408
    elif isinstance(exc, DatasetTooLargeError):
        code, message = "sql_result_row_limit", "SQL 结果超过当前行数上限，未截断或导入。"
    else:
        code, message = "sql_request_failed", "这次 SQL 查询提案未完成导入，活动数据未更新。"
    return SqlProposalError(code, message + " " + _RETRY_ACTION, status=status, execution_consumed=True)


def confirm_sql_import(settings, *, task_id, actor_id, token, registry):
    body = _unseal(settings, token, task_id, actor_id)
    lock_root = settings.workspace / "sql_import_locks"
    lock_root.mkdir(exist_ok=True)
    try:
        with FileLock(str(lock_root / (_hash([task_id, body["proposal_id"]]) + ".lock")), timeout=5):
            return _confirm_sql_import(settings, task_id=task_id, actor_id=actor_id, token=token, registry=registry)
    except Timeout:
        raise SqlProposalError("sql_import_in_progress", "这次 SQL 导入正在执行，请等待完成后读取结果；不会重复运行查询。") from None


def _confirm_sql_import(settings, *, task_id, actor_id, token, registry):
    body = _unseal(settings, token, task_id, actor_id)
    completed = _completed(settings, body)
    if completed:
        dataset = registry.get(completed["dataset_id"])
        if dataset.task_id != task_id or dataset.content_hash != completed["dataset_content_hash"]:
            raise SqlProposalError("sql_completion_dataset_changed")
        registry.resolve_verified_path(dataset.id)
        return dataset, DataWorkspaceRepository(settings.db_path).get_or_default(task_id), True
    if body["expires_at"] < int(time.time()):
        raise SqlProposalError("sql_proposal_expired")
    owner_token = _claim_execution(settings, body)
    try:
        return _execute_claimed_import(settings, task_id=task_id, actor_id=actor_id, body=body, owner_token=owner_token, registry=registry)
    except Exception as exc:
        failure = _consumed_error(exc)
        try:
            _fail_execution(settings, body, owner_token, failure.code)
        except Exception:
            raise SqlProposalError("sql_import_execution_uncertain", "执行状态尚无法验证。" + _RETRY_ACTION, status=409, execution_consumed=True) from None
        raise failure from None


def _execute_claimed_import(settings, *, task_id, actor_id, body, owner_token, registry):
    workspace = DataWorkspaceRepository(settings.db_path)
    if workspace.get_or_default(task_id).revision != body["workspace_revision"]:
        raise DataWorkspaceRevisionConflict("SQL 确认所依据的数据工作区版本已变化，请重新生成。")
    source = resolve_sql_source(settings, body["source_id"], actor_id, secret=_secret(settings))
    def authenticate_source():
        live = resolve_sql_source(settings, body["source_id"], actor_id, secret=_secret(settings))
        if live.source_fingerprint != body["source_fingerprint"] or live.schema_hash != body["schema_hash"]:
            raise SqlProposalError("sql_source_changed", "SQL 数据源配置、文件身份或字段结构已变化；当前确认已失效，请重新生成查询。")

    authenticate_source()
    sql = compile_select(body["spec"], source.schema(), max_rows=body["max_rows"])
    if sql != body["query"]:
        raise SqlProposalError("sql_proposal_query_changed")
    connection = read_only_connect(source.spec)
    try:
        frame = _execute_query(connection, source.backend, sql, max_rows=body["max_rows"], timeout_seconds=body["timeout_seconds"])
    finally:
        connection.close()
    authenticate_source()
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(settings.datasets_dir / task_id / "sql_imports", body["proposal_id"] + ".parquet")
    try:
        frame.to_parquet(artifact.path, index=False)
        def persist(connection):
            connection.execute("BEGIN IMMEDIATE")
            # Recheck under the same transaction as dataset + workspace + audit.
            dataset = registry.register_existing_on_connection(connection, artifact.final_path, task_id=task_id, role="sample")
            if sha256_file(artifact.final_path) != dataset.content_hash:
                raise SqlProposalError("sql_import_content_changed")
            snapshot = workspace.save_on_connection(connection, task_id,
                DataWorkspaceDraft(active_dataset_id=dataset.id, active_dataset_content_hash=dataset.content_hash),
                body["workspace_revision"], dataset_authenticator=authenticate_source)
            receipt = {"proposal_hash": _hash(body), "task_id": task_id, "actor_id": actor_id,
                "source_id": body["source_id"], "source_fingerprint": body["source_fingerprint"], "schema_hash": body["schema_hash"],
                "query_hash": _hash(sql), "dataset_id": dataset.id, "dataset_content_hash": dataset.content_hash,
                "row_count": dataset.row_count, "column_count": len(dataset.columns), "workspace_revision": snapshot.revision,
                "executed_at": datetime.now(UTC).isoformat()}
            _write_audit_row(connection, kind="sql.import.completed", target_ref=body["proposal_id"], actor=actor_id,
                inputs_hash=_hash(body), outcome="succeeded", detail={"receipt": receipt, "signature": _sign(settings, receipt)})
            changed = connection.execute("""UPDATE sql_import_executions SET state='completed',ended_at=?,dataset_id=?
              WHERE proposal_id=? AND proposal_hash=? AND owner_token=? AND state='running'""",
              (time.time(), dataset.id, body["proposal_id"], _hash(body), owner_token)).rowcount
            if changed != 1:
                raise SqlProposalError("sql_execution_claim_changed")
            if sha256_file(artifact.final_path) != dataset.content_hash:
                raise SqlProposalError("sql_import_content_changed")
            return dataset, snapshot, False

        return uow.finalize_with_connection(lambda: connect(settings.db_path), persist)
    except Exception:
        uow.rollback()
        raise


__all__ = ["SqlProposalError", "propose_sql_import", "confirm_sql_import"]
