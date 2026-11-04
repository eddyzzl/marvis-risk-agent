"""SQL queries started from the conversation (V3 D-6).

The Agent drafts a query through the SQL panel's own proposal path; nothing
runs until the user confirms that exact proposal card. Execution, allow-list,
audit and workspace CAS all stay in :mod:`marvis.agent.sql_import`.
"""
from __future__ import annotations

import hashlib

from filelock import FileLock, Timeout

from marvis.agent.sql_import import SqlProposalError, _secret, confirm_sql_import, propose_sql_import
from marvis.data.sql_sources import list_sql_sources
from marvis.db_schema import connect
from marvis.repositories.audit import _write_audit_row
from marvis.repositories.data_workspace import DataWorkspaceRepository


SQL_PROPOSAL_KIND = "sql_query_proposal"
SQL_RESULT_KIND = "sql_query_result"
SQL_SOURCE_CHOICE_KIND = "sql_source_choice"
_MAX_RESULT_COLUMNS = 60


def ready_sql_sources(settings, actor_id):
    """Sources this actor may query; any configuration failure means none."""
    if not actor_id:
        return []
    try:
        sources = list_sql_sources(settings, actor_id, secret=_secret(settings))
    except Exception:  # noqa: BLE001 - an unreadable source config offers nothing
        return []
    return [{"source_id": item["source_id"], "name": item.get("name") or item["source_id"]}
            for item in sources if item.get("status") == "ready"]


def _assistant(repo, task_id, content, action, **metadata):
    return repo.add_agent_message(task_id, role="assistant", stage="chat", content=content,
        metadata={"intent": "sql_query", "conversation_action": action, **metadata})


def draft_sql_in_conversation(settings, repo, task_id, *, actor_id, question, source_id=None, client=None):
    """Write a proposal card (or a source-choice card). Never executes SQL."""
    sources = ready_sql_sources(settings, actor_id)
    if not sources:
        raise SqlProposalError("sql_source_unavailable", "当前身份没有可用的 SQL 数据源，未起草查询。")
    by_id = {item["source_id"]: item for item in sources}
    if source_id is None and len(sources) > 1:
        return _assistant(repo, task_id, "要从哪个数据源取数？", {
            "kind": SQL_SOURCE_CHOICE_KIND, "question": question, "sources": sources, "actor_id": actor_id})
    source_id = source_id or sources[0]["source_id"]
    if source_id not in by_id:
        raise SqlProposalError("sql_source_unavailable", "所选数据源不可用或当前身份无权使用，未起草查询。")
    revision = DataWorkspaceRepository(settings.db_path).get_or_default(task_id).revision
    proposal = propose_sql_import(settings, task_id=task_id, actor_id=actor_id, source_id=source_id,
        expected_revision=revision, question=question, client=client)
    with connect(settings.db_path) as conn:
        _write_audit_row(conn, kind="sql.conversation.proposed", target_ref=proposal["proposal_id"], actor=actor_id,
            inputs_hash=hashlib.sha256(proposal["query"].encode()).hexdigest(), outcome="needs_confirmation",
            detail={"task_id": task_id, "source_id": source_id, "workspace_revision": revision})
    action = {"kind": SQL_PROPOSAL_KIND, "proposal_id": proposal["proposal_id"], "query": proposal["query"],
        "source_id": source_id, "source_name": by_id[source_id]["name"], "max_rows": proposal["max_rows"],
        "expires_at": proposal["expires_at"], "workspace_revision": revision, "actor_id": actor_id,
        "execution_started": False}
    return _assistant(repo, task_id, "查询已起草，确认后执行。", action, sql_proposal_token=proposal["proposal_token"])


def _result_for(repo, task_id, message_id):
    for message in repo.list_agent_messages(task_id):
        action = (message.get("metadata") or {}).get("conversation_action") or {}
        if action.get("kind") == SQL_RESULT_KIND and action.get("source_message_id") == message_id:
            return message
    return None


def confirm_sql_in_conversation(settings, repo, task_id, *, actor_id, message_id, registry):
    """Run the exact proposal on one card, once. Returns ``(result_message, replayed)``."""
    try:
        message = repo.get_agent_message(task_id, message_id)
    except KeyError:
        raise SqlProposalError("sql_proposal_invalid", "找不到这条待确认查询。") from None
    metadata = message.get("metadata") or {}
    action = metadata.get("conversation_action") or {}
    token = metadata.get("sql_proposal_token")
    if message.get("role") != "assistant" or action.get("kind") != SQL_PROPOSAL_KIND or not isinstance(token, str):
        raise SqlProposalError("sql_proposal_invalid", "这条消息不是待确认查询。")
    if action.get("actor_id") != actor_id:
        raise SqlProposalError("sql_proposal_identity_mismatch", "只有起草这条查询的身份可以确认执行。")
    # One card, one result card: concurrent confirms of the same card are
    # serialized here, so the second one sees the first one's result card.
    lock_root = settings.workspace / "sql_import_locks"
    lock_root.mkdir(exist_ok=True)
    lock_name = hashlib.sha256(f"conversation:{task_id}:{message_id}".encode()).hexdigest() + ".lock"
    try:
        with FileLock(str(lock_root / lock_name), timeout=60):
            return _confirm_card_locked(settings, repo, task_id, actor_id=actor_id, message_id=message_id,
                                        action=action, token=token, registry=registry)
    except Timeout:
        raise SqlProposalError("sql_import_in_progress", "这次 SQL 导入正在执行，请等待完成后读取结果；不会重复运行查询。") from None


def _confirm_card_locked(settings, repo, task_id, *, actor_id, message_id, action, token, registry):
    existing = _result_for(repo, task_id, message_id)
    if existing is not None:
        return existing, True
    dataset, snapshot, replayed = confirm_sql_import(settings, task_id=task_id, actor_id=actor_id, token=token,
                                                     registry=registry)
    if replayed:
        existing = _result_for(repo, task_id, message_id)
        if existing is not None:
            return existing, True
        # Replayed import without a card means the earlier request stopped
        # after the import committed; write the card once (still under lock).
    columns = [column.name for column in dataset.columns]
    result = _assistant(repo, task_id, f"已按确认的查询取数：{dataset.row_count} 行，已设为当前数据。", {
        "kind": SQL_RESULT_KIND, "source_message_id": message_id, "proposal_id": action["proposal_id"],
        "dataset": {"id": dataset.id, "row_count": dataset.row_count, "column_count": len(columns),
                    "column_names": columns[:_MAX_RESULT_COLUMNS], "content_hash": dataset.content_hash},
        "workspace_revision": snapshot.revision})
    return result, False


__all__ = ["SQL_PROPOSAL_KIND", "SQL_RESULT_KIND", "SQL_SOURCE_CHOICE_KIND", "ready_sql_sources",
           "draft_sql_in_conversation", "confirm_sql_in_conversation"]
