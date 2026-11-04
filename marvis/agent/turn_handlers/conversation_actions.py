"""Draft-only conversation lanes for SQL queries (D-6) and monitoring schedules (D-8).

Both lanes write the user's message and one Agent card. Neither runs SQL nor
creates a schedule: those happen only through the card's explicit confirm
endpoint, under the human principal who clicks it.
"""
from __future__ import annotations

from marvis.agent.conversation_sql import draft_sql_in_conversation
from marvis.agent.sql_import import SqlProposalError
from marvis.data.sql_query import SqlQueryError
from marvis.data.sql_sources import SqlSourceError
from marvis.llm_client import LLMClientError
from marvis.llm_outbound import OutboundContextError
from marvis.operations.conversation_schedule import draft_schedule_form_in_conversation
from . import responses as responses_lane


def _user(repo, task, user_text, intent):
    repo.add_agent_message(task.id, role="user", stage="chat", content=str(user_text or ""),
                           metadata={"intent": intent, "request_source": "semantic_intent"})


def _note(repo, task, content, intent):
    repo.add_agent_message(task.id, role="assistant", stage="chat", content=content,
                           metadata={"intent": intent, "error": True})


def handle_sql_query_turn(runtime, repo, task, *, user_text):
    _user(repo, task, user_text, "sql_query")
    actor = getattr(getattr(runtime, "local_principal", None), "id", None)
    try:
        draft_sql_in_conversation(runtime.settings, repo, task.id, actor_id=str(actor or ""),
                                  question=str(user_text), client=runtime.llm_client)
    except (SqlProposalError, SqlQueryError, SqlSourceError) as exc:
        _note(repo, task, f"未起草查询：{exc}", "sql_query")
    except OutboundContextError:
        _note(repo, task, "这段描述含有不能发送给模型的内容，未起草查询。", "sql_query")
    except LLMClientError:
        _note(repo, task, "查询起草暂时不可用，请重试或在 SQL 面板手动输入 SELECT；未执行查询。", "sql_query")
    return responses_lane.join_turn_response(repo, task.id)


def handle_monitoring_schedule_turn(runtime, repo, task, *, user_text):
    _user(repo, task, user_text, "monitoring_schedule")
    try:
        draft_schedule_form_in_conversation(runtime.settings, repo, task.id, instruction=str(user_text),
                                            client=runtime.llm_client)
    except OutboundContextError:
        _note(repo, task, "这段描述含有不能发送给模型的内容，未填写监控表单。", "monitoring_schedule")
    return responses_lane.join_turn_response(repo, task.id)
