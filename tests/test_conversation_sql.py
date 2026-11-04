"""V3 D-6: SQL drafted in the conversation runs only after explicit confirmation."""
import json
from types import SimpleNamespace

import pytest

from marvis.agent.conversation_sql import SQL_PROPOSAL_KIND, SQL_RESULT_KIND, SQL_SOURCE_CHOICE_KIND, draft_sql_in_conversation
from marvis.agent.sql_import import SqlProposalError
from marvis.agent.turn_handlers.conversation_actions import handle_sql_query_turn
from marvis.agent.turn_handlers.semantic import _conversation_action_intents
from marvis.data.sql_query import SqlQueryError
from marvis.data.sql_sources import source_identity
from marvis.db_schema import connect
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.tasks import TaskRepository
from tests.test_llm_outbound import Wire
from tests.test_sql_product import PRIVATE, _runtime

SPEC = {"spec": {"table": "loans", "select": [{"column": "channel"}, {"column": "amount", "aggregate": "avg", "alias": "mean_amount"}],
                 "filters": [{"column": "amount", "op": ">=", "value": 2000}], "group_by": ["channel"]}}
QUESTION = "从业务库取 amount 至少 2000 的记录，按 channel 看 amount 均值"


def _executions(settings):
    with connect(settings.db_path) as conn:
        return conn.execute("SELECT count(*) FROM sql_import_executions").fetchone()[0]


def _audit(settings, kind):
    with connect(settings.db_path) as conn:
        return conn.execute("SELECT * FROM audit WHERE kind=?", (kind,)).fetchall()


def _draft(monkeypatch, tmp_path, reply=SPEC):
    client, settings, task, source = _runtime(tmp_path)
    wire = Wire(monkeypatch)
    wire.reply = reply
    repo = TaskRepository(settings.db_path)
    return client, settings, task, repo, wire


def _card(monkeypatch, tmp_path):
    client, settings, task, repo, wire = _draft(monkeypatch, tmp_path)
    message = draft_sql_in_conversation(settings, repo, task.id, actor_id="maker", question=QUESTION, client=wire.client)
    return client, settings, task, repo, wire, message


def test_draft_writes_a_proposal_card_and_audit_but_runs_nothing(monkeypatch, tmp_path):
    _, settings, task, _, wire, message = _card(monkeypatch, tmp_path)
    action = message["metadata"]["conversation_action"]
    assert action["kind"] == SQL_PROPOSAL_KIND and action["execution_started"] is False
    assert action["query"].upper().startswith("SELECT") and action["actor_id"] == "maker"
    assert len(wire.requests) == 1 and PRIVATE not in json.dumps(wire.requests, ensure_ascii=False)
    assert _executions(settings) == 0
    assert DataWorkspaceRepository(settings.db_path).get_or_default(task.id).revision == 0
    rows = _audit(settings, "sql.conversation.proposed")
    assert len(rows) == 1 and rows[0]["target_ref"] == action["proposal_id"] and rows[0]["outcome"] == "needs_confirmation"


def test_confirm_runs_once_registers_dataset_and_writes_result_card(monkeypatch, tmp_path):
    client, settings, task, repo, _, message = _card(monkeypatch, tmp_path)
    response = client.post(f"/api/tasks/{task.id}/sql/conversation/confirm", json={"message_id": message["id"], "confirmed": True})
    assert response.status_code == 200, response.text
    result = response.json()["message"]["metadata"]["conversation_action"]
    assert result["kind"] == SQL_RESULT_KIND and result["source_message_id"] == message["id"]
    assert result["dataset"]["row_count"] == 2 and result["dataset"]["column_names"] == ["channel", "mean_amount"]
    assert DataWorkspaceRepository(settings.db_path).get_or_default(task.id).revision == 1
    assert len(_audit(settings, "sql.import.completed")) == 1 and _executions(settings) == 1
    again = client.post(f"/api/tasks/{task.id}/sql/conversation/confirm", json={"message_id": message["id"], "confirmed": True})
    assert again.status_code == 200 and again.json()["replayed"] is True
    assert _executions(settings) == 1
    results = [m for m in repo.list_agent_messages(task.id)
               if (m["metadata"].get("conversation_action") or {}).get("kind") == SQL_RESULT_KIND]
    assert len(results) == 1


@pytest.mark.parametrize("body", [{}, {"confirmed": False}, {"confirmed": "true"}])
def test_confirm_without_explicit_confirmation_is_refused(monkeypatch, tmp_path, body):
    client, settings, task, _, _, message = _card(monkeypatch, tmp_path)
    response = client.post(f"/api/tasks/{task.id}/sql/conversation/confirm", json={"message_id": message["id"], **body})
    assert response.status_code == 422
    assert _executions(settings) == 0 and DataWorkspaceRepository(settings.db_path).get_or_default(task.id).revision == 0


def test_other_actor_or_non_proposal_message_cannot_confirm(monkeypatch, tmp_path):
    client, settings, task, repo, _, message = _card(monkeypatch, tmp_path)
    other = client.post(f"/api/tasks/{task.id}/sql/conversation/confirm", json={"message_id": message["id"], "confirmed": True},
                        headers={"x-test-actor": "other-maker"})
    assert other.status_code == 422
    plain = repo.add_agent_message(task.id, role="assistant", stage="chat", content="x", metadata={})
    forged = client.post(f"/api/tasks/{task.id}/sql/conversation/confirm", json={"message_id": plain["id"], "confirmed": True})
    assert forged.status_code == 422
    assert _executions(settings) == 0


def test_actor_outside_the_source_allow_list_gets_no_source_and_no_draft(monkeypatch, tmp_path):
    _, settings, task, repo, wire = _draft(monkeypatch, tmp_path)
    with pytest.raises(SqlProposalError) as error:
        draft_sql_in_conversation(settings, repo, task.id, actor_id="stranger", question=QUESTION, client=wire.client)
    assert error.value.code == "sql_source_unavailable"
    assert wire.requests == [] and _audit(settings, "sql.conversation.proposed") == []
    runtime = SimpleNamespace(settings=settings, local_principal=SimpleNamespace(id="stranger"))
    assert "sql_query" not in _conversation_action_intents(runtime, task)
    runtime = SimpleNamespace(settings=settings, local_principal=SimpleNamespace(id="maker"))
    assert "sql_query" in _conversation_action_intents(runtime, task)


def test_model_output_outside_the_restricted_grammar_is_rejected(monkeypatch, tmp_path):
    _, settings, task, repo, wire = _draft(monkeypatch, tmp_path, reply={"spec": {"table": "secrets", "select": [{"column": "*"}]}})
    with pytest.raises((SqlQueryError, SqlProposalError)):
        draft_sql_in_conversation(settings, repo, task.id, actor_id="maker", question=QUESTION, client=wire.client)
    assert _audit(settings, "sql.conversation.proposed") == [] and _executions(settings) == 0
    assert not any(m["metadata"].get("conversation_action") for m in repo.list_agent_messages(task.id))


def test_turn_lane_logs_user_text_and_card_and_explains_clarification(monkeypatch, tmp_path):
    _, settings, task, repo, wire = _draft(monkeypatch, tmp_path)
    runtime = SimpleNamespace(settings=settings, llm_client=wire.client, local_principal=SimpleNamespace(id="maker"))
    response = handle_sql_query_turn(runtime, repo, task, user_text=QUESTION)
    roles = [(m["role"], (m["metadata"].get("conversation_action") or {}).get("kind")) for m in response["messages"]]
    assert roles[-2:] == [("user", None), ("assistant", SQL_PROPOSAL_KIND)]
    wire.reply = {"clarify": "需要说明表名"}
    response = handle_sql_query_turn(runtime, repo, task, user_text="取点数据")
    assert response["messages"][-1]["metadata"].get("error") is True and _executions(settings) == 0


def test_several_sources_ask_for_a_source_first_then_draft_through_http(monkeypatch, tmp_path):
    client, settings, task, repo, wire = _draft(monkeypatch, tmp_path)
    config = json.loads((settings.workspace / "sql_sources.json").read_text())
    config["second source"] = dict(config["business source"])
    (settings.workspace / "sql_sources.json").write_text(json.dumps(config))
    choice = draft_sql_in_conversation(settings, repo, task.id, actor_id="maker", question=QUESTION, client=wire.client)
    action = choice["metadata"]["conversation_action"]
    assert action["kind"] == SQL_SOURCE_CHOICE_KIND and len(action["sources"]) == 2 and wire.requests == []
    monkeypatch.setattr("marvis.routers.sql._llm", lambda *_: wire.client)
    response = client.post(f"/api/tasks/{task.id}/sql/conversation/drafts",
                           json={"question": QUESTION, "source_id": source_identity("business source")})
    assert response.status_code == 200, response.text
    assert response.json()["message"]["metadata"]["conversation_action"]["kind"] == SQL_PROPOSAL_KIND
    assert _executions(settings) == 0


def test_concurrent_confirms_run_once_and_write_one_result_card(monkeypatch, tmp_path):
    import threading

    from marvis.agent.conversation_sql import confirm_sql_in_conversation
    from marvis.data.backend import DataBackend
    from marvis.data.registry import DatasetRegistry
    from marvis.db import DatasetRepository

    _, settings, task, repo, _, message = _card(monkeypatch, tmp_path)
    registry = DatasetRegistry(DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir)
    barrier, outcomes = threading.Barrier(3), []

    def confirm():
        barrier.wait()
        outcomes.append(confirm_sql_in_conversation(settings, repo, task.id, actor_id="maker",
                                                    message_id=message["id"], registry=registry))

    threads = [threading.Thread(target=confirm) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(replayed for _, replayed in outcomes) == [False, True, True]
    assert len({result["id"] for result, _ in outcomes}) == 1
    cards = [m for m in repo.list_agent_messages(task.id)
             if (m["metadata"].get("conversation_action") or {}).get("kind") == SQL_RESULT_KIND]
    assert len(cards) == 1 and _executions(settings) == 1
