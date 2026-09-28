"""Native source authority follows the ordinary HTTP presentation carriers."""

from fastapi.testclient import TestClient
import pytest

from marvis.agent.plan_message_composer import PlanMessageComposer
from marvis.agent.driver_turn import DriverTurn
from marvis.agent.turn_handlers.shared import _append_driver_messages_core
from marvis.repositories.tasks import TaskRepository
from test_native_producer_recovery import _case
from test_event_runtime import runtime, prepare, gated_plan


def _completed(tmp_path, monkeypatch, kind):
    case = _case(tmp_path, monkeypatch, kind)
    response = case.rt.maker.post(
        f"/api/plans/{case.plan['id']}/reconcile", json={"target_id": case.target["id"]}
    )
    assert response.status_code == 200, response.text
    plan = case.rt.app.state.plan_repo.load_plan(case.plan["id"])
    assert plan.status.value == "done"
    repo = case.rt.app.state.plan_repo
    # Exercise the actual deterministic presenter and append-only message
    # producer, without asking an LLM to invent or paraphrase source results.
    composer = PlanMessageComposer(
        load_output=repo.load_step_output,
        load_step_presentation_binding=repo.load_step_presentation_binding,
    )
    message = composer.done_message(plan, run_seq=1)
    assert message.metadata["plan_id"] == plan.id
    assert "artifact_id=" in message.content
    _append_driver_messages_core(
        TaskRepository(case.rt.settings.db_path),
        case.task_id,
        DriverTurn(plan.id, "done", [message]),
    )
    return case


def _urls(case):
    step = case.plan["steps"][0]["id"]
    return [
        f"/api/step-outputs/{step}",
        f"/api/step-outputs/{step}:v1",
        f"/api/plans/{case.plan['id']}",
        f"/api/tasks/{case.task_id}/plans",
        f"/api/tasks/{case.task_id}/agent/messages",
        f"/api/plans/{case.plan['id']}/business-acceptance/xlsx",
        f"/api/plans/{case.plan['id']}/business-acceptance/docx",
    ]


@pytest.mark.parametrize("kind", ["event", "batch", "reconciliation"])
def test_current_grant_is_required_for_every_native_result_carrier(
    tmp_path, monkeypatch, kind
):
    case = _completed(tmp_path, monkeypatch, kind)
    urls = _urls(case)
    for url in urls:
        response = case.rt.maker.get(url)
        assert response.status_code == 200, (url, response.text)
    messages = case.rt.maker.get(urls[4]).json()["messages"]
    assert messages and "已完成" in messages[-1]["content"]
    for client in (case.rt.other, TestClient(case.rt.app)):
        for url in urls:
            response = client.get(url)
            assert response.status_code == 403, (url, response.text)
            assert (
                "features" not in response.text and "snapshot_hash" not in response.text
            )
    response = case.rt.admin.post(
        f"/api/tasks/{case.rt.task.id}/risk-events/grants/{case.rt.grant['grant_id']}/revoke"
    )
    assert response.status_code == 200, response.text
    for url in urls:
        response = case.rt.maker.get(url)
        assert response.status_code == 403, (url, response.text)


def test_pending_source_plan_contract_is_also_private(tmp_path):
    rt = runtime.__wrapped__(tmp_path)
    plan = gated_plan(rt, prepare(rt))
    urls = [f"/api/plans/{plan['id']}", f"/api/tasks/{rt.task.id}/plans"]
    for url in urls:
        assert rt.maker.get(url).status_code == 200
        for client in (rt.other, TestClient(rt.app)):
            response = client.get(url)
            assert response.status_code == 403, response.text


def test_agent_message_read_blocks_incremental_pages_and_derived_replies(
    tmp_path, monkeypatch
):
    case = _completed(tmp_path, monkeypatch, "event")
    base = f"/api/tasks/{case.task_id}/agent"
    message = case.rt.maker.get(base + "/messages").json()["messages"][-1]
    assert case.observed["output"]["artifact_id"] in message["content"]
    for client in (case.rt.other, TestClient(case.rt.app)):
        for query in ("?limit=1", f"?after_id={message['id']}&limit=1"):
            response = client.get(base + "/messages" + query)
            assert response.status_code == 403, response.text
        for route, body in (
            ("messages", {"content": "总结之前的结果"}),
            ("summarize", {}),
        ):
            response = client.post(base + "/" + route, json=body)
            assert response.status_code == 403, response.text


def test_guard_is_read_only_and_authorized_reads_survive_restart(tmp_path, monkeypatch):
    from marvis.app import create_app
    from marvis.db_schema import connect

    case = _completed(tmp_path, monkeypatch, "event")
    fresh = TestClient(create_app(case.rt.settings))
    fresh.cookies.update(case.rt.maker.cookies)
    with connect(case.rt.settings.db_path) as conn:
        before = conn.execute("SELECT count(*) FROM jobs").fetchone()[0]
        audits = conn.execute("SELECT count(*) FROM audit").fetchone()[0]
        database_before = tuple(conn.iterdump())
    fresh.app.state.plan_executor.reconciler.authorize_task_read(
        case.task_id, actor_id=case.rt.principal["id"]
    )
    with connect(case.rt.settings.db_path) as conn:
        assert tuple(conn.iterdump()) == database_before
    for url in _urls(case):
        assert fresh.get(url).status_code == 200
    with connect(case.rt.settings.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == before
        assert conn.execute("SELECT count(*) FROM audit").fetchone()[0] == audits


def test_expired_source_grant_cannot_read_presentations(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta
    import marvis.risk_context.event_repository as module

    case = _completed(tmp_path, monkeypatch, "event")
    monkeypatch.setattr(
        module, "_now", lambda: (datetime.now(UTC) + timedelta(hours=2)).isoformat()
    )
    for url in _urls(case):
        assert case.rt.maker.get(url).status_code == 403


def test_pinned_tool_ref_keeps_original_native_read_guard(tmp_path, monkeypatch):
    from dataclasses import replace
    from marvis.db_schema import connect
    from test_event_runtime import approve

    rt = runtime.__wrapped__(tmp_path)
    original = rt.app.state.planner.from_template

    def pinned(*args, **kwargs):
        plan = original(*args, **kwargs)
        plan.steps[0].tool_ref = replace(plan.steps[0].tool_ref, version="0.1.0")
        return plan

    monkeypatch.setattr(rt.app.state.planner, "from_template", pinned)
    plan = gated_plan(rt, prepare(rt))
    done = approve(rt, plan)
    assert done["status"] == "done", done
    run = rt.app.state.plan_repo.list_step_runs(plan["steps"][0]["id"])[0]
    assert run["tool_ref"] == "risk_context.replay_events"
    assert run["invocation_contract"]["tool_version"] == "0.1.0"
    assert done["steps"][0]["tool_ref"]["version"] == "0.1.0"
    registry = rt.app.state.plan_executor.reconciler.verifiers
    with connect(rt.settings.db_path) as conn:
        with registry.reader(rt.principal["id"]):
            registry.authorize_read(run["tool_ref"], run["id"], conn)
        with pytest.raises(PermissionError):
            registry.authorize_read(run["tool_ref"], run["id"], conn)
    url = f"/api/step-outputs/{plan['steps'][0]['id']}:v1"
    assert rt.maker.get(url).status_code == 200
    assert rt.other.get(url).status_code == 403


def test_http_read_rechecks_revocation_after_native_result_and_message_snapshot(
    tmp_path, monkeypatch
):
    from marvis.db_schema import connect

    case = _completed(tmp_path, monkeypatch, "event")
    repository = case.rt.app.state.plan_repo
    original_output = repository.load_step_output(case.plan["steps"][0]["id"])
    original = TaskRepository.list_agent_messages
    reads = []

    def revoke_after_snapshot(self, task_id, *args, **kwargs):
        messages = original(self, task_id, *args, **kwargs)
        if task_id == case.task_id and not reads:
            reads.append(messages)
            response = case.rt.admin.post(
                f"/api/tasks/{case.rt.task.id}/risk-events/grants/{case.rt.grant['grant_id']}/revoke"
            )
            assert response.status_code == 200
        return messages

    with connect(case.rt.settings.db_path) as conn:
        jobs = conn.execute("SELECT count(*) FROM jobs").fetchone()[0]
    monkeypatch.setattr(TaskRepository, "list_agent_messages", revoke_after_snapshot)
    response = case.rt.maker.get(f"/api/tasks/{case.task_id}/agent/messages")
    assert reads and reads[0]
    assert response.status_code == 403, response.text
    assert case.observed["output"]["artifact_id"] not in response.text
    assert repository.load_step_output(case.plan["steps"][0]["id"]) == original_output
    assert repository.load_plan(case.plan["id"]).status.value == "done"
    assert len(repository.list_step_runs(case.plan["steps"][0]["id"])) == 1
    with connect(case.rt.settings.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == jobs


def test_async_read_adapter_keeps_fastapi_signature_and_await_semantics(
    tmp_path, monkeypatch
):
    from fastapi import Request
    from marvis.governance.http_reads import native_task_read_scope

    case = _completed(tmp_path, monkeypatch, "event")

    @case.rt.app.get("/native-read-probe/{task_id}")
    @native_task_read_scope
    async def probe(task_id: str, request: Request):
        return {"task_id": task_id}

    assert case.rt.maker.get(f"/native-read-probe/{case.task_id}").json() == {
        "task_id": case.task_id
    }
    assert case.rt.other.get(f"/native-read-probe/{case.task_id}").status_code == 403
