"""V3 D-8: the Agent fills a monitoring form; the user's confirmation creates the schedule."""
from datetime import UTC, datetime
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pandas as pd

from marvis.agent.turn_handlers.conversation_actions import handle_monitoring_schedule_turn
from marvis.app import create_app
from marvis.db_schema import connect
from marvis.operations.conversation_schedule import (
    CREATED_KIND, FORM_KIND, draft_schedule_form_in_conversation, fill_schedule_form,
)
from marvis.repositories.tasks import TaskRepository
from marvis.settings import build_settings

from test_analysis_pack import _register, _runtime
from test_llm_outbound import Wire
from test_operations_api import _claim_role

NOW = datetime(2025, 4, 1, tzinfo=UTC)
PRIVATE_CELL = "PRIVATE_MONITORING_CELL_4471"


def _setup(tmp_path):
    _, _, registry, task = _runtime(tmp_path)
    frame = pd.DataFrame({
        "applied_at": pd.to_datetime(["2025-03-01", "2025-03-02", "2025-03-03"], utc=True),
        "score": [0.1, 0.5, 0.9], "bad": [0, 1, 0], "note": [PRIVATE_CELL] * 3,
    })
    dataset = _register(registry, tmp_path, task.id, frame, "monitoring-source")
    settings = build_settings(tmp_path / "workspace")
    with connect(settings.db_path) as conn:
        conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?)",
                     ("exp-1", task.id, "lgbm", "{}", None, "artifact-1", "trained", NOW.isoformat()))
    app = create_app(settings.workspace, operations_clock=lambda: NOW)
    client = TestClient(app)
    _claim_role(app, client, "admin")
    return app, client, settings, task, dataset, TaskRepository(settings.db_path)


def _proposal_rows(settings):
    with connect(settings.db_path) as conn:
        return conn.execute("SELECT count(*) FROM operations_schedule_proposals").fetchone()[0]


def test_agent_fills_form_from_choice_lists_only(monkeypatch, tmp_path):
    _, _, settings, task, dataset, _ = _setup(tmp_path)
    wire = Wire(monkeypatch)
    wire.reply = {"target_id": "exp-1", "dataset_id": dataset.id, "time_col": "applied_at", "frequency": "weekly",
                  "interval": 2, "start_date": "2025-04-07", "notify": "local_inbox"}
    form, options, filled_by, missing = fill_schedule_form(settings, task.id, "每两周监控 lgbm 模型，从 4 月 7 日开始", wire.client, now=NOW)
    assert filled_by == "agent" and missing == []
    assert (form["frequency"], form["interval"], form["start_date"], form["time_col"]) == ("weekly", 2, "2025-04-07", "applied_at")
    sent = json.dumps(wire.requests, ensure_ascii=False)
    assert PRIVATE_CELL not in sent and dataset.content_hash not in sent
    context = json.loads(wire.requests[0]["messages"][1]["content"])
    assert context["targets"][0]["id"] == "exp-1" and context["today"] == "2025-04-01"


def test_model_values_outside_the_choices_are_dropped(monkeypatch, tmp_path):
    _, _, settings, task, dataset, _ = _setup(tmp_path)
    wire = Wire(monkeypatch)
    wire.reply = {"target_id": "foreign-model", "dataset_id": "foreign", "time_col": "note; DROP", "frequency": "hourly",
                  "interval": 500, "start_date": "next week", "notify": "email:boss@example.com"}
    form, _, filled_by, _ = fill_schedule_form(settings, task.id, "监控模型", wire.client, now=NOW)
    assert filled_by == "platform_default"
    assert form["target_id"] == "exp-1" and form["dataset_id"] == dataset.id
    assert (form["frequency"], form["interval"], form["notify"]) == ("daily", 1, "local_inbox")


def test_without_a_model_the_platform_default_form_is_offered(tmp_path):
    _, _, settings, task, _, repo = _setup(tmp_path)
    message = draft_schedule_form_in_conversation(settings, repo, task.id, instruction="每天监控", client=None, now=NOW)
    action = message["metadata"]["conversation_action"]
    assert action["kind"] == FORM_KIND and action["filled_by"] == "platform_default"
    assert action["form"]["start_date"] == "2025-04-02" and action["form"]["advanced"]["timezone"] == "Asia/Shanghai"
    assert all("content_hash" not in item for item in action["options"]["datasets"])
    assert _proposal_rows(settings) == 0


def test_confirmed_form_creates_the_schedule_once_through_proposal_and_receipt(tmp_path):
    app, client, settings, task, dataset, repo = _setup(tmp_path)
    message = draft_schedule_form_in_conversation(settings, repo, task.id, instruction="每月监控", client=None, now=NOW)
    form = message["metadata"]["conversation_action"]["form"]
    form.update(frequency="monthly", interval=1, time_col="applied_at")
    body = {"task_id": task.id, "message_id": message["id"], "form": form}
    response = client.post("/api/operations/conversation/schedules", json=body)
    assert response.status_code == 201, response.text
    created = response.json()["message"]["metadata"]["conversation_action"]
    assert created["kind"] == CREATED_KIND and created["source_message_id"] == message["id"]
    record = app.state.operations_runtime.store.get_schedule(created["schedule_id"])
    contract = record.contract.to_dict()
    assert contract["monitoring_ref"] == "modeling.monitor_run" and contract["calendar"]["interval_months"] == 1
    assert contract["monitoring_binding"]["target_id"] == "exp-1"
    assert contract["monitoring_binding"]["label_mode"] == "not_applicable"
    with connect(settings.db_path) as conn:
        receipts = conn.execute("SELECT * FROM operations_schedule_confirmations").fetchall()
    assert len(receipts) == 1 and receipts[0]["schedule_id"] == created["schedule_id"]
    again = client.post("/api/operations/conversation/schedules", json=body)
    assert again.status_code == 201 and again.json()["replayed"] is True
    assert _proposal_rows(settings) == 1


def test_invalid_form_is_refused_with_fields_and_creates_nothing(tmp_path):
    app, client, settings, task, _, repo = _setup(tmp_path)
    message = draft_schedule_form_in_conversation(settings, repo, task.id, instruction="监控", client=None, now=NOW)
    form = message["metadata"]["conversation_action"]["form"]
    form.update(time_col="missing_col", interval=99)
    form["advanced"]["target_col"] = "bad"  # a label column needs a maturity
    response = client.post("/api/operations/conversation/schedules",
                           json={"task_id": task.id, "message_id": message["id"], "form": form})
    assert response.status_code == 422
    assert set(response.json()["detail"]["fields"]) == {"time_col", "interval", "advanced.label_maturity_days"}
    assert _proposal_rows(settings) == 0
    assert app.state.operations_runtime.store.list_schedules() == ()
    plain = repo.add_agent_message(task.id, role="assistant", stage="chat", content="x", metadata={})
    forged = client.post("/api/operations/conversation/schedules",
                         json={"task_id": task.id, "message_id": plain["id"], "form": form})
    assert forged.status_code == 404


def test_turn_lane_writes_user_text_and_form_card(monkeypatch, tmp_path):
    _, _, settings, task, dataset, repo = _setup(tmp_path)
    wire = Wire(monkeypatch)
    wire.reply = {"frequency": "daily", "time_col": "applied_at"}
    runtime = SimpleNamespace(settings=settings, llm_client=wire.client)
    response = handle_monitoring_schedule_turn(runtime, repo, task, user_text="每天监控这个模型，通知我")
    kinds = [(m["role"], (m["metadata"].get("conversation_action") or {}).get("kind")) for m in response["messages"]]
    assert kinds[-2:] == [("user", None), ("assistant", FORM_KIND)]
    assert _proposal_rows(settings) == 0


def _post_form(client, task, message, **changes):
    form = message["metadata"]["conversation_action"]["form"]
    form.update(time_col="applied_at")
    advanced = changes.pop("advanced", {})
    form.update(changes)
    form["advanced"].update(advanced)
    return client.post("/api/operations/conversation/schedules",
                       json={"task_id": task.id, "message_id": message["id"], "form": form})


def test_future_coverage_and_past_first_run_are_refused(tmp_path):
    app, client, settings, task, _, repo = _setup(tmp_path)
    message = draft_schedule_form_in_conversation(settings, repo, task.id, instruction="监控", client=None, now=NOW)
    future = _post_form(client, task, message, advanced={"complete_through": "2025-04-01T00:00:01+00:00"})
    assert future.status_code == 422 and future.json()["detail"]["fields"] == ["advanced.complete_through"]
    # A naive value is read in the form's timezone: 08:00:01 Shanghai is 00:00:01 UTC.
    naive = _post_form(client, task, message, advanced={"complete_through": "2025-04-01T08:00:01"})
    assert naive.status_code == 422 and naive.json()["detail"]["fields"] == ["advanced.complete_through"]
    past = _post_form(client, task, message, frequency="monthly", start_date="2025-03-01",
                      advanced={"complete_through": "2025-04-01T00:00:00+00:00"})
    assert past.status_code == 422 and past.json()["detail"]["fields"] == ["start_date"]
    assert _proposal_rows(settings) == 0 and app.state.operations_runtime.store.list_schedules() == ()
    exact = _post_form(client, task, message, frequency="daily", start_date="2025-04-01",
                       advanced={"complete_through": "2025-04-01T08:00:00"})
    assert exact.status_code == 201, exact.text


def test_create_path_never_revises_an_existing_schedule_id(tmp_path):
    app, client, settings, task, _, repo = _setup(tmp_path)
    first = draft_schedule_form_in_conversation(settings, repo, task.id, instruction="监控", client=None, now=NOW)
    created = _post_form(client, task, first)
    assert created.status_code == 201, created.text
    schedule_id = created.json()["message"]["metadata"]["conversation_action"]["schedule_id"]
    second = draft_schedule_form_in_conversation(settings, repo, task.id, instruction="监控", client=None, now=NOW)
    clash = _post_form(client, task, second, frequency="weekly", advanced={"schedule_id": schedule_id})
    assert clash.status_code == 422 and clash.json()["detail"]["fields"] == ["advanced.schedule_id"]
    record = app.state.operations_runtime.store.get_schedule(schedule_id)
    assert record.contract.revision == 1 and _proposal_rows(settings) == 1


def test_agent_past_start_date_is_not_accepted(monkeypatch, tmp_path):
    _, _, settings, task, _, _ = _setup(tmp_path)
    wire = Wire(monkeypatch)
    wire.reply = {"start_date": "2025-01-01", "frequency": "monthly"}
    form, _, _, _ = fill_schedule_form(settings, task.id, "从一月开始每月监控", wire.client, now=NOW)
    assert form["start_date"] == "2025-04-02" and form["frequency"] == "monthly"
