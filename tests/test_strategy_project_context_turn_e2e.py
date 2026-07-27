"""Focused Agent -> Workflow -> Tool project-context vertical."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.agent.renderers import render_tool_output
from marvis.packs.strategy.project_context_tools import (
    load_current_strategy_project_context,
)


class _ProjectContextLLM:
    def __init__(self, workflow_inputs: dict) -> None:
        self.workflow_inputs = workflow_inputs
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_project_context",
                "workflow_inputs": self.workflow_inputs,
            },
            ensure_ascii=False,
        )


def _create_task(client: TestClient, tmp_path: Path) -> tuple[str, Path]:
    source_dir = client.app.state.settings.workspace / f"source-{tmp_path.name}"
    source_dir.mkdir()
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "策略项目上下文",
            "validator": "qa",
            "source_dir": str(source_dir),
            "task_type": "strategy",
            "run_mode": "manual",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"], source_dir


def test_project_context_runs_without_workspace_and_binds_message_and_external_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id, source_dir = _create_task(client, tmp_path)
    (source_dir / "历史评审.xlsx").write_bytes(b"opaque report bytes")
    inputs = {
        "as_of": "2026-06-30",
        "scope": "自营中信借钱贷中提额项目",
        "business_context": {
            "project.background": "本次目标是优化存量客户提额策略",
        },
        "explicit_unavailable": ["current.status_fields.economics"],
        "external_report_filenames": ["历史评审.xlsx"],
    }
    llm = _ProjectContextLLM(inputs)
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    utterance = (
        "请整理策略项目现状，截止 2026-06-30，范围是自营中信借钱贷中提额项目；"
        "本次目标是优化存量客户提额策略；收益和成本暂时没有；"
        "外部材料是历史评审.xlsx。"
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": utterance},
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == ["strategy_project_context"]
    assert plans[0]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    user_message = next(
        message
        for message in reversed(response.json()["messages"])
        if message["role"] == "user"
    )
    assert stored.steps[0].inputs == {
        "expected_revision": 0,
        "user_message_ref": {
            "message_id": user_message["id"],
            "content_hash": hashlib.sha256(
                user_message["content"].encode("utf-8")
            ).hexdigest(),
        },
        **{key: value for key, value in inputs.items() if value is not None},
    }
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["created"] is True
    assert output["revision"]["revision"] == 1
    assert output["external_artifacts"][0]["kind"] == (
        "strategy_history_external_source"
    )
    assert client.get(output["context_artifact"]["download_url"]).status_code == 200
    loaded = load_current_strategy_project_context(
        SimpleNamespace(settings=client.app.state.settings),
        task_id=task_id,
    )
    assert loaded == output["revision"]
    assistant_text = "\n".join(
        message["content"]
        for message in response.json()["messages"]
        if message["role"] == "assistant"
    )
    assert "策略项目上下文已创建" in assistant_text
    assert "外部材料按原始字节快照" in assistant_text
    assert "不会填成 0" in assistant_text

    tampered = copy.deepcopy(output)
    tampered["revision"]["state"]["as_of"] = "2026-07-01"
    failure_text, failure_tables = render_tool_output(
        "materialize_project_context", tampered
    )
    assert "完整性校验失败" in failure_text
    assert failure_tables == []


def test_project_context_second_turn_uses_exact_current_cas(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id, _ = _create_task(client, tmp_path)
    llm = _ProjectContextLLM(
        {
            "as_of": "2026-06-30",
            "scope": None,
            "business_context": {},
            "explicit_unavailable": ["historical_strategy_reviews"],
            "external_report_filenames": [],
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    first = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "整理策略项目现状，截止 2026-06-30，历史策略暂时没有。"},
    )
    assert first.status_code == 202, first.text
    first_plan = client.get(f"/api/tasks/{task_id}/plans").json()["plans"][0]
    first_output = client.app.state.plan_repo.load_step_output(
        client.app.state.plan_repo.load_plan(first_plan["id"]).steps[0].id
    )

    llm.workflow_inputs = {
        "as_of": "2026-07-22",
        "scope": None,
        "business_context": {"project.change": "本次补充了项目变更背景"},
        "explicit_unavailable": ["historical_strategy_reviews"],
        "external_report_filenames": [],
    }
    second = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "刷新策略项目上下文，截止 2026-07-22，本次补充了项目变更背景，"
                "历史策略暂时没有。"
            )
        },
    )
    assert second.status_code == 202, second.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    second_plan = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert second_plan.steps[0].inputs["expected_revision"] == 1
    assert second_plan.steps[0].inputs["expected_revision_id"] == first_output[
        "revision"
    ]["revision_id"]
    assert second_plan.steps[0].inputs["expected_state_hash"] == first_output[
        "revision"
    ]["state_hash"]
    second_output = client.app.state.plan_repo.load_step_output(
        second_plan.steps[0].id
    )
    assert second_output["revision"]["revision"] == 2


def test_pending_context_answer_is_used_but_never_promoted_to_tool_metric(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id, _ = _create_task(client, tmp_path)
    llm = _ProjectContextLLM(
        {
            "as_of": "2026-06-30",
            "scope": "自营渠道",
            "business_context": {},
            "explicit_unavailable": ["historical_strategy_reviews"],
            "external_report_filenames": [],
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    first = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "整理策略项目现状，截止 2026-06-30，范围是自营渠道，"
                "历史策略暂时没有。"
            )
        },
    )
    assert first.status_code == 202, first.text
    llm_call_count = len(llm.calls)
    first_plan = client.get(f"/api/tasks/{task_id}/plans").json()["plans"][0]
    first_output = client.app.state.plan_repo.load_step_output(
        client.app.state.plan_repo.load_plan(first_plan["id"]).steps[0].id
    )
    first_scope = first_output["revision"]["state"]["current_project_snapshot"][
        "scope"
    ]
    first_history_missing = next(
        item
        for item in first_output["missing_information_records"]
        if item["field_path"] == "historical_strategy_reviews"
    )

    answer = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "当前审批通过率由业务侧提供为 82%，口径尚未经过平台校验。"},
    )

    assert answer.status_code == 202, answer.text
    assert len(llm.calls) == llm_call_count
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert len(plans) == 2
    latest = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert latest.steps[0].inputs["expected_revision"] == 1
    assert "scope" not in latest.steps[0].inputs
    output = client.app.state.plan_repo.load_step_output(latest.steps[0].id)
    assert output["revision"]["state"]["current_project_snapshot"]["scope"] == (
        first_scope
    )
    approval = output["revision"]["state"]["current_project_snapshot"][
        "status_fields"
    ]["approval"]
    assert approval["availability"] == "present"
    assert approval["origin"] == "user"
    assert approval["value"] == "当前审批通过率由业务侧提供为 82%，口径尚未经过平台校验。"
    assert "unverified" in approval["note"]
    missing = {
        item["field_path"]: item
        for item in output["missing_information_records"]
    }
    assert missing["current.status_fields.approval"]["status"] == "provided"
    assert missing["current.status_fields.approval"]["answer_source_ref"]["kind"] == (
        "agent_message"
    )
    assert missing["historical_strategy_reviews"]["answer_source_ref"] == (
        first_history_missing["answer_source_ref"]
    )
    assert missing["historical_strategy_reviews"]["answered_at"] == (
        first_history_missing["answered_at"]
    )


def test_pending_context_does_not_intercept_a_new_strategy_workflow_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id, _ = _create_task(client, tmp_path)
    llm = _ProjectContextLLM(
        {
            "as_of": "2026-06-30",
            "scope": None,
            "business_context": {},
            "explicit_unavailable": [],
            "external_report_filenames": [],
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    first = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "整理策略项目现状，截止 2026-06-30。"},
    )
    assert first.status_code == 202, first.text
    llm_call_count = len(llm.calls)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "分析 income 的坏账率，做单变量。"},
    )

    assert response.status_code == 202, response.text
    assert len(llm.calls) == llm_call_count + 1
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == ["strategy_project_context"]
    current = load_current_strategy_project_context(
        SimpleNamespace(settings=client.app.state.settings),
        task_id=task_id,
    )
    assert current is not None
    assert current["revision"] == 1


def test_pending_context_unavailable_answer_is_null_and_ambiguous_answer_clarifies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id, _ = _create_task(client, tmp_path)
    llm = _ProjectContextLLM(
        {
            "as_of": "2026-06-30",
            "scope": None,
            "business_context": {},
            "explicit_unavailable": [],
            "external_report_filenames": [],
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    first = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "整理策略项目现状，截止 2026-06-30。"},
    )
    assert first.status_code == 202, first.text

    ambiguous = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "暂时没有。"},
    )
    assert ambiguous.status_code == 202, ambiguous.text
    assert ambiguous.json()["code"] == "strategy_project_context_answer_field_required"
    assert len(client.get(f"/api/tasks/{task_id}/plans").json()["plans"]) == 1

    unavailable = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "以上全部暂时没有。"},
    )
    assert unavailable.status_code == 202, unavailable.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    latest = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    output = client.app.state.plan_repo.load_step_output(latest.steps[0].id)
    records = output["missing_information_records"]
    assert records
    assert {item["status"] for item in records} == {"unavailable"}
    assert all(item["answer_source_ref"] is not None for item in records)
    status_fields = output["revision"]["state"]["current_project_snapshot"][
        "status_fields"
    ]
    assert all(field["value"] is None for field in status_fields.values())
