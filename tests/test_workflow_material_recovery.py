from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.driver_turn import DriverMessage, DriverTurn
from marvis.app import create_app
from marvis.agent.workflow_error_diagnostics import build_workflow_error_diagnostic
from marvis.agent.workflow_recovery import (
    answer_workflow_recovery_message,
    deterministic_workflow_recovery_reply,
    is_explicit_cancelled_workflow_resume,
    is_explicit_workflow_retry,
    is_workflow_repair_request,
    latest_unresolved_workflow_failure,
)
from marvis.db import TaskRepository
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.plugins.manifest import ToolRef


_REAL_RESOURCE_RETRY_COMMAND = (
    "请帮我解决这次资源超限问题，并从当前失败的“调参”步骤继续重试，不要从头执行。"
    "沿用已经确认的 188 个特征、split_tag、无样本权重，lgb、xgb、catboost "
    "各 40 轮，OOT 只报告、不参与选优。"
)

_CHECKPOINT_RESTART_RETRY_COMMAND = (
    "服务已恢复，请从当前失败的“调参”步骤继续。"
    "沿用已确认参数并复用已完成的 lgb、xgb、catboost 调参检查点，"
    "只重新汇总调参结果，不要重跑前序步骤或已完成算法。"
)


def _write_xlsx_with_csv_suffix(path: Path, frame: pd.DataFrame) -> None:
    workbook_path = path.with_suffix(".xlsx")
    frame.to_excel(workbook_path, index=False)
    workbook_path.replace(path)


def test_timestamp_serialization_failure_is_diagnosed_as_platform_recoverable() -> None:
    diagnostic = build_workflow_error_diagnostic(
        workflow="modeling",
        exc=TypeError("Object of type Timestamp is not JSON serializable"),
    )

    assert diagnostic["code"] == "platform_metadata_serialization_failed"
    assert diagnostic["cause"] == (
        "平台在保存日期字段画像时未先把 Timestamp 转成 JSON 可保存的日期文本；"
        "材料本身没有损坏。"
    )
    assert diagnostic["location"] == "数据集字段画像持久化"
    assert diagnostic["retryable"] is True
    assert diagnostic["auto_recoverable"] is True
    assert diagnostic["recovery_actions"] == [
        {"label": "由 Agent 重新执行", "command": "请帮我解决并重试"}
    ]


@pytest.mark.parametrize(
    "text",
    ["你可以帮我解决吗？", "请帮我解决并重试", "能否为我修复这个异常"],
)
def test_auto_repair_request_recognizes_user_authorization(text: str) -> None:
    assert is_workflow_repair_request(text) is True


@pytest.mark.parametrize(
    "text",
    ["先不要帮我解决", "是什么问题？", "为什么不能解决", "我再考虑一下"],
)
def test_auto_repair_request_rejects_questions_and_negation(text: str) -> None:
    assert is_workflow_repair_request(text) is False


@pytest.mark.parametrize(
    "text",
    ["继续", "继续当前步骤", "从当前步骤继续执行", "恢复执行", "重试当前步骤"],
)
def test_cancelled_workflow_resume_requires_explicit_command(text: str) -> None:
    assert is_explicit_cancelled_workflow_resume(text) is True


@pytest.mark.parametrize(
    "text",
    ["先不要继续", "为什么停止了？", "能继续吗？", "把算法改成 xgb 再继续", "先这样"],
)
def test_cancelled_workflow_resume_rejects_negation_questions_and_adjustments(
    text: str,
) -> None:
    assert is_explicit_cancelled_workflow_resume(text) is False


def test_auto_recoverable_fallback_offers_agent_repair_without_blame() -> None:
    diagnostic = build_workflow_error_diagnostic(
        workflow="modeling",
        exc=TypeError("Object of type Timestamp is not JSON serializable"),
    )

    reply = deterministic_workflow_recovery_reply(diagnostic)

    assert "无需修改或重新上传材料" in reply
    assert "请帮我解决并重试" in reply
    assert "材料修正后" not in reply


def _single_sample_source(root: Path) -> Path:
    source = root / "misnamed_workbook"
    source.mkdir()
    count = 120
    split = np.array(["train"] * 60 + ["test"] * 30 + ["oot"] * 30)
    _write_xlsx_with_csv_suffix(
        source / "sample.csv",
        pd.DataFrame(
            {
                "cust_id": np.arange(count),
                "feature_a": np.linspace(0.0, 1.0, count),
                "feature_b": np.linspace(1.0, 0.0, count),
                "model_flag": split,
                "y": np.arange(count) % 2,
            }
        ),
    )
    return source


def _dated_parquet_source(root: Path) -> Path:
    source = root / "dated_parquet"
    source.mkdir()
    count = 120
    pd.DataFrame(
        {
            "cust_id": np.arange(count),
            "apply_date": pd.date_range("2026-01-01", periods=count, freq="D"),
            "feature_a": np.linspace(0.0, 1.0, count),
            "model_flag": ["train"] * 60 + ["test"] * 30 + ["oot"] * 30,
            "y": np.arange(count) % 2,
        }
    ).to_parquet(source / "sample.parquet", index=False)
    return source


def _join_source(root: Path) -> Path:
    source = root / "misnamed_join_workbooks"
    source.mkdir()
    _write_xlsx_with_csv_suffix(
        source / "sample.csv",
        pd.DataFrame({"cust_id": np.arange(20), "y": np.arange(20) % 2}),
    )
    _write_xlsx_with_csv_suffix(
        source / "features.csv",
        pd.DataFrame({"cust_id": np.arange(20), "feature_a": np.linspace(0.0, 1.0, 20)}),
    )
    return source


def _vintage_source(root: Path) -> Path:
    source = root / "misnamed_vintage_workbook"
    source.mkdir()
    _write_xlsx_with_csv_suffix(
        source / "vintage.csv",
        pd.DataFrame(
            {
                "cust_id": np.arange(120),
                "cohort": ["202601", "202602"] * 60,
                "mob": np.tile(np.arange(12), 10),
                "y": np.arange(120) % 2,
            }
        ),
    )
    return source


def _source_for_task(root: Path, task_type: str) -> Path:
    if task_type == "data_join":
        return _join_source(root)
    if task_type == "vintage":
        return _vintage_source(root)
    return _single_sample_source(root)


def _malformed_csv_source(root: Path) -> Path:
    source = root / "malformed_csv"
    source.mkdir()
    # Header + eight valid one-field rows means the malformed record is line 10.
    (source / "broken.csv").write_text(
        "value\n" + "\n".join(["ok"] * 8) + "\nbad,extra\n",
        encoding="utf-8",
    )
    return source


def _task_payload(*, source: Path, task_type: str, run_mode: str) -> dict:
    payload = {
        "model_name": "材料恢复回归",
        "validator": "qa",
        "source_dir": str(source),
        "task_type": task_type,
        "run_mode": run_mode,
    }
    if task_type == "strategy":
        payload.update(
            {
                "target_col": "y",
                "score_col": "feature_a",
                "strategy_input": {"entry_mode": "strategy_analysis"},
            }
        )
    elif task_type == "vintage":
        payload.update({"target_col": "y", "time_col": "cohort"})
    return payload


def _start_until_material_scan(
    client: TestClient,
    task_id: str,
    task_type: str,
):
    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    if task_type != "vintage" or started.json().get("status") == "error":
        return started
    selected = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "标准 Vintage"},
    )
    if selected.json().get("status") == "error":
        return selected
    return client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "材料已上传"},
    )


@pytest.mark.parametrize(
    "task_type",
    ["data_join", "feature_analysis", "modeling", "strategy", "vintage"],
)
@pytest.mark.e2e
def test_workflow_recovers_xlsx_content_with_csv_suffix_and_explains_it(
    tmp_path: Path,
    task_type: str,
) -> None:
    app_root = tmp_path / "app"
    client = TestClient(create_app(app_root))
    source = _source_for_task(app_root, task_type)
    created = client.post(
        "/api/tasks",
        json=_task_payload(source=source, task_type=task_type, run_mode="manual"),
    )
    assert created.status_code == 200, created.text

    started = _start_until_material_scan(
        client,
        created.json()["id"],
        task_type,
    )

    assert started.status_code == 202, started.text
    assert started.json()["status"] != "error"
    assistant_messages = [
        message for message in started.json()["messages"] if message["role"] == "assistant"
    ]
    assert assistant_messages
    assert all(not message.get("metadata", {}).get("error") for message in assistant_messages)
    notices = [
        notice
        for message in assistant_messages
        for notice in message.get("metadata", {}).get("ingest_notices", [])
    ]
    assert notices
    assert {notice["code"] for notice in notices} == {"extension_content_mismatch"}
    assert all(notice["detected_format"] == "xlsx" for notice in notices)
    assert any("扩展名是 `.csv`" in message["content"] for message in assistant_messages)
    assert any("已按 Excel 工作簿读取" in message["content"] for message in assistant_messages)


@pytest.mark.parametrize(
    "task_type",
    ["data_join", "feature_analysis", "modeling", "strategy", "vintage"],
)
@pytest.mark.e2e
def test_workflow_csv_parse_failure_is_structured_and_marks_driver_job_failed(
    tmp_path: Path,
    task_type: str,
) -> None:
    app_root = tmp_path / "app"
    client = TestClient(create_app(app_root))
    source = _malformed_csv_source(app_root)
    created = client.post(
        "/api/tasks",
        json=_task_payload(source=source, task_type=task_type, run_mode="manual"),
    )
    task_id = created.json()["id"]

    started = _start_until_material_scan(client, task_id, task_type)

    assert started.status_code == 202, started.text
    assert started.json()["status"] == "error"
    message = [item for item in started.json()["messages"] if item["role"] == "assistant"][-1]
    assert "broken.csv" in message["content"]
    assert "第 10 行" in message["content"]
    assert "预期 1 列" in message["content"]
    assert "实际 2 列" in message["content"]
    assert "处理建议" in message["content"]
    assert "Error tokenizing data" not in message["content"]
    metadata = message["metadata"]
    assert metadata["error"] is True
    diagnostic = metadata["error_diagnostic"]
    assert diagnostic["code"] == "csv_field_count_mismatch"
    assert diagnostic["location"] == "broken.csv · 第 10 行"
    assert diagnostic["line_number"] == 10
    assert diagnostic["expected_fields"] == 1
    assert diagnostic["actual_fields"] == 2
    assert diagnostic["retryable"] is True
    assert diagnostic["technical_detail"].startswith("ParserError:")
    failure = metadata["failure_envelope"]
    assert failure["schema_version"] == "failure.v1"
    assert failure["error_kind"] == "csv_parse"
    assert failure["retryable"] is True
    job = TaskRepository(client.app.state.settings.db_path).get_latest_job(
        task_id, kind="driver"
    )
    assert job is not None
    assert job["status"] == "failed"
    assert job["error_name"] == "CsvParseError"
    assert client.app.state.plan_repo.list_plans_for_task(task_id) == []


class _GateLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps({"action": "confirm", "reason": "继续"})


@pytest.mark.e2e
def test_historical_timestamp_failure_is_explained_and_agent_can_recover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_root = tmp_path / "app"
    client = TestClient(create_app(app_root))
    source = _dated_parquet_source(app_root)
    gate_llm = _GateLLM()
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda request, task, payload: gate_llm,
    )
    created = client.post(
        "/api/tasks",
        json=_task_payload(source=source, task_type="modeling", run_mode="agent"),
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]
    repo = TaskRepository(client.app.state.settings.db_path)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="模型开发未完成。",
        metadata={
            "error": True,
            "error_diagnostic": {
                "schema_version": "workflow_error.v1",
                "workflow": "modeling",
                "code": "workflow_execution_failed",
                "phase": "prepare",
                "title": "模型开发未完成",
                "summary": "模型开发在准备阶段停止。",
                "cause": "平台遇到了未能自动恢复的执行异常。",
                "location": "当前工作流步骤",
                "actions": ["展开技术信息。"],
                "retryable": True,
                "technical_detail": (
                    "TypeError: Object of type Timestamp is not JSON serializable"
                ),
            },
            "failure_envelope": {"retryable": True, "failed_step_id": None},
        },
    )

    history = client.get(f"/api/tasks/{task_id}/agent/messages")

    assert history.status_code == 200, history.text
    historical_error = history.json()["messages"][-1]["metadata"]["error_diagnostic"]
    assert historical_error["code"] == "platform_metadata_serialization_failed"
    assert historical_error["auto_recoverable"] is True
    assert "材料本身没有损坏" in historical_error["cause"]

    repaired = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "你可以帮我解决吗？"},
    )

    assert repaired.status_code == 202, repaired.text
    assert repaired.json()["status"] != "error"
    messages = repaired.json()["messages"]
    assert len(
        [item for item in messages if item.get("metadata", {}).get("error_diagnostic")]
    ) == 1
    assert any(
        item.get("role") == "assistant"
        and not item.get("metadata", {}).get("error")
        and "Timestamp" not in item.get("content", "")
        for item in messages
    )


@pytest.mark.parametrize(
    "task_type",
    ["data_join", "feature_analysis", "modeling", "strategy", "vintage"],
)
@pytest.mark.e2e
def test_agent_can_chat_after_material_failure_and_only_explicit_retry_reruns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_type: str,
) -> None:
    app_root = tmp_path / "app"
    client = TestClient(create_app(app_root))
    source = _malformed_csv_source(app_root)
    gate_llm = _GateLLM()
    router_llm = _GateLLM()
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda request, task, payload: gate_llm,
    )
    if task_type == "strategy":
        monkeypatch.setattr(
            "marvis.agent.validation_app_service.driver_llm_client",
            lambda request, task: router_llm,
        )
    created = client.post(
        "/api/tasks",
        json=_task_payload(source=source, task_type=task_type, run_mode="agent"),
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]

    started = _start_until_material_scan(client, task_id, task_type)
    assert started.status_code == 202, started.text
    assert started.json()["status"] == "error"
    initial_errors = [
        item
        for item in started.json()["messages"]
        if item.get("metadata", {}).get("error_diagnostic")
    ]
    assert len(initial_errors) == 1

    chatted = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "你好"},
    )

    assert chatted.status_code == 202, chatted.text
    assert chatted.json()["status"] == "message_saved"
    chat_messages = chatted.json()["messages"]
    assert len(
        [
            item
            for item in chat_messages
            if item.get("metadata", {}).get("error_diagnostic")
        ]
    ) == 1
    recovery = [item for item in chat_messages if item["role"] == "assistant"][-1]
    assert recovery["metadata"]["intent"] == "workflow_recovery_chat"
    assert recovery["metadata"]["recovery_of_message_id"] == initial_errors[0]["id"]
    assert "重新读取" in recovery["content"]

    question = (
        "为什么策略分析失败？"
        if task_type == "strategy"
        else "为什么重试会失败？"
    )
    questioned = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": question},
    )

    assert questioned.status_code == 202, questioned.text
    assert questioned.json()["status"] == "message_saved"
    assert _last_recovery_intent(questioned.json()["messages"]) == (
        "workflow_recovery_chat"
    )
    if task_type == "strategy":
        assert router_llm.calls == []
    assert len(
        [
            item
            for item in questioned.json()["messages"]
            if item.get("metadata", {}).get("error_diagnostic")
        ]
    ) == 1

    retried = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "重新读取"},
    )

    assert retried.status_code == 202, retried.text
    assert retried.json()["status"] == "error"
    assert len(
        [
            item
            for item in retried.json()["messages"]
            if item.get("metadata", {}).get("error_diagnostic")
        ]
    ) == 2


def _last_recovery_intent(messages: list[dict]) -> str | None:
    assistants = [item for item in messages if item.get("role") == "assistant"]
    if not assistants:
        return None
    return (assistants[-1].get("metadata") or {}).get("intent")


@pytest.mark.parametrize(
    "text",
    [
        "重新读取",
        "请帮我重试一下",
        "请帮我解决并从当前失败步骤重试。",
        _REAL_RESOURCE_RETRY_COMMAND,
        _CHECKPOINT_RESTART_RETRY_COMMAND,
        "从当前失败的调参步骤继续",
        "继续失败步骤",
        "复用检查点",
        "复用检查点，从当前失败步骤继续",
        "重新执行失败步骤",
        "重试当前步骤，不要从头执行",
        "重试当前步骤，不用样本权重，OOT 不参与选优",
        "请重试当前失败步骤，不要重新执行整个流程",
        "请重试当前失败步骤，不要重新执行已完成步骤",
        "重试当前失败的模型交付动作，不要从头执行。沿用已生成的模型报告和 LightGBM 冠军产物。",
        "不用样本权重并从当前失败的调参步骤继续重试",
        "OOT 不参与选优并从当前失败步骤继续重试",
        "不使用样本权重，重试当前步骤",
        "再试一次",
        "开始特征分析",
        "retry",
        "try again",
    ],
)
def test_explicit_workflow_retry_requires_a_command(text: str) -> None:
    assert is_explicit_workflow_retry(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "你好",
        "继续",
        "为什么重试失败",
        "怎么重新读取？",
        "可以重新读取吗",
        "可以重试吗",
        "先别重试",
        "不要重试",
        "不要重新读取",
        "重试当前步骤，但是不要了",
        "请重试，算了",
        "请重试，然后暂停",
        "重试会发生什么",
        "重试失败了",
        "重新执行还是失败了",
        "开始模型开发后报错了",
        "重试按钮没有用",
        "重新读取没有解决问题",
        "重试调参，改成5轮",
        "重试当前步骤，参数调整为5轮",
        "重试当前步骤，算法用 xgb",
        "从失败步骤继续，改用 catboost",
        "重试，去掉样本权重",
        "重试，增加到100轮",
        "重试，减少为10轮",
    ],
)
def test_retry_questions_and_negations_remain_chat(text: str) -> None:
    assert is_explicit_workflow_retry(text) is False


def test_real_failed_step_retry_command_routes_to_driver_not_recovery_chat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_root = tmp_path / "app"
    client = TestClient(create_app(app_root))
    source = _single_sample_source(app_root)
    created = client.post(
        "/api/tasks",
        json=_task_payload(source=source, task_type="modeling", run_mode="agent"),
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]
    plan_id = "retry-plan"
    step_id = "retry-tune"
    client.app.state.plan_repo.create_plan(
        Plan(
            id=plan_id,
            task_id=task_id,
            goal="modeling",
            source="template",
            template_id="modeling",
            autonomy_level=1,
            status=PlanStatus.FAILED,
            steps=[
                PlanStep(
                    id=step_id,
                    plan_id=plan_id,
                    index=0,
                    title="调参",
                    tool_ref=ToolRef("modeling", "tune_hyperparameters"),
                    inputs={},
                    depends_on=[],
                    post_checks=[],
                    needs_confirmation=True,
                    status=StepStatus.FAILED,
                    error="tool worker RSS exceeded memory limit",
                )
            ],
        )
    )
    repo = TaskRepository(client.app.state.settings.db_path)
    failure = repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="调参失败。",
        metadata={
            "error": True,
            "error_diagnostic": {
                "workflow": "plan_driver",
                "code": "workflow_step_failed",
                "summary": "调参未完成。",
                "cause": "tool worker RSS exceeded memory limit",
                "retryable": True,
            },
            "failure_envelope": {
                "failed_step_id": step_id,
                "retryable": True,
                "run_seq": 0,
            },
        },
    )

    class _RetryDriver:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def retry_failed_step(self, plan_id, step_id, **kwargs) -> DriverTurn:
            self.calls.append({"plan_id": plan_id, "step_id": step_id, **kwargs})
            return DriverTurn(
                plan_id,
                PlanStatus.RUNNING.value,
                [
                    DriverMessage(
                        "chat",
                        "已进入确定性失败步骤恢复路径。",
                        {"plan_id": plan_id, "step_id": step_id},
                    )
                ],
            )

    retry_driver = _RetryDriver()
    recovery_calls: list[str] = []

    def forbidden_recovery_response(**kwargs):
        recovery_calls.append(str(kwargs.get("user_message") or ""))
        raise AssertionError("explicit retry must not call the recovery chat LLM")

    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda request, task, payload: _GateLLM(),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._driver",
        lambda runtime: retry_driver,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service._driver_recovery_responder",
        lambda *args, **kwargs: forbidden_recovery_response,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _REAL_RESOURCE_RETRY_COMMAND},
    )

    assert response.status_code == 202, response.text
    assert retry_driver.calls == [
        {
            "plan_id": plan_id,
            "step_id": step_id,
            "run_seq": 1,
            "preserve_target_confirmation": True,
        }
    ]
    assert recovery_calls == []
    messages = response.json()["messages"]
    retry_message = next(
        item
        for item in messages
        if item.get("role") == "user"
        and (item.get("metadata") or {}).get("intent") == "workflow_recovery_retry"
    )
    assert retry_message["metadata"]["recovery_of_message_id"] == failure["id"]
    assert not any(
        (item.get("metadata") or {}).get("intent") == "workflow_recovery_chat"
        for item in messages
        if item.get("id") != failure["id"]
    )


def test_cancelled_plan_continue_resumes_same_step_without_rebuilding_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_root = tmp_path / "app"
    client = TestClient(create_app(app_root))
    source = _single_sample_source(app_root)
    created = client.post(
        "/api/tasks",
        json=_task_payload(source=source, task_type="modeling", run_mode="agent"),
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]
    plan_id = "cancelled-plan"
    step_id = "cancelled-tune"
    client.app.state.plan_repo.create_plan(
        Plan(
            id=plan_id,
            task_id=task_id,
            goal="modeling",
            source="template",
            template_id="modeling",
            autonomy_level=1,
            status=PlanStatus.CANCELLED,
            steps=[
                PlanStep(
                    id=step_id,
                    plan_id=plan_id,
                    index=0,
                    title="调参",
                    tool_ref=ToolRef("modeling", "tune_hyperparameters"),
                    inputs={},
                    depends_on=[],
                    post_checks=[],
                    needs_confirmation=True,
                    status=StepStatus.FAILED,
                    error="用户已停止当前动作",
                )
            ],
        )
    )
    repo = TaskRepository(client.app.state.settings.db_path)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="调参已停止；检查点已保留。",
        metadata={
            "intent": "execution_cancelled",
            "cancelled": True,
            "plan_id": plan_id,
            "step_id": step_id,
            "run_seq": 4,
        },
    )

    class _RetryDriver:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def retry_failed_step(self, plan_id, step_id, **kwargs) -> DriverTurn:
            self.calls.append({"plan_id": plan_id, "step_id": step_id, **kwargs})
            return DriverTurn(
                plan_id,
                PlanStatus.RUNNING.value,
                [
                    DriverMessage(
                        "chat",
                        "已从停止的调参步骤继续。",
                        {"plan_id": plan_id, "step_id": step_id},
                    )
                ],
            )

    retry_driver = _RetryDriver()
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda request, task, payload: _GateLLM(),
    )
    monkeypatch.setattr("marvis.agent.turn_handlers._driver", lambda runtime: retry_driver)

    question = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "为什么停止了？"},
    )
    assert question.status_code == 202, question.text
    assert retry_driver.calls == []
    question_messages = question.json()["messages"]
    assert (question_messages[-1].get("metadata") or {}).get("intent") == (
        "workflow_cancelled_chat"
    )
    assert not any(
        (item.get("metadata") or {}).get("join_c1") for item in question_messages
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "继续"},
    )

    assert response.status_code == 202, response.text
    assert retry_driver.calls == [
        {
            "plan_id": plan_id,
            "step_id": step_id,
            "run_seq": 5,
            "preserve_target_confirmation": True,
        }
    ]
    messages = response.json()["messages"]
    assert any(
        (item.get("metadata") or {}).get("intent")
        == "workflow_cancelled_resume"
        for item in messages
    )
    assert not any((item.get("metadata") or {}).get("join_c1") for item in messages)


def test_legacy_restart_notice_routes_explicit_retry_to_failed_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old restart notices lacked a failure envelope and must remain retryable.

    This is the exact persisted shape that previously made the explicit retry
    fall through to modeling setup and ask the user to confirm file roles again.
    """

    app_root = tmp_path / "app"
    client = TestClient(create_app(app_root))
    source = _single_sample_source(app_root)
    created = client.post(
        "/api/tasks",
        json=_task_payload(source=source, task_type="modeling", run_mode="agent"),
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]
    plan_id = "restart-plan"
    step_id = "restart-tune"
    client.app.state.plan_repo.create_plan(
        Plan(
            id=plan_id,
            task_id=task_id,
            goal="modeling",
            source="template",
            template_id="modeling",
            autonomy_level=1,
            status=PlanStatus.FAILED,
            steps=[
                PlanStep(
                    id=step_id,
                    plan_id=plan_id,
                    index=0,
                    title="调参",
                    tool_ref=ToolRef("modeling", "tune_hyperparameters"),
                    inputs={},
                    depends_on=[],
                    post_checks=[],
                    needs_confirmation=True,
                    status=StepStatus.FAILED,
                    error="interrupted during running; explicit retry required",
                )
            ],
        )
    )
    restart_notice = TaskRepository(client.app.state.settings.db_path).add_agent_message(
        task_id,
        role="assistant",
        stage="failure",
        content=(
            "服务已重启，计划已暂停在当前步骤；中间产物已保留，"
            "可点击『重试步骤』从失败步继续。"
        ),
        metadata={
            "plan_interrupted_by_restart": True,
            "plan_resumed_at_confirmation": False,
            "plan_id": plan_id,
            "streaming": False,
        },
    )
    # Reproduce the stale setup gate appended by the buggy fall-through.  The
    # FAILED plan remains authoritative and an explicit recovery command must
    # not be trapped by this role-confirmation prompt.
    TaskRepository(client.app.state.settings.db_path).add_agent_message(
        task_id,
        role="assistant",
        stage="gate",
        content="请先确认建模文件角色与目标列。",
        metadata={"join_c1": {"files": []}},
    )

    class _RetryDriver:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def retry_failed_step(self, plan_id, step_id, **kwargs) -> DriverTurn:
            self.calls.append({"plan_id": plan_id, "step_id": step_id, **kwargs})
            return DriverTurn(
                plan_id,
                PlanStatus.RUNNING.value,
                [
                    DriverMessage(
                        "chat",
                        "已进入确定性失败步骤恢复路径。",
                        {"plan_id": plan_id, "step_id": step_id},
                    )
                ],
            )

    retry_driver = _RetryDriver()
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda request, task, payload: _GateLLM(),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._driver",
        lambda runtime: retry_driver,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service._driver_recovery_responder",
        lambda *args, **kwargs: (
            lambda **response_kwargs: (
                "当前尚未启动重试，请明确回复“重试当前步骤”。",
                {"fallback": True},
            )
        ),
    )

    continued = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "继续"},
    )
    assert continued.status_code == 202, continued.text
    assert retry_driver.calls == []
    assert _last_recovery_intent(continued.json()["messages"]) == (
        "workflow_recovery_chat"
    )
    assert sum(
        "请先确认建模文件角色" in str(item.get("content") or "")
        for item in continued.json()["messages"]
    ) == 1

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _CHECKPOINT_RESTART_RETRY_COMMAND},
    )

    assert response.status_code == 202, response.text
    assert retry_driver.calls == [
        {
            "plan_id": plan_id,
            "step_id": step_id,
            "run_seq": 1,
            "preserve_target_confirmation": True,
        }
    ]
    messages = response.json()["messages"]
    retry_message = next(
        item
        for item in messages
        if item.get("role") == "user"
        and (item.get("metadata") or {}).get("intent") == "workflow_recovery_retry"
    )
    assert retry_message["metadata"]["recovery_of_message_id"] == restart_notice["id"]
    assert sum(
        "请先确认建模文件角色" in str(item.get("content") or "")
        for item in messages
    ) == 1


def test_existing_legacy_parser_error_is_available_as_recovery_context() -> None:
    failure = latest_unresolved_workflow_failure(
        [
            {
                "id": "old-error",
                "role": "assistant",
                "stage": "chat",
                "content": (
                    "特征分析出错：Error tokenizing data. C error: "
                    "Expected 1 fields in line 10, saw 2"
                ),
                "metadata": {"error": True},
            }
        ],
        workflow="feature_analysis",
    )

    assert failure is not None
    assert failure.message_id == "old-error"
    assert failure.diagnostic["code"] == "csv_field_count_mismatch"
    assert failure.diagnostic["line_number"] == 10


def test_historical_c0_parquet_alias_failure_is_platform_recoverable() -> None:
    failure = latest_unresolved_workflow_failure(
        [
            {
                "id": "old-c0-error",
                "role": "assistant",
                "stage": "chat",
                "content": "切分样本失败。",
                "metadata": {
                    "error": True,
                    "error_diagnostic": {
                        "workflow": "plan_driver",
                        "code": "workflow_step_failed",
                        "summary": "切分样本失败。",
                        "cause": "No match for FieldRef.Name(C0) in : int64",
                        "technical_detail": "No match for FieldRef.Name(C0) in : int64",
                        "retryable": True,
                    },
                    "failure_envelope": {
                        "failed_step_id": "plan-step-1",
                        "retryable": True,
                    },
                },
            }
        ],
        workflow="modeling",
    )

    assert failure is not None
    assert failure.diagnostic["code"] == "platform_parquet_column_alias_failed"
    assert failure.diagnostic["auto_recoverable"] is True
    assert "无需修改或重新上传材料" in failure.diagnostic["actions"][0]
    assert failure.failure_envelope["failed_step_id"] == "plan-step-1"


class _RecoveryLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return "我在。第 10 行多了一个字段，请先核对分隔符；修正后再明确重试。"


def test_recovery_llm_receives_only_allowlisted_failure_evidence() -> None:
    llm = _RecoveryLLM()
    diagnostic = {
        "workflow": "feature_analysis",
        "code": "csv_field_count_mismatch",
        "summary": "第 10 行字段数不一致。",
        "cause": "分隔符或引号不一致。",
        "location": "broken.csv · 第 10 行",
        "actions": ["核对分隔符。"],
        "retryable": True,
        "technical_detail": "secret traceback /Users/private/material.csv",
        "exception_type": "CsvParseError",
    }

    content, metadata = answer_workflow_recovery_message(
        user_message="这是什么问题？",
        diagnostic=diagnostic,
        client=llm,
    )

    assert "第 10 行" in content
    assert metadata["fallback"] is False
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["caller"] == "workflow_recovery_chat"
    assert "secret traceback" not in call["user_prompt"]
    assert "/Users/private" not in call["user_prompt"]
    assert "technical_detail" not in call["user_prompt"]


class _FalsePromiseRecoveryLLM:
    def complete(self, **kwargs) -> str:
        return (
            "收到您的授权。我将从失败的调参步骤重试，请稍候；"
            "重试过程中会自动增加容器内存限制。"
        )


class _StaticRecoveryLLM:
    def __init__(self, content: str):
        self.content = content

    def complete(self, **kwargs) -> str:
        return self.content


def test_recovery_chat_replaces_unsupported_execution_promise() -> None:
    content, metadata = answer_workflow_recovery_message(
        user_message="请先说明重试方案。",
        diagnostic={
            "workflow": "modeling",
            "code": "workflow_step_failed",
            "summary": "调参未完成。",
            "cause": "tool worker RSS exceeded memory limit",
            "retryable": True,
        },
        client=_FalsePromiseRecoveryLLM(),
    )

    assert content.startswith("当前尚未启动重试或修改执行配置。")
    assert "收到您的授权" not in content
    assert "请稍候" not in content
    assert "自动增加容器内存" not in content
    assert metadata == {
        "fallback": True,
        "fallback_reason": "unauthorized_execution_claim",
        "llm_response_replaced": True,
    }


@pytest.mark.parametrize(
    "promise",
    [
        "我会从失败步骤重试。",
        "接下来会重新执行。",
        "Agent 将会重试。",
        "接下来我会重新跑一遍失败步骤。",
        "已进入恢复流程。",
        "我会马上重跑当前调参步骤。",
        "已经发起对失败步骤的再次运行。",
        "恢复作业已经进入队列。",
    ],
)
def test_recovery_chat_replaces_future_execution_promise_variants(promise: str) -> None:
    content, metadata = answer_workflow_recovery_message(
        user_message="请先说明重试方案。",
        diagnostic={"summary": "调参未完成。", "cause": "memory limit", "retryable": True},
        client=_StaticRecoveryLLM(promise),
    )

    assert content.startswith("当前尚未启动重试或修改执行配置。")
    assert metadata["fallback_reason"] == "unauthorized_execution_claim"


def test_recovery_chat_allows_conditional_retry_notice() -> None:
    notice = "当前尚未启动重试；你明确授权后才会重试。"
    content, metadata = answer_workflow_recovery_message(
        user_message="请先说明重试方案。",
        diagnostic={"summary": "调参未完成。", "cause": "memory limit", "retryable": True},
        client=_StaticRecoveryLLM(notice),
    )

    assert content.startswith("当前尚未启动重试或修改执行配置。")
    assert notice in content
    assert metadata == {"fallback": False}
