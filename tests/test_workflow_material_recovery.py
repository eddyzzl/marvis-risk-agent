from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.agent.workflow_recovery import (
    answer_workflow_recovery_message,
    is_explicit_workflow_retry,
    latest_unresolved_workflow_failure,
)
from marvis.db import TaskRepository


def _write_xlsx_with_csv_suffix(path: Path, frame: pd.DataFrame) -> None:
    workbook_path = path.with_suffix(".xlsx")
    frame.to_excel(workbook_path, index=False)
    workbook_path.replace(path)


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
        "为什么重试失败",
        "怎么重新读取？",
        "可以重新读取吗",
        "先别重试",
        "不要重新读取",
        "重试会发生什么",
    ],
)
def test_retry_questions_and_negations_remain_chat(text: str) -> None:
    assert is_explicit_workflow_retry(text) is False


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
