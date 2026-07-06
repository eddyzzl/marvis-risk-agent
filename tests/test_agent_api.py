import json
from pathlib import Path
from types import SimpleNamespace

from docx import Document
from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient
import pytest

from marvis.app import create_app
from marvis.agent_memory.models import MemoryCandidate
from marvis.agent_memory.store import AgentMemoryStore
from marvis.db import PluginRepository, TaskRepository
from marvis.domain import TaskStatus
from marvis.pipeline import NOTEBOOK_STAGE_FAILURE_PREFIX


REQUIRED_AGENT_CONCLUSIONS = {
    "TEXT:pressure_test_summary": "压力测试显示模型在主要数据源缺失场景下整体稳定。",
    "TEXT:pressure_impact_recommendation": "建议继续使用模型，同时监控缺失率较高的数据源。",
    "TEXT:final_validation_conclusion": (
        "从当前验证结果看，模型开发过程、分数一致性、区分效果、稳定性和压力测试结果整体满足验证要求，"
        "建议在后续投产监测中持续关注 OOT 样本表现和关键变量稳定性。"
    ),
}


def _client(tmp_path: Path) -> TestClient:
    app = create_app(tmp_path)
    return TestClient(app)


def test_normalize_effort_falls_back_to_high():
    from marvis.agent.validation_app_service import normalize_effort

    assert normalize_effort("low") == "low"
    assert normalize_effort("Medium") == "medium"
    assert normalize_effort("high") == "high"
    assert normalize_effort("ultra") == "high"
    assert normalize_effort(None) == "high"
    assert normalize_effort("") == "high"


def test_validation_agent_job_loop_lives_outside_api_module():
    from marvis import api
    from marvis.agent import validation_runner

    assert validation_runner.run_agent_validation_job.__module__ == "marvis.agent.validation_runner"
    assert api._run_agent_validation_job_impl is validation_runner.run_agent_validation_job


def test_validation_agent_evidence_helper_lives_outside_api_module():
    from marvis import api
    from marvis.agent import validation_evidence

    assert (
        validation_evidence.agent_evidence_from_settings.__module__
        == "marvis.agent.validation_evidence"
    )
    assert api._agent_evidence_from_settings is validation_evidence.agent_evidence_from_settings


def test_validation_agent_stage_impl_lives_outside_api_module():
    from marvis import api
    from marvis.agent import validation_stages

    assert (
        validation_stages.run_agent_scan_stage.__module__
        == "marvis.agent.validation_stages"
    )
    assert api._run_agent_scan_stage_impl is validation_stages.run_agent_scan_stage
    assert api._run_agent_metrics_stage_impl is validation_stages.run_agent_metrics_stage


def _create_task(client: TestClient, tmp_path: Path, *, run_mode: str = "agent") -> str:
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "A卡",
            "validator": "qa",
            "source_dir": str(tmp_path),
            "run_mode": run_mode,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _advance_to_writing_artifacts(repo: TaskRepository, task_id: str) -> None:
    repo.update_status(task_id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task_id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task_id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task_id,
        TaskStatus.COMPUTING_METRICS,
        "metrics",
        expected=TaskStatus.EXECUTED,
    )
    repo.update_status(
        task_id,
        TaskStatus.WRITING_ARTIFACTS,
        "writing",
        expected=TaskStatus.COMPUTING_METRICS,
    )


def _configure_llm(client: TestClient) -> None:
    response = client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    assert response.status_code == 200, response.text


def test_llm_settings_api_masks_keys_and_lists_enabled_models(tmp_path):
    client = _client(tmp_path)

    response = client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
                {
                    "model_id": "m2",
                    "enabled": False,
                    "display_name": "停用模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "disabled-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["models"][0]["has_api_key"] is True
    assert "api_key" not in payload["models"][0]
    assert [model["model_id"] for model in payload["enabled_models"]] == ["m1"]

    loaded = client.get("/api/settings/llm")

    assert loaded.status_code == 200
    assert loaded.json() == payload


def test_agent_message_without_llm_config_returns_guidance(tmp_path):
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "开始验证"},
    )

    assert response.status_code == 409
    assert "请先在设置中配置至少一个启用的大模型" in response.json()["detail"]


def test_agent_messages_endpoint_supports_after_id_cursor(tmp_path):
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    first = repo.add_agent_message(task_id, role="user", stage="chat", content="first")
    second = repo.add_agent_message(task_id, role="assistant", stage="scan", content="second")

    response = client.get(
        f"/api/tasks/{task_id}/agent/messages",
        params={"after_id": first["id"]},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["incremental"] is True
    assert [message["id"] for message in payload["messages"]] == [second["id"]]
    assert payload["has_more"] is False
    assert payload["limit"] is None

    third = repo.add_agent_message(task_id, role="assistant", stage="scan", content="third")
    limited_response = client.get(
        f"/api/tasks/{task_id}/agent/messages",
        params={"after_id": first["id"], "limit": 1},
    )

    assert limited_response.status_code == 200, limited_response.text
    limited_payload = limited_response.json()
    assert limited_payload["incremental"] is True
    assert limited_payload["has_more"] is True
    assert limited_payload["limit"] == 1
    assert [message["id"] for message in limited_payload["messages"]] == [second["id"]]

    fallback_response = client.get(
        f"/api/tasks/{task_id}/agent/messages",
        params={"after_id": "missing-message"},
    )

    assert fallback_response.status_code == 200, fallback_response.text
    fallback_payload = fallback_response.json()
    assert fallback_payload["incremental"] is False
    assert [message["id"] for message in fallback_payload["messages"]] == [first["id"], second["id"], third["id"]]

    capped_response = client.get(
        f"/api/tasks/{task_id}/agent/messages",
        params={"limit": 9999},
    )
    assert capped_response.status_code == 200, capped_response.text
    assert capped_response.json()["limit"] == 500


def test_agent_chat_uses_relevant_memory_and_audits_use(tmp_path, monkeypatch):
    observed: dict = {}

    def fake_answer_chat_message(**kwargs):
        observed.update(kwargs)
        return "上一版A卡模型KS为20，当前版本KS为30，效果更好。", {
            "fallback": False,
            "memory_references": [
                {
                    "id": kwargs["memory_context"]["memories"][0]["id"],
                    "memory_type": "model_experience",
                    "source_task_id": "task-history",
                    "confidence": "high",
                    "use_reason": "chat",
                }
            ],
        }

    monkeypatch.setattr("marvis.routers.validation_agent.answer_chat_message", fake_answer_chat_message)
    client = _client(tmp_path)
    _configure_llm(client)
    task_id = _create_task(client, tmp_path)
    store = AgentMemoryStore(tmp_path / "marvis.sqlite")
    memory = store.create(
        MemoryCandidate(
            memory_type="model_experience",
            summary="上一版A卡模型V2025在202601自营渠道KS为20。",
            payload={
                "ks": 20,
                "auc": 0.68,
                "psi": 0.06,
                "month": "202601",
                "channel": "自营",
                "model_name": "A卡",
                "model_version": "V2025",
                "scope": "贷前A卡",
                "source_task_id": "task-history",
                "important_feature_sources": ["征信"],
            },
            source_task_id="task-history",
            confidence="high",
        )
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "和上一版A卡模型比怎么样？", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert observed["memory_context"]["memories"][0]["id"] == memory.id
    assistant = response.json()["messages"][-1]
    assert assistant["metadata"]["memory_references"][0]["id"] == memory.id
    events = store.list_events(memory.id)
    assert [event["event_type"] for event in events] == ["create", "retrieve", "use"]
    assert events[-1]["task_id"] == task_id
    assert events[-1]["message_id"] == assistant["id"]


def test_agent_chat_persists_explicit_user_preference_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "marvis.routers.validation_agent.answer_chat_message",
        lambda **kwargs: ("已记住。", {"fallback": False}),
    )
    client = _client(tmp_path)
    _configure_llm(client)
    task_id = _create_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "请记住：回答时先写核心风险，再写补充说明", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    store = AgentMemoryStore(tmp_path / "marvis.sqlite")
    memories = store.list_entries(memory_type="user_preference")
    assert len(memories) == 1
    memory = memories[0]
    assert memory.summary == "回答时先写核心风险，再写补充说明"
    assert memory.payload == {"preference": "回答时先写核心风险，再写补充说明"}
    assert memory.source_task_id == task_id
    assert memory.source_message_id == response.json()["messages"][-2]["id"]


def test_agent_start_queues_streaming_opening_before_background_scan(
    tmp_path,
    monkeypatch,
):
    queued: list[tuple[str, str]] = []

    def fake_run_agent_validation_job(
        job_id,
        settings,
        task_id,
        model_profile,
        opening_message_id,
        stage=None,
        stage_message_id=None,
        acceptance_mode=None,
        stage_instruction=None,
    ):
        queued.append((task_id, opening_message_id, acceptance_mode, stage_instruction))

    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_agent_validation_job",
        fake_run_agent_validation_job,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/start",
        json={"model_id": "m1", "effort": "high"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["stage"] == "scan"
    messages = payload["messages"]
    assert queued == [(task_id, messages[0]["id"], "normal", None)]
    assert [message["role"] for message in messages] == ["assistant"]
    assert messages[0]["stage"] == "chat"
    assert messages[0]["content"] == ""
    assert messages[0]["metadata"]["streaming"] is True
    assert messages[0]["metadata"]["model_id"] == "m1"


def test_agent_auto_accept_runs_all_remaining_stages_without_continue_prompts(
    tmp_path,
    monkeypatch,
):
    from marvis.api import _run_agent_validation_job

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    job_id = repo.start_job(task_id, "agent")
    calls: list[str] = []

    def fake_open_stage(_repo, **kwargs):
        calls.append(f"open:{kwargs['stage']}")

    def fake_scan_stage(repo, _settings, task_id, _model_profile, *, auto_accept=False):
        calls.append(f"scan:{auto_accept}")
        repo.update_status(task_id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
        return True

    def fake_repro_stage(repo, _settings, task_id, _model_profile, *, auto_accept=False):
        calls.append(f"reproducibility:{auto_accept}")
        repo.update_status(task_id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
        repo.update_status(task_id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
        return True

    def fake_metrics_stage(repo, _settings, task_id, _model_profile, *, auto_accept=False):
        calls.append(f"metrics:{auto_accept}")
        repo.update_status(
            task_id,
            TaskStatus.COMPUTING_METRICS,
            "metrics",
            expected=TaskStatus.EXECUTED,
        )
        repo.update_status(
            task_id,
            TaskStatus.WRITING_ARTIFACTS,
            "writing",
            expected=TaskStatus.COMPUTING_METRICS,
        )
        return True

    def fake_word_stage(
        repo,
        _settings,
        task_id,
        _model_profile,
        *,
        draft_message_id=None,
        auto_accept=False,
        rewrite_instruction=None,
    ):
        calls.append(f"word:{auto_accept}:{draft_message_id}:{rewrite_instruction}")
        repo.update_agent_report_conclusions(task_id, REQUIRED_AGENT_CONCLUSIONS, expected_revision=0)
        return True

    monkeypatch.setattr("marvis.agent.validation_app_service.open_agent_stage", fake_open_stage)
    monkeypatch.setattr("marvis.agent.validation_app_service.run_agent_scan_stage", fake_scan_stage)
    monkeypatch.setattr("marvis.agent.validation_app_service.run_agent_reproducibility_stage", fake_repro_stage)
    monkeypatch.setattr("marvis.agent.validation_app_service.run_agent_metrics_stage", fake_metrics_stage)
    monkeypatch.setattr("marvis.agent.validation_app_service.run_agent_word_conclusion_stage", fake_word_stage)

    _run_agent_validation_job(
        job_id,
        SimpleNamespace(db_path=tmp_path / "marvis.sqlite"),
        task_id,
        {"model_id": "m1", "effort": "high"},
        acceptance_mode="auto_accept",
    )

    assert calls == [
        "open:scan",
        "scan:True",
        "open:reproducibility",
        "reproducibility:True",
        "open:metrics",
        "metrics:True",
        "open:word_conclusion_draft",
        "word:True:None:None",
    ]
    assert not [
        message
        for message in repo.list_agent_messages(task_id)
        if message.get("metadata", {}).get("awaiting_next_stage")
    ]


def test_agent_auto_accept_does_not_add_received_intro_when_auto_advancing(
    tmp_path,
    monkeypatch,
):
    from marvis.api import _run_agent_validation_job

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    job_id = repo.start_job(task_id, "agent")

    monkeypatch.setattr(
        "marvis.agent.validation_app_service.compose_agent_start_message",
        lambda **_kwargs: ("开始验证材料。", {"fallback": True}),
    )

    def fake_scan_stage(repo, _settings, task_id, _model_profile, *, auto_accept=False):
        assert auto_accept is True
        repo.update_status(task_id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
        return True

    def fake_repro_stage(repo, _settings, task_id, _model_profile, *, auto_accept=False):
        assert auto_accept is True
        repo.update_status(task_id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
        repo.update_status(task_id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
        return True

    def fake_metrics_stage(repo, _settings, task_id, _model_profile, *, auto_accept=False):
        assert auto_accept is True
        repo.update_status(
            task_id,
            TaskStatus.COMPUTING_METRICS,
            "metrics",
            expected=TaskStatus.EXECUTED,
        )
        repo.update_status(
            task_id,
            TaskStatus.WRITING_ARTIFACTS,
            "writing",
            expected=TaskStatus.COMPUTING_METRICS,
        )
        return True

    def fake_word_stage(
        repo,
        _settings,
        task_id,
        _model_profile,
        *,
        draft_message_id=None,
        auto_accept=False,
        rewrite_instruction=None,
    ):
        assert auto_accept is True
        repo.update_agent_report_conclusions(task_id, REQUIRED_AGENT_CONCLUSIONS, expected_revision=0)
        return True

    monkeypatch.setattr("marvis.agent.validation_app_service.run_agent_scan_stage", fake_scan_stage)
    monkeypatch.setattr("marvis.agent.validation_app_service.run_agent_reproducibility_stage", fake_repro_stage)
    monkeypatch.setattr("marvis.agent.validation_app_service.run_agent_metrics_stage", fake_metrics_stage)
    monkeypatch.setattr("marvis.agent.validation_app_service.run_agent_word_conclusion_stage", fake_word_stage)

    _run_agent_validation_job(
        job_id,
        SimpleNamespace(db_path=tmp_path / "marvis.sqlite"),
        task_id,
        {"model_id": "m1", "effort": "high"},
        acceptance_mode="auto_accept",
    )

    assistant_texts = [
        message["content"]
        for message in repo.list_agent_messages(task_id)
        if message["role"] == "assistant"
    ]
    # Scan is the entry stage and emits its own substantive opening via
    # compose_agent_start_message, so the "接下来开始执行模型材料完备性验证。"
    # banner is intentionally suppressed for stage="scan". Later stages keep
    # the banner because it follows the previous stage's wrap-up.
    assert [text for text in assistant_texts if text.startswith("接下来开始执行")] == [
        "接下来开始执行模型可复现性验证。",
        "接下来开始执行模型效果&稳定性验证。",
        "接下来开始执行报告结论草稿生成。",
    ]
    assert "开始验证材料。" in assistant_texts
    assert not [text for text in assistant_texts if text.startswith("收到")]


def test_agent_auto_accept_dispatch_does_not_precreate_received_intro_for_next_stage(
    tmp_path,
    monkeypatch,
):
    from marvis.api import _dispatch_agent_validation_job

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    repo.update_status(task_id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    task = repo.get_task(task_id)
    background_tasks = BackgroundTasks()

    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_agent_validation_job",
        lambda *_args, **_kwargs: None,
    )

    payload = _dispatch_agent_validation_job(
        repo=repo,
        task=task,
        settings=SimpleNamespace(db_path=tmp_path / "marvis.sqlite"),
        model_profile={"model_id": "m1", "effort": "high"},
        acceptance_mode="auto_accept",
        background_tasks=background_tasks,
    )

    assert payload["stage"] == "reproducibility"
    assert not [
        message["content"]
        for message in payload["messages"]
        if message["role"] == "assistant" and message["content"].startswith("收到")
    ]


def test_agent_word_conclusions_auto_accept_confirms_and_generates_report(
    tmp_path,
    monkeypatch,
):
    from marvis.api import _run_agent_word_conclusion_stage

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    report_calls: list[str] = []

    def fake_generate_word_conclusions(**_kwargs):
        return REQUIRED_AGENT_CONCLUSIONS, {"source": "test"}

    def fake_run_report_stage(*, task_id, settings):
        report_calls.append(task_id)
        local_repo = TaskRepository(tmp_path / "marvis.sqlite")
        local_repo.update_status(
            task_id,
            TaskStatus.SUCCEEDED,
            "word generated",
            expected={TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED},
        )

    monkeypatch.setattr("marvis.agent.validation_app_service.generate_word_conclusions", fake_generate_word_conclusions)
    monkeypatch.setattr("marvis.agent.validation_app_service.run_report_stage", fake_run_report_stage)
    monkeypatch.setattr("marvis.agent.validation_app_service.agent_pipeline_settings", lambda _settings, _task: object())
    monkeypatch.setattr("marvis.agent.validation_app_service.agent_evidence_from_settings_impl", lambda _settings, _task_id: {})

    assert _run_agent_word_conclusion_stage(
        repo,
        SimpleNamespace(db_path=tmp_path / "marvis.sqlite"),
        task_id,
        {"model_id": "m1", "effort": "high"},
        auto_accept=True,
    )

    messages = repo.list_agent_messages(task_id)
    assert report_calls == [task_id]
    assert repo.get_task(task_id).status == TaskStatus.SUCCEEDED
    assert repo.get_report_values(task_id)[0] == REQUIRED_AGENT_CONCLUSIONS
    audit = PluginRepository(tmp_path / "marvis.sqlite").list_audit(
        kind="report.agent_conclusions.confirm",
    )
    assert len(audit) == 1
    assert audit[0]["target_ref"] == task_id
    assert audit[0]["detail"]["auto_accept"] is True
    assert audit[0]["detail"]["keys"] == sorted(REQUIRED_AGENT_CONCLUSIONS)
    assert [message["stage"] for message in messages] == [
        "word_conclusion_draft",
        "word_conclusion_confirmed",
        "word_report_ready",
    ]
    assert not any(message.get("metadata", {}).get("awaiting_confirmation") for message in messages)


def test_agent_word_conclusion_stage_passes_rewrite_instruction_to_llm(
    tmp_path,
    monkeypatch,
):
    from marvis.api import _run_agent_word_conclusion_stage

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    seen_instructions: list[str | None] = []

    def fake_generate_word_conclusions(**kwargs):
        seen_instructions.append(kwargs.get("user_instruction"))
        return REQUIRED_AGENT_CONCLUSIONS, {"source": "test"}

    monkeypatch.setattr("marvis.agent.validation_app_service.generate_word_conclusions", fake_generate_word_conclusions)
    monkeypatch.setattr("marvis.agent.validation_app_service.agent_evidence_from_settings_impl", lambda _settings, _task_id: {})

    assert _run_agent_word_conclusion_stage(
        repo,
        SimpleNamespace(db_path=tmp_path / "marvis.sqlite"),
        task_id,
        {"model_id": "m1", "effort": "high"},
        rewrite_instruction="重新写草稿，强化压力测试高风险数据源说明",
    )

    assert seen_instructions == ["重新写草稿，强化压力测试高风险数据源说明"]
    messages = repo.list_agent_messages(task_id)
    assert messages[-2]["stage"] == "word_conclusion_draft"
    assert messages[-1]["metadata"]["awaiting_confirmation"] is True


def test_agent_word_conclusion_stage_passes_prior_stage_summaries_to_llm(
    tmp_path,
    monkeypatch,
):
    from marvis.api import _run_agent_word_conclusion_stage

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="reproducibility",
        content="分数一致性阶段已通过，最大差异为 0。",
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="普通聊天不应进入 Word 草稿证据。",
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="metrics",
        content="效果稳定性解读：OOT KS 约 33 个点，PSI 总体可接受。",
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_draft",
        content="旧草稿不应作为新草稿证据。",
    )
    seen_evidence: list[dict] = []

    def fake_generate_word_conclusions(**kwargs):
        seen_evidence.append(kwargs["evidence"])
        return REQUIRED_AGENT_CONCLUSIONS, {"source": "test"}

    monkeypatch.setattr("marvis.agent.validation_app_service.generate_word_conclusions", fake_generate_word_conclusions)
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.agent_evidence_from_settings_impl",
        lambda _settings, _task_id: {"validation_results": {"model_name": "A卡"}},
    )

    assert _run_agent_word_conclusion_stage(
        repo,
        SimpleNamespace(db_path=tmp_path / "marvis.sqlite"),
        task_id,
        {"model_id": "m1", "effort": "high"},
    )

    assert seen_evidence
    assert seen_evidence[0]["visible_stage_summaries"] == [
        {
            "stage": "reproducibility",
            "content": "分数一致性阶段已通过，最大差异为 0。",
        },
        {
            "stage": "metrics",
            "content": "效果稳定性解读：OOT KS 约 33 个点，PSI 总体可接受。",
        },
    ]


def test_metrics_stage_evidence_drops_oversized_roc_curve_arrays():
    # Raw roc_ks_curves points are ~12 MB on realistic tasks; if they reach
    # the LLM the prompt is too large and the call falls back to the
    # one-line placeholder text. The slim helper must drop them while
    # preserving the AUC/KS summary numbers the analysis depends on.
    from marvis.agent.service import _metrics_stage_evidence

    payload_curve = [{"x": i / 100, "y": (i / 100) ** 0.5} for i in range(2000)]
    evidence = {
        "scan": {"checks": []},
        "validation_results": {
            "model_name": "demo",
            "effectiveness": {
                "overall": [{"split": "train", "ks": 0.34, "auc": 0.79}],
                "monthly_ks": [{"month": "202503", "ks": 0.33}],
                "psi_stability_table": [{"split": "oot", "psi": 0.05}],
                "roc_ks_curves": {
                    "train": {"roc": payload_curve, "ks": payload_curve},
                    "oot": {"roc": payload_curve, "ks": payload_curve},
                },
            },
            "stress_test": {"scenarios": []},
        },
    }

    scoped = _metrics_stage_evidence(evidence)

    effectiveness = scoped["validation_results"]["effectiveness"]
    assert "roc_ks_curves" not in effectiveness
    assert effectiveness["overall"] == [{"split": "train", "ks": 0.34, "auc": 0.79}]
    assert effectiveness["monthly_ks"] == [{"month": "202503", "ks": 0.33}]
    assert effectiveness["psi_stability_table"] == [{"split": "oot", "psi": 0.05}]


def test_word_conclusion_draft_evidence_drops_oversized_roc_curves():
    # The Word draft stage uses the global evidence dict; the same large
    # roc_ks_curves arrays must not reach the LLM or the three drafts fall
    # back to placeholder text.
    from marvis.agent.service import _stage_scoped_evidence

    payload_curve = [{"x": i, "y": i} for i in range(500)]
    evidence = {
        "scan": {"checks": []},
        "notebook_steps": [],
        "contract": {},
        "reproducibility": {"summary": {}},
        "validation_results": {
            "effectiveness": {
                "overall": [{"split": "train", "ks": 0.34}],
                "roc_ks_curves": {"train": payload_curve},
            },
        },
        "report_fields": {"text_values": {}},
        "visible_stage_summaries": [],
    }

    scoped = _stage_scoped_evidence("word_conclusion_draft", evidence)

    assert "roc_ks_curves" not in scoped["validation_results"]["effectiveness"]
    assert scoped["validation_results"]["effectiveness"]["overall"] == [
        {"split": "train", "ks": 0.34}
    ]
    # report_fields and other non-validation keys are kept intact.
    assert scoped["report_fields"] == {"text_values": {}}


def test_word_conclusion_draft_evidence_is_compact_and_keeps_business_summaries():
    from marvis.agent.service import _stage_scoped_evidence

    huge_rows = [
        {
            "row_index": index,
            "score_code_model": "0." + ("1" * 600),
            "score_submitted_pmml": "0." + ("2" * 600),
        }
        for index in range(80)
    ]
    huge_curve = [{"x": index, "y": index} for index in range(1200)]
    huge_bin_table = [{"bin": index, "bad_rate": index / 1000} for index in range(100)]
    evidence = {
        "scan": {
            "checks": [{"label": "Notebook 文件", "status": "success", "message": "已识别"}],
            "artifacts": [{"path": "/tmp/" + ("x" * 4000)}],
        },
        "notebook_steps": {
            "steps": [
                {
                    "id": "notebook-long",
                    "title": "长源码步骤",
                    "source_previews": ["print('" + ("x" * 5000) + "')"],
                }
            ]
        },
        "contract": {"source": "RMC_SAMPLE_DF = " + ("x" * 5000)},
        "reproducibility": {
            "summary": {"status": "pass", "max_abs_diff": 0.0, "mismatch_count": 0},
            "rows": huge_rows,
            "sample_size": 200,
            "seed": 42,
        },
        "validation_results": {
            "model_name": "A卡",
            "algorithm": "lgb",
            "reproducibility": {
                "summary": {"status": "pass", "max_abs_diff": 0.0},
                "rows": huge_rows,
            },
            "basic_info": {
                "split_summary": [{"split": "train", "sample_count": 1000}],
                "feature_importance": [
                    {"rank": index, "feature": f"f{index}", "importance": index}
                    for index in range(60)
                ],
            },
            "effectiveness": {
                "overall": [{"split": "oot", "ks": 0.33, "auc": 0.72}],
                "monthly_ks": [{"month": f"2025{index:02d}", "ks": 0.3} for index in range(1, 30)],
                "monthly_psi": [
                    {"month": f"2025{index:02d}", "psi_vs_train": 0.02}
                    for index in range(1, 30)
                ],
                "psi_stability_table": huge_bin_table,
                "bin_tables": {"train": huge_bin_table},
                "roc_ks_curves": {"train": huge_curve},
            },
            "stress_test": {
                "baseline": {"ks": 0.33, "sample_count": 1000, "bin_table": huge_bin_table},
                "per_category": [
                    {
                        "category": f"数据源{index}",
                        "ks_after": 0.30,
                        "ks_delta": -0.03,
                        "psi_vs_baseline": 0.02,
                        "bin_table": huge_bin_table,
                        "dropped_features": [f"f{index}", f"g{index}"],
                    }
                    for index in range(30)
                ],
            },
            "overfitting_check": {"status": "pass", "train_oot_abs_diff": 0.01},
        },
        "report_fields": {"text_values": {"TEXT:report_title": "A卡模型验证"}},
        "visible_stage_summaries": [
            {"stage": "reproducibility", "content": "分数一致性已通过，最大差异为 0。"},
            {"stage": "metrics", "content": "OOT KS 约 33 个点，PSI 总体可接受。"}
        ],
    }

    scoped = _stage_scoped_evidence("word_conclusion_draft", evidence)
    payload = json.dumps(scoped, ensure_ascii=False)

    assert len(payload) < 16000
    assert "notebook_steps" not in scoped
    assert "contract" not in scoped
    assert scoped["reproducibility"]["summary"]["status"] == "pass"
    assert "rows" not in scoped["reproducibility"]
    validation_results = scoped["validation_results"]
    assert "rows" not in validation_results["reproducibility"]
    assert "roc_ks_curves" not in validation_results["effectiveness"]
    assert "bin_tables" not in validation_results["effectiveness"]
    assert len(validation_results["effectiveness"]["monthly_ks"]) <= 12
    assert len(validation_results["basic_info"]["feature_importance"]) <= 20
    assert "bin_table" not in validation_results["stress_test"]["baseline"]
    assert "bin_table" not in validation_results["stress_test"]["per_category"][0]
    assert scoped["visible_stage_summaries"][0]["stage"] == "reproducibility"
    assert scoped["visible_stage_summaries"][1]["content"].startswith("OOT KS")


def test_word_conclusion_prompt_requires_detailed_final_conclusion(tmp_path):
    from marvis.agent.service import _stage_prompt

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    task = TaskRepository(tmp_path / "marvis.sqlite").get_task(task_id)

    prompt = json.loads(
        _stage_prompt(
            task=task,
            stage="word_conclusion_draft",
            evidence={
                "validation_results": {
                    "effectiveness": {
                        "overall": [{"split": "oot", "ks": 0.33, "auc": 0.72}]
                    }
                },
                "visible_stage_summaries": [
                    {"stage": "metrics", "content": "OOT KS 约 33 个点，PSI 总体可接受。"}
                ],
            },
        )
    )

    instructions = prompt["instructions"]
    assert "TEXT:final_validation_conclusion" in instructions
    assert "1 到 2 个自然段" in instructions
    assert "Notebook 可复现性" in instructions
    assert "分数一致性" in instructions
    assert "压力测试主要发现" in instructions
    assert "不能退化成一句泛泛结论" in instructions


def test_scan_stage_prompt_tells_llm_completed_materials_are_not_missing(tmp_path):
    from marvis.agent.service import _stage_prompt

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    task = TaskRepository(tmp_path / "marvis.sqlite").get_task(task_id)

    prompt = json.loads(
        _stage_prompt(
            task=task,
            stage="scan",
            evidence={
                "checks": [
                    {"label": "Notebook 文件", "status": "success", "message": "已识别：model.ipynb"},
                    {"label": "样本数据", "status": "success", "message": "已识别：sample.feather"},
                    {"label": "PMML 模型", "status": "success", "message": "已识别：model.pmml"},
                    {"label": "数据字典", "status": "success", "message": "已识别：dictionary.xlsx"},
                    {
                        "label": "Notebook RMC 契约",
                        "status": "success",
                        "message": "已定义 RMC_SAMPLE_DF / RMC_SCORE_FN / RMC_TARGET_COL / RMC_ALGORITHM",
                    },
                ],
            },
        )
    )

    interpretation = prompt["evidence"]["scan_interpretation"]
    assert interpretation["required_materials_complete"] is True
    assert interpretation["missing_required_materials"] == []
    assert "pickle" in prompt["instructions"]
    assert "KS/PSI/AUC" in prompt["instructions"]
    assert "不得说缺少" in prompt["instructions"]
    assert "未检测到模型文件" in prompt["instructions"]


def test_scan_stage_summary_overrides_llm_missing_claim_when_materials_complete(
    tmp_path,
    monkeypatch,
):
    from marvis.agent.service import summarize_stage

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **_kwargs):
            return "根据扫描结果，缺少 PMML 或 pickle、验证样本评分输出和 KS/PSI/AUC 中间结果。"

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    task = TaskRepository(tmp_path / "marvis.sqlite").get_task(task_id)

    content, metadata = summarize_stage(
        task=task,
        stage="scan",
        evidence={
            "checks": [
                {"label": "Notebook 文件", "status": "success", "message": "已识别：model.ipynb"},
                {"label": "样本数据", "status": "success", "message": "已识别：sample.feather"},
                {"label": "PMML 模型", "status": "success", "message": "已识别：model.pmml"},
                {"label": "数据字典", "status": "success", "message": "已识别：dictionary.xlsx"},
                {
                    "label": "Notebook RMC 契约",
                    "status": "success",
                    "message": "已定义 RMC_SAMPLE_DF / RMC_SCORE_FN / RMC_TARGET_COL / RMC_ALGORITHM",
                },
            ],
        },
        model_profile={"model_id": "m1"},
        fallback="材料扫描完成，平台已识别必需验证材料。",
    )

    assert content == (
        "材料完备性检查已完成，Notebook、样本数据、PMML 模型、数据字典和 "
        "Notebook RMC 契约均已通过；当前未发现必需材料缺失。"
    )
    assert metadata["guarded_scan_summary"] is True


def test_scan_stage_summary_overrides_llm_undetected_claim_when_materials_complete(
    tmp_path,
    monkeypatch,
):
    from marvis.agent.service import summarize_stage

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **_kwargs):
            return (
                "材料扫描完成，已识别出Notebook代码中包含样本划分、模型训练及变量重要性排序等步骤。"
                "但未检测到PMML模型文件、模型参数导出或验证输出指标。"
            )

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    task = TaskRepository(tmp_path / "marvis.sqlite").get_task(task_id)

    content, metadata = summarize_stage(
        task=task,
        stage="scan",
        evidence={
            "checks": [
                {"label": "Notebook 文件", "status": "success", "message": "已识别：model.ipynb"},
                {"label": "样本数据", "status": "success", "message": "已识别：sample.feather"},
                {"label": "PMML 模型", "status": "success", "message": "已识别：model.pmml"},
                {"label": "数据字典", "status": "success", "message": "已识别：dictionary.xlsx"},
                {
                    "label": "Notebook RMC 契约",
                    "status": "success",
                    "message": "已定义 RMC_SAMPLE_DF / RMC_SCORE_FN / RMC_TARGET_COL / RMC_ALGORITHM",
                },
            ],
        },
        model_profile={"model_id": "m1"},
        fallback="材料扫描完成，平台已识别必需验证材料。",
    )

    assert content == (
        "材料完备性检查已完成，Notebook、样本数据、PMML 模型、数据字典和 "
        "Notebook RMC 契约均已通过；当前未发现必需材料缺失。"
    )
    assert metadata["guarded_scan_summary"] is True


def test_agent_plain_chat_question_uses_llm_answer_instead_of_start_guidance(
    tmp_path,
    monkeypatch,
):
    calls = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            calls.append(kwargs)
            on_delta = kwargs.get("on_delta")
            if on_delta:
                on_delta("PMML 是一种")
                on_delta("模型部署交换格式。")
            return "PMML 是一种模型部署交换格式。"

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "可以给我解释一下什么是 PMML 吗", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "message_saved"
    messages = payload["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["stage"] == "chat"
    assert messages[1]["content"] == "PMML 是一种模型部署交换格式。"
    assert "需要开始时" not in messages[1]["content"]
    assert messages[1]["metadata"]["streaming"] is False
    assert messages[1]["metadata"]["streamed"] is True
    assert calls
    assert "PMML" in calls[0]["user_prompt"]
    prompt = json.loads(calls[0]["user_prompt"])
    assert "task_id" not in prompt["task"]
    assert task_id not in calls[0]["user_prompt"]
    assert prompt["task"]["model_name"] == "A卡"


def test_agent_plain_chat_follow_up_uses_same_task_conversation_memory(
    tmp_path,
    monkeypatch,
):
    calls = []
    responses = iter(
        [
            "模型可复现性验证的必要性在于确认开发环境分数与部署 PMML 分数一致。",
            "这会影响实际风控任务的审批一致性、策略阈值可信度和上线后问题定位。",
        ]
    )

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            calls.append(json.loads(kwargs["user_prompt"]))
            return next(responses)

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)

    first = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "做模型可复现性验证的必要性是什么", "model_id": "m1"},
    )
    second = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "对实际风控任务有什么影响", "model_id": "m1"},
    )

    assert first.status_code == 202, first.text
    assert second.status_code == 202, second.text
    assert len(calls) == 2
    follow_up_prompt = calls[1]
    assert "conversation" not in follow_up_prompt
    memory = follow_up_prompt["conversation_memory"]
    assert memory["scope"] == "same_agent_task"
    assert memory["previous_user_question"] == "做模型可复现性验证的必要性是什么"
    assert "模型可复现性验证的必要性" in memory["previous_assistant_answer"]
    assert "省略主语" in memory["follow_up_guidance"]
    assert any(
        message["content"] == "做模型可复现性验证的必要性是什么"
        for message in memory["messages"]
    )
    assert any(
        "模型可复现性验证的必要性" in message["content"]
        for message in memory["messages"]
    )
    assert "上一轮用户问题" in follow_up_prompt["instructions"]
    assert "承接" in follow_up_prompt["instructions"]


def test_agent_greeting_chat_instructs_llm_to_answer_naturally(
    tmp_path,
    monkeypatch,
):
    calls = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            calls.append(kwargs)
            return "你好，我在。"

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "你好", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "message_saved"
    assert payload["messages"][-1]["content"] == "你好，我在。"
    prompt = json.loads(calls[0]["user_prompt"])
    assert prompt["user_message"] == "你好"
    assert "问候" in prompt["instructions"]
    assert "固定启动口令" in prompt["instructions"]
    assert "开始验证" not in prompt["instructions"]


def test_agent_greeting_chat_suppresses_repeated_start_guidance(
    tmp_path,
    monkeypatch,
):
    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **_kwargs):
            return "我可以协助启动验证、解释阶段结果，并在报告生成前起草三段 Word 结论。需要开始时，请输入“开始验证”。"

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "你好", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assistant_content = payload["messages"][-1]["content"]
    assert "你好" in assistant_content
    assert "开始验证" not in assistant_content
    assert payload["messages"][-1]["metadata"]["fallback"] is True
    assert payload["messages"][-1]["metadata"]["llm_response_replaced"] is True


def test_agent_chat_llm_error_fallback_does_not_show_start_command(
    tmp_path,
    monkeypatch,
):
    from marvis.llm_client import LLMClientError

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **_kwargs):
            raise LLMClientError("network failed")

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "今天聊点别的", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assistant_content = response.json()["messages"][-1]["content"]
    assert "无法调用大模型" in assistant_content
    assert "开始验证" not in assistant_content
    assert "输入" not in assistant_content


def test_agent_chat_question_receives_current_validation_context(
    tmp_path,
    monkeypatch,
):
    calls = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            calls.append(kwargs)
            return "这张图对应的 OOT KS 为 0.3215，PSI 为 0.0123，整体区分和稳定性需要结合阈值复核。"

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    output_dir = tmp_path / "tasks" / task_id / "outputs"
    output_dir.mkdir(parents=True)
    (output_dir / "validation_results.json").write_text(
        json.dumps(
            {
                "effectiveness": {
                    "overall": [
                        {
                            "split": "oot",
                            "ks": 0.321456,
                            "psi_vs_train": 0.012345,
                        }
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="metrics",
        content=f"效果图显示任务 {task_id} 的 OOT KS 为 0.3215，PSI 为 0.0123。",
    )
    repo.update_agent_report_conclusions(
        task_id,
        REQUIRED_AGENT_CONCLUSIONS,
        expected_revision=0,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "这张效果图和最终结论说明了什么？", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    prompt = json.loads(calls[0]["user_prompt"])
    assert "task_id" not in prompt["task"]
    assert task_id not in calls[0]["user_prompt"]
    assert prompt["task"]["model_name"] == "A卡"
    assert "conversation" not in prompt
    assert any(
        "A卡" in message["content"]
        for message in prompt["conversation_memory"]["messages"]
    )
    evidence = prompt["available_evidence"]
    assert evidence["validation_results"]["effectiveness"]["overall"][0]["ks"] == 0.321456
    assert evidence["report_fields"]["metric_values"]["TEXT:oot_ks"] == "0.3215"
    assert (
        evidence["report_fields"]["text_values"]["TEXT:final_validation_conclusion"]
        == REQUIRED_AGENT_CONCLUSIONS["TEXT:final_validation_conclusion"]
    )
    assert evidence["visible_stage_summaries"] == [
        {
            "stage": "metrics",
            "content": "效果图显示任务 A卡 的 OOT KS 为 0.3215，PSI 为 0.0123。",
        }
    ]
    assert "PSI 小于 0.10" in calls[0]["system_prompt"]
    assert "KS 0.30" in calls[0]["system_prompt"]
    assert "过拟合" in calls[0]["system_prompt"]
    assert "train-test" in calls[0]["system_prompt"]
    assert "相对 10%" in calls[0]["system_prompt"]
    assert "5 个点" in calls[0]["system_prompt"]
    assert "不能脱离模型场景" in prompt["instructions"]


def test_agent_evidence_adds_overfitting_check_from_split_ks(tmp_path):
    from marvis.api import _agent_evidence_from_settings

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    output_dir = tmp_path / "tasks" / task_id / "outputs"
    output_dir.mkdir(parents=True)
    (output_dir / "validation_results.json").write_text(
        json.dumps(
            {
                "effectiveness": {
                    "overall": [
                        {"split": "train", "ks": 0.42, "psi_vs_train": 0.0},
                        {"split": "test", "ks": 0.36, "psi_vs_train": 0.03},
                        {"split": "oot", "ks": 0.34, "psi_vs_train": 0.08},
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    evidence = _agent_evidence_from_settings(client.app.state.settings, task_id)
    overfitting = evidence["validation_results"]["overfitting_check"]

    assert overfitting["metric"] == "ks"
    assert overfitting["status"] == "fail"
    assert overfitting["train_ks"] == 0.42
    assert overfitting["test_ks"] == 0.36
    assert overfitting["oot_ks"] == 0.34
    assert round(overfitting["train_test_relative_diff"], 4) == 0.1429
    assert overfitting["train_test_threshold"] == 0.10
    assert overfitting["train_test_status"] == "fail"
    assert round(overfitting["train_oot_abs_diff"], 4) == 0.08
    assert overfitting["train_oot_threshold"] == 0.05
    assert overfitting["train_oot_status"] == "fail"


def test_agent_metrics_summary_prompt_uses_contextual_metric_guidance(
    tmp_path,
    monkeypatch,
):
    calls = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            calls.append(kwargs)
            return "OOT KS 为 0.3215，PSI 为 0.0123，整体表现可接受。"

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/summarize",
        json={"model_id": "m1"},
    )

    assert response.status_code == 200, response.text
    assert calls
    prompt = json.loads(calls[0]["user_prompt"])
    assert "PSI 小于 0.10" in calls[0]["system_prompt"]
    assert "KS 0.30" in calls[0]["system_prompt"]
    assert "过拟合" in calls[0]["system_prompt"]
    assert "train-test" in calls[0]["system_prompt"]
    assert "相对 10%" in calls[0]["system_prompt"]
    assert "5 个点" in calls[0]["system_prompt"]
    assert "高风险数据源" in calls[0]["system_prompt"]
    assert "中风险数据源" in calls[0]["system_prompt"]
    assert "低风险数据源" in calls[0]["system_prompt"]
    assert "PSI 小于 0.10" in prompt["instructions"]
    assert "KS 0.30" in prompt["instructions"]
    assert "过拟合" in prompt["instructions"]
    assert "train-test" in prompt["instructions"]
    assert "高风险数据源" in prompt["instructions"]
    assert "中风险数据源" in prompt["instructions"]
    assert "低风险数据源" in prompt["instructions"]
    assert "不能脱离模型场景" in prompt["instructions"]


def test_agent_stage_summary_strips_instruction_acknowledgement_preamble(
    tmp_path,
    monkeypatch,
):
    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **_kwargs):
            return (
                "好的，遵照您的指示。以下是针对模型 A卡 在“可复现性/分数一致性”阶段的验证分析。\n\n"
                "***\n\n"
                "结论\n当前阶段应重点关注 PMML 分数一致性。"
            )

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/summarize",
        json={"model_id": "m1"},
    )

    assert response.status_code == 200, response.text
    content = response.json()["message"]["content"]
    assert content == "结论\n当前阶段应重点关注 PMML 分数一致性。"
    assert "好的" not in content
    assert "遵照" not in content
    assert "以下是" not in content
    assert "***" not in content


def test_agent_stage_summary_prompt_uses_model_name_not_task_id(
    tmp_path,
    monkeypatch,
):
    calls = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            calls.append(kwargs)
            return "将基于 A卡 的验证结果进行分析。"

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/summarize",
        json={"model_id": "m1"},
    )

    assert response.status_code == 200, response.text
    assert calls
    prompt = json.loads(calls[0]["user_prompt"])
    assert "task_id" not in prompt["task"]
    assert task_id not in calls[0]["user_prompt"]
    assert prompt["task"]["model_name"] == "A卡"
    assert "模型名称" in prompt["instructions"]


def test_agent_question_about_validation_conclusion_does_not_restart_validation(
    tmp_path,
    monkeypatch,
):
    queued = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **_kwargs):
            return "这是一段围绕当前验证结论的解释。"

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_agent_validation_job",
        lambda *args: queued.append(args),
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "这个验证结论说明了什么？", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "message_saved"
    assert queued == []
    assert payload["messages"][-1]["content"] == "这是一段围绕当前验证结论的解释。"


def test_agent_start_message_runs_only_material_scan_and_waits_for_continue(
    tmp_path,
    monkeypatch,
):
    unexpected: list[str] = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            if prompt["stage"] == "agent_start":
                return "我将先说明任务，然后调用材料识别工具。"
            if prompt["stage"] == "scan":
                return "材料完备性检查已完成，四类材料均已通过。"
            return f"{prompt['stage']} summary"

    def fake_scan(repo, task, _settings):
        repo.update_status(
            task.id,
            TaskStatus.SCANNED,
            "scan ok",
            expected=TaskStatus.CREATED,
        )
        return {"checks": [], "artifacts": []}

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    monkeypatch.setattr("marvis.agent.validation_app_service.perform_scan_task", fake_scan)
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_notebook_stage",
        lambda **_kwargs: unexpected.append("notebook"),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_metrics_stage",
        lambda **_kwargs: unexpected.append("metrics"),
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "开始验证", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "accepted"
    assert unexpected == []
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    assert repo.get_task(task_id).status == TaskStatus.SCANNED
    messages = repo.list_agent_messages(task_id)
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "开始验证"
    assert messages[0]["metadata"]["intent"] == "advance"
    assert [message["stage"] for message in messages[1:]] == [
        "chat",
        "scan",
        "scan",
        "chat",
    ]
    assert "是否继续执行" in messages[-1]["content"]
    assert "模型可复现性验证" in messages[-1]["content"]


def test_agent_continue_from_scanned_runs_only_notebook_stage(
    tmp_path,
    monkeypatch,
):
    unexpected: list[str] = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            if prompt["stage"] == "reproducibility":
                return "模型可复现性验证已完成，分数一致性通过。"
            return f"{prompt['stage']} summary"

    def fake_notebook_stage(*, task_id, settings, stage_claimed):
        assert stage_claimed is True
        repo = TaskRepository(settings.db_path)
        repo.update_status(
            task_id,
            TaskStatus.EXECUTED,
            "notebook executed",
            expected=TaskStatus.RUNNING,
        )

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.perform_scan_task",
        lambda *_args, **_kwargs: unexpected.append("scan"),
    )
    monkeypatch.setattr("marvis.agent.validation_app_service.run_notebook_stage", fake_notebook_stage)
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_metrics_stage",
        lambda **_kwargs: unexpected.append("metrics"),
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    repo.update_status(
        task_id,
        TaskStatus.SCANNED,
        "scan ok",
        expected=TaskStatus.CREATED,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "继续下一步", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "accepted"
    assert unexpected == []
    assert repo.get_task(task_id).status == TaskStatus.EXECUTED
    messages = repo.list_agent_messages(task_id)
    assert messages[0]["content"] == "继续下一步"
    assert any(message["stage"] == "reproducibility" for message in messages)
    assert "模型效果&稳定性验证" in messages[-1]["content"]


def test_agent_continue_after_intervening_chat_dispatches_metrics_stage(
    tmp_path,
    monkeypatch,
):
    metrics_calls: list[str] = []
    chat_prompts: list[dict] = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            chat_prompts.append(prompt)
            if prompt["stage"] == "metrics":
                return "效果与稳定性验证已完成。"
            return "普通聊天回复"

    def fake_metrics_stage(*, task_id, settings, stage_claimed):
        assert stage_claimed is True
        metrics_calls.append(task_id)
        repo = TaskRepository(settings.db_path)
        repo.update_status(
            task_id,
            TaskStatus.WRITING_ARTIFACTS,
            "metrics generated",
            expected=TaskStatus.COMPUTING_METRICS,
        )

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_notebook_stage",
        lambda **_kwargs: pytest.fail("notebook stage should not rerun"),
    )
    monkeypatch.setattr("marvis.agent.validation_app_service.run_metrics_stage", fake_metrics_stage)
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    repo.update_status(
        task_id,
        TaskStatus.SCANNED,
        "scan ok",
        expected=TaskStatus.CREATED,
    )
    repo.update_status(
        task_id,
        TaskStatus.RUNNING,
        "notebook running",
        expected=TaskStatus.SCANNED,
    )
    repo.update_status(
        task_id,
        TaskStatus.EXECUTED,
        "模型可复现性验证完成。",
        expected=TaskStatus.RUNNING,
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="是否继续执行【模型效果&稳定性验证】？你可以先继续提问；需要继续时，请明确回复“继续”。",
        metadata={"awaiting_next_stage": "metrics"},
    )
    repo.add_agent_message(
        task_id,
        role="user",
        stage="chat",
        content="这个问题严重吗",
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="分数不一致会影响上线决策。",
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "先继续吧", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "accepted"
    assert metrics_calls == [task_id]
    assert repo.get_task(task_id).status == TaskStatus.WRITING_ARTIFACTS
    assert not any(prompt["stage"] == "chat" for prompt in chat_prompts)
    messages = repo.list_agent_messages(task_id)
    assert messages[-1]["metadata"]["awaiting_next_stage"] == "word_conclusion_draft"


def test_agent_continue_generate_report_after_intervening_chat_dispatches_word_draft(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []
    chat_prompts: list[dict] = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            calls.append(prompt["stage"])
            if prompt["stage"] == "word_conclusion_draft":
                return json.dumps(REQUIRED_AGENT_CONCLUSIONS, ensure_ascii=False)
            chat_prompts.append(prompt)
            return "普通分析总结"

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="是否继续执行【报告结论草稿生成】？你可以先继续提问；需要继续时，请明确回复“继续”。",
        metadata={"awaiting_next_stage": "word_conclusion_draft"},
    )
    repo.add_agent_message(
        task_id,
        role="user",
        stage="chat",
        content="压力测试的数据源后续怎么排查？",
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="可以先看高风险数据源的缺失率和 OOT 表现。",
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "好的，先继续生成报告吧，我后续排查", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "accepted"
    assert calls == ["word_conclusion_draft"]
    assert chat_prompts == []
    messages = repo.list_agent_messages(task_id)
    assert messages[-2]["stage"] == "word_conclusion_draft"
    assert messages[-2]["metadata"]["draft_values"] == REQUIRED_AGENT_CONCLUSIONS
    assert "压力测试总结" in messages[-2]["content"]
    assert messages[-1]["metadata"]["awaiting_confirmation"] is True


def test_agent_reproducibility_summary_prompt_excludes_other_stage_evidence(
    tmp_path,
    monkeypatch,
):
    calls = []
    unexpected: list[str] = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            calls.append(prompt)
            if prompt["stage"] == "reproducibility":
                return "分数一致性通过。"
            return f"{prompt['stage']} summary"

    def fake_notebook_stage(*, task_id, settings, stage_claimed):
        assert stage_claimed is True
        task_dir = settings.workspace / "tasks" / task_id
        execution_dir = task_dir / "execution"
        output_dir = task_dir / "outputs"
        execution_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        (execution_dir / "runtime_contract.json").write_text(
            json.dumps(
                {
                    "target_col": "long_y",
                    "algorithm": "lgb",
                    "pmml_output_field": "probability_1",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (execution_dir / "notebook_steps.json").write_text(
            json.dumps(
                [{"title": "分数一致性对比", "status": "passed"}],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (output_dir / "reproducibility_result.json").write_text(
            json.dumps(
                {
                    "sample_size": 10,
                    "summary": {
                        "match_count": 10,
                        "mismatch_count": 0,
                        "max_abs_diff": 0.0,
                        "status": "pass",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        repo = TaskRepository(settings.db_path)
        repo.update_status(
            task_id,
            TaskStatus.EXECUTED,
            "notebook executed",
            expected=TaskStatus.RUNNING,
        )

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.perform_scan_task",
        lambda *_args, **_kwargs: unexpected.append("scan"),
    )
    monkeypatch.setattr("marvis.agent.validation_app_service.run_notebook_stage", fake_notebook_stage)
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_metrics_stage",
        lambda **_kwargs: unexpected.append("metrics"),
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    repo.update_status(
        task_id,
        TaskStatus.SCANNED,
        "scan ok",
        expected=TaskStatus.CREATED,
    )
    task_dir = tmp_path / "tasks" / task_id
    (task_dir / "execution").mkdir(parents=True, exist_ok=True)
    (task_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (task_dir / "execution" / "scan_result.json").write_text(
        json.dumps({"checks": [{"name": "PMML 文件", "status": "passed"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (task_dir / "outputs" / "validation_results.json").write_text(
        json.dumps(
            {
                "effectiveness": {"overall": [{"split": "oot", "ks": 0.3215, "auc": 0.77}]},
                "stability": {"psi": 0.0123},
                "stress_test": {"per_category": [{"category": "征信", "psi_vs_baseline": 0.05}]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "继续下一步", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert unexpected == []
    repro_prompt = next(prompt for prompt in calls if prompt["stage"] == "reproducibility")
    assert set(repro_prompt["evidence"]) == {
        "notebook_steps",
        "contract",
        "reproducibility",
    }
    prompt_text = json.dumps(repro_prompt, ensure_ascii=False)
    assert "PMML 文件" not in prompt_text
    assert "0.3215" not in prompt_text
    assert "0.0123" not in prompt_text
    assert "不要分析材料完备性" in repro_prompt["instructions"]
    assert "不要分析 AUC、KS、PSI" in repro_prompt["instructions"]
    assert "不要给出整体验证结论" in repro_prompt["instructions"]


def test_agent_reproducibility_summary_prompt_compacts_large_score_rows(tmp_path):
    from marvis.agent.service import _stage_prompt

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    task = TaskRepository(tmp_path / "marvis.sqlite").get_task(task_id)
    rows = [
        {
            "row_index": index,
            "score_code_model": 0.100001 + index / 1_000_000,
            "score_submitted_pmml": 0.100002 + index / 1_000_000,
            "abs_diff": 0.000001,
            "matched": False,
        }
        for index in range(1000)
    ]

    prompt = json.loads(
        _stage_prompt(
            task=task,
            stage="reproducibility",
            evidence={
                "notebook_steps": {
                    "steps": [
                        {
                            "id": "system-repro-compare",
                            "title": "分数一致性对比",
                            "status": "succeeded",
                            "source_previews": ["very large source that should be dropped"],
                        }
                    ],
                    "cells": [
                        {
                            "cell_index": 39,
                            "outputs": ["large cell outputs should be dropped"],
                        }
                    ],
                },
                "contract": {
                    "algorithm": "lgb",
                    "target_col": "y",
                    "pmml_output_field": "probability_1",
                    "feature_columns": [f"x{i}" for i in range(200)],
                },
                "reproducibility": {
                    "sample_size": 1000,
                    "seed": 42,
                    "summary": {
                        "match_count": 0,
                        "mismatch_count": 1000,
                        "max_abs_diff": 0.0223,
                        "status": "fail",
                    },
                    "rows": rows,
                },
                "validation_results": {
                    "reproducibility": {
                        "summary": {"status": "pass"},
                        "rows": rows,
                    },
                    "effectiveness": {"overall": [{"split": "oot", "ks": 0.33}]},
                },
            },
        )
    )

    evidence = prompt["evidence"]
    assert evidence["reproducibility"]["summary"]["status"] == "fail"
    assert evidence["reproducibility"]["sample_size"] == 1000
    assert "rows" not in evidence["reproducibility"]
    assert evidence["contract"]["feature_count"] == 200
    assert len(evidence["contract"]["feature_columns_sample"]) == 20
    assert "source_previews" not in evidence["notebook_steps"][0]
    prompt_text = json.dumps(prompt, ensure_ascii=False)
    assert "score_code_model" not in prompt_text
    assert "0.3215" not in prompt_text


def test_agent_continue_retries_notebook_stage_after_notebook_failure(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            prompt = json.loads(kwargs["user_prompt"])
            if prompt["stage"] == "reproducibility":
                return "模型可复现性已重新执行完成。"
            return f"{prompt['stage']} summary"

    def fake_notebook_stage(*, task_id, settings, stage_claimed):
        assert stage_claimed is True
        calls.append(task_id)
        repo = TaskRepository(settings.db_path)
        repo.update_status(
            task_id,
            TaskStatus.EXECUTED,
            "notebook executed",
            expected=TaskStatus.RUNNING,
        )

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    monkeypatch.setattr("marvis.agent.validation_app_service.run_notebook_stage", fake_notebook_stage)
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_metrics_stage",
        lambda **_kwargs: calls.append("metrics"),
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    repo.update_status(
        task_id,
        TaskStatus.FAILED,
        f"{NOTEBOOK_STAGE_FAILURE_PREFIX}cell failed",
        expected=TaskStatus.CREATED,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "继续", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "accepted"
    assert calls == [task_id]
    assert repo.get_task(task_id).status == TaskStatus.EXECUTED


def test_agent_stop_message_requests_active_agent_cancellation_without_llm_config(
    tmp_path,
    monkeypatch,
):
    requested: list[str] = []
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.request_notebook_cancellation",
        lambda task_id: requested.append(task_id) or True,
    )
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    repo.start_job(task_id, "agent")

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "请停止当前任务"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "cancel_requested"
    assert requested == [task_id]
    assert repo.get_active_job_kind(task_id) == "agent"
    messages = payload["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["metadata"]["intent"] == "stop"
    assert messages[1]["content"] == "已停止当前动作，请问有什么指示？"


def test_agent_stop_endpoint_requests_active_agent_cancellation_without_user_message(
    tmp_path,
    monkeypatch,
):
    requested_agent: list[str] = []
    requested_notebook: list[str] = []
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.request_agent_cancellation",
        lambda task_id: requested_agent.append(task_id),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.request_notebook_cancellation",
        lambda task_id: requested_notebook.append(task_id) or True,
    )
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    repo.start_job(task_id, "agent")

    response = client.post(f"/api/tasks/{task_id}/agent/stop")

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "cancel_requested"
    assert requested_agent == [task_id]
    assert requested_notebook == [task_id]
    assert repo.get_active_job_kind(task_id) == "agent"
    messages = payload["messages"]
    assert [message["role"] for message in messages] == ["assistant"]
    assert messages[0]["content"] == "已停止当前动作，请问有什么指示？"
    assert messages[0]["metadata"]["intent"] == "stop"
    assert messages[0]["metadata"]["cancel_requested"] is True


def test_agent_stop_marks_scanned_task_stopped_without_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.request_notebook_cancellation",
        lambda _task_id: True,
    )
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    repo.update_status(task_id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.start_job(task_id, "agent")

    response = client.post(f"/api/tasks/{task_id}/agent/stop")

    assert response.status_code == 202, response.text
    task = repo.get_task(task_id)
    assert task.status == TaskStatus.SCANNED
    assert task.status_message == "已停止当前动作"
    assert "失败" not in task.status_message


def test_pipeline_cancel_resume_status_is_idempotent_and_stopped(tmp_path):
    from marvis.pipeline import _mark_cancelled

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    repo.update_status(task_id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)

    _mark_cancelled(repo, task_id, TaskStatus.SCANNED, "已停止当前动作")

    task = repo.get_task(task_id)
    assert task.status == TaskStatus.SCANNED
    assert task.status_message == "已停止当前动作"


def test_agent_stop_keeps_active_job_guard_until_worker_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.request_notebook_cancellation",
        lambda _task_id: True,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    active_job_id = repo.start_job(task_id, "agent")

    stop_response = client.post(f"/api/tasks/{task_id}/agent/stop")
    continue_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "继续", "model_id": "m1"},
    )

    assert stop_response.status_code == 202, stop_response.text
    assert repo.get_active_job_kind(task_id) == "agent"
    assert continue_response.status_code == 409, continue_response.text
    assert continue_response.json()["detail"] == "task already has an active stage"
    repo.finish_job(active_job_id, status="cancelled")
    assert repo.get_active_job_kind(task_id) is None


def test_agent_streaming_message_consumes_cancellation_after_quiet_producer(tmp_path):
    from marvis.api import (
        AgentValidationCancelled,
        _add_streaming_agent_message,
        _clear_agent_cancellation,
        _request_agent_cancellation,
        _stream_agent_message,
    )

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    message = _add_streaming_agent_message(
        repo,
        task_id,
        stage="chat",
        model_profile={"model_id": "m1", "display_name": "主模型", "model_name": "gpt"},
    )

    def quiet_producer(_on_delta):
        _request_agent_cancellation(task_id)
        return "不应写入的最终内容", {}

    try:
        with pytest.raises(AgentValidationCancelled):
            _stream_agent_message(
                repo,
                message["id"],
                task_id=task_id,
                model_profile={
                    "model_id": "m1",
                    "display_name": "主模型",
                    "model_name": "gpt",
                },
                producer=quiet_producer,
            )
    finally:
        _clear_agent_cancellation(task_id)

    [stored_message] = repo.list_agent_messages(task_id)
    assert stored_message["content"] == ""
    assert stored_message["metadata"]["streaming"] is False
    assert stored_message["metadata"]["cancelled"] is True


def test_agent_word_conclusion_stage_shows_thinking_while_llm_generates_draft(
    tmp_path,
    monkeypatch,
):
    observed_messages_during_llm: list[list[dict]] = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **_kwargs):
            observed_messages_during_llm.append(repo.list_agent_messages(task_id))
            return json.dumps(REQUIRED_AGENT_CONCLUSIONS, ensure_ascii=False)

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    from marvis.api import _run_agent_word_conclusion_stage

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)

    finished = _run_agent_word_conclusion_stage(
        repo,
        client.app.state.settings,
        task_id,
        model_profile={
            "model_id": "m1",
            "display_name": "主模型",
            "model_name": "credit-risk-gpt",
        },
    )

    assert finished is True
    assert observed_messages_during_llm
    [in_flight_message] = observed_messages_during_llm[0]
    assert in_flight_message["role"] == "assistant"
    assert in_flight_message["stage"] == "word_conclusion_draft"
    assert in_flight_message["content"] == ""
    assert in_flight_message["metadata"]["streaming"] is True

    messages = repo.list_agent_messages(task_id)
    assert [message["stage"] for message in messages] == [
        "word_conclusion_draft",
        "chat",
    ]
    draft = messages[0]
    assert draft["metadata"]["streaming"] is False
    assert draft["metadata"]["draft_values"] == REQUIRED_AGENT_CONCLUSIONS
    assert draft["metadata"]["report_revision"] == 0
    assert "压力测试总结" in draft["content"]
    assert "最终验证结论" in draft["content"]
    assert messages[1]["metadata"]["awaiting_confirmation"] is True


def test_agent_word_conclusion_stage_rejects_empty_draft_without_confirmation(
    tmp_path,
    monkeypatch,
):
    from marvis.api import _run_agent_word_conclusion_stage

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)

    monkeypatch.setattr(
        "marvis.agent.validation_app_service.generate_word_conclusions",
        lambda **_kwargs: (
            {},
            {
                "llm_error": "上下文过长：prompt 超过模型窗口。",
                "fallback": True,
                "confirmable": False,
            },
        ),
    )
    monkeypatch.setattr("marvis.agent.validation_app_service.agent_evidence_from_settings_impl", lambda _settings, _task_id: {})

    finished = _run_agent_word_conclusion_stage(
        repo,
        client.app.state.settings,
        task_id,
        model_profile={
            "model_id": "m1",
            "display_name": "主模型",
            "model_name": "credit-risk-gpt",
        },
    )

    assert finished is False
    messages = repo.list_agent_messages(task_id)
    assert [message["stage"] for message in messages] == [
        "word_conclusion_draft",
        "chat",
    ]
    assert messages[0]["content"] == ""
    assert messages[0]["metadata"]["draft_values"] == {}
    assert "三段 Word 结论草稿已生成" not in messages[1]["content"]
    assert "草稿生成失败" in messages[1]["content"]
    assert "上下文过长" in messages[1]["content"]
    assert messages[1]["metadata"]["word_draft_failed"] is True
    assert "awaiting_confirmation" not in messages[1]["metadata"]


def test_agent_word_conclusion_display_uses_fixed_business_order():
    from marvis.api import _format_conclusion_values

    content = _format_conclusion_values(
        {
            "TEXT:final_validation_conclusion": "最终结论内容。",
            "TEXT:pressure_impact_recommendation": "影响建议内容。",
            "TEXT:pressure_test_summary": "压力测试内容。",
        }
    )

    assert content.split("\n\n") == [
        "压力测试总结\n压力测试内容。",
        "压力影响建议\n影响建议内容。",
        "最终验证结论\n最终结论内容。",
    ]


def test_agent_word_conclusion_dispatch_returns_draft_thinking_message(tmp_path):
    from marvis.api import _dispatch_agent_validation_job

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)

    result = _dispatch_agent_validation_job(
        repo=repo,
        task=repo.get_task(task_id),
        settings=client.app.state.settings,
        model_profile={
            "model_id": "m1",
            "display_name": "主模型",
            "model_name": "credit-risk-gpt",
        },
        background_tasks=BackgroundTasks(),
    )

    assert result["status"] == "accepted"
    assert result["stage"] == "word_conclusion_draft"
    messages = result["messages"]
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["stage"] == "chat"
    assert messages[-2]["content"] == "收到，我将基于已完成的验证结果起草 Word 报告中的三段结论，完成后会等你确认。"
    assert messages[-2]["metadata"]["streaming"] is False
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["stage"] == "word_conclusion_draft"
    assert messages[-1]["content"] == ""
    assert messages[-1]["metadata"]["streaming"] is True
    assert repo.get_active_job_kind(task_id) == "agent"


def test_agent_word_conclusion_prompt_uses_contextual_metric_guidance(
    tmp_path,
    monkeypatch,
):
    calls = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            calls.append(kwargs)
            return json.dumps(REQUIRED_AGENT_CONCLUSIONS, ensure_ascii=False)

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/report-draft",
        json={"model_id": "m1"},
    )

    assert response.status_code == 200, response.text
    assert calls
    assert "PSI 小于 0.10" in calls[0]["system_prompt"]
    assert "KS 0.30" in calls[0]["system_prompt"]
    assert "过拟合" in calls[0]["system_prompt"]
    assert "train-test" in calls[0]["system_prompt"]
    assert "相对 10%" in calls[0]["system_prompt"]
    assert "5 个点" in calls[0]["system_prompt"]
    assert "高风险数据源" in calls[0]["system_prompt"]
    assert "中风险数据源" in calls[0]["system_prompt"]
    assert "低风险数据源" in calls[0]["system_prompt"]
    assert "不能脱离模型场景" in calls[0]["system_prompt"]


def test_agent_chat_confirm_report_draft_dispatches_report_without_llm_chat(
    tmp_path,
    monkeypatch,
):
    llm_calls: list[dict] = []
    report_calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            llm_calls.append(kwargs)
            return "不应调用普通聊天"

    def fake_report_stage(*, task_id, settings):
        report_calls.append(task_id)
        repo = TaskRepository(settings.db_path)
        repo.update_status(
            task_id,
            TaskStatus.SUCCEEDED,
            "pipeline succeeded",
            expected={TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED},
        )

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    monkeypatch.setattr("marvis.agent.validation_app_service.run_report_stage", fake_report_stage)
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_draft",
        content="压力测试总结\n压力测试显示模型整体稳定。",
        metadata={"draft_values": REQUIRED_AGENT_CONCLUSIONS, "report_revision": 0},
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="三段 Word 结论草稿已生成。请先查看；需要写入 Word 时，请直接回复“确认”。",
        metadata={"awaiting_confirmation": True},
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "accepted"
    assert llm_calls == []
    assert report_calls == [task_id]
    values, revision = repo.get_report_values(task_id)
    assert revision == 1
    assert REQUIRED_AGENT_CONCLUSIONS.items() <= values.items()
    assert repo.get_task(task_id).status == TaskStatus.SUCCEEDED
    messages = repo.list_agent_messages(task_id)
    assert messages[-3]["role"] == "user"
    assert messages[-3]["metadata"]["intent"] == "confirm_report"
    assert messages[-2]["stage"] == "word_conclusion_confirmed"
    assert messages[-1]["stage"] == "word_report_ready"


def test_agent_chat_confirm_report_draft_does_not_need_enabled_llm(tmp_path, monkeypatch):
    report_calls: list[str] = []

    def fake_report_stage(*, task_id, settings):
        report_calls.append(task_id)
        repo = TaskRepository(settings.db_path)
        repo.update_status(
            task_id,
            TaskStatus.SUCCEEDED,
            "pipeline succeeded",
            expected={TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED},
        )

    monkeypatch.setattr("marvis.agent.validation_app_service.run_report_stage", fake_report_stage)
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_draft",
        content="压力测试总结\n压力测试显示模型整体稳定。",
        metadata={"draft_values": REQUIRED_AGENT_CONCLUSIONS, "report_revision": 0},
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "accepted"
    assert report_calls == [task_id]
    assert repo.get_task(task_id).status == TaskStatus.SUCCEEDED


def test_agent_chat_confirm_report_draft_rejects_stale_revision_without_mutation(
    tmp_path,
    monkeypatch,
):
    report_calls: list[str] = []
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_report_stage",
        lambda **kwargs: report_calls.append(kwargs["task_id"]),
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_draft",
        content="压力测试总结\n旧草稿。",
        metadata={"draft_values": REQUIRED_AGENT_CONCLUSIONS, "report_revision": 0},
    )
    repo.update_report_values(task_id, {"TEXT:report_title": "人工修改"}, expected_revision=0)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "model_id": "m1"},
    )

    assert response.status_code == 409
    assert "stale report values revision" in response.json()["detail"]
    values, revision = repo.get_report_values(task_id)
    assert revision == 1
    assert values == {"TEXT:report_title": "人工修改"}
    assert report_calls == []
    assert not any(
        message["stage"] == "word_conclusion_confirmed"
        for message in repo.list_agent_messages(task_id)
    )


def test_agent_chat_confirm_ignores_freeform_assistant_report_headings(
    tmp_path,
    monkeypatch,
):
    llm_calls: list[str] = []
    report_calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **_kwargs):
            llm_calls.append("chat")
            return "这是普通对话回复。"

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_report_stage",
        lambda **kwargs: report_calls.append(kwargs["task_id"]),
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=(
            "压力测试总结\n普通解释。\n\n"
            "压力影响建议\n普通建议。\n\n"
            "最终验证结论\n普通结论。"
        ),
        metadata={},
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "message_saved"
    assert llm_calls == ["chat"]
    assert report_calls == []
    values, revision = repo.get_report_values(task_id)
    assert revision == 0
    assert values == {}


def test_agent_chat_confirm_ignores_draft_after_report_was_confirmed(
    tmp_path,
    monkeypatch,
):
    llm_calls: list[str] = []
    report_calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **_kwargs):
            llm_calls.append("chat")
            return "这是普通对话回复。"

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_report_stage",
        lambda **kwargs: report_calls.append(kwargs["task_id"]),
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_draft",
        content="压力测试总结\n旧草稿。",
        metadata={"draft_values": REQUIRED_AGENT_CONCLUSIONS, "report_revision": 0},
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_confirmed",
        content="三段报告结论已确认，将开始生成最终 Word 报告。",
        metadata={"revision": 1},
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "message_saved"
    assert llm_calls == ["chat"]
    assert report_calls == []
    assert not any(
        message["metadata"].get("intent") == "confirm_report"
        for message in repo.list_agent_messages(task_id)
    )


def test_agent_chat_regenerate_report_creates_structured_draft_not_plain_chat(
    tmp_path,
    monkeypatch,
):
    report_calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **_kwargs):
            return json.dumps(REQUIRED_AGENT_CONCLUSIONS, ensure_ascii=False)

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_report_stage",
        lambda **kwargs: report_calls.append(kwargs["task_id"]),
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_draft",
        content="压力测试总结\n旧草稿。",
        metadata={"draft_values": REQUIRED_AGENT_CONCLUSIONS, "report_revision": 0},
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="三段 Word 结论草稿已生成。请先查看；需要写入 Word 时，请直接回复“确认”。",
        metadata={"awaiting_confirmation": True},
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "重新生成报告", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "accepted"
    assert response.json()["stage"] == "word_conclusion_draft"
    assert report_calls == []
    messages = repo.list_agent_messages(task_id)
    assert messages[-4]["role"] == "user"
    assert messages[-4]["metadata"]["intent"] == "regenerate_report_draft"
    assert messages[-3]["stage"] == "chat"
    assert messages[-3]["metadata"]["streaming"] is False
    assert messages[-2]["stage"] == "word_conclusion_draft"
    assert messages[-2]["metadata"]["draft_values"] == REQUIRED_AGENT_CONCLUSIONS
    assert messages[-1]["metadata"]["awaiting_confirmation"] is True


def test_agent_chat_regenerate_report_rejects_active_job_without_new_draft(
    tmp_path,
    monkeypatch,
):
    llm_calls: list[str] = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **_kwargs):
            llm_calls.append("called")
            return json.dumps(REQUIRED_AGENT_CONCLUSIONS, ensure_ascii=False)

    monkeypatch.setattr(
        "marvis.agent.service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_draft",
        content="压力测试总结\n旧草稿。",
        metadata={"draft_values": REQUIRED_AGENT_CONCLUSIONS, "report_revision": 0},
    )
    before_messages = repo.list_agent_messages(task_id)
    repo.start_job(task_id, "metrics")

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "重新生成报告", "model_id": "m1"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "task already has an active stage"
    assert llm_calls == []
    assert repo.list_agent_messages(task_id) == before_messages


def test_agent_rerun_scan_resets_steps_without_deleting_history(
    tmp_path,
    monkeypatch,
):
    calls: list[dict] = []

    def fake_run_agent_validation_job(
        job_id,
        settings,
        task_id,
        model_profile,
        opening_message_id,
        stage=None,
        stage_message_id=None,
        acceptance_mode=None,
        stage_instruction=None,
    ):
        calls.append(
            {
                "task_id": task_id,
                "stage": stage,
                "opening_message_id": opening_message_id,
                "stage_instruction": stage_instruction,
            }
        )

    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_agent_validation_job",
        fake_run_agent_validation_job,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.update_agent_report_conclusions(
        task_id,
        REQUIRED_AGENT_CONCLUSIONS,
        expected_revision=0,
    )
    repo.update_status(
        task_id,
        TaskStatus.SUCCEEDED,
        "done",
        expected=TaskStatus.WRITING_ARTIFACTS,
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="metrics",
        content="旧的指标分析仍应留在聊天历史里。",
        metadata={"source": "old"},
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "重新读取材料", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["stage"] == "scan"
    assert calls[0]["stage"] == "scan"
    assert calls[0]["stage_instruction"] == "重新读取材料"
    task = repo.get_task(task_id)
    assert task.status == TaskStatus.CREATED
    assert task.status_message == "agent rerun requested: scan"
    values, revision = repo.get_report_values(task_id)
    assert revision == 2
    assert all(values.get(key, "") == "" for key in REQUIRED_AGENT_CONCLUSIONS)
    messages = repo.list_agent_messages(task_id)
    assert any(message["content"] == "旧的指标分析仍应留在聊天历史里。" for message in messages)
    rerun_user = next(message for message in messages if message["content"] == "重新读取材料")
    assert rerun_user["metadata"]["intent"] == "rerun_stage"
    assert rerun_user["metadata"]["target_stage"] == "scan"


@pytest.mark.parametrize(
    ("content", "stage"),
    [
        ("重新执行第一步", "scan"),
        ("重跑步骤1", "scan"),
        ("从头执行一遍", "scan"),
        ("重新执行一遍全部", "scan"),
        ("全部重新执行", "scan"),
        ("完整流程重跑", "scan"),
        ("从第一步开始重新跑", "scan"),
        ("重新跑材料完备性验证", "scan"),
        ("再执行一下完备性检查", "scan"),
        ("重新执行第二步", "reproducibility"),
        ("重跑步骤2", "reproducibility"),
        ("重新执行复现性验证", "reproducibility"),
        ("重新做模型可复现性", "reproducibility"),
        ("再跑分数一致性", "reproducibility"),
        ("重新执行第三步", "metrics"),
        ("重跑步骤3", "metrics"),
        ("重新执行效果稳定性验证", "metrics"),
        ("重新跑效果与稳定性", "metrics"),
        ("再执行压力测试", "metrics"),
        ("重新执行第四步", "word_conclusion_draft"),
        ("重跑步骤4", "word_conclusion_draft"),
        ("重新写三段结论草稿", "word_conclusion_draft"),
        ("重新生成报告", "word_conclusion_draft"),
        ("再生成 Word 报告", "word_conclusion_draft"),
    ],
)
def test_agent_rerun_stage_recognizes_step_numbers_and_business_aliases(content, stage):
    from marvis.agent.service import agent_rerun_stage

    assert agent_rerun_stage(content) == stage


def test_agent_rerun_material_completeness_after_stop_dispatches_scan(
    tmp_path,
    monkeypatch,
):
    calls: list[dict] = []
    chat_calls: list[str] = []

    def fake_run_agent_validation_job(
        job_id,
        settings,
        task_id,
        model_profile,
        opening_message_id,
        stage=None,
        stage_message_id=None,
        acceptance_mode=None,
        stage_instruction=None,
    ):
        calls.append(
            {
                "task_id": task_id,
                "stage": stage,
                "opening_message_id": opening_message_id,
                "acceptance_mode": acceptance_mode,
                "stage_instruction": stage_instruction,
            }
        )

    def fake_answer_chat_message(**kwargs):
        chat_calls.append(kwargs["user_message"])
        return "不应进入普通问答。", {"fallback": True}

    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_agent_validation_job",
        fake_run_agent_validation_job,
    )
    monkeypatch.setattr("marvis.routers.validation_agent.answer_chat_message", fake_answer_chat_message)
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    repo.update_status(task_id, TaskStatus.SCANNED, "已停止当前动作", expected=TaskStatus.CREATED)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="已停止当前动作，请问有什么指示？",
        metadata={"intent": "stop", "cancel_requested": True},
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "重新执行一下完备性验证", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["stage"] == "scan"
    assert calls[0]["stage"] == "scan"
    assert calls[0]["stage_instruction"] == "重新执行一下完备性验证"
    assert chat_calls == []
    task = repo.get_task(task_id)
    assert task.status == TaskStatus.CREATED
    assert task.status_message == "agent rerun requested: scan"
    rerun_user = next(
        message
        for message in repo.list_agent_messages(task_id)
        if message["content"] == "重新执行一下完备性验证"
    )
    assert rerun_user["metadata"]["intent"] == "rerun_stage"
    assert rerun_user["metadata"]["target_stage"] == "scan"


def test_agent_rerun_report_draft_resets_report_step_and_keeps_rewrite_instruction(
    tmp_path,
    monkeypatch,
):
    calls: list[dict] = []

    def fake_run_agent_validation_job(
        job_id,
        settings,
        task_id,
        model_profile,
        opening_message_id,
        stage=None,
        stage_message_id=None,
        acceptance_mode=None,
        stage_instruction=None,
    ):
        calls.append(
            {
                "stage": stage,
                "stage_message_id": stage_message_id,
                "stage_instruction": stage_instruction,
            }
        )

    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_agent_validation_job",
        fake_run_agent_validation_job,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.update_agent_report_conclusions(
        task_id,
        REQUIRED_AGENT_CONCLUSIONS,
        expected_revision=0,
    )
    repo.update_status(
        task_id,
        TaskStatus.SUCCEEDED,
        "done",
        expected=TaskStatus.WRITING_ARTIFACTS,
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_draft",
        content="旧草稿也要保留。",
        metadata={"draft_values": REQUIRED_AGENT_CONCLUSIONS, "report_revision": 1},
    )

    instruction = "重新给我写一个草稿，要修改压力测试部分的措辞"
    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": instruction, "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["stage"] == "word_conclusion_draft"
    assert calls[0]["stage"] == "word_conclusion_draft"
    assert calls[0]["stage_message_id"]
    assert calls[0]["stage_instruction"] == instruction
    task = repo.get_task(task_id)
    assert task.status == TaskStatus.WRITING_ARTIFACTS
    assert task.status_message == "agent rerun requested: word_conclusion_draft"
    values, revision = repo.get_report_values(task_id)
    assert revision == 2
    assert all(values.get(key, "") == "" for key in REQUIRED_AGENT_CONCLUSIONS)
    messages = repo.list_agent_messages(task_id)
    assert any(message["content"] == "旧草稿也要保留。" for message in messages)
    rerun_user = next(message for message in messages if message["content"] == instruction)
    assert rerun_user["metadata"]["intent"] == "rerun_stage"
    assert rerun_user["metadata"]["target_stage"] == "word_conclusion_draft"


def test_agent_rerun_rejects_stage_that_has_not_been_reached(tmp_path, monkeypatch):
    def unexpected_dispatch(*_args, **_kwargs):
        pytest.fail("unreached rerun stage should not dispatch an agent job")

    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_agent_validation_job",
        unexpected_dispatch,
    )
    client = _client(tmp_path)
    client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "主模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "credit-risk-gpt",
                    "api_key": "secret",
                    "timeout_seconds": 45,
                },
            ],
        },
    )
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "重新生成报告", "model_id": "m1"},
    )

    assert response.status_code == 409, response.text
    assert "尚未执行到该阶段" in response.json()["detail"]
    assert repo.get_task(task_id).status == TaskStatus.CREATED
    assert repo.list_agent_messages(task_id) == []


def test_agent_report_confirm_writes_three_conclusions_and_dispatches_report(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.run_report_stage",
        lambda **kwargs: calls.append(kwargs["task_id"]),
    )
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)

    response = client.post(
        f"/api/tasks/{task_id}/agent/report-draft/confirm",
        json={"revision": 0, "text_values": REQUIRED_AGENT_CONCLUSIONS},
    )

    assert response.status_code == 202, response.text
    values, revision = repo.get_report_values(task_id)
    assert revision == 1
    assert REQUIRED_AGENT_CONCLUSIONS.items() <= values.items()
    assert calls == [task_id]
    messages = repo.list_agent_messages(task_id)
    assert messages[-1]["stage"] == "word_report_ready"
    assert messages[-1]["content"] == (
        "报告已生成。右侧步骤里的“预览”可以在线查看 Word，"
        "“下载Word”用于下载验证报告，“下载Excel”用于下载指标分析明细。"
    )


def test_agent_report_confirm_rejects_extra_report_keys(tmp_path):
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    payload = {
        **REQUIRED_AGENT_CONCLUSIONS,
        "TEXT:report_title": "不允许覆盖",
    }

    response = client.post(
        f"/api/tasks/{task_id}/agent/report-draft/confirm",
        json={"revision": 0, "text_values": payload},
    )

    assert response.status_code == 422
    assert "only update agent conclusion keys" in response.json()["detail"]


def test_agent_report_confirm_rejects_active_job_without_mutating_conclusions(tmp_path):
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.start_job(task_id, "metrics")

    response = client.post(
        f"/api/tasks/{task_id}/agent/report-draft/confirm",
        json={"revision": 0, "text_values": REQUIRED_AGENT_CONCLUSIONS},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "task already has an active stage"
    values, revision = repo.get_report_values(task_id)
    assert revision == 0
    assert not REQUIRED_AGENT_CONCLUSIONS.items() <= values.items()
    assert not any(
        message["stage"] == "word_conclusion_confirmed"
        for message in repo.list_agent_messages(task_id)
    )


def test_agent_report_confirm_claims_job_before_mutating_conclusions(
    tmp_path,
    monkeypatch,
):
    start_attempts: list[tuple[str, str]] = []

    def fail_start_job(_repo, task_id, kind):
        start_attempts.append((task_id, kind))
        raise HTTPException(status_code=409, detail="task already has an active stage")

    monkeypatch.setattr("marvis.agent.validation_app_service.start_task_job", fail_start_job)
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)

    response = client.post(
        f"/api/tasks/{task_id}/agent/report-draft/confirm",
        json={"revision": 0, "text_values": REQUIRED_AGENT_CONCLUSIONS},
    )

    assert response.status_code == 409
    assert start_attempts == [(task_id, "report")]
    values, revision = repo.get_report_values(task_id)
    assert revision == 0
    assert not REQUIRED_AGENT_CONCLUSIONS.items() <= values.items()
    assert not any(
        message["stage"] == "word_conclusion_confirmed"
        for message in repo.list_agent_messages(task_id)
    )


def test_agent_report_confirm_rejects_non_reportable_status_without_mutating_conclusions(
    tmp_path,
):
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")

    response = client.post(
        f"/api/tasks/{task_id}/agent/report-draft/confirm",
        json={"revision": 0, "text_values": REQUIRED_AGENT_CONCLUSIONS},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "cannot generate report in status created"
    values, revision = repo.get_report_values(task_id)
    assert revision == 0
    assert not REQUIRED_AGENT_CONCLUSIONS.items() <= values.items()
    assert repo.list_agent_messages(task_id) == []


def test_agent_report_generation_requires_confirmed_conclusions(tmp_path):
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)

    response = client.post(f"/api/tasks/{task_id}/report")

    assert response.status_code == 409
    assert "请先确认三段报告结论" in response.json()["detail"]


def test_agent_report_preview_requires_confirmed_agent_conclusions(tmp_path):
    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.update_status(
        task_id,
        TaskStatus.SUCCEEDED,
        "done",
        expected=TaskStatus.WRITING_ARTIFACTS,
    )
    output_dir = tmp_path / "tasks" / task_id / "outputs"
    output_dir.mkdir(parents=True)
    document = Document()
    document.add_paragraph("报告已生成但未确认 Agent 结论")
    document.save(output_dir / "validation_report.docx")

    preview = client.get(f"/api/tasks/{task_id}/report/preview")
    download = client.get(f"/api/tasks/{task_id}/report/download")

    assert preview.status_code == 409
    assert download.status_code == 409
    assert "请先确认三段报告结论" in preview.json()["detail"]


def test_stream_agent_message_throttles_db_writes():
    from marvis.agent.validation_messages import stream_agent_message

    class _SpyRepo:
        def __init__(self):
            self.update_calls = 0
            self.last_content = ""

        def update_agent_message(self, message_id, *, content, metadata=None):
            self.update_calls += 1
            self.last_content = content
            return {
                "id": message_id,
                "content": content,
                "metadata": metadata or {},
            }

    repo = _SpyRepo()
    delta_count = 200

    def producer(on_delta):
        for _ in range(delta_count):
            on_delta("x")
        return "x" * delta_count, {}

    result = stream_agent_message(
        repo,
        "msg-1",
        task_id="task-1",
        model_profile={"model_id": "m1", "display_name": "d", "model_name": "n"},
        producer=producer,
        raise_if_cancelled=lambda _task_id: None,
    )

    # Final content is complete and correct.
    assert result["content"] == "x" * delta_count
    assert repo.last_content == "x" * delta_count
    # Writes are throttled: far fewer DB updates than deltas emitted.
    assert repo.update_calls < delta_count
    # 200 single-char deltas (< 512 chars, sub-second) => only the final flush.
    assert repo.update_calls <= 2
