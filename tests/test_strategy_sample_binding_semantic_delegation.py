from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.llm_client import OpenAICompatibleLLMClient
from marvis.llm_settings import resolve_llm_model
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.repositories.tasks import TaskRepository


_INSTRUCTION = (
    "读取材料目录里的 strategy_sample_v2.csv，把它作为当前策略样本，"
    "坏样本字段用 bad_flag。"
)
_INTENT = "strategy_sample_binding"
_LIVE_WORKSPACE_ENV = "MARVIS_SEMANTIC_DIAGNOSTIC_WORKSPACE"
_LIVE_MODEL_ENV = "MARVIS_SEMANTIC_DIAGNOSTIC_MODEL_ID"


class _StrategySampleBindingClient:
    def __init__(
        self,
        *,
        overrides: dict | None = None,
        mutation=None,
    ) -> None:
        self.calls: list[dict] = []
        self.requests: list[dict] = []
        self.overrides = dict(overrides or {})
        self.mutation = mutation

    def complete(self, **kwargs):
        caller = str(kwargs.get("caller") or "")
        self.calls.append(kwargs)
        assert caller in {"semantic_intent_router", "semantic_intent_reviewer"}
        request = json.loads(kwargs["user_prompt"])
        self.requests.append(request)
        payload = {
            "intent": _INTENT,
            "evidence_quote": request["instruction"],
            "reason": "用户只明确授权绑定当前唯一策略样本及其坏样本字段。",
            "confidence": "high",
            "is_question": False,
            "is_conditional": False,
            "requests_change": False,
            "withholds_action": False,
        }
        payload.update(self.overrides)
        if caller == "semantic_intent_reviewer" and self.mutation is not None:
            mutation, self.mutation = self.mutation, None
            mutation()
        return json.dumps(payload, ensure_ascii=False)


def _install_client(monkeypatch, decision_client) -> None:
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda _request, _task, _payload: decision_client,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda _request, _task: decision_client,
    )


def _source_dir(tmp_path: Path) -> Path:
    source = tmp_path / "strategy-binding-materials"
    source.mkdir(parents=True)
    rows = 120
    pd.DataFrame(
        {
            "application_id": np.arange(rows),
            "score": np.linspace(300.0, 850.0, rows),
            "bad_flag": np.resize([0, 1], rows),
        }
    ).to_csv(source / "strategy_sample_v2.csv", index=False)
    return source


def _nested_source_dir(tmp_path: Path) -> Path:
    source = tmp_path / "strategy-binding-nested-materials"
    nested = source / "nested"
    nested.mkdir(parents=True)
    rows = 120
    pd.DataFrame(
        {
            "application_id": np.arange(rows),
            "score": np.linspace(300.0, 850.0, rows),
            "bad_flag": np.resize([0, 1], rows),
        }
    ).to_parquet(nested / "only.parquet")
    return source


def _full_strategy_source_dir(tmp_path: Path) -> Path:
    """Mirror the 480-row browser fixture used by Candidate Lab acceptance."""

    source = tmp_path / "strategy-binding-full-flow"
    source.mkdir(parents=True)
    partitions = [
        ("2025-10-15", "202510", 240),
        ("2025-12-15", "202512", 120),
        ("2026-01-15", "202601", 120),
    ]
    apply_dates = [
        apply_date
        for apply_date, _apply_month, count in partitions
        for _ in range(count)
    ]
    apply_months = [
        apply_month
        for _apply_date, apply_month, count in partitions
        for _ in range(count)
    ]
    rows = len(apply_dates)
    pd.DataFrame(
        {
            "cust_id": [f"C{index:04d}" for index in range(rows)],
            "sig1": np.linspace(-3.0, 3.0, rows),
            "apply_month": apply_months,
            "apply_date": apply_dates,
            "loan_amount": np.linspace(1000.0, 5000.0, rows),
            "bad_flag": np.resize([0, 1, 0, 0, 1], rows),
        }
    ).to_csv(source / "strategy_sample_v2.csv", index=False)
    return source


def _sample_design_request(*, observation_available: bool) -> dict:
    observation_window = (
        {
            "status": "provided",
            "start": "2025-10-15",
            "end": "2026-01-15",
        }
        if observation_available
        else {"status": "unavailable", "start": None, "end": None}
    )
    return {
        "request_kind": "standard_workflow",
        "workflow": "strategy_sample_design_v2",
        "workflow_inputs": {
            "target_bad_value": 1,
            "drop_nan_labels": True,
            "relationship": "nested_same_cohort",
            "approval_population": {"inclusion": None, "exclusion": None},
            "risk_population": {"inclusion": None, "exclusion": None},
            "partitioning": {
                "method": "time_ranges",
                "column": "apply_date",
                "ranges": {
                    "development": {
                        "start": "2025-10-15",
                        "end": "2025-11-15",
                    },
                    "validation": {
                        "start": "2025-12-15",
                        "end": "2025-12-15",
                    },
                    "oot": {
                        "start": "2026-01-15",
                        "end": "2026-01-15",
                    },
                },
            },
            "maturity": {
                "status": "confirmed_matured",
                "performance_window_days": 90,
                "cutoff_date": "2026-08-08",
                "reason": None,
            },
            "performance_window": {"status": "provided", "days": 90},
            "observation_window": observation_window,
            "field_bindings": {
                "entity_field": "cust_id",
                "time_field": "apply_date",
                "group_field": None,
                "month_field": "apply_month",
                "weight_field": None,
                "loan_amount_field": "loan_amount",
                "overdue_amount_field": None,
            },
            "historical_score": {
                "status": "unavailable",
                "column": None,
                "direction": None,
                "reason": "未提供历史评分",
            },
        },
    }


def _univariate_request() -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "univariate_candidate_analysis",
        "workflow_inputs": {
            "features": ["sig1"],
            "methods": ["equal_frequency"],
            "bin_count": 10,
            "min_bin_pct": 0.02,
            "loan_amount_col": "loan_amount",
            "sentinel_values": [],
        },
    }


def _latest_assistant(response) -> dict:
    return [
        message
        for message in response.json()["messages"]
        if message["role"] == "assistant"
    ][-1]


def _start_rendered_plan(client: TestClient, task_id: str, response) -> dict:
    overview = _latest_assistant(response)
    if overview["metadata"].get("kind") != "plan_overview":
        plan_id = overview["metadata"].get("plan_id")
        assert isinstance(plan_id, str), json.dumps(overview, ensure_ascii=False)
        assert client.app.state.plan_repo.load_plan(plan_id).status.value == "done"
        return response.json()
    started = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "开始执行当前计划",
            "ui_action": "start_plan",
            "expected_plan_id": overview["metadata"]["plan_id"],
            **overview["metadata"]["confirmation_snapshot"],
        },
    )
    assert started.status_code == 202, started.text
    return started.json()


def test_strategy_sample_binding_intent_persists_authenticated_workspace_without_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _source_dir(tmp_path)
    decision_client = _StrategySampleBindingClient()
    _install_client(monkeypatch, decision_client)

    with TestClient(create_app(tmp_path)) as client:
        created = client.post(
            "/api/tasks",
            json={
                "model_name": "策略样本绑定语义回归",
                "validator": "qa",
                "source_dir": str(source),
                "task_type": "strategy",
                "run_mode": "agent",
                # Reproduce the stale task-level default observed in the UI.
                "target_col": "y",
            },
        )
        assert created.status_code == 200, created.text
        task_id = str(created.json()["id"])

        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": _INSTRUCTION, "model_id": "semantic-test"},
        )

        assert response.status_code == 202, response.text
        assert client.app.state.plan_repo.list_plans_for_task(task_id) == []
        assistant = [
            message
            for message in response.json()["messages"]
            if message["role"] == "assistant"
        ][-1]
        assert assistant["metadata"]["intent"] == _INTENT, json.dumps(
            assistant, ensure_ascii=False
        )
        assert assistant["metadata"]["code"] == "strategy_sample_binding_complete"
        assert "strategy_sample_v2.csv" in assistant["content"]
        assert "bad_flag" in assistant["content"]
        assert "已绑定" in assistant["content"]
        assert "坏样本取值" in assistant["content"]
        assert "缺失标签" in assistant["content"]
        assert "人群" in assistant["content"]
        assert "成熟度" in assistant["content"]

        settings = client.app.state.settings
        workspace = DataWorkspaceRepository(settings.db_path).get_or_default(task_id)
        assert workspace.active_dataset_id is not None
        assert workspace.active_dataset_content_hash
        assert workspace.semantic_mapping.target_col == "bad_flag"
        assert workspace.semantic_mapping.field_roles == {"bad_flag": "target"}
        assert TaskRepository(settings.db_path).get_task(task_id).target_col == (
            "bad_flag"
        )

        registry = DatasetRegistry(
            DatasetRepository(settings.db_path),
            DataBackend(settings.datasets_dir),
            settings.datasets_dir,
        )
        binding = registry.authenticate_dataset_binding(
            workspace.active_dataset_id,
            expected_task_id=task_id,
            expected_content_hash=workspace.active_dataset_content_hash,
        )
        assert registry.verify_dataset_binding(binding) == binding
        assert registry.source_identity(binding.dataset_id)["original_name"] == (
            "strategy_sample_v2.csv"
        )

        assert [call["caller"] for call in decision_client.calls] == [
            "semantic_intent_router",
            "semantic_intent_reviewer",
        ]
        for request in decision_client.requests:
            assert _INTENT in request["allowed_intents"]
            assert request["current_context"]["available_strategy_samples"] == [
                "strategy_sample_v2.csv"
            ]
            assert request["current_context"]["strategy_sample_target_candidate"] == (
                "bad_flag"
            )


def test_strategy_sample_binding_accepts_unique_nested_relative_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _nested_source_dir(tmp_path)
    decision_client = _StrategySampleBindingClient()
    _install_client(monkeypatch, decision_client)
    instruction = (
        "读取材料目录里的 nested/only.parquet，把它作为当前策略样本，"
        "坏样本字段用 bad_flag。"
    )

    with TestClient(create_app(tmp_path)) as client:
        task_id = str(
            client.post(
                "/api/tasks",
                json={
                    "model_name": "嵌套策略样本绑定语义回归",
                    "validator": "qa",
                    "source_dir": str(source),
                    "task_type": "strategy",
                    "run_mode": "agent",
                    "target_col": "y",
                },
            ).json()["id"]
        )

        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": instruction, "model_id": "semantic-test"},
        )

        assert response.status_code == 202, response.text
        assistant = _latest_assistant(response)
        assert assistant["metadata"]["code"] == "strategy_sample_binding_complete"
        assert assistant["metadata"]["dataset_name"] == "only.parquet"
        assert assistant["metadata"]["dataset_relative_path"] == (
            "nested/only.parquet"
        )
        workspace = DataWorkspaceRepository(
            client.app.state.settings.db_path
        ).get_or_default(task_id)
        assert workspace.active_dataset_id is not None
        identity = DatasetRegistry(
            DatasetRepository(client.app.state.settings.db_path),
            DataBackend(client.app.state.settings.datasets_dir),
            client.app.state.settings.datasets_dir,
        ).source_identity(workspace.active_dataset_id)
        assert identity["original_name"] == "only.parquet"
        assert Path(identity["resolved_path"]).relative_to(source).as_posix() == (
            "nested/only.parquet"
        )
        for request in decision_client.requests:
            assert request["current_context"]["available_strategy_samples"] == [
                "nested/only.parquet"
            ]


@pytest.mark.parametrize("duplicate_basename", [False, True])
def test_strategy_sample_binding_rejects_multiple_nested_candidates(
    tmp_path: Path,
    monkeypatch,
    duplicate_basename: bool,
) -> None:
    source = tmp_path / "strategy-binding-ambiguous-materials"
    paths = (
        [source / "a" / "only.parquet", source / "b" / "only.parquet"]
        if duplicate_basename
        else [source / "a.parquet", source / "b.parquet"]
    )
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "application_id": np.arange(20),
                "signal": np.arange(20) + index,
                "bad_flag": np.resize([0, 1], 20),
            }
        ).to_parquet(path)
    decision_client = _StrategySampleBindingClient()
    _install_client(monkeypatch, decision_client)

    with TestClient(create_app(tmp_path)) as client:
        task_id = str(
            client.post(
                "/api/tasks",
                json={
                    "model_name": "歧义策略样本绑定语义回归",
                    "validator": "qa",
                    "source_dir": str(source),
                    "task_type": "strategy",
                    "run_mode": "agent",
                    "target_col": "y",
                },
            ).json()["id"]
        )
        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={
                "content": "读取 only.parquet 作为策略样本，坏样本字段用 bad_flag。",
                "model_id": "semantic-test",
            },
        )

        assert response.status_code == 202, response.text
        assert response.json()["status"] == "clarification_required"
        assert (
            DataWorkspaceRepository(client.app.state.settings.db_path)
            .get_or_default(task_id)
            .active_dataset_id
            is None
        )
        assert DatasetRepository(
            client.app.state.settings.db_path
        ).list_datasets(task_id) == []
        assert all(
            _INTENT not in request["allowed_intents"]
            for request in decision_client.requests
        )


@pytest.mark.slow
@pytest.mark.e2e
def test_semantic_binding_native_sample_then_univariate_keeps_scope_and_identity_distinct(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exercise the browser sequence without accepting a V1 fallback.

    An exploration-only sample is valid immutable evidence, but it is not a
    strategy-development execution sample.  The platform must name that scope
    boundary instead of reporting a provenance failure.  Once the user supplies
    the missing observation window, the newest authenticated native sample must
    drive exactly the 240 development rows into univariate analysis.
    """

    app_workspace = tmp_path / "workspace"
    source = _full_strategy_source_dir(app_workspace)
    decision_client = _StrategySampleBindingClient()
    _install_client(monkeypatch, decision_client)

    with TestClient(create_app(app_workspace)) as client:
        created = client.post(
            "/api/tasks",
            json={
                "model_name": "策略绑定到原生单变量回归",
                "validator": "qa",
                "source_dir": str(source),
                "task_type": "strategy",
                "run_mode": "agent",
                "target_col": "y",
            },
        )
        assert created.status_code == 200, created.text
        task_id = str(created.json()["id"])

        bound = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": _INSTRUCTION, "model_id": "semantic-test"},
        )
        assert bound.status_code == 202, bound.text
        assert _latest_assistant(bound)["metadata"]["code"] == (
            "strategy_sample_binding_complete"
        )

        exploratory = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={
                "content": "先固化探索口径的双人群样本",
                "acceptance_mode": "manual",
                "strategy_request": _sample_design_request(
                    observation_available=False
                ),
            },
        )
        assert exploratory.status_code == 202, exploratory.text
        _start_rendered_plan(client, task_id, exploratory)

        blocked = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={
                "content": "启动单变量候选分析",
                "acceptance_mode": "manual",
                "strategy_request": _univariate_request(),
            },
        )
        assert blocked.status_code == 202, blocked.text
        blocked_message = _latest_assistant(blocked)
        assert blocked_message["metadata"]["code"] == (
            "strategy_sample_design_scope_ineligible"
        ), blocked_message
        assert "exploration_only" in blocked_message["content"]
        assert "观察窗" in blocked_message["content"]
        assert "无法认证" not in blocked_message["content"]

        governed = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={
                "content": "补全观察窗并固化可开发的双人群样本",
                "acceptance_mode": "manual",
                "strategy_request": _sample_design_request(
                    observation_available=True
                ),
            },
        )
        assert governed.status_code == 202, governed.text
        _start_rendered_plan(client, task_id, governed)

        candidate = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={
                "content": "启动单变量候选分析",
                "acceptance_mode": "manual",
                "strategy_request": _univariate_request(),
            },
        )
        assert candidate.status_code == 202, candidate.text
        _start_rendered_plan(client, task_id, candidate)

        plans = client.app.state.plan_repo.list_plans_for_task(task_id)
        assert [plan.template_id for plan in plans] == [
            "strategy_sample_design_v2_native",
            "strategy_sample_design_v2_native",
            "strategy_univariate_candidate_analysis",
        ]
        assert all(plan.status.value == "done" for plan in plans)
        candidate_plan = plans[-1]
        candidate_output = client.app.state.plan_repo.load_step_output(
            candidate_plan.steps[0].id
        )
        assert candidate_output["candidate_evidence"]["analysis"]["row_count"] == 240

        artifacts = TaskArtifactRepository(
            client.app.state.settings.db_path
        ).list_for_task(task_id)
        newest_native_bundle = [
            artifact
            for artifact in artifacts
            if artifact["kind"] == "strategy_sample_design_v2_json"
            and artifact["origin_tool"]
            == "strategy.materialize_sample_design_v2_native"
        ][-1]
        sample_ref = candidate_plan.steps[0].inputs["sample_design_ref"]
        assert sample_ref == {
            "artifact_id": newest_native_bundle["id"],
            "artifact_content_hash": newest_native_bundle["content_hash"],
            "sample_design_id": newest_native_bundle["provenance"][
                "sample_design_id"
            ],
            "sample_design_content_hash": newest_native_bundle["provenance"][
                "sample_design_content_hash"
            ],
            "partition": "risk/development",
        }


@pytest.mark.parametrize(
    ("instruction", "overrides"),
    [
        (
            "能不能把 strategy_sample_v2.csv 作为当前策略样本？",
            {"is_question": True},
        ),
        (
            "如果数据没问题，再把 strategy_sample_v2.csv 作为当前策略样本。",
            {"is_conditional": True},
        ),
        (
            "先不要绑定 strategy_sample_v2.csv。",
            {"withholds_action": True},
        ),
        (
            "读取 strategy_sample_v2.csv 作为当前策略样本，坏样本字段用 bad_flag，另外删除 score。",
            {"requests_change": True},
        ),
    ],
)
def test_strategy_sample_binding_safety_signals_leave_data_state_unchanged(
    tmp_path: Path,
    monkeypatch,
    instruction: str,
    overrides: dict,
) -> None:
    source = _source_dir(tmp_path)
    decision_client = _StrategySampleBindingClient(overrides=overrides)
    _install_client(monkeypatch, decision_client)

    with TestClient(create_app(tmp_path)) as client:
        created = client.post(
            "/api/tasks",
            json={
                "model_name": "策略样本绑定语义安全信号",
                "validator": "qa",
                "source_dir": str(source),
                "task_type": "strategy",
                "run_mode": "agent",
                "target_col": "y",
            },
        )
        task_id = str(created.json()["id"])

        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": instruction, "model_id": "semantic-test"},
        )

        assert response.status_code == 202, response.text
        assert response.json()["status"] == "clarification_required"
        assert response.json()["messages"][-1]["metadata"]["code"] == (
            "semantic_intent_clarification"
        )
        settings = client.app.state.settings
        workspace = DataWorkspaceRepository(settings.db_path).get_or_default(task_id)
        assert workspace.active_dataset_id is None
        assert DatasetRepository(settings.db_path).list_datasets(task_id) == []
        assert TaskRepository(settings.db_path).get_task(task_id).target_col == "y"
        assert client.app.state.plan_repo.list_plans_for_task(task_id) == []


def test_strategy_source_change_during_independent_review_discards_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _source_dir(tmp_path)
    sample_path = source / "strategy_sample_v2.csv"

    def mutate_source() -> None:
        frame = pd.read_csv(sample_path)
        frame.loc[len(frame)] = [999, 700.0, 1]
        frame.to_csv(sample_path, index=False)

    decision_client = _StrategySampleBindingClient(mutation=mutate_source)
    _install_client(monkeypatch, decision_client)

    with TestClient(create_app(tmp_path)) as client:
        created = client.post(
            "/api/tasks",
            json={
                "model_name": "策略样本绑定源材料漂移",
                "validator": "qa",
                "source_dir": str(source),
                "task_type": "strategy",
                "run_mode": "agent",
                "target_col": "y",
            },
        )
        task_id = str(created.json()["id"])

        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": _INSTRUCTION, "model_id": "semantic-test"},
        )

        assert response.status_code == 202, response.text
        assert response.json()["status"] == "clarification_required"
        last = response.json()["messages"][-1]
        assert last["metadata"]["code"] == "semantic_intent_clarification"
        assert "状态已变化" in last["metadata"]["reason"]
        settings = client.app.state.settings
        workspace = DataWorkspaceRepository(settings.db_path).get_or_default(task_id)
        assert workspace.active_dataset_id is None
        assert DatasetRepository(settings.db_path).list_datasets(task_id) == []
        assert TaskRepository(settings.db_path).get_task(task_id).target_col == "y"
        assert client.app.state.plan_repo.list_plans_for_task(task_id) == []


@pytest.mark.parametrize(
    "instruction",
    [
        "读取材料目录里的 another_sample.csv，把它作为当前策略样本，坏样本字段用 bad_flag。",
        "读取材料目录里的 strategy_sample_v2.csv，把它作为当前策略样本，坏样本字段用 another_flag。",
    ],
)
def test_llm_binding_misroute_cannot_substitute_unmentioned_file_or_target(
    tmp_path: Path,
    monkeypatch,
    instruction: str,
) -> None:
    source = _source_dir(tmp_path)
    decision_client = _StrategySampleBindingClient()
    _install_client(monkeypatch, decision_client)

    with TestClient(create_app(tmp_path)) as client:
        created = client.post(
            "/api/tasks",
            json={
                "model_name": "策略样本绑定操作数防替换",
                "validator": "qa",
                "source_dir": str(source),
                "task_type": "strategy",
                "run_mode": "agent",
                "target_col": "y",
            },
        )
        task_id = str(created.json()["id"])

        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": instruction, "model_id": "semantic-test"},
        )

        assert response.status_code == 202, response.text
        assert response.json()["status"] == "clarification_required"
        assert response.json()["messages"][-1]["metadata"]["code"] == (
            "semantic_intent_clarification"
        )
        settings = client.app.state.settings
        assert (
            DataWorkspaceRepository(settings.db_path)
            .get_or_default(task_id)
            .active_dataset_id
            is None
        )
        assert DatasetRepository(settings.db_path).list_datasets(task_id) == []
        assert TaskRepository(settings.db_path).get_task(task_id).target_col == "y"
        assert client.app.state.plan_repo.list_plans_for_task(task_id) == []


@pytest.mark.skipif(
    not os.environ.get(_LIVE_WORKSPACE_ENV),
    reason=f"set {_LIVE_WORKSPACE_ENV} to run the live DeepSeek integration",
)
def test_live_deepseek_http_turn_binds_exact_strategy_sample_without_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile = resolve_llm_model(
        Path(os.environ[_LIVE_WORKSPACE_ENV]).expanduser().resolve(),
        model_id=os.environ.get(_LIVE_MODEL_ENV) or None,
        role="router_intent",
    )
    if "deepseek" not in str(profile.get("model_name") or "").lower():
        pytest.skip("configured diagnostic profile is not DeepSeek")
    call_records: list[dict] = []
    decision_client = OpenAICompatibleLLMClient(
        profile,
        on_call_recorded=call_records.append,
    )
    _install_client(monkeypatch, decision_client)
    source = _source_dir(tmp_path)

    with TestClient(create_app(tmp_path)) as client:
        created = client.post(
            "/api/tasks",
            json={
                "model_name": "真实 DeepSeek 策略样本绑定",
                "validator": "qa",
                "source_dir": str(source),
                "task_type": "strategy",
                "run_mode": "agent",
                "target_col": "y",
            },
        )
        task_id = str(created.json()["id"])

        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": _INSTRUCTION, "model_id": "live-semantic-test"},
        )

        diagnostic = json.dumps(
            {
                "status_code": response.status_code,
                "response_status": (
                    response.json().get("status")
                    if response.headers.get("content-type", "").startswith(
                        "application/json"
                    )
                    else None
                ),
                "last_code": (
                    (response.json().get("messages") or [{}])[-1]
                    .get("metadata", {})
                    .get("code")
                    if response.headers.get("content-type", "").startswith(
                        "application/json"
                    )
                    else None
                ),
                "call_records": call_records,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        assert response.status_code == 202, diagnostic
        assert response.json()["status"] == "ok", diagnostic
        assert response.json()["messages"][-1]["metadata"]["code"] == (
            "strategy_sample_binding_complete"
        ), diagnostic
        workspace = DataWorkspaceRepository(
            client.app.state.settings.db_path
        ).get_or_default(task_id)
        assert workspace.active_dataset_id is not None
        assert workspace.semantic_mapping.target_col == "bad_flag"
        assert client.app.state.plan_repo.list_plans_for_task(task_id) == []
