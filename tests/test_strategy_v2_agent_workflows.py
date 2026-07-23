"""Natural-language V2 strategy evidence workflows at the real Agent seam."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, TaskRepository
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.pending_strategy_requests import (
    PendingStrategyRequestRepository,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactDataError,
    TaskArtifactRepository,
)


def _eq(column: str, value: object) -> dict:
    return {
        "op": "eq",
        "left": {"column": column},
        "right": {"literal": value},
    }


def _sample_v2_inputs(*, drop_nan_labels: bool = False) -> dict:
    return {
        "target_bad_value": 1,
        "drop_nan_labels": drop_nan_labels,
        "approval_population": {"inclusion": None, "exclusion": None},
        "risk_population": {"inclusion": None, "exclusion": None},
        "partitioning": {
            "method": "predicate_ast",
            "selectors": {
                "development": _eq("sample_role", "dev"),
                "validation": _eq("sample_role", "valid"),
                "oot": _eq("sample_role", "oot"),
            },
        },
        "maturity": {
            "status": "confirmed_matured",
            "performance_window_days": 30,
            "cutoff_date": "2026-04-30",
            "reason": None,
        },
        "performance_window": {"status": "provided", "days": 30},
        "observation_window": {
            "status": "provided",
            "start": "2026-01-01",
            "end": "2026-04-30",
        },
        "field_bindings": {
            "entity_field": "customer_id",
            "time_field": "apply_date",
            "group_field": None,
            "month_field": "apply_month",
            "weight_field": "weight",
            "loan_amount_field": "loan_amount",
            "overdue_amount_field": "overdue_amount",
        },
        "historical_score": {
            "status": "available",
            "column": "legacy_score",
            "direction": "higher_is_riskier",
            "reason": None,
        },
    }


def _sample_v2_utterance(*, drop_nan_labels: bool = False) -> str:
    missing_policy = "丢弃缺失标签" if drop_nan_labels else "不丢弃缺失标签"
    return (
        f"固化 V2 策略样本设计；1 代表坏样本；{missing_policy}；"
        "审批总体无纳排条件；风险总体无纳排条件；切分列 sample_role；"
        "开发值 dev；验证值 valid；OOT 值 oot；表现窗 30 天；"
        "观察窗 2026-01-01 至 2026-04-30；成熟度已确认成熟；"
        "成熟表现窗 30 天；成熟度截止日 2026-04-30；实体字段 customer_id；"
        "时间字段 apply_date；分组字段暂无；月份字段 apply_month；"
        "权重字段 weight；放款金额字段 loan_amount；逾期金额字段 overdue_amount；"
        "历史分 legacy_score，越高越风险。"
    )


def _univariate_payload(*, feature: str = "legacy_score", method: str = "equal_width") -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "univariate_candidate_analysis",
        "workflow_inputs": {
            "features": [feature],
            "methods": [method],
            "bin_count": 3,
            "min_bin_pct": 0.02,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "sentinel_values": [],
        },
    }


def _model_evidence_payload() -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "strategy_model_evidence_v2",
        "workflow_inputs": {},
    }


def _legacy_sample_payload() -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "strategy_sample_design",
        "workflow_inputs": {
            "target_bad_value": 1,
            "performance_window_status": "provided",
            "performance_window_days": 30,
            "observation_window_status": "provided",
            "observation_start": "2026-01-01",
            "observation_end": "2026-04-30",
            "maturity_status": "confirmed_matured",
            "split_col": "sample_role",
            "development_values": ["dev"],
            "validation_values": ["valid"],
            "oot_values": ["oot"],
            "month_col": "apply_month",
            "weight_col": "weight",
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "drop_nan_labels": False,
        },
    }


class _SequencedStrategyLLM:
    def __init__(self, *payloads: dict) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        assert self.payloads, "unexpected compiler LLM call"
        return json.dumps(self.payloads.pop(0), ensure_ascii=False)


def _create_strategy_task(client: TestClient, tmp_path: Path) -> str:
    source_dir = client.app.state.settings.workspace / f"source-{tmp_path.name}"
    source_dir.mkdir(exist_ok=True)
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "V2 策略证据 Agent",
            "validator": "qa",
            "source_dir": str(source_dir),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _register_workspace_sample(
    client: TestClient,
    task_id: str,
    tmp_path: Path,
    *,
    nan_label: bool = False,
    target_col: str | None = "bad",
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bad: list[object] = [0, 1, 0, 1, 0, 1]
    if nan_label:
        bad[1] = None
    frame = pd.DataFrame(
        {
            "sample_role": ["dev", "dev", "valid", "valid", "oot", "oot"],
            "customer_id": ["a", "b", "c", "d", "e", "f"],
            "apply_date": [
                "2026-01-01",
                "2026-01-10",
                "2026-02-01",
                "2026-02-10",
                "2026-03-01",
                "2026-03-10",
            ],
            "apply_month": [
                "202601",
                "202601",
                "202602",
                "202602",
                "202603",
                "202603",
            ],
            "legacy_score": [100.0, 200.0, 120.0, 220.0, 140.0, 240.0],
            "weight": [1.0] * 6,
            "loan_amount": [100.0, 200.0, 150.0, 180.0, 300.0, 250.0],
            "overdue_amount": [0.0, 20.0, 0.0, 10.0, 0.0, 30.0],
            "bad": bad,
        }
    )
    source = tmp_path / f"{task_id}.parquet"
    frame.to_parquet(source, index=False)
    settings = client.app.state.settings
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_from_upload(task_id, source, role="strategy_sample")
    repository = DataWorkspaceRepository(settings.db_path)
    activated = repository.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    mapping = DataSemanticMapping(
        target_col=target_col,
        field_roles={
            "sample_role": "segment",
            "customer_id": "id",
            "apply_date": "date",
            "apply_month": "month",
            "legacy_score": "score",
            "weight": "weight",
            "loan_amount": "loan_amount",
            "overdue_amount": "overdue_amount",
            **({"bad": "target"} if target_col is not None else {}),
        },
    )
    workspace = repository.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=mapping,
        ),
        expected_revision=activated.revision,
    )
    return dataset, workspace, mapping


def _register_candidate_registry_stub(
    client: TestClient,
    task_id: str,
    *,
    suffix: str,
    sample_design_ref: object,
) -> dict:
    records = TaskArtifactRepository(
        client.app.state.settings.db_path
    ).list_for_task(task_id)
    bundle_record = next(
        record
        for record in reversed(records)
        if record["kind"] == "strategy_sample_design_v2_json"
    )
    bundle = json.loads(Path(bundle_record["path"]).read_text(encoding="utf-8"))
    identity = bundle["sample_design"]["identity"]
    dataset_ref = identity["dataset_ref"]
    workspace_ref = identity["workspace_ref"]
    path = (
        client.app.state.settings.tasks_dir
        / task_id
        / f"candidate-registry-stub-{suffix}.json"
    )
    raw = b"{}"
    path.write_bytes(raw)
    return TaskArtifactRepository(client.app.state.settings.db_path).register(
        task_id=task_id,
        kind="strategy_candidate_json",
        path=str(path),
        content_hash=hashlib.sha256(raw).hexdigest(),
        origin_tool="strategy.analyze_univariate_candidates",
        provenance={
            "candidate_id": f"candidate-stub-{suffix}",
            "evidence_hash": hashlib.sha256(
                f"evidence-{suffix}".encode("utf-8")
            ).hexdigest(),
            "dataset_id": dataset_ref["dataset_id"],
            "dataset_content_hash": dataset_ref["content_hash"],
            "workspace_revision": workspace_ref["revision"],
            "workspace_generation": workspace_ref["generation"],
            "semantic_mapping_hash": workspace_ref["semantic_mapping_hash"],
            "generation_parameters": {
                "sample_design_ref": sample_design_ref,
            },
        },
    )


def test_fresh_sample_v2_executes_two_real_tools_with_platform_owned_bindings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    dataset, workspace, mapping = _register_workspace_sample(client, task_id, tmp_path)
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2"
    ], json.dumps(response.json()["messages"][-1], ensure_ascii=False)
    assert plans[0]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    assert len(stored.steps) == 2
    first, second = stored.steps
    assert first.status.value == "done"
    assert second.status.value == "done"
    assert first.inputs["dataset_id"] == dataset.id
    assert first.inputs["expected_dataset_content_hash"] == dataset.content_hash
    assert first.inputs["workspace_revision"] == workspace.revision
    assert first.inputs["workspace_generation"] == workspace.analysis_generation
    assert first.inputs["semantic_mapping_hash"] == data_semantic_mapping_hash(mapping)
    assert first.inputs["target_col"] == "bad"
    assert first.inputs["split_col"] == "sample_role"
    assert first.inputs["month_col"] == "apply_month"
    assert first.inputs["weight_col"] == "weight"
    assert first.inputs["loan_amount_col"] == "loan_amount"
    assert first.inputs["overdue_amount_col"] == "overdue_amount"
    assert second.inputs["legacy_sample_design_ref"] == {
        "artifact_id": f"$ref:{first.id}.output.artifact.artifact_id",
        "artifact_content_hash": f"$ref:{first.id}.output.artifact.content_hash",
        "sample_design_id": f"$ref:{first.id}.output.sample_design_id",
        "sample_design_content_hash": f"$ref:{first.id}.output.content_hash",
        "partition": "development",
    }
    assert second.inputs["relationship"] == "nested_same_cohort"
    assert second.inputs["scope"] == "strategy_development"
    assert second.inputs["policy"]["diagnostic_severities"]["maturity"] == "fail"
    output = client.app.state.plan_repo.load_step_output(second.id)
    assert output["not_created_strategy"] is True
    assert output["not_adopted"] is True
    assert output["not_deployed"] is True
    assert len(llm.calls) == 1


def test_fresh_sample_v2_requires_workspace_before_compiler_llm(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    )

    assert response.status_code == 202, response.text
    assert response.json()["code"] == "strategy_sample_design_workspace_required"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert llm.calls == []


def test_fresh_sample_v2_requires_workspace_target_before_compiler_llm(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(client, task_id, tmp_path, target_col=None)
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    )

    assert response.status_code == 202, response.text
    assert response.json()["code"] == "strategy_sample_design_workspace_required"
    assert "二元目标列" in response.json()["messages"][-1]["content"]
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert llm.calls == []


def test_fresh_sample_v2_nan_labels_pause_until_explicit_exclusion_authorization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(client, task_id, tmp_path, nan_label=True)
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(drop_nan_labels=False),
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance(drop_nan_labels=False)},
    )

    assert opened.status_code == 202, opened.text
    assert opened.json()["code"] == "strategy_drop_nan_labels_confirmation_required"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []

    resumed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认将空标签仅从风险分母排除并继续"},
    )

    assert resumed.status_code == 202, resumed.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == ["strategy_sample_design_v2"]
    assert plans[0]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    assert stored.steps[0].inputs["drop_nan_labels"] is True
    output = client.app.state.plan_repo.load_step_output(stored.steps[1].id)
    assert output["bundle"]["sample_design"]["target_selector"]["drop_missing"] is True
    assert len(llm.calls) == 1


def test_fresh_model_evidence_v2_binds_live_sample_and_candidate_registry_refs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(client, task_id, tmp_path)
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        },
        _univariate_payload(),
        _model_evidence_payload(),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    sample_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    )
    assert sample_response.status_code == 202, sample_response.text

    candidate_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "对 legacy_score 用 equal_width 做单变量分析，目标箱数 3，"
                "最小箱占比 2%，放款金额列 loan_amount，"
                "逾期金额列 overdue_amount，不设置哨兵值"
            )
        },
    )
    assert candidate_response.status_code == 202, candidate_response.text

    evidence_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "生成 Strategy ModelEvidence V2，汇总当前已认证单变量候选证据"},
    )

    assert evidence_response.status_code == 202, evidence_response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2",
        "strategy_univariate_candidate_analysis",
        "strategy_model_evidence_v2",
    ], json.dumps(evidence_response.json()["messages"][-1], ensure_ascii=False)
    assert plans[-1]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert len(stored.steps) == 1
    step = stored.steps[0]
    records = TaskArtifactRepository(
        client.app.state.settings.db_path
    ).list_for_task(task_id)
    membership = next(
        record
        for record in records
        if record["kind"] == "strategy_sample_membership_v2_binary"
    )
    sample_bundle = next(
        record
        for record in records
        if record["kind"] == "strategy_sample_design_v2_json"
    )
    candidate = next(
        record for record in records if record["kind"] == "strategy_candidate_json"
    )
    sample_provenance = sample_bundle["provenance"]
    candidate_provenance = candidate["provenance"]
    assert step.inputs["sample_design_ref"] == {
        "membership_artifact_id": membership["id"],
        "expected_membership_artifact_content_hash": membership["content_hash"],
        "bundle_artifact_id": sample_bundle["id"],
        "expected_bundle_artifact_content_hash": sample_bundle["content_hash"],
        "expected_bundle_id": sample_provenance["bundle_id"],
        "expected_sample_design_id": sample_provenance["sample_design_id"],
        "expected_sample_design_content_hash": sample_provenance[
            "sample_design_content_hash"
        ],
    }
    assert step.inputs["univariate_sources"] == [
        {
            "artifact_id": candidate["id"],
            "expected_artifact_content_hash": candidate["content_hash"],
            "expected_candidate_id": candidate_provenance["candidate_id"],
            "expected_evidence_hash": candidate_provenance["evidence_hash"],
        }
    ]
    output = client.app.state.plan_repo.load_step_output(step.id)
    assert output["source_artifacts"] == [
        {
            "artifact_id": candidate["id"],
            "kind": "strategy_candidate_json",
            "content_hash": candidate["content_hash"],
        }
    ]
    assert output["univariate_only"] is True
    assert len(llm.calls) == 3
    model_prompt = json.dumps(llm.calls[-1], ensure_ascii=False, default=str)
    assert membership["id"] not in model_prompt
    assert sample_bundle["id"] not in model_prompt
    assert candidate["id"] not in model_prompt


def test_model_evidence_v2_needs_no_dataset_preview_but_requires_live_sample_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    llm = _SequencedStrategyLLM(_model_evidence_payload())
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "归集认证单变量候选证据，生成模型证据 V2"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["code"] == "strategy_model_evidence_v2_sample_required"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert len(llm.calls) == 1


def test_model_evidence_v2_excludes_cross_task_or_incompatible_registry_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(client, task_id, tmp_path)
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        },
        _model_evidence_payload(),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    sample_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    )
    assert sample_response.status_code == 202, sample_response.text

    foreign_path = (
        client.app.state.settings.tasks_dir / task_id / "foreign-candidate.json"
    )
    foreign_path.write_text("{}", encoding="utf-8")
    TaskArtifactRepository(client.app.state.settings.db_path).register(
        task_id=task_id,
        kind="strategy_candidate_json",
        path=str(foreign_path),
        content_hash=hashlib.sha256(b"{}").hexdigest(),
        origin_tool="strategy.analyze_univariate_candidates",
        provenance={
            "candidate_id": "candidate-" + "1" * 32,
            "evidence_hash": "2" * 64,
            "dataset_id": "foreign-task-dataset",
            "dataset_content_hash": "3" * 64,
            "workspace_revision": 1,
            "workspace_generation": 1,
            "semantic_mapping_hash": "4" * 64,
            "generation_parameters": {},
        },
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "生成 Strategy ModelEvidence V2，汇总当前已认证单变量候选证据"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["code"] == "strategy_model_evidence_v2_candidate_required"
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == ["strategy_sample_design_v2"]
    assert len(llm.calls) == 2


def test_model_evidence_v2_excludes_candidate_file_drift_and_creates_no_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(client, task_id, tmp_path)
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        },
        _univariate_payload(),
        _model_evidence_payload(),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    ).status_code == 202
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "对 legacy_score 用 equal_width 做单变量分析，目标箱数 3，"
                "最小箱占比 2%，放款金额列 loan_amount，"
                "逾期金额列 overdue_amount，不设置哨兵值"
            )
        },
    ).status_code == 202
    candidate = next(
        record
        for record in TaskArtifactRepository(
            client.app.state.settings.db_path
        ).list_for_task(task_id)
        if record["kind"] == "strategy_candidate_json"
    )
    Path(candidate["path"]).write_text("{}", encoding="utf-8")

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "生成 Strategy ModelEvidence V2，汇总当前已认证单变量候选证据"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["code"] == "strategy_model_evidence_v2_candidate_invalid"
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2",
        "strategy_univariate_candidate_analysis",
    ]


def test_model_evidence_v2_normalizes_candidate_registry_load_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(client, task_id, tmp_path)
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        },
        _univariate_payload(),
        _model_evidence_payload(),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    ).status_code == 202
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "对 legacy_score 用 equal_width 做单变量分析，目标箱数 3，"
                "最小箱占比 2%，放款金额列 loan_amount，"
                "逾期金额列 overdue_amount，不设置哨兵值"
            )
        },
    ).status_code == 202
    candidate = next(
        record
        for record in TaskArtifactRepository(
            client.app.state.settings.db_path
        ).list_for_task(task_id)
        if record["kind"] == "strategy_candidate_json"
    )
    original_get_for_task = TaskArtifactRepository.get_for_task

    def broken_candidate_load(self, owner_task_id: str, artifact_id: str):
        if owner_task_id == task_id and artifact_id == candidate["id"]:
            raise TaskArtifactDataError("persisted candidate provenance is invalid")
        return original_get_for_task(self, owner_task_id, artifact_id)

    monkeypatch.setattr(
        TaskArtifactRepository,
        "get_for_task",
        broken_candidate_load,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "生成 Strategy ModelEvidence V2，汇总当前已认证单变量候选证据"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["code"] == "strategy_model_evidence_v2_candidate_invalid"
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2",
        "strategy_univariate_candidate_analysis",
    ]


def test_model_evidence_v2_rejects_snapshot_candidate_with_incomplete_ref(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(client, task_id, tmp_path)
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        },
        _model_evidence_payload(),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    ).status_code == 202
    _register_candidate_registry_stub(
        client,
        task_id,
        suffix="incomplete-ref",
        sample_design_ref={},
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "生成 Strategy ModelEvidence V2，汇总当前已认证单变量候选证据"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["code"] == "strategy_model_evidence_v2_candidate_invalid"
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == ["strategy_sample_design_v2"]


def test_model_evidence_v2_fails_closed_on_latest_sample_bundle_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(client, task_id, tmp_path)
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        },
        _model_evidence_payload(),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    ).status_code == 202
    bundle = next(
        record
        for record in TaskArtifactRepository(
            client.app.state.settings.db_path
        ).list_for_task(task_id)
        if record["kind"] == "strategy_sample_design_v2_json"
    )
    Path(bundle["path"]).write_text("{}", encoding="utf-8")

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "生成 Strategy ModelEvidence V2，汇总当前已认证单变量候选证据"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["code"] == "strategy_model_evidence_v2_sample_invalid"
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == ["strategy_sample_design_v2"]


def test_model_evidence_v2_collects_multiple_compatible_sources_in_stable_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(client, task_id, tmp_path)
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        },
        _univariate_payload(method="equal_width"),
        _univariate_payload(method="equal_frequency"),
        _model_evidence_payload(),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    ).status_code == 202
    for method, label in (
        ("equal_width", "等距"),
        ("equal_frequency", "等频"),
    ):
        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={
                "content": (
                    f"对 legacy_score 用{label}法做单变量分析，"
                    "目标箱数 3，最小箱占比 2%，"
                    "放款金额列 loan_amount，逾期金额列 overdue_amount，"
                    "不设置哨兵值"
                )
            },
        )
        assert response.status_code == 202, response.text
        candidate_plans = [
            plan
            for plan in client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
            if plan["template_id"] == "strategy_univariate_candidate_analysis"
        ]
        assert len(candidate_plans) == (1 if method == "equal_width" else 2), (
            json.dumps(response.json()["messages"][-1], ensure_ascii=False)
        )

    evidence = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "生成 Strategy ModelEvidence V2，汇总当前已认证单变量候选证据"},
    )

    assert evidence.status_code == 202, evidence.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2",
        "strategy_univariate_candidate_analysis",
        "strategy_univariate_candidate_analysis",
        "strategy_model_evidence_v2",
    ]
    stored = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    sources = stored.steps[0].inputs["univariate_sources"]
    assert len(sources) == 2
    assert sources == sorted(
        sources,
        key=lambda item: (item["expected_candidate_id"], item["artifact_id"]),
    )
    assert len({item["artifact_id"] for item in sources}) == 2
    assert len({item["expected_candidate_id"] for item in sources}) == 2


def test_model_evidence_v2_rejects_global_candidate_source_limit_before_loading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(client, task_id, tmp_path)
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        },
        _model_evidence_payload(),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    ).status_code == 202
    bundle_record = next(
        record
        for record in TaskArtifactRepository(
            client.app.state.settings.db_path
        ).list_for_task(task_id)
        if record["kind"] == "strategy_sample_design_v2_json"
    )
    bundle = json.loads(Path(bundle_record["path"]).read_text(encoding="utf-8"))
    legacy_ref = bundle["sample_design"]["compatibility"][
        "legacy_development_ref"
    ]
    for index in range(101):
        _register_candidate_registry_stub(
            client,
            task_id,
            suffix=f"limit-{index:03d}",
            sample_design_ref=legacy_ref,
        )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "生成 Strategy ModelEvidence V2，汇总当前已认证单变量候选证据"},
    )

    assert response.status_code == 202, response.text
    assert (
        response.json()["code"]
        == "strategy_model_evidence_v2_candidate_budget_exceeded"
    )
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == ["strategy_sample_design_v2"]


def test_model_evidence_v2_rechecks_registry_snapshot_before_plan_creation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(client, task_id, tmp_path)
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        },
        _univariate_payload(method="equal_width"),
        _univariate_payload(method="equal_frequency"),
        _model_evidence_payload(),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    ).status_code == 202
    for method, label in (
        ("equal_width", "等距"),
        ("equal_frequency", "等频"),
    ):
        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={
                "content": (
                    f"对 legacy_score 用{label}法做单变量分析，目标箱数 3，"
                    "最小箱占比 2%，放款金额列 loan_amount，"
                    "逾期金额列 overdue_amount，不设置哨兵值"
                )
            },
        )
        assert response.status_code == 202, (method, response.text)

    repository = TaskArtifactRepository(client.app.state.settings.db_path)
    candidates = [
        record
        for record in repository.list_for_task(task_id)
        if record["kind"] == "strategy_candidate_json"
    ]
    assert len(candidates) == 2
    delayed = candidates[-1]
    with repository.transaction() as conn:
        conn.execute(
            "DELETE FROM task_artifacts WHERE task_id = ? AND id = ?",
            (task_id, delayed["id"]),
        )
        conn.commit()

    original_list_for_task = TaskArtifactRepository.list_for_task
    task_snapshot_reads = 0
    published = False

    def publish_between_discovery_and_cas(self, owner_task_id: str):
        nonlocal task_snapshot_reads, published
        records = original_list_for_task(self, owner_task_id)
        if owner_task_id == task_id:
            task_snapshot_reads += 1
            if task_snapshot_reads == 2 and not published:
                repository.register(
                    task_id=task_id,
                    kind=delayed["kind"],
                    path=delayed["path"],
                    content_hash=delayed["content_hash"],
                    origin_tool=delayed["origin_tool"],
                    provenance=delayed["provenance"],
                    created_at=delayed["created_at"],
                )
                published = True
        return records

    monkeypatch.setattr(
        TaskArtifactRepository,
        "list_for_task",
        publish_between_discovery_and_cas,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "生成 Strategy ModelEvidence V2，汇总当前已认证单变量候选证据"},
    )

    assert response.status_code == 202, response.text
    assert (
        response.json().get("code")
        == "strategy_model_evidence_v2_registry_changed"
    ), response.json()
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2",
        "strategy_univariate_candidate_analysis",
        "strategy_univariate_candidate_analysis",
    ]


def test_persisted_legacy_sample_draft_replays_but_fresh_legacy_route_is_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recovery_client = TestClient(create_app(tmp_path / "recovery-workspace"))
    recovery_task = _create_strategy_task(recovery_client, tmp_path / "recovery")
    _register_workspace_sample(
        recovery_client,
        recovery_task,
        tmp_path / "recovery",
    )
    settings = recovery_client.app.state.settings
    pending = PendingStrategyRequestRepository(settings.db_path).create(
        task_id=recovery_task,
        validated_draft=_legacy_sample_payload(),
        dataset_identity=None,
        target_col=None,
    )
    TaskRepository(settings.db_path).add_agent_message(
        recovery_task,
        role="assistant",
        stage="chat",
        content="历史 V1 样本草案等待恢复确认。",
        metadata={
            "strategy_request": pending.to_metadata_reference(),
            "intent": "strategy_request_confirmation",
            "kind": "confirmation",
        },
    )
    recovery_llm = _SequencedStrategyLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: recovery_llm,
    )

    confirmed = recovery_client.post(
        f"/api/tasks/{recovery_task}/agent/messages",
        json={"content": "确认"},
    )

    assert confirmed.status_code == 202, confirmed.text
    recovery_plans = recovery_client.get(
        f"/api/tasks/{recovery_task}/plans"
    ).json()["plans"]
    assert [plan["template_id"] for plan in recovery_plans] == [
        "strategy_sample_design"
    ]
    assert PendingStrategyRequestRepository(settings.db_path).get(
        recovery_task,
        pending.id,
    ).status == "consumed"
    assert recovery_llm.calls == []

    fresh_client = TestClient(create_app(tmp_path / "fresh-workspace"))
    fresh_task = _create_strategy_task(fresh_client, tmp_path / "fresh")
    _register_workspace_sample(fresh_client, fresh_task, tmp_path / "fresh")
    fresh_llm = _SequencedStrategyLLM(
        _legacy_sample_payload(),
        _legacy_sample_payload(),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: fresh_llm,
    )

    rejected = fresh_client.post(
        f"/api/tasks/{fresh_task}/agent/messages",
        json={"content": _sample_v2_utterance()},
    )

    assert rejected.status_code == 202, rejected.text
    assert rejected.json()["code"] == "invalid_strategy_request"
    assert fresh_client.get(f"/api/tasks/{fresh_task}/plans").json()["plans"] == []
    assert len(fresh_llm.calls) == 2
