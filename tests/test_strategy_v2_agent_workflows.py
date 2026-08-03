"""Natural-language V2 strategy evidence workflows at the real Agent seam."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
import pytest

from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, StrategyRepository, TaskRepository
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy import pool_tools
from marvis.packs.strategy.sample_design_v2_native_tools import (
    run_materialize_sample_design_v2_native,
)
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.pending_strategy_requests import (
    PendingStrategyRequestRepository,
)
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
from marvis.repositories.task_artifacts import (
    TaskArtifactDataError,
    TaskArtifactRepository,
)
import marvis.repositories.task_artifacts as task_artifact_repository


def _eq(column: str, value: object) -> dict:
    return {
        "op": "eq",
        "left": {"column": column},
        "right": {"literal": value},
    }


def _sample_v2_inputs(
    *,
    drop_nan_labels: bool = False,
    relationship: str = "nested_same_cohort",
) -> dict:
    return {
        "target_bad_value": 1,
        "drop_nan_labels": drop_nan_labels,
        "relationship": relationship,
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


def _sample_v2_utterance(
    *,
    drop_nan_labels: bool = False,
    relationship: str = "nested_same_cohort",
) -> str:
    missing_policy = "丢弃缺失标签" if drop_nan_labels else "不丢弃缺失标签"
    relationship_text = (
        "审批总体与风险总体是同批 cohort 的嵌套关系"
        if relationship == "nested_same_cohort"
        else "审批总体与风险总体是平行时间 cohort"
    )
    return (
        f"固化 V2 策略样本设计；1 代表坏样本；{missing_policy}；"
        f"{relationship_text}；"
        "审批总体无纳排条件；风险总体无纳排条件；切分列 sample_role；"
        "开发值 dev；验证值 valid；OOT 值 oot；表现窗 30 天；"
        "观察窗 2026-01-01 至 2026-04-30；成熟度已确认成熟；"
        "成熟表现窗 30 天；成熟度截止日 2026-04-30；实体字段 customer_id；"
        "时间字段 apply_date；分组字段暂无；月份字段 apply_month；"
        "权重字段 weight；放款金额字段 loan_amount；逾期金额字段 overdue_amount；"
        "历史分 legacy_score，越高越风险。"
    )


def _parallel_population_sample_v2_inputs() -> dict:
    inputs = _sample_v2_inputs(relationship="parallel_time_cohorts")
    inputs["approval_population"]["inclusion"] = {
        "match": "all",
        "conditions": [
            {"column": "approval_flag", "operator": "eq", "value": 1}
        ],
    }
    inputs["risk_population"]["inclusion"] = {
        "match": "all",
        "conditions": [
            {"column": "risk_flag", "operator": "eq", "value": 1}
        ],
    }
    return inputs


def _parallel_population_sample_v2_utterance() -> str:
    return _sample_v2_utterance(
        relationship="parallel_time_cohorts"
    ).replace(
        "审批总体无纳排条件；风险总体无纳排条件；",
        "审批总体纳入 approval_flag 等于 1，无排除条件；"
        "风险总体纳入 risk_flag 等于 1，无排除条件；",
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


def _automatic_tree_payload() -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "automatic_tree_candidate_build",
        "workflow_inputs": {
            "features": ["legacy_score"],
            "max_depth": 2,
            "min_leaf_count": 2,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
        },
    }


def _refinement_payload() -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "univariate_candidate_refinement",
        "workflow_inputs": {
            "feature": "legacy_score",
            "method": "equal_width",
            "bin_count": 3,
            "min_bin_pct": 0.02,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "selection": {
                "risk_threshold": {"operator": ">=", "value": 0.5}
            },
            "selection_reason": "保留观测坏率达到 50% 的风险箱",
        },
    }


def _cross_matrix_payload() -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "cross_matrix_analysis",
        "workflow_inputs": {
            "x_feature": "legacy_score",
            "x_method": "equal_width",
            "y_feature": "age",
            "y_method": "equal_width",
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


def _deterministic_candidate_payload(strategy_type: str) -> dict:
    if strategy_type == "limit":
        return {
            "operation": "develop",
            "strategy_type": "limit",
            "candidate_design": {
                "method": "score_band_limit",
                "score_col": "legacy_score",
                "n_bands": 2,
                "limit_grid": [1000, 2000],
                "max_expected_loss_per_account": 200,
            },
            "economics_inputs": {
                "pd_value": 0.10,
                "lgd_value": 0.50,
                "utilization_value": 0.60,
            },
        }
    if strategy_type == "pricing":
        return {
            "operation": "develop",
            "strategy_type": "pricing",
            "candidate_design": {
                "method": "score_band_pricing",
                "score_col": "legacy_score",
                "n_bands": 2,
                "rate_grid": [0.12, 0.18],
                "min_roa": 0.0,
            },
            "economics_inputs": {
                "ead_col": "ead",
                "pd_col": "pd",
                "lgd_value": 0.50,
                "funding_rate_value": 0.04,
                "term_months_value": 12,
                "operating_cost_per_loan_value": 10,
            },
        }
    assert strategy_type == "segmentation"
    return {
        "operation": "develop",
        "strategy_type": "segmentation",
        "candidate_design": {
            "method": "single_variable_segmentation",
            "feature_col": "legacy_score",
            "n_bands": 2,
        },
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


def _approval_strategy_spec(*, threshold: float = 250.0) -> dict:
    return {
        "strategy_type": "approval",
        "default_action": {"type": "approval"},
        "rules": [
            {
                "rule_id": f"legacy-score-above-{threshold:g}",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": "legacy_score",
                    "operator": ">=",
                    "value": threshold,
                },
                "action": {"type": "reject"},
            }
        ],
    }


def _create_stored_approval_strategy(
    client: TestClient,
    task_id: str,
    *,
    threshold: float,
) -> str:
    strategy = build_strategy_from_spec(
        _approval_strategy_spec(threshold=threshold),
        description=f"native-parallel-baseline-{threshold:g}",
    )
    StrategyRepository(client.app.state.settings.db_path).create_strategy(
        task_id,
        strategy,
    )
    return strategy.id


def _materialize_native_parallel_sample_from_agent(
    client: TestClient,
    task_id: str,
    monkeypatch,
) -> dict[str, str]:
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _parallel_population_sample_v2_inputs(),
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _parallel_population_sample_v2_utterance()},
    )
    assert response.status_code == 202, response.text
    plan = client.app.state.plan_repo.list_plans_for_task(task_id)[-1]
    assert plan.template_id == "strategy_sample_design_v2_native"
    assert plan.status == "done"
    output = client.app.state.plan_repo.load_step_output(plan.steps[0].id)
    assert output["membership"]["counts"]["approval"]["development"] == 5
    assert output["membership"]["counts"]["risk"]["development"] == 4
    bundle = next(
        record
        for record in reversed(
            TaskArtifactRepository(
                client.app.state.settings.db_path
            ).list_for_task(task_id)
        )
        if record["kind"] == "strategy_sample_design_v2_json"
        and record["origin_tool"]
        == "strategy.materialize_sample_design_v2_native"
    )
    return {
        "artifact_id": bundle["id"],
        "artifact_content_hash": bundle["content_hash"],
        "sample_design_id": bundle["provenance"]["sample_design_id"],
        "sample_design_content_hash": bundle["provenance"][
            "sample_design_content_hash"
        ],
        "partition": "risk/development",
    }


def _register_workspace_sample(
    client: TestClient,
    task_id: str,
    tmp_path: Path,
    *,
    nan_label: bool = False,
    target_col: str | None = "bad",
    parallel_populations: bool = False,
    parallel_bad_pattern: list[object] | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    if parallel_populations:
        sample_roles = [
            role
            for role in ("dev", "valid", "oot")
            for _ in range(6)
        ]
        population_pattern = [
            "approval_only_a",
            "approval_only_b",
            "risk_only",
            "both_a",
            "both_b",
            "both_c",
        ]
        population = population_pattern * 3
        approval_flag = [1, 1, 0, 1, 1, 1] * 3
        risk_flag = [0, 0, 1, 1, 1, 1] * 3
        if parallel_bad_pattern is not None:
            assert len(parallel_bad_pattern) == 6
        bad: list[object] = list(
            (parallel_bad_pattern or [0, 1, 0, 0, 1, 1]) * 3
        )
        apply_dates = [
            f"2026-{month:02d}-{day:02d}"
            for month in (1, 2, 3)
            for day in range(1, 7)
        ]
        apply_months = [
            f"2026{month:02d}"
            for month in (1, 2, 3)
            for _ in range(6)
        ]
        legacy_score = [
            float(month * 100 + offset)
            for month in range(3)
            for offset in (10, 20, 100, 200, 300, 400)
        ]
        age = [
            float(20 + month * 10 + offset)
            for month in range(3)
            for offset in range(6)
        ]
        customer_ids = [
            f"customer-{index:02d}" for index in range(len(sample_roles))
        ]
        loan_amount = [
            float(100 + index * 10) for index in range(len(sample_roles))
        ]
        overdue_amount = [
            float((index % 4) * 5) for index in range(len(sample_roles))
        ]
        ead = [float(800 + index * 25) for index in range(len(sample_roles))]
        pd_values = [
            float(0.02 + (index % 6) * 0.03)
            for index in range(len(sample_roles))
        ]
    else:
        sample_roles = ["dev", "dev", "valid", "valid", "oot", "oot"]
        population = None
        approval_flag = None
        risk_flag = None
        bad = [0, 1, 0, 1, 0, 1]
        apply_dates = [
            "2026-01-01",
            "2026-01-10",
            "2026-02-01",
            "2026-02-10",
            "2026-03-01",
            "2026-03-10",
        ]
        apply_months = [
            "202601",
            "202601",
            "202602",
            "202602",
            "202603",
            "202603",
        ]
        legacy_score = [100.0, 200.0, 120.0, 220.0, 140.0, 240.0]
        age = [21.0, 35.0, 23.0, 37.0, 25.0, 39.0]
        customer_ids = ["a", "b", "c", "d", "e", "f"]
        loan_amount = [100.0, 200.0, 150.0, 180.0, 300.0, 250.0]
        overdue_amount = [0.0, 20.0, 0.0, 10.0, 0.0, 30.0]
        ead = [800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0]
        pd_values = [0.02, 0.08, 0.04, 0.10, 0.06, 0.12]
    if nan_label:
        bad[1] = None
    row_count = len(sample_roles)
    frame_data = {
        "sample_role": sample_roles,
        "customer_id": customer_ids,
        "apply_date": apply_dates,
        "apply_month": apply_months,
        "legacy_score": legacy_score,
        "age": age,
        "weight": [1.0] * row_count,
        "loan_amount": loan_amount,
        "overdue_amount": overdue_amount,
        "ead": ead,
        "pd": pd_values,
        "bad": bad,
    }
    if parallel_populations:
        frame_data.update(
            {
                "population": population,
                "approval_flag": approval_flag,
                "risk_flag": risk_flag,
            }
        )
    frame = pd.DataFrame(frame_data)
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
            "age": "feature",
            "weight": "weight",
            "loan_amount": "loan_amount",
            "overdue_amount": "overdue_amount",
            "ead": "feature",
            "pd": "feature",
            **(
                {
                    "population": "segment",
                    "approval_flag": "segment",
                    "risk_flag": "segment",
                }
                if parallel_populations
                else {}
            ),
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


def test_parallel_sample_v2_manual_and_natural_requests_share_native_tool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    dataset, workspace, mapping = _register_workspace_sample(
        client,
        task_id,
        tmp_path,
    )
    inputs = _sample_v2_inputs(relationship="parallel_time_cohorts")
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": inputs,
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    natural = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": _sample_v2_utterance(
                relationship="parallel_time_cohorts"
            )
        },
    )
    manual = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "人工界面原生固化平行时间 cohort 样本",
            "strategy_request": {
                "request_kind": "standard_workflow",
                "workflow": "strategy_sample_design_v2",
                "workflow_inputs": inputs,
            },
        },
    )

    assert natural.status_code == 202, natural.text
    assert manual.status_code == 202, manual.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        "strategy_sample_design_v2_native",
    ], json.dumps(
        {"natural": natural.json(), "manual": manual.json()},
        ensure_ascii=False,
    )
    assert all(plan["status"] == "done" for plan in plans)
    natural_plan = client.app.state.plan_repo.load_plan(plans[0]["id"])
    manual_plan = client.app.state.plan_repo.load_plan(plans[1]["id"])
    assert len(natural_plan.steps) == len(manual_plan.steps) == 1
    natural_step = natural_plan.steps[0]
    manual_step = manual_plan.steps[0]
    assert natural_step.tool_ref == manual_step.tool_ref
    assert natural_step.inputs == manual_step.inputs
    assert natural_step.inputs["source_mode"] == "native_active_dataset"
    assert natural_step.inputs["dataset_id"] == dataset.id
    assert (
        natural_step.inputs["expected_dataset_content_hash"]
        == dataset.content_hash
    )
    assert natural_step.inputs["workspace_revision"] == workspace.revision
    assert (
        natural_step.inputs["semantic_mapping_hash"]
        == data_semantic_mapping_hash(mapping)
    )
    assert natural_step.inputs["relationship"] == "parallel_time_cohorts"
    natural_output = client.app.state.plan_repo.load_step_output(natural_step.id)
    manual_output = client.app.state.plan_repo.load_step_output(manual_step.id)
    assert natural_output["content_hash"] == manual_output["content_hash"]
    assert natural_output["source_binding"]["source_mode"] == "native_active_dataset"
    assert (
        natural_output["source_binding"]["development_partition"]
        == "risk/development"
    )
    assert len(llm.calls) == 1


@pytest.mark.parametrize("request_source", ["natural_language", "manual_ui"])
def test_native_parallel_sample_drives_univariate_on_exact_risk_development_rows(
    tmp_path: Path,
    monkeypatch,
    request_source: str,
) -> None:
    client = TestClient(create_app(tmp_path / request_source / "workspace"))
    task_id = _create_strategy_task(client, tmp_path / request_source)
    _register_workspace_sample(
        client,
        task_id,
        tmp_path / request_source,
        parallel_populations=True,
    )
    sample_request = {
        "request_kind": "standard_workflow",
        "workflow": "strategy_sample_design_v2",
        "workflow_inputs": _parallel_population_sample_v2_inputs(),
    }
    univariate_request = _univariate_payload()
    llm = (
        _SequencedStrategyLLM(sample_request, univariate_request)
        if request_source == "natural_language"
        else _SequencedStrategyLLM()
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    sample_body = {"content": _parallel_population_sample_v2_utterance()}
    candidate_body = {
        "content": (
            "对 legacy_score 用 equal_width 做单变量分析，目标箱数 3，"
            "最小箱占比 2%，放款金额列 loan_amount，"
            "逾期金额列 overdue_amount，不设置哨兵值"
        )
    }
    if request_source == "manual_ui":
        sample_body["strategy_request"] = sample_request
        candidate_body["strategy_request"] = univariate_request

    sample_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=sample_body,
    )
    candidate_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=candidate_body,
    )

    assert sample_response.status_code == 202, sample_response.text
    assert candidate_response.status_code == 202, candidate_response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        "strategy_univariate_candidate_analysis",
    ], json.dumps(candidate_response.json()["messages"][-1], ensure_ascii=False)
    assert all(plan["status"] == "done" for plan in plans)

    sample_plan = client.app.state.plan_repo.load_plan(plans[0]["id"])
    candidate_plan = client.app.state.plan_repo.load_plan(plans[1]["id"])
    sample_output = client.app.state.plan_repo.load_step_output(
        sample_plan.steps[0].id
    )
    assert sample_output["membership"]["counts"]["approval"]["development"] == 5
    assert sample_output["membership"]["counts"]["risk"]["development"] == 4

    native_bundle = next(
        record
        for record in TaskArtifactRepository(
            client.app.state.settings.db_path
        ).list_for_task(task_id)
        if record["kind"] == "strategy_sample_design_v2_json"
        and record["origin_tool"]
        == "strategy.materialize_sample_design_v2_native"
    )
    expected_ref = {
        "artifact_id": native_bundle["id"],
        "artifact_content_hash": native_bundle["content_hash"],
        "sample_design_id": native_bundle["provenance"]["sample_design_id"],
        "sample_design_content_hash": native_bundle["provenance"][
            "sample_design_content_hash"
        ],
        "partition": "risk/development",
    }
    assert candidate_plan.steps[0].inputs["sample_design_ref"] == expected_ref
    assert set(candidate_plan.steps[0].inputs["sample_design_ref"]) == {
        "artifact_id",
        "artifact_content_hash",
        "sample_design_id",
        "sample_design_content_hash",
        "partition",
    }
    candidate_output = client.app.state.plan_repo.load_step_output(
        candidate_plan.steps[0].id
    )
    evidence = candidate_output["candidate_evidence"]
    assert evidence["generation"]["parameters"]["sample_design_ref"] == expected_ref
    assert evidence["analysis"]["row_count"] == 4
    assert len(llm.calls) == (2 if request_source == "natural_language" else 0)


def test_manual_candidate_inherits_native_sample_drop_nan_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(
        client,
        task_id,
        tmp_path,
        parallel_populations=True,
    )
    sample_inputs = _parallel_population_sample_v2_inputs()
    sample_inputs["drop_nan_labels"] = True
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: _SequencedStrategyLLM(),
    )

    sample_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": _parallel_population_sample_v2_utterance().replace(
                "不丢弃缺失标签",
                "丢弃缺失标签",
            ),
            "strategy_request": {
                "request_kind": "standard_workflow",
                "workflow": "strategy_sample_design_v2",
                "workflow_inputs": sample_inputs,
            },
        },
    )
    candidate_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "对 legacy_score 用 equal_width 做单变量分析，目标箱数 3，"
                "最小箱占比 2%，放款金额列 loan_amount，"
                "逾期金额列 overdue_amount，不设置哨兵值"
            ),
            "strategy_request": _univariate_payload(),
        },
    )

    assert sample_response.status_code == 202, sample_response.text
    assert candidate_response.status_code == 202, candidate_response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        "strategy_univariate_candidate_analysis",
    ], json.dumps(candidate_response.json()["messages"][-1], ensure_ascii=False)
    candidate_plan = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert candidate_plan.steps[0].inputs["drop_nan_labels"] is True
    assert candidate_plan.status.value == "done"


def test_manual_adoption_inherits_native_sample_drop_nan_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(
        client,
        task_id,
        tmp_path,
        parallel_populations=True,
        nan_label=True,
    )
    sample_inputs = _parallel_population_sample_v2_inputs()
    sample_inputs["drop_nan_labels"] = True
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: _SequencedStrategyLLM(),
    )

    sample_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": _parallel_population_sample_v2_utterance().replace(
                "不丢弃缺失标签",
                "丢弃缺失标签",
            ),
            "strategy_request": {
                "request_kind": "standard_workflow",
                "workflow": "strategy_sample_design_v2",
                "workflow_inputs": sample_inputs,
            },
        },
    )
    strategy_id = _create_stored_approval_strategy(
        client,
        task_id,
        threshold=250.0,
    )
    adoption_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "提交本地采纳复核",
            "strategy_request": {
                "request_kind": "strategy_lifecycle",
                "operation": "adopt",
                "strategy_type": "approval",
                "strategy_id": strategy_id,
                "adoption_reason": "已复核独立验证、影响测算和报告证据，同意本地采纳",
            },
        },
    )

    assert sample_response.status_code == 202, sample_response.text
    assert adoption_response.status_code == 202, adoption_response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        "stored_strategy_adoption",
    ], json.dumps(adoption_response.json()["messages"][-1], ensure_ascii=False)
    adoption_plan = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert adoption_plan.status.value == "awaiting_confirm"
    assert adoption_plan.steps[0].status.value == "done"
    assert adoption_plan.steps[0].inputs["drop_nan_labels"] is True
    assert (
        adoption_plan.steps[0].inputs["sample_design_ref"]["partition"]
        == "risk/development"
    )
    assert adoption_plan.steps[1].status.value == "awaiting_confirm"


@pytest.mark.parametrize("request_source", ["natural_language", "manual_ui"])
def test_native_parallel_sample_drives_automatic_tree_on_risk_development(
    tmp_path: Path,
    monkeypatch,
    request_source: str,
) -> None:
    scoped = tmp_path / request_source
    client = TestClient(create_app(scoped / "workspace"))
    task_id = _create_strategy_task(client, scoped)
    _register_workspace_sample(
        client,
        task_id,
        scoped,
        parallel_populations=True,
    )
    sample_request = {
        "request_kind": "standard_workflow",
        "workflow": "strategy_sample_design_v2",
        "workflow_inputs": _parallel_population_sample_v2_inputs(),
    }
    tree_request = _automatic_tree_payload()
    llm = (
        _SequencedStrategyLLM(sample_request, tree_request)
        if request_source == "natural_language"
        else _SequencedStrategyLLM()
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    sample_body = {"content": _parallel_population_sample_v2_utterance()}
    tree_body = {
        "content": (
            "用 legacy_score 在原生风险开发样本上构建自动决策树候选，"
            "最大深度 2，最小叶节点 2，放款金额列 loan_amount，"
            "逾期金额列 overdue_amount"
        )
    }
    if request_source == "manual_ui":
        sample_body["strategy_request"] = sample_request
        tree_body["strategy_request"] = tree_request
    sample_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=sample_body,
    )
    tree_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=tree_body,
    )

    assert sample_response.status_code == 202, sample_response.text
    assert tree_response.status_code == 202, tree_response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        "strategy_automatic_tree_candidate_build",
    ]
    assert all(plan["status"] == "done" for plan in plans)
    sample_plan = client.app.state.plan_repo.load_plan(plans[0]["id"])
    tree_plan = client.app.state.plan_repo.load_plan(plans[1]["id"])
    sample_output = client.app.state.plan_repo.load_step_output(
        sample_plan.steps[0].id
    )
    assert sample_output["membership"]["counts"]["approval"]["development"] == 5
    assert sample_output["membership"]["counts"]["risk"]["development"] == 4
    native_bundle = next(
        record
        for record in TaskArtifactRepository(
            client.app.state.settings.db_path
        ).list_for_task(task_id)
        if record["kind"] == "strategy_sample_design_v2_json"
        and record["origin_tool"]
        == "strategy.materialize_sample_design_v2_native"
    )
    expected_ref = {
        "artifact_id": native_bundle["id"],
        "artifact_content_hash": native_bundle["content_hash"],
        "sample_design_id": native_bundle["provenance"]["sample_design_id"],
        "sample_design_content_hash": native_bundle["provenance"][
            "sample_design_content_hash"
        ],
        "partition": "risk/development",
    }
    assert tree_plan.steps[0].inputs["sample_design_ref"] == expected_ref
    tree_output = client.app.state.plan_repo.load_step_output(
        tree_plan.steps[0].id
    )
    assert tree_output["summary"]["sample_design_ref"] == expected_ref
    assert tree_output["summary"]["training_row_count"] == 4
    assert len(llm.calls) == (2 if request_source == "natural_language" else 0)


@pytest.mark.parametrize(
    ("workflow_request", "utterance", "template_id"),
    [
        (
            _refinement_payload(),
            (
                "对 legacy_score 做等距 3 箱并保留观测坏率大于等于 50% "
                "的候选箱，最小箱占比 2%，放款金额列 loan_amount，"
                "逾期金额列 overdue_amount"
            ),
            "strategy_univariate_candidate_refinement",
        ),
        (
            _cross_matrix_payload(),
            (
                "构建 legacy_score 等距 3 箱乘以 age 等距 3 箱的二维"
                "交叉矩阵，最小箱占比 2%，放款金额列 loan_amount，"
                "逾期金额列 overdue_amount"
            ),
            "strategy_cross_matrix_analysis",
        ),
    ],
)
def test_native_parallel_sample_drives_composed_candidate_workflows(
    tmp_path: Path,
    monkeypatch,
    workflow_request: dict,
    utterance: str,
    template_id: str,
) -> None:
    scoped = tmp_path / template_id
    client = TestClient(create_app(scoped / "workspace"))
    task_id = _create_strategy_task(client, scoped)
    _register_workspace_sample(
        client,
        task_id,
        scoped,
        parallel_populations=True,
    )
    sample_request = {
        "request_kind": "standard_workflow",
        "workflow": "strategy_sample_design_v2",
        "workflow_inputs": _parallel_population_sample_v2_inputs(),
    }
    llm = _SequencedStrategyLLM(sample_request, workflow_request)
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    sample_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _parallel_population_sample_v2_utterance()},
    )
    workflow_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": utterance},
    )

    assert sample_response.status_code == 202, sample_response.text
    assert workflow_response.status_code == 202, workflow_response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        template_id,
    ], json.dumps(workflow_response.json()["messages"][-1], ensure_ascii=False)
    assert all(plan["status"] == "done" for plan in plans)
    sample_plan = client.app.state.plan_repo.load_plan(plans[0]["id"])
    candidate_plan = client.app.state.plan_repo.load_plan(plans[1]["id"])
    sample_output = client.app.state.plan_repo.load_step_output(
        sample_plan.steps[0].id
    )
    assert sample_output["membership"]["counts"]["approval"]["development"] == 5
    assert sample_output["membership"]["counts"]["risk"]["development"] == 4
    first_output = client.app.state.plan_repo.load_step_output(
        candidate_plan.steps[0].id
    )
    exact_ref = candidate_plan.steps[0].inputs["sample_design_ref"]
    assert exact_ref["partition"] == "risk/development"
    assert first_output["candidate_evidence"]["analysis"]["row_count"] == 4
    assert (
        first_output["candidate_evidence"]["generation"]["parameters"][
            "sample_design_ref"
        ]
        == exact_ref
    )
    second_output = client.app.state.plan_repo.load_step_output(
        candidate_plan.steps[1].id
    )
    assert second_output["parent_candidate_id"] == first_output["candidate_id"]
    assert second_output["parent_evidence_hash"] == first_output["evidence_hash"]
    if template_id == "strategy_univariate_candidate_refinement":
        assert second_output["candidate_asset"]["rule"]["condition"]["op"] in {
            "compare",
            "between",
            "or",
        }
        assert second_output["effect"]["selected_count"] > 0
    else:
        measurement = second_output["cross_matrix_candidate"]["measurement"]
        assert measurement["population_count"] == 4
        assert sum(cell["count"] for cell in measurement["cells"]) == 4
    assert len(llm.calls) == 2


@pytest.mark.parametrize("strategy_type", ["limit", "pricing", "segmentation"])
def test_native_parallel_sample_drives_deterministic_candidate_agent_workflow(
    tmp_path: Path,
    monkeypatch,
    strategy_type: str,
) -> None:
    scoped = tmp_path / strategy_type
    client = TestClient(create_app(scoped / "workspace"))
    task_id = _create_strategy_task(client, scoped)
    _register_workspace_sample(
        client,
        task_id,
        scoped,
        parallel_populations=True,
    )
    exact_ref = _materialize_native_parallel_sample_from_agent(
        client,
        task_id,
        monkeypatch,
    )
    llm = _SequencedStrategyLLM(_deterministic_candidate_payload(strategy_type))
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": f"基于 legacy_score 开发{strategy_type}确定性候选策略"},
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        "deterministic_strategy_candidate_development",
    ], json.dumps(response.json()["messages"][-1], ensure_ascii=False)
    assert plans[-1]["status"] == "awaiting_confirm"
    plan = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert plan.steps[0].inputs["sample_design_ref"] == exact_ref
    assert plan.steps[2].inputs["sample_design_ref"] == exact_ref
    design_inputs = plan.steps[0].inputs["candidate_design"]
    grid_name = {
        "limit": "limit_grid",
        "pricing": "rate_grid",
        "segmentation": None,
    }[strategy_type]
    if grid_name is not None:
        assert isinstance(design_inputs[grid_name], list)
    design = client.app.state.plan_repo.load_step_output(plan.steps[0].id)
    assert design["design_evidence"]["development_population_count"] == 4
    assert design["sample_design_ref"] == exact_ref
    backtest = client.app.state.plan_repo.load_step_output(plan.steps[2].id)
    assert backtest["population_count"] == 4
    assert len(llm.calls) == 1


def test_native_parallel_sample_drives_rule_mining_and_evaluation_agent_workflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(
        client,
        task_id,
        tmp_path,
        parallel_populations=True,
    )
    exact_ref = _materialize_native_parallel_sample_from_agent(
        client,
        task_id,
        monkeypatch,
    )
    llm = _SequencedStrategyLLM(
        {
            "operation": "mine_rules",
            "strategy_type": "reject",
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "挖掘并评估当前样本的审批拒绝规则"},
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        "rule_strategy",
    ], json.dumps(response.json()["messages"][-1], ensure_ascii=False)
    assert plans[-1]["status"] == "awaiting_confirm"
    plan = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert plan.steps[0].inputs["sample_design_ref"] == exact_ref
    assert plan.steps[2].inputs["sample_design_ref"] == exact_ref
    assert plan.steps[4].inputs["sample_design_ref"] == exact_ref
    mined = client.app.state.plan_repo.load_step_output(plan.steps[0].id)
    evaluated = client.app.state.plan_repo.load_step_output(plan.steps[2].id)
    backtest = client.app.state.plan_repo.load_step_output(plan.steps[4].id)
    assert mined["sample_design_ref"] == exact_ref
    assert mined["n_rows"] == 4
    assert evaluated["sample_design_ref"] == exact_ref
    assert backtest["population_count"] == 4
    assert len(llm.calls) == 1


def test_native_parallel_sample_drives_tradeoff_cutoff_backtest_and_compare(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(
        client,
        task_id,
        tmp_path,
        parallel_populations=True,
        # The development workflow's default direction is higher_is_better.
        # Keep this binding regression semantically aligned so the deterministic
        # max-approval design yields a real cutoff instead of the valid
        # approve-all result (whose empty-rule representation is a separate
        # legacy build_strategy contract gap).
        parallel_bad_pattern=[0, 1, 1, 1, 0, 0],
    )
    exact_ref = _materialize_native_parallel_sample_from_agent(
        client,
        task_id,
        monkeypatch,
    )
    baseline_strategy_id = _create_stored_approval_strategy(
        client,
        task_id,
        threshold=350.0,
    )
    llm = _SequencedStrategyLLM(
        {
            "operation": "develop",
            "strategy_type": "approval",
            "objective": "max_approval",
            "max_bad_rate": 0.25,
            "min_approval_rate": 0.25,
            "baseline_strategy_id": baseline_strategy_id,
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "扫描权衡并设计审批 cutoff，回测后与基线策略比较"},
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        "strategy_development",
    ], json.dumps(response.json()["messages"][-1], ensure_ascii=False)
    assert plans[-1]["status"] == "awaiting_confirm"
    plan = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    for step_index in (0, 1, 3, 4):
        assert plan.steps[step_index].inputs["sample_design_ref"] == exact_ref
    tradeoff = client.app.state.plan_repo.load_step_output(plan.steps[0].id)
    bands = client.app.state.plan_repo.load_step_output(plan.steps[1].id)
    backtest = client.app.state.plan_repo.load_step_output(plan.steps[3].id)
    comparison = client.app.state.plan_repo.load_step_output(plan.steps[4].id)
    assert tradeoff["sample_design_ref"] == exact_ref
    assert bands["sample_design_ref"] == exact_ref
    assert backtest["population_count"] == 4
    assert comparison["status"] == "compared"
    assert len(llm.calls) == 1


def test_native_parallel_sample_drives_limit_pricing_matrix_agent_workflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(
        client,
        task_id,
        tmp_path,
        parallel_populations=True,
    )
    exact_ref = _materialize_native_parallel_sample_from_agent(
        client,
        task_id,
        monkeypatch,
    )
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "limit_pricing_matrix",
            "workflow_inputs": {
                "score_col": "legacy_score",
                "pd_col": "pd",
                "n_bands": 2,
                "limit_grid": [1000, 2000],
                "rate_grid": [0.12, 0.18],
                "lgd": 0.50,
                "funding_rate": 0.04,
                "term_months": 12,
                "cost_per_loan": 10,
                "el_ead_max": 0.20,
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "按 legacy_score 和 pd 测算额度利率定价矩阵"},
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        "strategy_limit_pricing_analysis",
    ], json.dumps(response.json()["messages"][-1], ensure_ascii=False)
    assert plans[-1]["status"] == "done"
    plan = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert plan.steps[0].inputs["sample_design_ref"] == exact_ref
    assert plan.steps[1].inputs["sample_design_ref"] == exact_ref
    output = client.app.state.plan_repo.load_step_output(plan.steps[0].id)
    assert output["source_evidence"]["sample_design_partition"] == "risk/development"
    assert output["sample_design_ref"] == exact_ref
    band_counts = {
        cell["band"]: cell["count"]
        for cell in output["matrix"]
    }
    assert sum(band_counts.values()) == 4
    assert len(llm.calls) == 1


def test_native_parallel_sample_drives_stored_strategy_backtest_and_compare(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(
        client,
        task_id,
        tmp_path,
        parallel_populations=True,
    )
    exact_ref = _materialize_native_parallel_sample_from_agent(
        client,
        task_id,
        monkeypatch,
    )
    challenger_id = _create_stored_approval_strategy(
        client,
        task_id,
        threshold=250.0,
    )
    baseline_id = _create_stored_approval_strategy(
        client,
        task_id,
        threshold=350.0,
    )
    llm = _SequencedStrategyLLM(
        {
            "operation": "compare",
            "strategy_type": "approval",
            "strategy_id": challenger_id,
            "baseline_strategy_id": baseline_id,
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "回测并比较当前审批候选策略与基线策略"},
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        "stored_strategy_evaluation",
    ], json.dumps(response.json()["messages"][-1], ensure_ascii=False)
    assert plans[-1]["status"] == "done"
    plan = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert plan.steps[0].inputs["sample_design_ref"] == exact_ref
    backtest = client.app.state.plan_repo.load_step_output(plan.steps[0].id)
    assert backtest["population_count"] == 4
    assert backtest["normalized_input"]["sample_design_ref"] == exact_ref
    assert sum(row["count"] for row in backtest["transitions"]) == 4
    assert len(llm.calls) == 1


def test_native_parallel_sample_drives_stored_adoption_on_risk_development(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(
        client,
        task_id,
        tmp_path,
        parallel_populations=True,
    )
    _materialize_native_parallel_sample_from_agent(
        client,
        task_id,
        monkeypatch,
    )
    strategy_id = _create_stored_approval_strategy(
        client,
        task_id,
        threshold=250.0,
    )
    llm = _SequencedStrategyLLM(
        {
            "operation": "adopt",
            "strategy_type": "approval",
            "strategy_id": strategy_id,
            "adoption_reason": "仅用于验证原生样本不会绕过既有采纳边界",
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "采纳这个已有审批策略"},
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        "stored_strategy_adoption",
    ]
    adoption = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert adoption.status.value == "awaiting_confirm"
    assert adoption.steps[0].status.value == "done"
    assert (
        adoption.steps[0].inputs["sample_design_ref"]["partition"]
        == "risk/development"
    )
    assert adoption.steps[1].status.value == "awaiting_confirm"
    assert len(llm.calls) == 1


def test_native_parallel_candidate_natural_language_pool_add_compile_and_impact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(
        client,
        task_id,
        tmp_path,
        parallel_populations=True,
    )
    sample_request = {
        "request_kind": "standard_workflow",
        "workflow": "strategy_sample_design_v2",
        "workflow_inputs": _parallel_population_sample_v2_inputs(),
    }
    refinement_request = _refinement_payload()
    initial_llm = _SequencedStrategyLLM(
        sample_request,
        refinement_request,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: initial_llm,
    )

    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _parallel_population_sample_v2_utterance()},
    ).status_code == 202
    refined = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "对 legacy_score 做等距 3 箱并保留观测坏率大于等于 50% "
                "的候选箱，最小箱占比 2%，放款金额列 loan_amount，"
                "逾期金额列 overdue_amount"
            )
        },
    )
    assert refined.status_code == 202, refined.text
    refinement_plan = client.app.state.plan_repo.list_plans_for_task(task_id)[-1]
    asset_output = client.app.state.plan_repo.load_step_output(
        refinement_plan.steps[-1].id
    )
    asset_id = asset_output["asset_id"]

    add_llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_add_candidate",
            "workflow_inputs": {
                "candidate_asset_id": asset_id,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: add_llm,
    )
    added = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"把 {asset_id} 加入审批 Strategy Pool；默认动作 approval，"
                "命中动作 reject"
            )
        },
    )
    assert added.status_code == 202, added.text

    compile_llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_compile",
            "workflow_inputs": {"strategy_type": "approval"},
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: compile_llm,
    )
    compiled = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "只预览并编译审批 Strategy Pool 草案，不要采纳或部署"},
    )
    assert compiled.status_code == 202, compiled.text

    impact_llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_impact",
            "workflow_inputs": {
                "strategy_type": "approval",
                "comparison_mode": "absolute",
                "month_col": "apply_month",
                "loan_amount_col": "loan_amount",
                "overdue_amount_col": "overdue_amount",
                "drop_nan_labels": False,
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: impact_llm,
    )
    measured = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "计算审批策略池的通过率和坏账率；月份列 apply_month，"
                "放款金额列 loan_amount，逾期金额列 overdue_amount"
            )
        },
    )
    assert measured.status_code == 202, measured.text

    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        "strategy_univariate_candidate_refinement",
        "strategy_pool_add_candidate",
        "strategy_pool_compile",
        "strategy_pool_impact",
    ], json.dumps(measured.json()["messages"][-1], ensure_ascii=False)
    assert all(plan["status"] == "done" for plan in plans)
    pool = StrategyCandidatePoolRepository(
        client.app.state.settings.db_path
    ).get_current(task_id, "approval")
    assert pool is not None
    assert [entry["source"]["asset_id"] for entry in pool["entries"]] == [
        asset_id
    ]
    runtime = strategy_tools._runtime(
        ToolContext(
            task_id=task_id,
            seed=0,
            datasets_root=client.app.state.settings.datasets_dir,
            workspace=client.app.state.settings.workspace,
        )
    )
    binding = pool_tools.load_current_strategy_candidate_pool_artifact(
        runtime,
        task_id=task_id,
        strategy_type="approval",
    )
    development = pool_tools.bind_strategy_pool_development_execution(
        runtime,
        binding,
    )
    assert development.sample_design.source_mode == "native_active_dataset"
    assert development.sample_design.reference.partition == "risk/development"
    assert development.sample_design.development_population_count == 4
    compile_plan = client.app.state.plan_repo.load_plan(plans[-2]["id"])
    compile_output = client.app.state.plan_repo.load_step_output(
        compile_plan.steps[0].id
    )
    assert compile_output["strategy_spec"]["rules"]
    impact_plan = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert (
        impact_plan.steps[0].inputs["sample_design_ref"]["partition"]
        == "risk/development"
    )
    impact_output = client.app.state.plan_repo.load_step_output(
        impact_plan.steps[0].id
    )
    assert impact_output["population_count"] == 4
    assert len(initial_llm.calls) == 2
    assert len(add_llm.calls) == len(compile_llm.calls) == 1
    assert len(impact_llm.calls) == 1


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
    assert len(step.inputs["expected_registry_token"]) == 64
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


def test_native_parallel_sample_drives_model_evidence_on_exact_risk_development(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    _register_workspace_sample(
        client,
        task_id,
        tmp_path,
        parallel_populations=True,
    )
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _parallel_population_sample_v2_inputs(),
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
        json={"content": _parallel_population_sample_v2_utterance()},
    )
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
    evidence_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "生成 Strategy ModelEvidence V2，汇总当前已认证单变量候选证据"},
    )

    assert sample_response.status_code == 202, sample_response.text
    assert candidate_response.status_code == 202, candidate_response.text
    assert evidence_response.status_code == 202, evidence_response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2_native",
        "strategy_univariate_candidate_analysis",
        "strategy_model_evidence_v2",
    ], json.dumps(evidence_response.json()["messages"][-1], ensure_ascii=False)
    assert all(plan["status"] == "done" for plan in plans)

    candidate_plan = client.app.state.plan_repo.load_plan(plans[1]["id"])
    evidence_plan = client.app.state.plan_repo.load_plan(plans[2]["id"])
    exact_ref = candidate_plan.steps[0].inputs["sample_design_ref"]
    assert exact_ref["partition"] == "risk/development"
    assert len(evidence_plan.steps[0].inputs["expected_registry_token"]) == 64
    output = client.app.state.plan_repo.load_step_output(
        evidence_plan.steps[0].id
    )
    assert {
        (item["sample_ref"]["population"], item["sample_ref"]["partition"])
        for item in output["bundle"]["univariate_evidence"]
    } == {("risk", "development")}
    assert {
        (
            item["analysis_ref"]["population"],
            item["analysis_ref"]["partition"],
        )
        for item in output["bundle"]["univariate_evidence"]
    } == {("risk", "development")}
    model_record = next(
        record
        for record in TaskArtifactRepository(
            client.app.state.settings.db_path
        ).list_for_task(task_id)
        if record["kind"] == "strategy_model_evidence_v2_json"
    )
    assert model_record["provenance"]["legacy_sample_design_ref"] == exact_ref
    assert len(llm.calls) == 3


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


def test_model_evidence_v2_never_falls_back_behind_latest_native_sample(
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

    native_path = (
        client.app.state.settings.tasks_dir
        / task_id
        / "latest-native-sample-bundle.json"
    )
    native_bytes = b"{}"
    native_path.write_bytes(native_bytes)
    TaskArtifactRepository(client.app.state.settings.db_path).register(
        task_id=task_id,
        kind="strategy_sample_design_v2_json",
        path=str(native_path),
        content_hash=hashlib.sha256(native_bytes).hexdigest(),
        origin_tool="strategy.materialize_sample_design_v2_native",
        provenance={},
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "生成 Strategy ModelEvidence V2，汇总当前已认证单变量候选证据"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["code"] == "strategy_model_evidence_v2_sample_invalid"
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2"
    ]


@pytest.mark.parametrize(
    "native_damage",
    [
        "none",
        "missing_provenance",
        "wrong_artifact_hash",
        "wrong_artifact_path",
    ],
)
def test_candidate_workflow_never_falls_back_to_older_v1_after_native_sample(
    tmp_path: Path,
    monkeypatch,
    native_damage: str,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    dataset, workspace, mapping = _register_workspace_sample(
        client,
        task_id,
        tmp_path,
    )
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        },
        _univariate_payload(),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    ).status_code == 202

    native_path = (
        client.app.state.settings.tasks_dir
        / task_id
        / "newer-native-sample-bundle.json"
    )
    native_bytes = b"{}"
    native_path.write_bytes(native_bytes)
    registered_path = (
        native_path.with_name("missing-native-sample-bundle.json")
        if native_damage == "wrong_artifact_path"
        else native_path
    )
    registered_hash = (
        "0" * 64
        if native_damage == "wrong_artifact_hash"
        else hashlib.sha256(native_bytes).hexdigest()
    )
    provenance = {
        "task_id": task_id,
        "dataset_id": dataset.id,
        "dataset_content_hash": dataset.content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "target_col": "bad",
        "drop_nan_labels": False,
    }
    TaskArtifactRepository(client.app.state.settings.db_path).register(
        task_id=task_id,
        kind="strategy_sample_design_v2_json",
        path=str(registered_path),
        content_hash=registered_hash,
        origin_tool="strategy.materialize_sample_design_v2_native",
        provenance={} if native_damage == "missing_provenance" else provenance,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "对 legacy_score 用 equal_width 做单变量分析，目标箱数 3，"
                "最小箱占比 2%，放款金额列 loan_amount，"
                "逾期金额列 overdue_amount，不设置哨兵值"
            )
        },
    )

    assert response.status_code == 202, response.text
    assert (
        response.json()["code"]
        == "strategy_sample_design_v2_native_source_unsupported"
    )
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2"
    ]


def test_newer_matching_v1_recovers_from_older_damaged_native_sample(
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
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    assert client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    ).status_code == 202

    damaged_path = (
        client.app.state.settings.tasks_dir
        / task_id
        / "older-damaged-native-sample-bundle.json"
    )
    damaged_bytes = b"{}"
    damaged_path.write_bytes(damaged_bytes)
    TaskArtifactRepository(client.app.state.settings.db_path).register(
        task_id=task_id,
        kind="strategy_sample_design_v2_json",
        path=str(damaged_path),
        content_hash=hashlib.sha256(damaged_bytes).hexdigest(),
        origin_tool="strategy.materialize_sample_design_v2_native",
        provenance={},
        created_at="2020-01-01T00:00:00+00:00",
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "对 legacy_score 用 equal_width 做单变量分析，目标箱数 3，"
                "最小箱占比 2%，放款金额列 loan_amount，"
                "逾期金额列 overdue_amount，不设置哨兵值"
            )
        },
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2",
        "strategy_univariate_candidate_analysis",
    ]
    assert all(plan["status"] == "done" for plan in plans)


def test_candidate_ignores_newer_authenticated_native_from_other_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_strategy_task(client, tmp_path)
    first_dataset, first_workspace, mapping = _register_workspace_sample(
        client,
        task_id,
        tmp_path / "context-a",
    )
    llm = _SequencedStrategyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_v2_inputs(),
        },
        _univariate_payload(),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    native_user_inputs = _sample_v2_inputs(
        relationship="parallel_time_cohorts"
    )
    native_request = {
        "source_mode": "native_active_dataset",
        "dataset_id": first_dataset.id,
        "expected_dataset_content_hash": first_dataset.content_hash,
        "workspace_revision": first_workspace.revision,
        "workspace_generation": first_workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "target_col": "bad",
        "scope": "strategy_development",
        **native_user_inputs,
        "policy": {
            "minimum_partition_count": 1,
            "minimum_bad_count": 1,
            "minimum_label_coverage": 0.8,
            "minimum_historical_score_coverage": 0.8,
            "maximum_group_coverage_gap": 0.2,
            "diagnostic_severities": {
                "entity_overlap": "fail",
                "temporal_oot": "fail",
                "risk_outside_approval": "fail",
                "maturity": "fail",
                "label_coverage": "fail",
                "historical_score_coverage": "warn",
                "group_coverage_gap": "warn",
                "sufficiency": "fail",
            },
        },
    }
    original_now = task_artifact_repository._now
    monkeypatch.setattr(
        task_artifact_repository,
        "_now",
        lambda: "2999-01-01T00:00:00+00:00",
    )
    run_materialize_sample_design_v2_native(
        native_request,
        ToolContext(
            task_id=task_id,
            seed=0,
            datasets_root=client.app.state.settings.datasets_dir,
            workspace=client.app.state.settings.workspace,
        ),
        strategy_tools._runtime(
            ToolContext(
                task_id=task_id,
                seed=0,
                datasets_root=client.app.state.settings.datasets_dir,
                workspace=client.app.state.settings.workspace,
            )
        ),
    )
    monkeypatch.setattr(task_artifact_repository, "_now", original_now)

    settings = client.app.state.settings
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    second_frame = pd.read_parquet(
        registry.resolve_verified_path(first_dataset.id)
    )
    second_frame.loc[0, "legacy_score"] = 101.0
    second_source = tmp_path / "context-b.parquet"
    second_frame.to_parquet(second_source, index=False)
    second_dataset = registry.register_from_upload(
        task_id,
        second_source,
        role="strategy_sample",
    )
    assert second_dataset.id != first_dataset.id
    workspaces = DataWorkspaceRepository(settings.db_path)
    reset_workspace = workspaces.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=second_dataset.id,
            active_dataset_content_hash=second_dataset.content_hash,
        ),
        expected_revision=first_workspace.revision,
    )
    second_workspace = workspaces.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=second_dataset.id,
            active_dataset_content_hash=second_dataset.content_hash,
            semantic_mapping=mapping,
        ),
        expected_revision=reset_workspace.revision,
    )
    assert second_workspace.active_dataset_id == second_dataset.id

    legacy_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _sample_v2_utterance()},
    )
    assert legacy_response.status_code == 202, legacy_response.text

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
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design_v2",
        "strategy_univariate_candidate_analysis",
    ]
    assert all(plan["status"] == "done" for plan in plans)
    legacy_plan = client.app.state.plan_repo.load_plan(plans[0]["id"])
    candidate_plan = client.app.state.plan_repo.load_plan(plans[1]["id"])
    legacy_output = client.app.state.plan_repo.load_step_output(
        legacy_plan.steps[0].id
    )
    registered = TaskArtifactRepository(settings.db_path)
    native_bundle = next(
        item
        for item in registered.list_for_task(task_id)
        if item["kind"] == "strategy_sample_design_v2_json"
        and item["origin_tool"]
        == "strategy.materialize_sample_design_v2_native"
    )
    legacy_artifact = registered.get_for_task(
        task_id,
        legacy_output["artifact"]["artifact_id"],
    )
    assert legacy_artifact is not None
    assert native_bundle["created_at"] > legacy_artifact["created_at"]
    assert candidate_plan.steps[0].inputs["sample_design_ref"] == {
        "artifact_id": legacy_output["artifact"]["artifact_id"],
        "artifact_content_hash": legacy_output["artifact"]["content_hash"],
        "sample_design_id": legacy_output["sample_design_id"],
        "sample_design_content_hash": legacy_output["content_hash"],
        "partition": "development",
    }


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
