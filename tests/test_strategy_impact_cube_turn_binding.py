"""Agent-side evidence binding for the unified Strategy ImpactCube."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import _strategy_impact_cube_plan_slots
import marvis.agent.turn_handlers as turn_handlers
from marvis.app import create_app
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.strategy.pool import compile_strategy_pool
from marvis.packs.strategy.strategy import build_strategy_from_spec
from tests.test_strategy_pool_validation_tools import _setup


def _runtime(fx: dict) -> SimpleNamespace:
    return SimpleNamespace(settings=fx["settings"])


class _PayloadLLM:
    def complete(self, **kwargs) -> str:
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_impact_cube",
                "workflow_inputs": {"strategy_type": "approval"},
            },
            ensure_ascii=False,
        )


def test_impact_cube_turn_binds_exact_latest_sample_pool_and_dimensions(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_impact_cube",
        workflow_inputs={"strategy_type": "approval"},
    )

    slots = _strategy_impact_cube_plan_slots(
        _runtime(fx),
        fx["task"],
        draft,
    )

    assert slots["strategy_type"] == "approval"
    assert slots["pool_ref"] == {
        "artifact_id": fx["pool_artifact"]["artifact_id"],
        "expected_artifact_content_hash": fx["pool_artifact"]["content_hash"],
        "expected_pool_id": fx["pool"]["pool_id"],
        "expected_revision": fx["pool"]["revision"],
        "expected_revision_id": fx["pool"]["pool"]["revision_id"],
        "expected_snapshot_hash": fx["pool"]["snapshot_hash"],
    }
    assert slots["sample_design_ref"] == fx["sample_ref"]
    assert slots["partitions"] == ["development", "validation", "oot"]
    assert slots["population"] == "risk"
    assert slots["dimension_bindings"] == {
        "month_col": "apply_month",
        "group_col": "channel",
        "segment_col": "sample_split",
    }
    assert slots["current_strategy_ref"] is None
    assert slots["economics_inputs"] is None


def test_impact_cube_turn_rejects_sample_changed_after_compiler_preview(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_impact_cube",
        workflow_inputs={"strategy_type": "approval"},
    )

    with pytest.raises(
        StrategySetupError,
        match="请求编译与计划创建之间发生变化",
    ):
        _strategy_impact_cube_plan_slots(
            _runtime(fx),
            fx["task"],
            draft,
            expected_sample_binding={
                "kind": "strategy_sample_design_v2",
                "sample_design_ref": {
                    **fx["sample_ref"],
                    "expected_sample_design_content_hash": "0" * 64,
                },
                "dataset_id": fx["dataset"].id,
                "dataset_content_hash": fx["dataset"].content_hash,
            },
        )


def test_impact_cube_turn_binds_explicit_partitions_economics_and_current_strategy(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    spec = compile_strategy_pool(fx["pool"]["pool"])["strategy_spec"]
    current = build_strategy_from_spec(spec)
    fx["runtime"].strategies.create_strategy(fx["task"].id, current)
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_impact_cube",
        workflow_inputs={
            "strategy_type": "approval",
            "partitions": ["oot", "development"],
            "month_col": "apply_month",
            "group_col": "channel",
            "segment_col": "sample_split",
            "current_strategy_id": current.id,
            "economics_inputs": {
                "ead": {"kind": "column", "column": "loan_amount"},
                "lgd": {"kind": "scalar", "value": 0.5},
            },
        },
    )

    slots = _strategy_impact_cube_plan_slots(
        _runtime(fx),
        fx["task"],
        draft,
    )

    assert slots["partitions"] == ["development", "oot"]
    assert slots["current_strategy_ref"] == {
        "strategy_id": current.id,
        "expected_strategy_spec_hash": strategy_spec_hash(current.spec),
    }
    assert slots["economics_inputs"] == {
        "ead": {"kind": "column", "column": "loan_amount"},
        "lgd": {"kind": "scalar", "value": 0.5},
    }


def test_impact_cube_turn_allows_pool_with_resolvable_score_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    original_load = (
        turn_handlers.load_current_strategy_candidate_pool_artifact
    )
    binding = original_load(
        fx["runtime"],
        task_id=fx["task"].id,
        strategy_type="approval",
        expected_pool_revision=fx["pool"]["revision"],
        expected_pool_snapshot_hash=fx["pool"]["snapshot_hash"],
    )
    requirement = {
        "rule_id": binding.pool["entries"][0]["rule_id"],
        "fragment_id": binding.pool["entries"][0]["source"]["fragment_id"],
        "requirement": {
            "type": "model_score_vector.v1",
            "virtual_field": "__marvis_model_pd_" + "1" * 16,
            "score_product": "raw_native_uncalibrated_bad_probability",
            "score_evidence_artifact_id": "2" * 64,
            "score_evidence_artifact_content_hash": "3" * 64,
            "score_vector_artifact_id": "1" * 64,
            "score_vector_artifact_content_hash": "4" * 64,
        },
    }
    controlled = replace(
        binding,
        compiled_design={
            **binding.compiled_design,
            "requirements": [requirement],
        },
    )
    monkeypatch.setattr(
        turn_handlers,
        "load_current_strategy_candidate_pool_artifact",
        lambda *args, **kwargs: controlled,
    )
    monkeypatch.setattr(
        turn_handlers,
        "resolve_pool_requirements",
        lambda *args, **kwargs: SimpleNamespace(
            requirements=tuple(controlled.compiled_design["requirements"])
        ),
    )
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_impact_cube",
        workflow_inputs={"strategy_type": "approval"},
    )

    slots = _strategy_impact_cube_plan_slots(
        _runtime(fx),
        fx["task"],
        draft,
    )

    assert slots["pool_ref"]["expected_pool_id"] == binding.pool["pool_id"]


def test_impact_cube_turn_rejects_sensitive_or_unknown_dimension(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    sensitive = StandardWorkflowRequestDraft(
        workflow="strategy_impact_cube",
        workflow_inputs={
            "strategy_type": "approval",
            "group_col": "customer_id",
        },
    )
    with pytest.raises(StrategySetupError, match="不能作为 ImpactCube 聚合维度"):
        _strategy_impact_cube_plan_slots(
            _runtime(fx),
            fx["task"],
            sensitive,
        )

    unknown = StandardWorkflowRequestDraft(
        workflow="strategy_impact_cube",
        workflow_inputs={
            "strategy_type": "approval",
            "group_col": "not_a_column",
        },
    )
    with pytest.raises(StrategySetupError, match="不在最新样本绑定的数据列"):
        _strategy_impact_cube_plan_slots(
            _runtime(fx),
            fx["task"],
            unknown,
        )


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_impact_cube_executes_exact_read_only_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    client = TestClient(create_app(fx["settings"].workspace))
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: _PayloadLLM(),
    )

    response = client.post(
        f"/api/tasks/{fx['task'].id}/agent/messages",
        json={"content": "请测算 approval 审批策略池的统一影响。"},
    )

    assert response.status_code == 202, response.text
    plans = client.get(
        f"/api/tasks/{fx['task'].id}/plans"
    ).json()["plans"]
    latest_message = response.json()["messages"][-1]
    assert plans, (
        latest_message["content"],
        latest_message.get("metadata"),
    )
    assert plans[-1]["template_id"] == "strategy_impact_cube"
    assert plans[-1]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert len(stored.steps) == 1
    step = stored.steps[0]
    assert step.inputs["pool_ref"]["expected_pool_id"] == fx["pool"]["pool_id"]
    assert step.inputs["sample_design_ref"] == fx["sample_ref"]
    output = client.app.state.plan_repo.load_step_output(step.id)
    assert output["strategy_type"] == "approval"
    assert output["partitions"] == ["development", "validation", "oot"]
    assert output["not_mutated_pool"] is True
    assert output["not_created_strategy"] is True
    assert output["not_adopted"] is True
    assert output["not_promoted"] is True
    assert output["not_deployed"] is True


@pytest.mark.slow
@pytest.mark.e2e
def test_manual_ui_impact_cube_uses_authenticated_sample_preview(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    client = TestClient(create_app(fx["settings"].workspace))

    response = client.post(
        f"/api/tasks/{fx['task'].id}/agent/messages",
        json={
            "content": "从 Candidate Lab 生成 approval Pool 的统一 ImpactCube",
            "strategy_request": {
                "request_kind": "standard_workflow",
                "workflow": "strategy_impact_cube",
                "workflow_inputs": {
                    "strategy_type": "approval",
                    "partitions": ["development", "validation", "oot"],
                    "month_col": "apply_month",
                    "group_col": "channel",
                },
            },
        },
    )

    assert response.status_code == 202, response.text
    plans = client.get(
        f"/api/tasks/{fx['task'].id}/plans"
    ).json()["plans"]
    assert plans, response.json()["messages"][-1]
    assert plans[-1]["template_id"] == "strategy_impact_cube"
    assert plans[-1]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert stored.steps[0].inputs["sample_design_ref"] == fx["sample_ref"]
    assert stored.steps[0].inputs["dimension_bindings"] == {
        "month_col": "apply_month",
        "group_col": "channel",
        "segment_col": "sample_split",
    }
