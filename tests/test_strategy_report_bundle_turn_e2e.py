"""Turn-boundary binding for natural-language StrategyReportBundle V2."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.turn_handlers import (
    _StrategyV2EvidenceSetupError,
    _strategy_report_bundle_v2_plan_slots,
    _strategy_report_current_pool_binding,
    _strategy_report_identity,
    _strategy_report_latest_pool_impact_binding,
    _strategy_report_latest_sample_binding,
    _strategy_report_optional_model_evidence,
    _strategy_report_optional_score_evidence,
    _strategy_report_read_runtime,
    _strategy_report_requested_pool_type,
)
from marvis.app import create_app
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.model_evidence_tools import (
    MODEL_EVIDENCE_V2_ARTIFACT_KIND,
)
from marvis.packs.strategy.report_bundle_tools import (
    run_build_strategy_report_bundle_v2,
    validate_build_strategy_report_bundle_v2_tool_output,
)
from marvis.plugins.manifest import ToolRef
from test_strategy_report_bundle_tools import _run, _setup


class _ReportLLM:
    def __init__(self, workflow_inputs: dict | None = None) -> None:
        self.workflow_inputs = workflow_inputs or {}
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_report_bundle_v2",
                "workflow_inputs": self.workflow_inputs,
            },
            ensure_ascii=False,
        )


def _draft(
    *,
    title: str = "策略迭代评审报告",
    status: str = "partial",
) -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="strategy_report_bundle_v2",
        workflow_inputs={"title": title, "status": status},
    )


def _runtime(fixture: dict) -> SimpleNamespace:
    return SimpleNamespace(settings=fixture["settings"])


def test_report_turn_binds_exact_current_sources_and_first_head(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)

    slots = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略迭代评审报告。"},
    )

    assert slots["title"] == "策略迭代评审报告"
    assert slots["status"] == "partial"
    for field in (
        "project_context_ref",
        "sample_design_ref",
        "candidate_pool_ref",
        "pool_impact_ref",
    ):
        assert slots[field] == fixture["request"][field]
    assert slots["report_revision"] == 1
    assert slots["previous_report_id"] is None
    assert slots["previous_report_content_hash"] is None
    assert slots["strategy_identity"] is None
    assert slots["model_evidence_ref"] is None
    assert slots["training_evidence_ref"] is None
    assert slots["score_evidence_ref"] is None
    generated_at = datetime.fromisoformat(slots["generated_at"])
    assert generated_at.utcoffset() == timedelta(0)


def test_report_turn_reloads_next_exact_report_head(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    first = _run(fixture)

    slots = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(status="final"),
        source_message={"content": "请再生成当前审批策略评审报告，状态 final。"},
    )

    assert slots["report_revision"] == 2
    assert slots["previous_report_id"] == first["report_id"]
    assert slots["previous_report_content_hash"] == first["content_hash"]


def test_report_planned_refs_do_not_rebind_after_head_drift(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    planned = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略评审报告。"},
    )
    frozen = deepcopy(planned)
    _run(fixture)

    assert planned == frozen
    with pytest.raises(StrategyError):
        run_build_strategy_report_bundle_v2(
            planned,
            fixture["ctx"],
            fixture["runtime"],
        )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("生成审批策略评审报告", "approval"),
        ("生成拒绝策略评审报告", "reject"),
        ("generate approval strategy review report", "approval"),
        ("generate reject strategy review report", "reject"),
    ],
)
def test_report_turn_uses_only_explicit_pool_type_selection(
    message: str,
    expected: str,
) -> None:
    assert _strategy_report_requested_pool_type({"content": message}) == expected


def test_report_turn_clarifies_when_both_nonempty_pool_types_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DualPoolRepository:
        def __init__(self, db_path: Path) -> None:
            pass

        def get_current(self, task_id: str, strategy_type: str) -> dict:
            return {"entries": [{"entry_id": strategy_type}]}

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        _DualPoolRepository,
    )

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_current_pool_binding(
            SimpleNamespace(settings=SimpleNamespace(db_path=tmp_path / "db.sqlite")),
            task_id="task-1",
            requested_type=None,
        )

    assert raised.value.code == "strategy_report_bundle_v2_pool_type_required"


def test_report_turn_clarifies_for_missing_context_sample_pool_or_impact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    runtime = _runtime(fixture)
    read_runtime = _strategy_report_read_runtime(runtime)
    artifacts = tuple(
        read_runtime.task_artifacts.list_for_task(fixture["task"].id)
    )

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_current_strategy_project_context_artifact",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(_StrategyV2EvidenceSetupError) as missing_context:
        _strategy_report_bundle_v2_plan_slots(
            runtime,
            fixture["task"],
            _draft(),
            source_message={"content": "生成审批策略评审报告"},
        )
    assert missing_context.value.code == (
        "strategy_report_bundle_v2_project_context_required"
    )

    with pytest.raises(_StrategyV2EvidenceSetupError) as missing_sample:
        _strategy_report_latest_sample_binding(
            read_runtime,
            task_id=fixture["task"].id,
            artifacts=(),
        )
    assert missing_sample.value.code == "strategy_report_bundle_v2_sample_required"

    class _EmptyPoolRepository:
        def __init__(self, db_path: Path) -> None:
            pass

        def get_current(self, task_id: str, strategy_type: str):
            return None

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        _EmptyPoolRepository,
    )
    with pytest.raises(_StrategyV2EvidenceSetupError) as missing_pool:
        _strategy_report_current_pool_binding(
            read_runtime,
            task_id=fixture["task"].id,
            requested_type=None,
        )
    assert missing_pool.value.code == "strategy_report_bundle_v2_pool_required"

    monkeypatch.undo()
    pool = _strategy_report_current_pool_binding(
        read_runtime,
        task_id=fixture["task"].id,
        requested_type="approval",
    )
    without_impacts = tuple(
        item
        for item in artifacts
        if item["kind"] != "strategy_pool_impact_json"
    )
    with pytest.raises(_StrategyV2EvidenceSetupError) as missing_impact:
        _strategy_report_latest_pool_impact_binding(
            read_runtime,
            task_id=fixture["task"].id,
            artifacts=without_impacts,
            pool=pool,
        )
    assert missing_impact.value.code == (
        "strategy_report_bundle_v2_pool_impact_required"
    )


def test_report_turn_latest_optional_evidence_is_compatible_or_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_ref = {
        "membership_artifact_id": "1" * 64,
        "expected_membership_artifact_content_hash": "2" * 64,
        "bundle_artifact_id": "3" * 64,
        "expected_bundle_artifact_content_hash": "4" * 64,
        "expected_bundle_id": "strategy-sample-design-bundle-" + "5" * 24,
        "expected_sample_design_id": "strategy-sample-design-" + "6" * 24,
        "expected_sample_design_content_hash": "7" * 64,
    }

    def _sample_binding(ref: dict) -> SimpleNamespace:
        return SimpleNamespace(
            membership_artifact_id=ref["membership_artifact_id"],
            membership_artifact_content_hash=ref[
                "expected_membership_artifact_content_hash"
            ],
            bundle_artifact_id=ref["bundle_artifact_id"],
            bundle_artifact_content_hash=ref[
                "expected_bundle_artifact_content_hash"
            ],
            bundle={
                "bundle_id": ref["expected_bundle_id"],
                "sample_design": {
                    "sample_design_id": ref["expected_sample_design_id"],
                    "content_hash": ref[
                        "expected_sample_design_content_hash"
                    ],
                },
            },
        )

    model_binding = SimpleNamespace(
        artifact_id="8" * 64,
        artifact_content_hash="9" * 64,
        bundle={"bundle_id": "model-bundle", "content_hash": "a" * 64},
        sample_design_binding=_sample_binding(sample_ref),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_strategy_model_evidence_v2_artifact",
        lambda *args, **kwargs: model_binding,
    )
    records = (
        {
            "kind": MODEL_EVIDENCE_V2_ARTIFACT_KIND,
            "id": model_binding.artifact_id,
            "content_hash": model_binding.artifact_content_hash,
            "provenance": {
                "bundle_id": "model-bundle",
                "bundle_content_hash": "a" * 64,
                "sample_design_ref": sample_ref,
            },
        },
    )

    _, reference = _strategy_report_optional_model_evidence(
        SimpleNamespace(),
        task_id="task-1",
        artifacts=records,
        sample_ref=sample_ref,
    )
    assert reference == {
        "artifact_id": "8" * 64,
        "expected_artifact_content_hash": "9" * 64,
        "expected_bundle_id": "model-bundle",
        "expected_bundle_content_hash": "a" * 64,
    }

    incompatible_ref = {**sample_ref, "expected_bundle_id": "other-bundle"}
    model_binding.sample_design_binding = _sample_binding(incompatible_ref)
    binding, reference = _strategy_report_optional_model_evidence(
        SimpleNamespace(),
        task_id="task-1",
        artifacts=records,
        sample_ref=sample_ref,
    )
    assert binding is None
    assert reference is None


def test_report_turn_compatible_score_can_supply_its_own_training_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_ref = {"expected_sample_design_id": "sample-1"}
    score_binding = SimpleNamespace(
        training=object(),
        evidence_record={"id": "a" * 64, "content_hash": "b" * 64},
        vector_record={"id": "c" * 64, "content_hash": "d" * 64},
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_model_score_evidence_artifacts",
        lambda *args, **kwargs: score_binding,
    )
    score_training_ref = {
        "sample_design_ref": sample_ref,
        "expected_experiment_id": "experiment-1",
    }
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.build_training_evidence_ref",
        lambda binding: score_training_ref,
    )
    records = (
        {
            "kind": "model_score_evidence_json",
            "id": "a" * 64,
            "content_hash": "b" * 64,
            "provenance": {
                "score_vector_artifact_id": "c" * 64,
                "score_vector_artifact_content_hash": "d" * 64,
            },
        },
    )

    binding, reference = _strategy_report_optional_score_evidence(
        SimpleNamespace(),
        task_id="task-1",
        artifacts=records,
        sample_ref=sample_ref,
        training_ref=None,
    )

    assert binding is score_binding
    assert reference == {
        "evidence_artifact_id": "a" * 64,
        "expected_evidence_artifact_content_hash": "b" * 64,
        "score_vector_artifact_id": "c" * 64,
        "expected_score_vector_artifact_content_hash": "d" * 64,
    }


def test_report_turn_corrupt_latest_same_kind_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        {
            "kind": MODEL_EVIDENCE_V2_ARTIFACT_KIND,
            "id": "1" * 64,
            "content_hash": "2" * 64,
            "provenance": {"older": "otherwise-valid"},
        },
        {
            "kind": MODEL_EVIDENCE_V2_ARTIFACT_KIND,
            "id": "3" * 64,
            "content_hash": "4" * 64,
            "provenance": None,
        },
    )

    def _must_not_fallback(*args, **kwargs):
        pytest.fail("corrupt newest same-kind evidence must stop selection")

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_strategy_model_evidence_v2_artifact",
        _must_not_fallback,
    )

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_optional_model_evidence(
            SimpleNamespace(),
            task_id="task-1",
            artifacts=records,
            sample_ref={},
        )

    assert raised.value.code == (
        "strategy_report_bundle_v2_optional_evidence_invalid"
    )


def test_report_turn_strategy_identity_requires_one_exact_same_type_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StrategyRepository:
        matches = [
            {
                "id": "strategy-1",
                "task_id": "task-1",
                "strategy_type": "approval",
                "version": 3,
            }
        ]

        def __init__(self, db_path: Path) -> None:
            pass

        def list_meta_for_task(self, task_id: str) -> list[dict]:
            return list(self.matches)

        def get_strategy(self, strategy_id: str):
            return SimpleNamespace(
                spec={"schema_version": "strategy.v2"},
                strategy_type="approval",
            )

        def get_strategy_spec_hash(self, strategy_id: str) -> str:
            return "a" * 64

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyRepository",
        _StrategyRepository,
    )
    runtime = SimpleNamespace(
        settings=SimpleNamespace(db_path=tmp_path / "db.sqlite")
    )

    assert _strategy_report_identity(
        runtime,
        task_id="task-1",
        strategy_type="approval",
    ) == {
        "strategy_id": "strategy-1",
        "strategy_version": "3",
        "strategy_type": "approval",
    }
    _StrategyRepository.matches.append(
        {
            "id": "strategy-2",
            "task_id": "task-1",
            "strategy_type": "approval",
            "version": 1,
        }
    )
    assert (
        _strategy_report_identity(
            runtime,
            task_id="task-1",
            strategy_type="approval",
        )
        is None
    )


def test_report_command_autostarts_exact_one_step_without_dataset_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    client = TestClient(create_app(fixture["settings"].workspace))
    llm = _ReportLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    def _unexpected_preview(*args, **kwargs):
        pytest.fail("report workflow must not require an active dataset preview")

    for name in (
        "_strategy_dataset_preview",
        "_strategy_sample_design_dataset_preview",
        "_strategy_pool_impact_dataset_preview",
    ):
        monkeypatch.setattr(f"marvis.agent.turn_handlers.{name}", _unexpected_preview)

    response = client.post(
        f"/api/tasks/{fixture['task'].id}/agent/messages",
        json={"content": "请生成当前审批策略迭代评审报告。"},
    )

    assert response.status_code == 202, response.text
    plans = client.get(
        f"/api/tasks/{fixture['task'].id}/plans"
    ).json()["plans"]
    assert len(plans) == 1, response.json()
    assert plans[0]["template_id"] == "strategy_report_bundle_v2"
    assert plans[0]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    assert len(stored.steps) == 1
    step = stored.steps[0]
    assert step.tool_ref == ToolRef("strategy", "build_report_bundle_v2")
    assert step.needs_confirmation is False
    assert step.inputs["title"] == "策略迭代评审报告"
    assert step.inputs["status"] == "partial"
    assert "previous_report_id" not in step.inputs
    assert "previous_report_content_hash" not in step.inputs
    assert step.inputs["project_context_ref"] == fixture["request"][
        "project_context_ref"
    ]
    output = client.app.state.plan_repo.load_step_output(step.id)
    validate_build_strategy_report_bundle_v2_tool_output(output)
    rendered_messages = "\n".join(
        str(message.get("content") or "")
        for message in response.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert output["report_id"] in rendered_messages
    assert "未创建策略、未采纳、未部署或上线" in rendered_messages
    assert rendered_messages.count("](/api/tasks/") == 3
    assert len(llm.calls) == 1
