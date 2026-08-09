from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from marvis.agent.strategy_request_compiler import (
    compile_strategy_request,
    validate_strategy_request,
)
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent import turn_handlers
from marvis.agent.strategy_workflows import (
    MANUAL_STANDARD_STRATEGY_WORKFLOWS,
    StrategyWorkflowPreparationContext,
    StrategyWorkflowResolutionContext,
    prepare_strategy_plan,
    resolve_strategy_request,
)
from marvis.api_schemas import ManualStrategyRequest
from marvis.app import create_app
from marvis.db_schema import connect
from marvis.packs.modeling.score_evidence import (
    MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
)
from marvis.packs.modeling.score_evidence_tools import (
    MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
)
from marvis.packs.strategy.model_score_comparison_tools import (
    MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND,
    MODEL_SCORE_COMPARISON_V2_AUDIT_KIND,
)
from tests.test_strategy_model_score_comparison import _comparison_fixture


WORKFLOW = "strategy_model_score_comparison_v2"


def _request_inputs() -> dict[str, str]:
    return {"population": "risk", "partition": "development"}


def test_score_comparison_is_a_canonical_manual_workflow_with_business_inputs_only(
) -> None:
    assert WORKFLOW in MANUAL_STANDARD_STRATEGY_WORKFLOWS

    request = ManualStrategyRequest.model_validate(
        {
            "request_kind": "standard_workflow",
            "workflow": WORKFLOW,
            "workflow_inputs": _request_inputs(),
        },
        strict=True,
    )
    compilation = validate_strategy_request(
        request.model_dump(mode="python"),
        allowed_columns=(),
        target_col=None,
    )

    assert compilation.draft is not None
    assert compilation.draft.to_dict() == {
        "request_kind": "standard_workflow",
        "workflow": WORKFLOW,
        "workflow_inputs": _request_inputs(),
    }
    assert "不选择" in str(compilation.confirmation)
    assert "不采纳" in str(compilation.confirmation)
    assert "不部署" in str(compilation.confirmation)


@pytest.mark.parametrize(
    "forbidden",
    [
        {"sample_design_ref": {"artifact_id": "a" * 64}},
        {"model_score_evidence_refs": []},
        {"expected_registry_token": "a" * 64},
        {"dataset_id": "dataset-owned-by-platform"},
        {"selected_model_evidence_ref": {"evidence_id": "forged"}},
    ],
)
def test_manual_score_comparison_rejects_platform_refs_and_selection(
    forbidden: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            {
                "request_kind": "standard_workflow",
                "workflow": WORKFLOW,
                "workflow_inputs": {**_request_inputs(), **forbidden},
            },
            strict=True,
        )


def test_score_comparison_catalog_preparer_uses_only_server_side_binding() -> None:
    bound = {
        "sample_design_ref": {"membership_artifact_id": "a" * 64},
        "model_score_evidence_refs": [
            {"evidence_artifact_id": "b" * 64},
            {"evidence_artifact_id": "c" * 64},
        ],
        "expected_registry_token": "d" * 64,
    }
    calls: list[tuple[str, str]] = []

    def bind(population: str, partition: str):
        calls.append((population, partition))
        return MappingProxyType(bound)

    resolved = resolve_strategy_request(
        WORKFLOW,
        _request_inputs(),
        context=StrategyWorkflowResolutionContext(
            allowed_columns=(),
            target_col=None,
        ),
    )
    prepared = prepare_strategy_plan(
        resolved.workflow_id,
        resolved.workflow_inputs,
        context=StrategyWorkflowPreparationContext(
            dataset_id=None,
            bind_model_score_comparison=bind,
        ),
    )

    assert calls == [("risk", "development")]
    assert prepared.template_id == WORKFLOW
    assert prepared.to_runtime_slots() == {
        **bound,
        "population": "risk",
        "partition": "development",
    }


class _ComparisonLLM:
    def complete(self, **_kwargs) -> str:
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": WORKFLOW,
                "workflow_inputs": _request_inputs(),
            },
            ensure_ascii=False,
        )


def test_natural_language_compiler_routes_only_grounded_business_dimensions() -> None:
    utterance = (
        "物化当前任务的模型评分比较证据：总体 risk，分区 development；"
        "只比较，不选择冠军、不采纳、不部署。"
    )

    assert turn_handlers._is_strategy_request_intent(utterance)
    compilation = compile_strategy_request(
        utterance,
        allowed_columns=(),
        target_col=None,
        llm=_ComparisonLLM(),
    )

    assert compilation.draft is not None
    assert compilation.draft.to_dict()["workflow_inputs"] == _request_inputs()


class _Artifacts:
    def __init__(self, snapshots: list[list[dict[str, object]]]) -> None:
        self.snapshots = snapshots
        self.calls = 0

    def list_for_task(self, _task_id: str) -> list[dict[str, object]]:
        index = min(self.calls, len(self.snapshots) - 1)
        self.calls += 1
        return [dict(item) for item in self.snapshots[index]]


def _score_record(
    marker: str,
    *,
    model_id: str,
    sample_id: str = "sample-current",
    sample_hash: str = "1" * 64,
) -> dict[str, object]:
    return {
        "id": marker * 64,
        "kind": MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
        "origin_tool": MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
        "content_hash": marker.upper() * 64,
        "created_at": f"2026-08-01T00:00:0{marker}Z",
        "provenance": {
            "model_artifact_id": model_id,
            "sample_design_id": sample_id,
            "sample_design_content_hash": sample_hash,
            "score_vector_artifact_id": (marker + "v") * 32,
            "score_vector_artifact_content_hash": (marker + "h") * 32,
        },
    }


def _sample_binding() -> SimpleNamespace:
    design = {
        "sample_design_id": "sample-current",
        "content_hash": "1" * 64,
    }
    return SimpleNamespace(
        task_id="task-1",
        membership_artifact_id="2" * 64,
        membership_artifact_content_hash="3" * 64,
        bundle_artifact_id="4" * 64,
        bundle_artifact_content_hash="5" * 64,
        bundle={
            "bundle_id": "sample-bundle-current",
            "sample_design": design,
        },
    )


def test_turn_binding_uses_latest_compatible_authenticated_candidate_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = _score_record(
        "a",
        model_id="model-stale",
        sample_id="sample-old",
        sample_hash="0" * 64,
    )
    old_model_a = _score_record("b", model_id="model-a")
    newest_model_a = _score_record("c", model_id="model-a")
    model_b = _score_record("d", model_id="model-b")
    records = [stale, old_model_a, newest_model_a, model_b]
    artifacts = _Artifacts([records, records])
    read_runtime = SimpleNamespace(task_artifacts=artifacts)
    sample = _sample_binding()
    load_calls: list[str] = []

    monkeypatch.setattr(
        turn_handlers,
        "_strategy_report_read_runtime",
        lambda _runtime: read_runtime,
    )
    monkeypatch.setattr(
        turn_handlers,
        "_latest_verified_strategy_sample_design_v2_binding",
        lambda *_args, **_kwargs: sample,
    )
    monkeypatch.setattr(
        turn_handlers,
        "model_score_comparison_registry_snapshot_token",
        lambda _records: "9" * 64,
        raising=False,
    )

    def load_score(_runtime, **kwargs):
        evidence_id = kwargs["evidence_artifact_id"]
        load_calls.append(evidence_id)
        record = next(item for item in records if item["id"] == evidence_id)
        provenance = record["provenance"]
        bound_sample = (
            sample.bundle
            if provenance["sample_design_id"] == "sample-current"
            else {
                "bundle_id": "sample-bundle-old",
                "sample_design": {
                    "sample_design_id": provenance["sample_design_id"],
                    "content_hash": provenance[
                        "sample_design_content_hash"
                    ],
                },
            }
        )
        vector_record = {
            "id": provenance["score_vector_artifact_id"],
            "content_hash": provenance[
                "score_vector_artifact_content_hash"
            ],
        }
        return SimpleNamespace(
            task_id="task-1",
            training=SimpleNamespace(
                task_id="task-1",
                model_artifact=SimpleNamespace(
                    id=provenance["model_artifact_id"]
                ),
                sample=SimpleNamespace(task_id="task-1", bundle=bound_sample),
            ),
            evidence_record=record,
            vector_record=vector_record,
            envelope={"single_model_evidence": {"model": evidence_id}},
        )

    monkeypatch.setattr(
        turn_handlers,
        "load_historical_model_score_evidence_artifacts",
        load_score,
    )
    comparison_calls: list[dict[str, object]] = []

    def compare(**kwargs):
        comparison_calls.append(kwargs)
        return {"selection": {"status": "no_selection"}}

    monkeypatch.setattr(
        turn_handlers,
        "build_model_score_comparison",
        compare,
        raising=False,
    )

    slots = turn_handlers._model_score_comparison_plan_slots(
        SimpleNamespace(),
        SimpleNamespace(id="task-1"),
        population="risk",
        partition="development",
    )

    assert load_calls == [
        stale["id"],
        old_model_a["id"],
        newest_model_a["id"],
        model_b["id"],
    ]
    assert comparison_calls[0]["population"] == "risk"
    assert comparison_calls[0]["partition"] == "development"
    assert comparison_calls[0]["model_evidence"] == [
        {"model": newest_model_a["id"]},
        {"model": model_b["id"]},
    ]
    assert len(slots["model_score_evidence_refs"]) == 2
    assert slots["sample_design_ref"]["expected_sample_design_id"] == (
        "sample-current"
    )
    assert slots["expected_registry_token"] == "9" * 64
    assert artifacts.calls == 2


def test_turn_binding_fails_closed_when_registry_changes_before_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = [
        _score_record("c", model_id="model-a"),
        _score_record("d", model_id="model-b"),
    ]
    changed = [*first, _score_record("e", model_id="model-c")]
    artifacts = _Artifacts([first, changed])
    read_runtime = SimpleNamespace(task_artifacts=artifacts)
    sample = _sample_binding()
    monkeypatch.setattr(
        turn_handlers,
        "_strategy_report_read_runtime",
        lambda _runtime: read_runtime,
    )
    monkeypatch.setattr(
        turn_handlers,
        "_latest_verified_strategy_sample_design_v2_binding",
        lambda *_args, **_kwargs: sample,
    )

    def load_score(_runtime, **kwargs):
        record = next(
            item for item in first if item["id"] == kwargs["evidence_artifact_id"]
        )
        provenance = record["provenance"]
        return SimpleNamespace(
            task_id="task-1",
            training=SimpleNamespace(
                task_id="task-1",
                model_artifact=SimpleNamespace(
                    id=provenance["model_artifact_id"]
                ),
                sample=SimpleNamespace(task_id="task-1", bundle=sample.bundle),
            ),
            evidence_record=record,
            vector_record={
                "id": provenance["score_vector_artifact_id"],
                "content_hash": provenance[
                    "score_vector_artifact_content_hash"
                ],
            },
            envelope={"single_model_evidence": {"model": record["id"]}},
        )

    monkeypatch.setattr(
        turn_handlers,
        "load_historical_model_score_evidence_artifacts",
        load_score,
    )
    monkeypatch.setattr(
        turn_handlers,
        "build_model_score_comparison",
        lambda **_kwargs: {"selection": {"status": "no_selection"}},
        raising=False,
    )

    with pytest.raises(StrategySetupError, match="发生变化"):
        turn_handlers._model_score_comparison_plan_slots(
            SimpleNamespace(),
            SimpleNamespace(id="task-1"),
            population="risk",
            partition="development",
        )


def test_turn_binding_does_not_fall_back_from_drifted_latest_model_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_model_a = _score_record("b", model_id="model-a")
    newest_model_a = _score_record("c", model_id="model-a")
    model_b = _score_record("d", model_id="model-b")
    records = [old_model_a, newest_model_a, model_b]
    artifacts = _Artifacts([records])
    sample = _sample_binding()
    monkeypatch.setattr(
        turn_handlers,
        "_strategy_report_read_runtime",
        lambda _runtime: SimpleNamespace(task_artifacts=artifacts),
    )
    monkeypatch.setattr(
        turn_handlers,
        "_latest_verified_strategy_sample_design_v2_binding",
        lambda *_args, **_kwargs: sample,
    )
    load_calls: list[str] = []

    def reject_latest(_runtime, **kwargs):
        evidence_id = kwargs["evidence_artifact_id"]
        load_calls.append(evidence_id)
        if evidence_id == newest_model_a["id"]:
            raise turn_handlers.ModelingError("evidence bytes drifted")
        record = next(item for item in records if item["id"] == evidence_id)
        provenance = record["provenance"]
        return SimpleNamespace(
            task_id="task-1",
            training=SimpleNamespace(
                task_id="task-1",
                model_artifact=SimpleNamespace(
                    id=provenance["model_artifact_id"]
                ),
                sample=SimpleNamespace(task_id="task-1", bundle=sample.bundle),
            ),
            evidence_record=record,
            vector_record={
                "id": provenance["score_vector_artifact_id"],
                "content_hash": provenance[
                    "score_vector_artifact_content_hash"
                ],
            },
            envelope={"single_model_evidence": {"model": evidence_id}},
        )

    monkeypatch.setattr(
        turn_handlers,
        "load_historical_model_score_evidence_artifacts",
        reject_latest,
    )

    with pytest.raises(StrategySetupError, match="不会回退"):
        turn_handlers._model_score_comparison_plan_slots(
            SimpleNamespace(),
            SimpleNamespace(id="task-1"),
            population="risk",
            partition="development",
        )

    assert load_calls == [old_model_a["id"], newest_model_a["id"]]
    assert model_b["id"] not in load_calls


@pytest.mark.slow
@pytest.mark.e2e
def test_manual_api_materializes_downloadable_audited_nonselecting_comparison(
    tmp_path,
) -> None:
    fixture, _opaque_tool_inputs = _comparison_fixture(tmp_path)
    task_id = fixture["task"].id
    with connect(fixture["settings"].db_path) as conn:
        conn.execute(
            "UPDATE tasks SET run_mode = 'agent' WHERE id = ?",
            (task_id,),
        )
        conn.commit()

    with TestClient(create_app(fixture["settings"])) as client:
        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={
                "content": "比较当前风险开发样本上的两个模型，只生成比较证据。",
                "strategy_request": {
                    "request_kind": "standard_workflow",
                    "workflow": WORKFLOW,
                    "workflow_inputs": _request_inputs(),
                },
            },
        )

        assert response.status_code == 202, response.text
        plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
        assert [(item["template_id"], item["status"]) for item in plans] == [
            (WORKFLOW, "done")
        ], [
            (item.get("content"), item.get("metadata"))
            for item in response.json()["messages"]
        ]
        plan = client.app.state.plan_repo.load_plan(plans[0]["id"])
        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert step.inputs["population"] == "risk"
        assert step.inputs["partition"] == "development"
        assert len(step.inputs["model_score_evidence_refs"]) == 2
        assert "sample_design_ref" in step.inputs
        assert "expected_registry_token" in step.inputs

        output = client.app.state.plan_repo.load_step_output(step.id)
        assert output["governance"] == {
            "selection_status": "no_selection",
            "winner_selected": False,
            "not_adopted": True,
            "not_deployed": True,
        }
        assert output["comparison"]["selection"]["status"] == "no_selection"
        downloaded = client.get(output["artifact"]["download_url"])
        assert downloaded.status_code == 200, downloaded.text
        document = json.loads(downloaded.content)
        assert document["comparison"] == output["comparison"]
        assert document["governance"] == output["governance"]

    with connect(fixture["settings"].db_path) as conn:
        audit = conn.execute(
            "SELECT outcome, detail_json FROM audit WHERE kind = ?",
            (MODEL_SCORE_COMPARISON_V2_AUDIT_KIND,),
        ).fetchone()
        artifact = conn.execute(
            "SELECT kind FROM task_artifacts WHERE id = ? AND task_id = ?",
            (output["artifact"]["artifact_id"], task_id),
        ).fetchone()
    assert audit is not None and audit["outcome"] == "succeeded"
    assert json.loads(audit["detail_json"])["winner_selected"] is False
    assert artifact is not None
    assert artifact["kind"] == MODEL_SCORE_COMPARISON_V2_ARTIFACT_KIND
