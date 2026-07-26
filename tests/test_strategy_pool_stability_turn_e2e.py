"""Turn-boundary and execution contract for current-Pool stability."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.turn_handlers import (
    _is_strategy_request_intent,
    _run_validated_strategy_request,
    _standard_workflow_request_preflight,
    _strategy_pool_stability_plan_slots,
    _strategy_request_requires_dataset,
    _strategy_request_requires_target,
)
from marvis.app import create_app
from tests.test_strategy_pool_validation_tools import _setup


def _runtime(fixture: dict) -> SimpleNamespace:
    return SimpleNamespace(settings=fixture["settings"])


def _draft(
    strategy_type: str = "approval",
) -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="strategy_pool_stability",
        workflow_inputs={"strategy_type": strategy_type},
    )


class _PayloadLLM:
    def complete(self, **kwargs) -> str:
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_pool_stability",
                "workflow_inputs": {"strategy_type": "approval"},
            },
            ensure_ascii=False,
        )


def test_turn_freezes_exact_pool_sample_and_all_available_comparisons(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)

    slots = _strategy_pool_stability_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
    )

    assert slots == {
        "strategy_type": "approval",
        "pool_ref": {
            "artifact_id": fixture["pool_artifact"]["artifact_id"],
            "expected_artifact_content_hash": fixture["pool_artifact"][
                "content_hash"
            ],
            "expected_pool_id": fixture["pool"]["pool_id"],
            "expected_revision": fixture["pool"]["revision"],
            "expected_revision_id": fixture["pool"]["pool"]["revision_id"],
            "expected_snapshot_hash": fixture["pool"]["snapshot_hash"],
        },
        "sample_design_ref": fixture["sample_ref"],
        "partitions": ["development", "validation", "oot"],
    }


def test_turn_routes_without_dataset_or_target_and_starts_two_step_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = _draft("pricing")
    captured: dict = {}
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_pool_stability_plan_slots",
        lambda runtime, task, candidate: {
            "strategy_type": "pricing",
            "pool_ref": {"exact": "pool"},
            "sample_design_ref": {"exact": "sample"},
            "partitions": ["development", "oot"],
        },
    )

    def _start(runtime, repo, task, **kwargs):
        captured.update(kwargs)
        return {"status": "started"}

    monkeypatch.setattr(
        "marvis.agent.turn_handlers._start_confirmed_strategy_plan",
        _start,
    )

    assert _strategy_request_requires_dataset(draft) is False
    assert _strategy_request_requires_target(draft) is False
    response = _run_validated_strategy_request(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(id="task-1"),
        draft,
        context=None,
        auto_start=True,
        drop_nan_labels=False,
    )

    assert response == {"status": "started"}
    assert captured["template_id"] == "strategy_pool_stability"
    assert captured["slots"]["strategy_type"] == "pricing"
    assert captured["auto_start"] is True


def test_stability_preflight_does_not_open_a_second_binding_window(
    tmp_path: Path,
) -> None:
    assert (
        _standard_workflow_request_preflight(
            SimpleNamespace(
                settings=SimpleNamespace(
                    db_path=tmp_path / "marvis.sqlite",
                )
            ),
            SimpleNamespace(id="task-1"),
            _draft(),
        )
        is None
    )


@pytest.mark.parametrize(
    "utterance",
    [
        "测量当前审批策略池的跨分区 PSI 稳定性",
        "measure current pricing pool cross-partition stability",
    ],
)
def test_turn_router_recognizes_pool_stability_commands(
    utterance: str,
) -> None:
    assert _is_strategy_request_intent(utterance) is True


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_pool_stability_creates_and_consumes_exact_cube(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    client = TestClient(create_app(fixture["settings"].workspace))
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: _PayloadLLM(),
    )

    response = client.post(
        f"/api/tasks/{fixture['task'].id}/agent/messages",
        json={"content": "测量当前 approval 审批策略池的跨分区 PSI 稳定性"},
    )

    assert response.status_code == 202, response.text
    plans = client.get(
        f"/api/tasks/{fixture['task'].id}/plans"
    ).json()["plans"]
    latest_message = response.json()["messages"][-1]
    assert plans, (
        latest_message["content"],
        latest_message.get("metadata"),
    )
    assert plans[-1]["template_id"] == "strategy_pool_stability"
    assert plans[-1]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert len(stored.steps) == 2
    cube_step, stability_step = stored.steps
    assert cube_step.inputs["pool_ref"]["expected_pool_id"] == (
        fixture["pool"]["pool_id"]
    )
    assert cube_step.inputs["sample_design_ref"] == fixture["sample_ref"]
    assert cube_step.inputs["dimension_bindings"] == {
        "month_col": None,
        "group_col": None,
        "segment_col": None,
    }
    assert stability_step.depends_on == [cube_step.id]
    assert stability_step.inputs == {
        "artifact_id": (
            f"$ref:{cube_step.id}.output.artifact.artifact_id"
        ),
        "expected_artifact_content_hash": (
            f"$ref:{cube_step.id}.output.artifact.content_hash"
        ),
        "expected_cube_id": f"$ref:{cube_step.id}.output.cube_id",
        "expected_cube_content_hash": (
            f"$ref:{cube_step.id}.output.content_hash"
        ),
    }

    cube_output = client.app.state.plan_repo.load_step_output(cube_step.id)
    stability_output = client.app.state.plan_repo.load_step_output(
        stability_step.id
    )
    expected_cube_ref = {
        "artifact_id": cube_output["artifact"]["artifact_id"],
        "expected_artifact_content_hash": cube_output["artifact"][
            "content_hash"
        ],
        "expected_cube_id": cube_output["cube_id"],
        "expected_cube_content_hash": cube_output["content_hash"],
    }
    assert stability_output["stability"]["source_bindings"][
        "impact_cube"
    ] == expected_cube_ref
    assert stability_output["read_only"] is True
    assert stability_output["effect_validation"] is False
    assert stability_output["not_mutated_pool"] is True
    assert stability_output["not_created_strategy"] is True
    assert stability_output["not_adopted"] is True
    assert stability_output["not_promoted"] is True
    assert stability_output["not_deployed"] is True
