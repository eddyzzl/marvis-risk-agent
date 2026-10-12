"""Independent opt-in modeling journey; only the LLM is a protocol fixture."""

import copy
import json
import time

import httpx
import pytest
from pydantic import ValidationError

from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeAction, RuntimeCase
from marvis.orchestrator.eval.runtime_runner import (
    Journey,
    RuntimeJourneyError,
    run_runtime_suite,
)
from test_runtime_agent_benchmark import fixture_model
from test_runtime_workflow_families import _business_protocol


def _modeling_protocol(request, answer, payload):
    from test_runtime_workflow_families import _gate_route_request
    if _gate_route_request(payload):
        return {
            "action": "adjust",
            "params": {"recipes": ["lr"], "n_trials": 1, "target_type": "binary"},
            "constraint": "",
            "reason": "按用户公开声明的单一逻辑回归规格继续。",
            "confidence": "high",
            "explicit_authorization": False,
        }
    return _business_protocol(request, answer, payload)


def test_normal_modeling_crosses_real_http_training_selection_and_delivery(tmp_path):
    paths = write_synthetic_suite(tmp_path / "suite", normal_modeling_only=True)
    with fixture_model(answer_factory=_modeling_protocol) as (model, calls):
        report = run_runtime_suite(
            cases_path=paths["cases"],
            expected_path=paths["expected"],
            dataset_root=paths["dataset_root"],
            output_dir=tmp_path / "runs",
            model=model,
            model_source="fixture_model",
        )
    assert report["all_passed"], json.dumps(report, ensure_ascii=False, indent=2)
    assert report["denominator"] == 1
    assert report["acceptance_claim"] == "not_established"
    assert report["a_evidence_case_count"] == 0
    assert calls


def _selection_journey():
    snapshot = {
        "expected_plan_status": "awaiting_confirm",
        "expected_plan_revision": 0,
        "expected_plan_fingerprint": "a" * 64,
        "expected_step_fingerprint": "b" * 64,
    }
    step = {
        "id": "selection",
        "status": "awaiting_confirm",
        "tool_ref": {"plugin": "modeling", "tool": "select_experiment"},
        "confirmation_snapshot": snapshot,
    }
    plan = {"id": "plan", "status": "awaiting_confirm", "steps": [step]}
    gate = {
        "role": "assistant",
        "metadata": {
            "kind": "gate",
            "plan_id": "plan",
            "step_id": "selection",
            "confirmation_snapshot": copy.deepcopy(snapshot),
            "model_delivery": {
                "recommended_experiment_id": "exp-current",
                "candidates": [
                    {"id": "exp-current", "recommended": True},
                    {"id": "exp-other", "recommended": False},
                ],
            },
        },
    }

    class Client:
        def __init__(self):
            self.posts = []

        def request(self, method, path, **kwargs):
            if method == "POST":
                self.posts.append((path, kwargs["json"]))
                return httpx.Response(202, json={})
            return httpx.Response(
                200,
                json={"plans": [plan]}
                if path.endswith("/plans")
                else {"messages": [gate]},
            )

    client = Client()
    case = RuntimeCase(
        id="test",
        revision="1",
        family="modeling",
        task={"task_type": "modeling"},
        business_constraints_source="Public declared recommendation selection",
    )
    journey = Journey(client, case, time.monotonic() + 20)
    journey.task_id = "task"
    journey.wait_idle = lambda **kwargs: None
    action = RuntimeAction(
        kind="select_recommended_experiment",
        tool="modeling.select_experiment",
        content="明确采用平台展示的推荐候选。",
    )
    return journey, client, action, plan, gate


def test_declared_recommendation_policy_uses_displayed_id_and_full_snapshot():
    journey, client, action, plan, gate = _selection_journey()
    journey.select_recommended_experiment(action)
    assert len(client.posts) == 1
    path, body = client.posts[0]
    assert path == "/api/tasks/task/agent/messages"
    assert body["adjust_params"] == {"selected_experiment_id": "exp-current"}
    assert body["ui_action"] == "confirm_gate"
    assert (
        body["expected_plan_id"] == "plan" and body["expected_step_id"] == "selection"
    )
    assert all(
        body[k] == v for k, v in gate["metadata"]["confirmation_snapshot"].items()
    )
    assert journey.interventions == 1


@pytest.mark.parametrize("refit", [False, True])
def test_declared_refit_policy_reaches_exact_current_selection_cas_body(refit):
    journey, client, legacy_action, plan, gate = _selection_journey()
    action = RuntimeAction.model_validate({**legacy_action.model_dump(), "refit_on_train_plus_test": refit})
    journey.select_recommended_experiment(action)
    assert len(client.posts) == 1
    path, body = client.posts[0]
    assert path == "/api/tasks/task/agent/messages"
    assert body["adjust_params"] == {"selected_experiment_id": "exp-current", "refit_on_train_plus_test": refit}
    assert body["ui_action"] == "confirm_gate"
    assert body["expected_plan_id"] == plan["id"] and body["expected_step_id"] == plan["steps"][0]["id"]
    assert all(body[key] == value for key, value in gate["metadata"]["confirmation_snapshot"].items())
    assert journey.approval == (path, body) and journey.interventions == 1


def test_omitted_refit_policy_preserves_legacy_serialized_action_and_wire():
    journey, client, action, _plan, _gate = _selection_journey()
    legacy = {"kind": "select_recommended_experiment", "tool": "modeling.select_experiment",
              "content": "明确采用平台展示的推荐候选。"}
    assert action.model_dump() == legacy
    assert json.loads(action.model_dump_json()) == legacy
    assert RuntimeAction.model_validate({**legacy, "refit_on_train_plus_test": None}).model_dump() == legacy
    journey.select_recommended_experiment(action)
    assert client.posts[0][1]["adjust_params"] == {"selected_experiment_id": "exp-current"}


@pytest.mark.parametrize("value", [0, 1, "false", "true", [], {}, 0.0, 1.0])
def test_selection_refit_policy_requires_an_actual_strict_boolean(value):
    with pytest.raises(ValidationError, match="refit_on_train_plus_test"):
        RuntimeAction(kind="select_recommended_experiment", tool="modeling.select_experiment",
                      content="明确采用平台推荐。", refit_on_train_plus_test=value)


@pytest.mark.parametrize("kind", ["message", "approve_step", "reject_step", "retry_step", "stop"])
@pytest.mark.parametrize("value", [False, True])
def test_refit_policy_is_not_a_generic_message_or_other_action_control(kind, value):
    with pytest.raises(ValidationError, match="refit_on_train_plus_test belongs only"):
        RuntimeAction(kind=kind, tool="modeling.select_experiment", content="公开人工声明。",
                      refit_on_train_plus_test=value)


@pytest.mark.parametrize("override", [{"refit_on_train_plus_test": "false"}, {"refit_on_train_plus_test": 0}, {"kind": "message"}])
def test_unvalidated_selection_refit_or_wrong_action_never_posts(override):
    journey, client, action, _plan, _gate = _selection_journey()
    with pytest.raises(RuntimeJourneyError):
        journey.select_recommended_experiment(action.model_copy(update=override))
    assert client.posts == [] and journey.interventions == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "no_recommendation",
        "multiple",
        "not_candidate",
        "duplicates",
        "missing_snapshot",
        "stale_snapshot",
        "invalid_snapshot",
        "wrong_gate",
        "wrong_plan",
        "user_message",
        "no_pending",
    ],
)
def test_recommendation_selection_fails_closed_before_post(mutation):
    journey, client, action, plan, gate = _selection_journey()
    delivery = gate["metadata"]["model_delivery"]
    if mutation == "no_recommendation":
        delivery["recommended_experiment_id"] = ""
    elif mutation == "multiple":
        delivery["candidates"][1]["recommended"] = True
    elif mutation == "not_candidate":
        delivery["recommended_experiment_id"] = "foreign"
    elif mutation == "duplicates":
        delivery["candidates"][1]["id"] = "exp-current"
    elif mutation == "missing_snapshot":
        gate["metadata"].pop("confirmation_snapshot")
    elif mutation == "stale_snapshot":
        gate["metadata"]["confirmation_snapshot"]["expected_plan_revision"] = 1
    elif mutation == "invalid_snapshot":
        gate["metadata"]["confirmation_snapshot"]["expected_plan_revision"] = True
        plan["steps"][0]["confirmation_snapshot"]["expected_plan_revision"] = True
    elif mutation == "wrong_gate":
        action = action.model_copy(update={"tool": "modeling.train_models"})
    elif mutation == "wrong_plan":
        gate["metadata"]["plan_id"] = "foreign"
    elif mutation == "user_message":
        gate["role"] = "user"
    elif mutation == "no_pending":
        plan["steps"][0]["status"] = "done"
    with pytest.raises(RuntimeJourneyError):
        journey.select_recommended_experiment(action)
    assert client.posts == [] and journey.interventions == 0


@pytest.mark.parametrize(
    "override",
    [
        {"tool": "modeling.train_models"},
        {"content": ""},
        {"selected_experiment_id": "invented"},
        {"route": "/unsafe"},
        {"callback": "anything"},
        {"selection_policy": "highest_private_expected_score"},
    ],
)
def test_selection_action_never_accepts_arbitrary_dispatch_or_candidate(override):
    with pytest.raises(ValidationError):
        RuntimeAction.model_validate(
            {
                "kind": "select_recommended_experiment",
                "tool": "modeling.select_experiment",
                "content": "明确采用平台推荐。",
                **override,
            }
        )


def test_normal_modeling_is_an_independent_explicit_suite(tmp_path):
    original = write_synthetic_suite(tmp_path / "nine", include_workflow_families=True)
    normal = write_synthetic_suite(tmp_path / "normal", normal_modeling_only=True)
    assert len(json.loads(original["cases"].read_text())["cases"]) == 9
    cases = json.loads(normal["cases"].read_text())["cases"]
    assert [c["id"] for c in cases] == ["synthetic_normal_modeling"]
    selection = next(action for action in cases[0]["actions"] if action["kind"] == "select_recommended_experiment")
    assert selection["refit_on_train_plus_test"] is False
    assert "不重训" in selection["content"]
    assert "fold-frozen training retains the authenticated selected candidate without legacy refit" in cases[0]["business_constraints_source"]
    assert set(json.loads(normal["expected"].read_text())["cases"]) == {
        "synthetic_normal_modeling"
    }
    assert list(normal["dataset_root"].iterdir()) == [
        normal["dataset_root"] / "normal_modeling.parquet"
    ]
    with pytest.raises(ValueError):
        write_synthetic_suite(
            tmp_path / "invalid",
            include_workflow_families=True,
            normal_modeling_only=True,
        )
    assert not (tmp_path / "invalid").exists()
