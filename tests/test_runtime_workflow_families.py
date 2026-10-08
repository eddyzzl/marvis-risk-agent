"""Additional family journeys through the isolated HTTP benchmark carrier."""

import json

import pytest
from pydantic import ValidationError

from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeAction, RuntimeSuite
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from marvis.orchestrator.eval.runtime_scoring import (
    Assertion,
    _assertion,
    runtime_task_coverage,
)
from test_runtime_agent_benchmark import fixture_model


def _business_protocol(request, answer, payload):
    if "allowed_intents" in request:
        answer = {
            key: answer[key]
            for key in (
                "intent",
                "evidence_quote",
                "reason",
                "confidence",
                "is_question",
                "is_conditional",
                "requests_change",
                "withholds_action",
            )
        }
        if request.get("current_context", {}).get(
            "risk_setup_phase"
        ) == "ask_goal" and "标准 Vintage" in request.get("instruction", ""):
            answer["intent"] = "risk_standard_vintage"
    elif _gate_route_request(payload):
        return {
            "action": "confirm",
            "params": {},
            "constraint": "",
            "reason": "合成用例明确授权当前计划开始。",
            "confidence": "high",
            "explicit_authorization": True,
        }
    elif "proposed_params" in request:
        return {
            "verdict": "authorize",
            "evidence_quote": request["instruction"],
            "reason": "合成用例明确授权当前计划开始。",
            "confidence": "high",
            "is_question": False,
            "is_conditional": False,
            "requests_change": False,
            "withholds_authorization": False,
        }
    return answer


def _gate_route_request(payload):
    # Route/review share the structural user envelope. Distinguish their actual
    # system contracts, not the retired free-text gate-context formatting.
    from marvis.llm_prompts import GATE_INSTRUCTION_ROUTER_SYS
    return payload["messages"][0]["content"].startswith(GATE_INSTRUCTION_ROUTER_SYS.text)


def test_typed_business_message_preserves_old_contract_and_rejects_hidden_actions():
    legacy = {"kind": "message", "content": "确认", "tool": ""}
    assert RuntimeAction.model_validate(legacy).model_dump() == legacy
    portfolio = {
        "id_col": "loan_id",
        "snapshot_col": "month",
        "bucket_col": "bucket",
        "balance_col": "balance",
        "segment_col": "segment",
        "loss_state": "charged_off",
        "lgd": 0.45,
        "horizon_months": 18,
    }
    action = RuntimeAction.model_validate({**legacy, "portfolio_request": portfolio})
    assert action.portfolio_request.lgd == 0.45
    for mutation in (
        {"kind": "stop", "portfolio_request": portfolio},
        {**legacy, "portfolio_request": {**portfolio, "lgd": "0.45"}},
        {**legacy, "portfolio_request": {**portfolio, "horizon_months": True}},
        {**legacy, "portfolio_request": {**portfolio, "route": "/bypass"}},
        {**legacy, "route": "/bypass"},
    ):
        with pytest.raises(ValidationError):
            RuntimeAction.model_validate(mutation)


def test_clarification_requires_latest_assistant_state_not_prior_or_user_metadata():
    assertion = Assertion(
        kind="latest_assistant_metadata", path=["kind"], value="clarify"
    )
    messages = [
        {"role": "assistant", "metadata": {"kind": "clarify"}},
        {"role": "assistant", "metadata": {"kind": "error"}},
        {"role": "user", "metadata": {"kind": "clarify"}},
    ]
    assert not _assertion(assertion, {}, {"messages": messages})
    messages.append({"role": "assistant", "metadata": {"kind": "clarify"}})
    assert _assertion(assertion, {}, {"messages": messages})


def test_nonexecution_requires_existing_steps_without_any_producer_attempt():
    assertion = Assertion(kind="tool_not_executed", tool="analysis.portfolio_report")
    step = {"tool": assertion.tool, "status": "pending", "runs": []}
    for wrong in (
        [],
        [{**step, "status": "failed"}],
        [{**step, "runs": [{"status": "failed"}]}],
        [{**step, "producer_invocation_id": "attempt"}],
        [{**step, "output_ref": "artifact"}],
    ):
        assert not _assertion(assertion, {"execution": {"steps": wrong}}, {})
    assert _assertion(assertion, {"execution": {"steps": [step]}}, {})


def test_successful_subset_keeps_missing_family_scenarios_visible():
    record = {
        "case_id": "one",
        "task_type": "portfolio",
        "family": "misleading-name",
        "scenario": "normal",
        "score": {"passed": True, "a_evidence_eligible": False},
    }
    result = runtime_task_coverage([record])
    assert result["cells"]["portfolio"]["normal"]["passed"] == 1
    assert result["cells"]["portfolio"]["normal"]["real_model_runtime_eligible"] == 0
    assert len(result["missing_cells"]) == 27
    assert not result["all_cells_represented"]
    del record["task_type"]
    assert runtime_task_coverage([record])["unclassified_case_ids"] == ["one"]


def test_opt_in_families_leave_original_public_cases_unchanged(tmp_path):
    old = write_synthetic_suite(tmp_path / "original")
    new = write_synthetic_suite(tmp_path / "expanded", include_workflow_families=True)
    old_cases = json.loads(old["cases"].read_text())["cases"]
    new_cases = json.loads(new["cases"].read_text())["cases"]
    assert new_cases[:2] == old_cases
    assert len(RuntimeSuite.model_validate_json(new["cases"].read_bytes()).cases) == 9
    expected = json.loads(new["expected"].read_text())["cases"]
    assert set(expected) == {case["id"] for case in new_cases}
    assert {case["case_set"] for case in new_cases} == {"development"}
    assert json.loads(old["expected"].read_text())["cases"] == {
        key: expected[key] for key in ("synthetic_join", "synthetic_feature")
    }


def test_portfolio_and_vintage_real_runtime_journeys(tmp_path):
    paths = write_synthetic_suite(tmp_path / "suite", include_workflow_families=True)
    # Run only the additional cases; the original HTTP journeys retain their own
    # separate regression test and historical denominator.
    suite = json.loads(paths["cases"].read_text())
    suite["cases"] = suite["cases"][2:]
    paths["cases"].write_text(json.dumps(suite))
    with fixture_model(answer_factory=_business_protocol) as (model, calls):
        report = run_runtime_suite(
            cases_path=paths["cases"],
            expected_path=paths["expected"],
            dataset_root=paths["dataset_root"],
            output_dir=tmp_path / "runs",
            model=model,
            model_source="fixture_model",
        )
    assert report["all_passed"], json.dumps(report, ensure_ascii=False, indent=2)
    assert report["denominator"] == 7
    assert report["a_evidence_case_count"] == 0
    assert report["acceptance_claim"] == "not_established"
    assert {case["scenario"] for case in report["cases"]} == {
        "normal",
        "clarification",
        "rejection",
    }
    assert calls
    for case in report["cases"]:
        if case["scenario"] == "clarification":
            assert case["execution"]["plans"] == []
        else:
            assert case["score"]["passed"]
            assert case["execution"]["steps"]
