import pytest

from marvis.business_acceptance import BusinessCriterion, evaluate_business_acceptance
from marvis.llm_outbound import OutboundContextError
from marvis.llm_outbound_business import business_acceptance_context
from marvis.orchestrator.evidence import payload_hash
from marvis.orchestrator.reviewer import Reviewer
from tests.test_business_acceptance import evidence, objective
from tests.test_llm_outbound import PRIVATE, Wire
from tests.test_orch_reviewer import _plan, _step


def _send(wire, result):
    wire.reply = {"summary": "依据平台验收结果解释", "goal_met": True}
    summary, _, _, opinion = Reviewer(lambda: wire.client)._llm_summarize(
        "模型验证", _plan(_step([])), {"candidate": {"metrics": {"oot_ks": .99}}}, result)
    assert summary == "依据平台验收结果解释"
    assert opinion is True
    return wire.last_context()["business_acceptance"]


@pytest.mark.parametrize("status", ["passed", "failed", "insufficient_evidence", "not_configured", "not_applicable"])
def test_native_business_status_and_numeric_thresholds_survive_actual_wire(monkeypatch, status):
    wire = Wire(monkeypatch)
    contract = objective(business_line=PRIVATE[0], responsibility_source=PRIVATE[1])
    measured = evidence(business_line=PRIVATE[0], label_provenance=PRIVATE[2], source_ref=PRIVATE[3])
    if status == "failed":
        measured = evidence(business_line=PRIVATE[0], metrics={"oot_ks": .2})
    elif status == "insufficient_evidence":
        measured = evidence(business_line=PRIVATE[0], labels_mature=False)
    elif status == "not_configured":
        contract = None
    elif status == "not_applicable":
        contract = objective(applicable=False, allow_not_applicable=True, not_applicable_reason=PRIVATE[4])
    result = evaluate_business_acceptance(contract, measured)
    result["execution_binding"] = payload_hash({"plan": "fixture"})
    bound = _send(wire, result)
    assert bound["status"] == result["status"] == status
    assert bound["execution_binding"] == result["execution_binding"]
    assert "objective" not in bound
    if result["target"]:
        assert bound["target"]["binding_hash"] == payload_hash(result["target"])
        assert bound["target"]["role"] == "adopted_target"
        assert bound["evidence"]["labels_mature"] is measured.labels_mature
        assert bound["evidence"]["period_start"] == "2026-01-01"
        criterion = bound["criteria"][0]
        assert criterion["minimum"] == .3
        assert criterion["maximum"] is None
        assert criterion["value"] == result["criteria"][0]["value"]
        assert criterion["comparison"] == "absolute"
        assert criterion["metric"] == "oot_ks"
        assert criterion["unit"] == "ratio"
        assert criterion["denominator"] == "mature_loans"
    if status == "insufficient_evidence":
        assert bound["criteria"][0]["status"] == "passed"
        assert "mature_observed_labels_missing" in bound["reason_codes"]


def test_native_delta_baseline_and_private_target_binding(monkeypatch):
    wire = Wire(monkeypatch)
    baseline_ref = PRIVATE[3]
    criterion = BusinessCriterion("oot_ks", "ratio", "mature_loans", minimum=.1,
                                  maximum=.2, comparison="delta", baseline_ref=baseline_ref)
    contract = objective(criteria=(criterion,), target_id=PRIVATE[0], target_version=PRIVATE[1])
    measured = evidence(target_id=PRIVATE[0], target_version=PRIVATE[1], baseline_ref=baseline_ref,
                        baseline_metrics={"oot_ks": .25}, baseline_binding_hash=payload_hash({"baseline": .25}),
                        baseline_context={"business_line": "consumer_credit", "decision_node": "approval",
                                          "population": "existing_customers", "period_start": "2026-01-01",
                                          "period_end": "2026-03-31", "currency": None,
                                          "metric_units": {"oot_ks": "ratio"},
                                          "denominators": {"oot_ks": "mature_loans"}})
    result = evaluate_business_acceptance(contract, measured)
    bound = _send(wire, result)
    assert bound["status"] == "passed"
    assert bound["target"]["binding_hash"] == payload_hash(result["target"])
    assert "id" not in bound["target"] and "version" not in bound["target"]
    assert bound["criteria"][0]["value"] == pytest.approx(.15)
    assert bound["criteria"][0]["baseline_value"] == .25
    assert bound["criteria"][0]["comparison"] == "delta"
    assert bound["evidence"]["baseline_binding_hash"] == measured.baseline_binding_hash


def test_all_native_criteria_preserved_including_private_tail_failure(monkeypatch):
    wire = Wire(monkeypatch)
    names = [f"{split}_{metric}" for split in ("train", "test", "oot")
             for metric in ("ks", "auc", "psi", "iv", "rmse", "mae", "r2")]
    names.append(PRIVATE[0])
    criteria = tuple(BusinessCriterion(name, "ratio", "mature_loans", minimum=.3) for name in names)
    measured = evidence(metrics={name: .4 if i < 21 else .2 for i, name in enumerate(names)},
                        metric_units=dict.fromkeys(names, "ratio"), denominators=dict.fromkeys(names, "mature_loans"))
    result = evaluate_business_acceptance(objective(criteria=criteria), measured)
    bound = _send(wire, result)
    assert bound["status"] == "failed"
    assert len(bound["criteria"]) == bound["criteria_count"] == 22
    assert bound["criteria"][-1]["status"] == "failed"
    assert bound["criteria"][-1]["value"] == .2
    assert bound["criteria"][-1]["criterion_index"] == 22
    assert bound["criteria"][-1]["metric_name_withheld"] is True
    assert bound["criteria"][-1]["reason_code"] == "threshold_not_met"


@pytest.mark.parametrize("change,code", [
    ({"metric_units": {"oot_ks": "percent"}}, "unit_mismatch"),
    ({"denominators": {"oot_ks": PRIVATE[0]}}, "denominator_mismatch"),
    ({"metrics": {"oot_ks": None}}, "metric_not_finite"),
])
def test_native_missing_metric_reason_has_fixed_code(monkeypatch, change, code):
    result = evaluate_business_acceptance(objective(), evidence(**change))
    bound = _send(Wire(monkeypatch), result)
    assert bound["status"] == "insufficient_evidence"
    assert bound["criteria"][0]["value"] is None
    assert bound["criteria"][0]["reason_code"] == code


@pytest.mark.parametrize("mature", [False, None])
def test_native_missing_label_boolean_is_not_stringified(monkeypatch, mature):
    result = evaluate_business_acceptance(objective(), evidence(labels_mature=mature))
    bound = _send(Wire(monkeypatch), result)
    assert bound["status"] == "insufficient_evidence"
    assert bound["evidence"]["labels_mature"] is mature


def test_oversized_acceptance_fails_before_request(monkeypatch):
    wire = Wire(monkeypatch)
    result = evaluate_business_acceptance(objective(), evidence())
    result["criteria"] *= 10001
    with pytest.raises(OutboundContextError):
        business_acceptance_context(result)
    summary, _, _, _ = Reviewer(lambda: wire.client)._llm_summarize("验证", _plan(_step([])), {}, result)
    assert summary == "Plan execution reviewed."
    assert wire.requests == []
