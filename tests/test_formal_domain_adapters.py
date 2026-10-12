from types import SimpleNamespace

import pytest

from marvis.orchestrator.eval.acceptance_trust import TrustInputError
from marvis.orchestrator.eval.formal_acceptance import _domain, _domain_result
from marvis.orchestrator.eval.formal_domain_adapters import independently_expected, _final_native_binding
from marvis.orchestrator.eval.runtime_scoring import ExpectedCase


def recomputed(values):
    return {"supported_checks_verified": True, "values": values,
            "checks": {"native_steps": {"status": "verified"}}, "unsupported": {"business_acceptance": "unavailable"}}


def expected(tool, path, value, kind="output_equals"):
    return ExpectedCase(result="done", assertions=[{"kind": kind, "tool": tool, "path": path, "value": value}])


def test_model_child_and_parent_metrics_cannot_be_conflated():
    data = recomputed({"metrics": {"test": None}, "scope": {"fit_scope": "full_development"},
                       "selection_evidence": {"metrics": {"test": {"auc": .7}}}})
    assert independently_expected(expected("modeling.select_experiment", ["refit", "applied"], True), {}, data, family="modeling")
    assert not independently_expected(expected("modeling.select_experiment", ["refit", "applied"], False), {}, data, family="modeling")
    assert independently_expected(expected("modeling.select_experiment", ["metrics", "test"], None), {}, data, family="modeling")
    with pytest.raises(TrustInputError, match="predicate_unavailable"):
        independently_expected(expected("modeling.select_experiment", ["metrics", "test", "auc"], .7), {}, data, family="modeling")


@pytest.mark.parametrize("family,tool", [("strategy", "strategy.backtest_strategy"), ("monitoring", "modeling.monitor_run")])
def test_recomputed_scalars_and_unknown_predicates(family, tool):
    data = recomputed({"metric": .2})
    assert independently_expected(expected(tool, ["metric"], .2), {}, data, family=family)
    assert not independently_expected(expected(tool, ["metric"], .3), {"candidate_passed": True}, data, family=family)
    with pytest.raises(TrustInputError, match="predicate_unavailable"):
        independently_expected(expected(tool, ["business_accepted"], True), {}, data, family=family)


def test_extension_arithmetic_does_not_establish_unchecked_safety():
    data = recomputed({"decisions": [{"action_type": "reject"}]})
    answer = independently_expected(expected("", ["decisions", 0, "action_type"], "reject", "extension_value"), {}, data, family="online_decision")
    result = _domain_result(data, answer)
    assert result["actual_result"] == "done"
    assert dict(result["safety"])["authorization"] == "unavailable"
    assert dict(result["safety"])["point_in_time"] == "unavailable"
    assert dict(result["safety"])["duplicate_side_effects"] == "unavailable"


def test_final_native_pointer_must_match_original_done_plan_and_exact_step_hash():
    binding = SimpleNamespace(plan_id="plan", step_ids={"select": "step"}, output_sha256={"select": "a" * 64})
    observed = {"execution": {"plans": [{"id": "plan", "status": "done"}],
                              "steps": [{"id": "step", "status": "done", "tool": "modeling.select_experiment", "output_sha256": "a" * 64}]}}
    native_tools = {"step": "modeling.select_experiment"}
    assert _final_native_binding(binding, observed, native_tools=native_tools) == {"modeling.select_experiment"}
    observed["execution"]["steps"][0]["output_sha256"] = "b" * 64
    with pytest.raises(TrustInputError, match="step_binding_mismatch"):
        _final_native_binding(binding, observed, native_tools=native_tools)


def test_public_tool_name_cannot_substitute_for_authenticated_native_tool():
    binding = SimpleNamespace(plan_id="plan", step_ids={"select": "step"}, output_sha256={"select": "a" * 64})
    observed = {"execution": {"plans": [{"id": "plan", "status": "done"}],
                              "steps": [{"id": "step", "status": "done", "tool": "uninstalled.absent_tool", "output_sha256": "a" * 64}]}}
    with pytest.raises(TrustInputError, match="step_binding_mismatch"):
        _final_native_binding(binding, observed, native_tools={"step": "modeling.select_experiment"})
    with pytest.raises(TrustInputError, match="step_binding_mismatch"):
        _final_native_binding(binding, observed, native_tools={})


@pytest.mark.parametrize("family", ["modeling", "strategy", "monitoring", "online_decision", "connector_antifraud", "collections"])
def test_each_installed_domain_preserves_unavailable_originals_and_wrong_cohort(family):
    cohort = "extension" if family in {"online_decision", "connector_antifraud", "collections"} else "normal"
    case = SimpleNamespace(family=family, cohort=cohort)
    assert _domain(case, None, None, None, None, None)[1] == "domain_original_evidence_unavailable"
    case.cohort = "recovery"
    assert "boundary_or_recovery" in _domain(case, None, None, None, None, None)[1]
