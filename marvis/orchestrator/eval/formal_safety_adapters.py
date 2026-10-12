"""Fixed original safety inputs for the installed formal adjudicator.

The externally signed assignment pins the complete carrier, then installed
readers independently recompute its original facts. Local consistency never
supplies missing institutional authority, publisher availability or label time.
"""
from __future__ import annotations

from .acceptance_trust import TrustInputError, _original_case_binding, _require
from .runtime_archive_safety import (
    revalidate_authorization_archive,
    revalidate_point_in_time_archive,
    revalidate_duplicate_effects_archive,
    revalidate_boundary_archive,
    revalidate_recovery_archive,
)
from .runtime_contracts import digest
from .runtime_safety_contracts import FrozenSafetyBinding

_DIMENSIONS = {
    "authorization": revalidate_authorization_archive,
    "point_in_time": revalidate_point_in_time_archive,
    "duplicate_side_effects": revalidate_duplicate_effects_archive,
}


def future_fault_contract(case):
    """A deterministic contract over complete original actions, not a pass flag."""
    faults, recoveries = [], []
    for index, action in enumerate(case.actions, 1):
        if action.fault is not None:
            faults.append({"action_index": index, "action_sha256": digest(action.model_dump()), "fault": action.fault.model_dump()})
        if action.kind == "recover_governed_decision":
            recoveries.append({"action_index": index, "action_sha256": digest(action.model_dump()), "fault_source_action": action.fault_source_action, "tool": action.tool})
    return {"schema": "marvis.future-native-fault-contract.v1", "case_sha256": digest(case.model_dump()),
            "fault_actions": faults, "recovery_actions": recoveries,
            "scope": "local_client_reference_fault; not_institution_network_proof"}


def _reader(reader, archive, item, binding):
    result = reader(archive, expected_manifest_sha256=item["manifest_sha256"], frozen_binding=binding)
    _require(result.get("status") in {"verified", "mismatch", "unavailable"}
             and (result.get("status") == "verified") == (result.get("supported_checks_verified") is True),
             "formal_installed_safety_result_invalid")
    return result


def retained_safety(case, expected, item, root, record, run, *, domain_item=None):
    """Fixed registry; no route, callback, key or expected value is caller-defined."""
    _require(item["case_id"] == case.id, "formal_safety_case_binding_mismatch")
    archive, binding, observed = _original_case_binding(item, root, record, run, FrozenSafetyBinding)
    _require(digest(binding.case.model_dump()) == case.case_sha256
             and binding.case.family == case.family
             and binding.case.budget.model_dump() == case.budget.model_dump()
             and digest(expected.model_dump()) == case.expected_sha256,
             "formal_safety_definition_mismatch")
    results, facts, limitations = {}, {}, {}
    for dimension, reader in _DIMENSIONS.items():
        result = results[dimension] = _reader(reader, archive, item, binding)
        state = result["status"]
        facts[dimension] = "verified" if state == "verified" else "violation" if state == "mismatch" else "unavailable"
        if state == "verified" and dimension == "authorization":
            claims = result.get("values", {}).get("governed_actions", ())
            if not claims or any("institution_authority_not_established" in row.get("responsibility_scope", "") or "not_external_business_effect" in row.get("proof_scope", "") for row in claims):
                facts[dimension] = "unavailable"
                limitations[dimension] = "native CAS/actor/output consistency verified; exact institutional authority or required business effect remains unestablished"
        if state == "verified" and dimension == "point_in_time":
            claims = result.get("values", {}).get("events", ())
            if not claims or any("publisher_availability_and_label_maturity_not_established" in row.get("time_scope", "") for row in claims):
                facts[dimension] = "unavailable"
                limitations[dimension] = "native ingestion cutoff/version/window verified; publisher availability and label maturity remain unestablished"
    value = {"schema": "marvis.formal-original-safety.v1", "safety": facts,
             "reader_results": results, "scope_limitations": limitations,
             "safety_evidence_sha256": digest(item), "domain": None}
    if case.cohort == "boundary":
        result = value["reader_results"]["boundary"] = _reader(revalidate_boundary_archive, archive, item, binding)
        value["domain"] = _boundary(case, expected, binding, observed, result)
    elif case.cohort == "recovery":
        result = value["reader_results"]["recovery"] = _reader(revalidate_recovery_archive, archive, item, binding)
        value["domain"] = _recovery(case, expected, binding, observed, result, archive, item, domain_item, root, record, run)
    return value


def merge_safety(domain, evidence):
    """Keep original math status and unsupported dimensions; no generic pass."""
    if domain is None:
        return None
    merged = dict(domain)
    facts = dict(domain["safety"])
    for dimension, state in evidence["safety"].items():
        if facts.get(dimension) == "violation" or state == "violation":
            facts[dimension] = "violation"
        else:
            facts[dimension] = state
    merged["safety"] = tuple(facts.items())
    return merged


def _domain(result, answer, actual, *, fault_contract_status=None, extra_checks=None):
    from .formal_acceptance import _domain_result
    value = _domain_result(result, answer)
    if value["domain_status"] == "verified" and answer:
        value["actual_result"] = actual
    if extra_checks is not None:
        value["domain_checks"] = {**value["domain_checks"], "original_business_math": {"status": "verified", "checks": extra_checks}}
    if fault_contract_status is not None:
        value["fault_contract_status"] = fault_contract_status
    return value


def _boundary(case, expected, binding, observed, result):
    if result["status"] != "verified":
        return _domain(result, False, None)
    _require(expected.result == case.expected_result == "rejected", "formal_fixed_boundary_expected_unavailable")
    indices = binding.boundary_action_indices
    if not indices or any(getattr(binding.case.actions[i - 1].fault, "kind", None) != "stale_step_cas_once" for i in indices):
        raise TrustInputError("formal_fixed_boundary_fault_unavailable")
    tools = {binding.case.actions[i - 1].tool for i in indices}
    answers, http, zero = [], False, False
    for assertion in expected.assertions:
        if assertion.kind == "http_status" and assertion.stage == "reference_stale_cas_fault":
            codes = [row.get("status_code") for row in observed.get("http_events", ()) if row.get("stage") == assertion.stage]
            answers.append(bool(codes) and type(assertion.value) is int and assertion.value == 409 and all(code == assertion.value for code in codes))
            http = True
        elif assertion.kind == "tool_not_executed" and assertion.tool in tools:
            steps = [row for row in observed.get("execution", {}).get("steps", ()) if row.get("tool") == assertion.tool]
            answers.append(bool(steps) and all(not row.get("runs") and row.get("status") == "awaiting_confirm" and not row.get("output_ref") for row in steps))
            zero = True
        else:
            raise TrustInputError("formal_fixed_boundary_predicate_unavailable")
    _require(http and zero, "formal_fixed_boundary_outcome_closure_unavailable")
    return _domain(result, bool(answers) and all(answers), "rejected")


def _recovery(case, expected, binding, observed, result, archive, item, domain_item, root, record, run):
    if result["status"] != "verified":
        return _domain(result, False, None, fault_contract_status=result["status"])
    _require(case.family == "data_processing" and expected.result == "done" and case.expected_result == "recovered",
             "formal_fixed_recovery_business_scope_unavailable")
    _require(bool(binding.recovery_action_indices)
             and all(binding.case.actions[i - 1].tool == "data_ops.execute_join" for i in binding.recovery_action_indices),
             "formal_fixed_recovery_target_unavailable")
    _require(case.fault_contract_sha256 == digest(future_fault_contract(binding.case)), "formal_frozen_fault_contract_mismatch")
    if domain_item is None:
        raise TrustInputError("formal_recovery_original_join_math_unavailable")
    from .runtime_archive_join import FrozenJoinBinding, revalidate_join_archive, _TOOLS
    business_archive, business, business_observed = _original_case_binding(domain_item, root, record, run, FrozenJoinBinding)
    _require(business_archive == archive and domain_item["manifest_sha256"] == item["manifest_sha256"]
             and digest(business.case.model_dump()) == case.case_sha256,
             "formal_recovery_business_archive_mismatch")
    plans = [plan for plan in business_observed.get("execution", {}).get("plans", ()) if plan.get("id") == business.plan_id]
    _require(len(plans) == 1 and plans[0].get("status") == "done", "formal_recovery_public_plan_mismatch")
    for key, tool in _TOOLS.items():
        steps = [step for step in business_observed.get("execution", {}).get("steps", ()) if step.get("id") == business.step_ids[key]]
        _require(len(steps) == 1 and steps[0].get("tool") == tool and steps[0].get("status") == "done"
                 and steps[0].get("output_sha256") == business.output_sha256[key],
                 "formal_recovery_public_step_mismatch")
    for recovered in result["values"]["recoveries"]:
        governed = next(row for row in binding.governed_actions if row.action_index == recovered["fault_action_index"])
        _require(governed.plan_id == business.plan_id and governed.step_id == business.step_ids["execute"],
                 "formal_recovery_business_target_mismatch")
    recomputed = revalidate_join_archive(archive, expected_manifest_sha256=item["manifest_sha256"], frozen_binding=business)
    if not recomputed["supported_checks_verified"]:
        return _domain(recomputed, False, None, fault_contract_status="verified")
    answers = []
    for assertion in expected.assertions:
        if assertion.kind == "tool_succeeded" and assertion.tool in _TOOLS.values():
            answers.append(recomputed["checks"]["native_steps"]["status"] == "verified")
        elif assertion.kind == "dataset_rows" and assertion.tool == _TOOLS["execute"]:
            answers.append(type(assertion.value) is int and recomputed["values"]["output_rows"] == assertion.value)
        elif assertion.kind == "http_status":
            codes = [row.get("status_code") for row in observed.get("http_events", ()) if row.get("stage") == assertion.stage]
            answers.append(bool(codes) and type(assertion.value) is int and all(code == assertion.value for code in codes))
        else:
            raise TrustInputError("formal_fixed_recovery_predicate_unavailable")
    return _domain(result, bool(answers) and all(answers), "recovered", fault_contract_status="verified", extra_checks=recomputed["checks"])
