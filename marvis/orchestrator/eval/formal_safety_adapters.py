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
        if action.kind in {"recover_governed_decision", "recover_validation_report", "recover_validation_execution", "recover_monitoring_execution"}:
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
    if case.cohort in {"boundary", "extension_safety"} and case.expected_result == "clarification":
        from .runtime_archive_boundary import revalidate_clarification_archive
        result = value["reader_results"]["clarification"] = _reader(revalidate_clarification_archive, archive, item, binding)
        value["domain"] = _clarification(case, expected, observed, result)
    elif case.cohort == "boundary":
        reader = revalidate_boundary_archive
        if case.family == "validation":
            from .runtime_validation_boundary import revalidate_validation_boundary_archive
            reader = revalidate_validation_boundary_archive
        result = value["reader_results"]["boundary"] = _reader(reader, archive, item, binding)
        value["domain"] = (_validation_boundary(case, expected, binding, observed, result)
                           if case.family == "validation" else _boundary(case, expected, binding, observed, result))
    elif case.cohort == "extension_safety":
        from .runtime_archive_boundary import revalidate_extension_negative_archive
        result = value["reader_results"]["extension_negative"] = _reader(revalidate_extension_negative_archive, archive, item, binding)
        value["domain"] = _extension_negative(case, expected, observed, result)
    elif case.cohort == "recovery":
        reader = revalidate_recovery_archive
        if case.family == "validation":
            from .runtime_archive_validation_recovery import revalidate_validation_recovery_archive
            reader = revalidate_validation_recovery_archive
            if any(a.kind == "recover_validation_execution" for a in binding.case.actions):
                from .runtime_archive_validation_timeout import revalidate_validation_timeout_archive
                reader = revalidate_validation_timeout_archive
        if case.family == "monitoring" and any(a.kind == "recover_monitoring_execution" for a in binding.case.actions):
            from .runtime_archive_monitoring_recovery import revalidate_monitoring_recovery_archive
            reader = revalidate_monitoring_recovery_archive
        result = value["reader_results"]["recovery"] = _reader(reader, archive, item, binding)
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
    if indices and all(binding.case.actions[i - 1].kind == "reject_step" and binding.case.actions[i - 1].fault is None for i in indices):
        return _human_rejection(expected, binding, observed, result)
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


def _validation_boundary(case, expected, binding, observed, result):
    from .runtime_validation_boundary import KIND, STAGE

    if result["status"] != "verified":
        return _domain(result, False, None)
    _require(expected.result == case.expected_result == "rejected", "formal_validation_boundary_expected_unavailable")
    _require(binding.boundary_action_indices == [2] and len(binding.case.actions) == 2
             and getattr(binding.case.actions[1].fault, "kind", None) == KIND,
             "formal_validation_boundary_fault_unavailable")
    answers, required = [], set()
    for assertion in expected.assertions:
        if assertion.kind == "http_status" and assertion.stage == STAGE:
            codes = [r.get("status_code") for r in observed.get("http_events", []) if r.get("stage") == STAGE]
            answers.append(bool(codes) and type(assertion.value) is int and assertion.value == 409
                           and all(code == 409 for code in codes))
            required.add("http_status")
        elif (assertion.kind == "validation_boundary_equals" and not assertion.tool and len(assertion.path) == 1
              and assertion.path[0] in {"confirmation_count", "new_jobs", "metrics_unchanged", "revision_unchanged"}):
            actual = result["values"][assertion.path[0]]
            answers.append(type(actual) is type(assertion.value) and actual == assertion.value)
            required.add(assertion.path[0])
        else:
            raise TrustInputError("formal_validation_boundary_predicate_unavailable")
    _require(required >= {"http_status", "confirmation_count", "new_jobs", "metrics_unchanged", "revision_unchanged"},
             "formal_validation_boundary_outcome_closure_unavailable")
    return _domain(result, all(answers), "rejected", fault_contract_status="verified")


def _recovery(case, expected, binding, observed, result, archive, item, domain_item, root, record, run):
    if result["status"] != "verified":
        return _domain(result, False, None, fault_contract_status=result["status"])
    if case.family == "validation":
        return _validation_recovery(case, expected, binding, result, archive, item, domain_item, root, record, run)
    if case.family == "monitoring" and any(a.kind == "recover_monitoring_execution" for a in binding.case.actions):
        return _monitoring_recovery(case, expected, binding, result, archive, item, domain_item, root, record, run)
    if case.family != "data_processing":
        return _family_recovery(case, expected, binding, observed, result, archive, item, domain_item, root, record, run)
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


def _validation_recovery(case, expected, binding, result, archive, item, domain_item, root, record, run):
    from .acceptance_trust import _validation_case_binding
    from .formal_acceptance import _validation_domain

    _require(expected.result == "done" and case.expected_result == "recovered", "formal_validation_recovery_scope_unavailable")
    _require(case.fault_contract_sha256 == digest(future_fault_contract(binding.case)), "formal_frozen_fault_contract_mismatch")
    if result.get("values", {}).get("recovery_completed") is False:
        return _domain(result, False, "recovered", fault_contract_status="verified")
    if domain_item is None:
        raise TrustInputError("formal_recovery_original_validation_math_unavailable")
    business_archive, business, _ = _validation_case_binding(domain_item, root, record, run)
    _require(business_archive == archive and domain_item["manifest_sha256"] == item["manifest_sha256"]
             and digest(business.case.model_dump()) == case.case_sha256, "formal_recovery_business_archive_mismatch")
    request = result["values"]["confirmation_request"]
    _require(request["revision"] + 1 == business.report_revision
             and digest(request["text_values"]) == business.confirmed_draft_sha256
             and {d["kind"]: d["sha256"] for d in result["values"]["downloads"]} == business.report_sha256,
             "formal_validation_recovery_confirmation_mismatch")
    domain, reason = _validation_domain(case.model_copy(update={"cohort": "normal", "fault_contract_sha256": None}),
                                        expected, domain_item, root, record, run)
    if domain is None:
        raise TrustInputError(reason or "formal_recovery_original_validation_math_unavailable")
    verified = domain["domain_status"] == "verified" and domain["actual_result"] == "done"
    combined = {**result, "checks": {**result["checks"], "original_business_math": {
        "status": domain["domain_status"], "actual_result": domain["actual_result"],
        "checks": domain["domain_checks"],
    }}, "unsupported": {**result.get("unsupported", {}),
                        "original_business_math": domain.get("unsupported", {})}}
    return _domain(combined, verified, "recovered", fault_contract_status="verified")


_RECOVERY_BINDINGS = {
    "labeling": ("labeling", "FrozenLabelingBinding"),
    "feature": ("feature", "FrozenFeatureBinding"),
    "risk_portfolio": ("portfolio", "FrozenPortfolioBinding"),
    "modeling": ("modeling", "FrozenModelingBinding"),
    "strategy": ("strategy", "FrozenStrategyBinding"),
    "monitoring": ("monitoring", "FrozenMonitoringBinding"),
}



def _monitoring_recovery(case, expected, binding, result, archive, item, domain_item, root, record, run):
    from .formal_acceptance import _domain as family_domain
    from .runtime_archive_monitoring import FrozenMonitoringBinding

    _require(expected.result == "done" and case.expected_result == "recovered"
             and case.fault_contract_sha256 == digest(future_fault_contract(binding.case)),
             "formal_monitoring_recovery_scope_mismatch")
    if result["values"]["recovery_completed"] is False:
        return _domain(result, False, "recovered", fault_contract_status="verified")
    if domain_item is None:
        raise TrustInputError("formal_monitoring_recovery_math_unavailable")
    business_archive, business, _ = _original_case_binding(domain_item, root, record, run, FrozenMonitoringBinding)
    _require(business_archive == archive and domain_item["manifest_sha256"] == item["manifest_sha256"]
             and digest(business.case.model_dump()) == case.case_sha256
             and business.plan_id == result["values"]["monitoring_plan_id"]
             and business.step_ids["modeling.score_dataset"] == result["values"]["score_step_id"],
             "formal_monitoring_recovery_business_binding_mismatch")
    domain, reason = family_domain(case.model_copy(update={"cohort": "normal", "fault_contract_sha256": None}),
                                  expected, domain_item, root, record, run)
    if domain is None:
        raise TrustInputError(reason or "formal_monitoring_recovery_math_unavailable")
    combined = {**result, "checks": {**result["checks"], "original_business_math": {
        "status": domain["domain_status"], "actual_result": domain["actual_result"],
        "checks": domain["domain_checks"]}}, "unsupported": {**result.get("unsupported", {}),
        "original_business_math": domain.get("unsupported", {})}}
    return _domain(combined, domain["domain_status"] == "verified" and domain["actual_result"] == "done",
                   "recovered", fault_contract_status="verified")


def _family_recovery(case, expected, binding, observed, result, archive, item, domain_item, root, record, run):
    """Fault/readback closure plus the family's own installed normal adjudication."""
    import importlib

    from .formal_acceptance import _domain as family_domain

    _require(case.family in _RECOVERY_BINDINGS and expected.result == "done" and case.expected_result == "recovered",
             "formal_fixed_recovery_business_scope_unavailable")
    _require(bool(binding.recovery_action_indices), "formal_fixed_recovery_target_unavailable")
    _require(case.fault_contract_sha256 == digest(future_fault_contract(binding.case)), "formal_frozen_fault_contract_mismatch")
    if domain_item is None:
        raise TrustInputError("formal_recovery_original_family_math_unavailable")
    suffix, type_name = _RECOVERY_BINDINGS[case.family]
    module = importlib.import_module(f".runtime_archive_{suffix}", package=__package__)
    business_archive, business, _ = _original_case_binding(domain_item, root, record, run, getattr(module, type_name))
    _require(business_archive == archive and domain_item["manifest_sha256"] == item["manifest_sha256"]
             and digest(business.case.model_dump()) == case.case_sha256,
             "formal_recovery_business_archive_mismatch")
    public_steps = {row.get("id"): row for row in observed.get("execution", {}).get("steps", ())}
    for recovered in result["values"]["recoveries"]:
        governed = next((row for row in binding.governed_actions if row.action_index == recovered.get("governed_action_index", recovered["fault_action_index"])), None)
        tool = binding.case.actions[recovered["fault_action_index"] - 1].tool
        _require(governed is not None and governed.plan_id == business.plan_id
                 and governed.step_id in business.step_ids.values()
                 and public_steps.get(governed.step_id, {}).get("tool") == tool,
                 "formal_recovery_business_target_mismatch")
    # The family's normal adjudication is the business-math oracle; the fault
    # contract and readback above are what make the attempt a recovery.
    normal = case.model_copy(update={"cohort": "normal", "fault_contract_sha256": None})
    domain, reason = family_domain(normal, expected, domain_item, root, record, run)
    if domain is None:
        raise TrustInputError(reason or "formal_recovery_original_family_math_unavailable")
    checks = dict(domain.pop("domain_checks", {}))
    domain.pop("unsupported", None)
    verified = (domain["domain_status"] == "verified" and domain["actual_result"] == "done"
                and result.get("values", {}).get("recovery_completed", True) is True)
    return _domain(result, verified, "recovered", fault_contract_status="verified",
                   extra_checks=checks)


def _latest_metadata_value(metadata, path):
    value = metadata
    for part in path:
        if not isinstance(value, dict) or part not in value:
            raise TrustInputError("formal_clarification_predicate_unavailable")
        value = value[part]
    return value


def _http_answer(observed, assertion):
    codes = [row.get("status_code") for row in observed.get("http_events", ()) if row.get("stage") == assertion.stage]
    return bool(codes) and type(assertion.value) is int and all(code == assertion.value for code in codes)


def _clarification(case, expected, observed, result):
    """Expected clarification: native zero effect plus the native assistant reply."""
    if result["status"] != "verified":
        return _domain(result, False, None)
    _require(expected.result == case.expected_result == "clarification", "formal_fixed_clarification_expected_unavailable")
    metadata = result["values"]["latest_assistant_metadata"]
    answers, asked = [], False
    for assertion in expected.assertions:
        if assertion.kind == "latest_assistant_metadata":
            actual = _latest_metadata_value(metadata, assertion.path)
            answers.append(type(actual) is type(assertion.value) and actual == assertion.value)
            asked = True
        elif assertion.kind == "http_status":
            answers.append(_http_answer(observed, assertion))
        else:
            raise TrustInputError("formal_fixed_clarification_predicate_unavailable")
    _require(asked, "formal_fixed_clarification_outcome_closure_unavailable")
    return _domain(result, all(answers), "clarification")


def _human_rejection(expected, binding, observed, result):
    tools = {binding.case.actions[i - 1].tool for i in binding.boundary_action_indices}
    answers, http, zero = [], False, False
    refused = {row["step_id"] for row in result["values"]["boundaries"]}
    for assertion in expected.assertions:
        if assertion.kind == "http_status" and assertion.stage == "human_rejection":
            answers.append(_http_answer(observed, assertion))
            http = True
        elif assertion.kind == "tool_not_executed" and assertion.tool in tools:
            steps = [row for row in observed.get("execution", {}).get("steps", ()) if row.get("tool") == assertion.tool]
            # Zero execution is the reader's native fact; public rows only bind identity.
            answers.append(bool(steps) and all(row.get("id") in refused for row in steps))
            zero = True
        else:
            raise TrustInputError("formal_fixed_boundary_predicate_unavailable")
    _require(http and zero, "formal_fixed_boundary_outcome_closure_unavailable")
    return _domain(result, bool(answers) and all(answers), "rejected")


def _extension_negative(case, expected, observed, result):
    if result["status"] != "verified":
        return _domain(result, False, None)
    _require(expected.result == case.expected_result == "rejected", "formal_extension_negative_expected_unavailable")
    statuses = {row["http_status"] for row in result["values"]["refusals"]}
    answers = []
    for assertion in expected.assertions:
        if assertion.kind == "http_status":
            answers.append(_http_answer(observed, assertion) and assertion.value in statuses)
        elif assertion.kind == "extension_value" and not assertion.tool and assertion.path == ["status"]:
            answers.append(assertion.value in {"rejected", "http_rejected"})
        else:
            raise TrustInputError("formal_extension_negative_predicate_unavailable")
    _require(bool(answers), "formal_extension_negative_outcome_closure_unavailable")
    return _domain(result, all(answers), "rejected")
