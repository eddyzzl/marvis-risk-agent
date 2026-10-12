"""Installed adjudication for retained modeling, policy and extension carriers.

Only independently recomputed values enter predicates. Unknown predicates and
missing original bindings remain unavailable; public runtime success flags never
substitute for the installed reader or for independent safety adjudication.
"""

from __future__ import annotations

import importlib
import math

from .acceptance_trust import TrustInputError, _original_case_binding, _require
from .runtime_contracts import digest


_READERS = {
    "modeling": ("modeling", "FrozenModelingBinding"),
    "strategy": ("strategy", "FrozenStrategyBinding"),
    "monitoring": ("monitoring", "FrozenMonitoringBinding"),
    "online_decision": ("online", "FrozenOnlineBinding"),
    "connector_antifraud": ("connector", "FrozenConnectorBinding"),
    "collections": ("collections", "FrozenCollectionsBinding"),
}


def _final_native_binding(binding, observed, *, native_tools):
    plans = [p for p in observed.get("execution", {}).get("plans", [])
             if p.get("id") == binding.plan_id]
    _require(len(plans) == 1 and plans[0].get("status") == "done",
             "final_domain_plan_binding_mismatch")
    tools = {}
    for key, identity in binding.step_ids.items():
        rows = [s for s in observed.get("execution", {}).get("steps", [])
                if s.get("id") == identity]
        _require(len(rows) == 1 and rows[0].get("status") == "done"
                 and rows[0].get("output_sha256") == binding.output_sha256[key]
                 and rows[0].get("tool") == native_tools.get(identity)
                 and isinstance(native_tools.get(identity), str),
                 "final_domain_step_binding_mismatch")
        tools[identity] = rows[0]["tool"]
    current = frozenset(tools.values())
    for name in ("modeling", "sample_design"):
        child = getattr(binding, name, None)
        if child is not None:
            current |= _final_native_binding(child, observed, native_tools=native_tools)
    return current


def _predicate_value(assertion, values):
    try:
        actual = values
        for part in assertion.path:
            if isinstance(part, bool):
                raise TypeError
            actual = actual[part]
    except (KeyError, TypeError, IndexError):
        raise TrustInputError("formal_domain_predicate_unavailable") from None
    if assertion.kind == "output_equals" or assertion.kind == "extension_value":
        return type(actual) is type(assertion.value) and actual == assertion.value
    if assertion.kind == "output_length":
        return isinstance(actual, (list, dict)) and type(assertion.value) is int and len(actual) == assertion.value
    if assertion.kind == "output_close":
        return (type(actual) in {int, float} and type(assertion.value) in {int, float}
                and math.isfinite(actual) and math.isfinite(assertion.value)
                and abs(actual - assertion.value) <= assertion.tolerance)
    raise TrustInputError("formal_domain_predicate_unavailable")


def independently_expected(expected, observed, recomputed, *, family, tools=frozenset(), native_facts=None):
    if not recomputed["supported_checks_verified"]:
        return False
    values, answers = recomputed.get("values", {}), []
    native_facts = native_facts or {"values_by_tool": {}, "artifacts": {}}
    for assertion in expected.assertions:
        if assertion.kind == "http_status":
            codes = [e.get("status_code") for e in observed.get("http_events", [])
                     if e.get("stage") == assertion.stage]
            answers.append(bool(codes) and all(type(c) is type(assertion.value) and c == assertion.value for c in codes))
        elif assertion.kind == "tool_succeeded" and assertion.tool in tools:
            answers.append(recomputed["checks"]["native_steps"]["status"] == "verified")
        elif assertion.kind == "artifact_exists" and assertion.tool in native_facts["artifacts"]:
            answers.append(len(assertion.path) == 1 and assertion.path[0] in native_facts["artifacts"][assertion.tool])
        elif assertion.kind == "dataset_rows" and assertion.tool == "modeling.score_dataset":
            answers.append(type(assertion.value) is int and native_facts["values_by_tool"].get(assertion.tool, {}).get("dataset_rows") == assertion.value)
        elif assertion.tool in native_facts["values_by_tool"]:
            answers.append(_predicate_value(assertion, native_facts["values_by_tool"][assertion.tool]))
        elif family == "modeling" and assertion.tool == "modeling.select_experiment":
            # Refit is established by the authenticated child/parent closure;
            # absent test/OOT metrics remain None rather than a fabricated score.
            projected = {"metrics": values["metrics"], "selection_evidence": values["selection_evidence"],
                         "refit": {"applied": values["scope"]["fit_scope"] == "full_development"}}
            answers.append(_predicate_value(assertion, projected))
        elif family == "strategy" and assertion.tool == "strategy.backtest_strategy":
            answers.append(_predicate_value(assertion, values))
        elif family == "monitoring" and assertion.tool == "modeling.monitor_run":
            if assertion.kind == "monitoring_evidence":
                # This describes the public entry only, not independent metrics.
                monitoring = observed.get("execution", {}).get("monitoring", {})
                answers.append(monitoring.get("runtime_entry") == assertion.value
                               and any(e.get("stage") == "human_monitoring_plan_start" and e.get("status_code") == 202
                                       for e in observed.get("http_events", [])))
            else:
                answers.append(_predicate_value(assertion, values))
        elif family in {"online_decision", "connector_antifraud", "collections"} and assertion.kind == "extension_value" and not assertion.tool:
            answers.append(_predicate_value(assertion, values))
        else:
            raise TrustInputError("formal_domain_predicate_unavailable")
    return expected.result == "done" and bool(answers) and all(answers)


def retained_domain(case, expected, item, root, record, run):
    if case.family not in _READERS:
        return None, "deterministic_family_revalidator_not_implemented"
    cohort = "extension" if case.family in {"online_decision", "connector_antifraud", "collections"} else "normal"
    if case.cohort != cohort:
        return None, "deterministic_boundary_or_recovery_revalidator_not_implemented"
    if item is None:
        return None, "domain_original_evidence_unavailable"
    suffix, type_name = _READERS[case.family]
    module = importlib.import_module(f".runtime_archive_{suffix}", package=__package__)
    _require(item["case_id"] == case.id, "formal_domain_case_binding_mismatch")
    archive, binding, observed = _original_case_binding(item, root, record, run, getattr(module, type_name))
    _require(digest(binding.case.model_dump()) == case.case_sha256, "formal_domain_definition_mismatch")
    tools = frozenset()
    recomputed = getattr(module, f"revalidate_{suffix}_archive")(
        archive, expected_manifest_sha256=item["manifest_sha256"], frozen_binding=binding)
    native_facts = None
    if cohort == "normal" and recomputed["supported_checks_verified"]:
        model_values = recomputed["values"] if case.family == "modeling" else None
        if case.family == "monitoring":
            from .runtime_archive_modeling import revalidate_modeling_archive
            model_result = revalidate_modeling_archive(archive, expected_manifest_sha256=item["manifest_sha256"],
                                                       frozen_binding=binding.modeling)
            _require(model_result["supported_checks_verified"], "formal_monitoring_model_predicates_unavailable")
            model_values = model_result["values"]
        from .formal_native_predicates import native_predicates
        native_facts = native_predicates(archive, item["manifest_sha256"], binding,
                                        family=case.family, model_values=model_values)
        tools = _final_native_binding(binding, observed, native_tools=native_facts["tools_by_step"])
    from .formal_acceptance import _domain_result
    return _domain_result(recomputed, independently_expected(expected, observed, recomputed,
                                                           family=case.family, tools=tools, native_facts=native_facts)), None
