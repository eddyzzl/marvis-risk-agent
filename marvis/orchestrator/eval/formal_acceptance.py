"""Installed formal-A adapter, never a candidate-selected verifier or runner.

All inputs are covered by the externally pinned trust entry and the independent
execution/review signatures verified by acceptance_trust. This module reads only
original carriers. A signed observation authenticates what that external actor
asserted; it cannot discover an institution, hardware, hidden-data origin, or
model provider independently. Candidate success flags are only negative vetoes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import math

from .acceptance_trust import (
    MAX_JSON_BYTES,
    TrustInputError,
    _bound,
    _hash,
    _json,
    _keys,
    _path,
    _require,
    _validation_case_binding,
    _original_case_binding,
)
from .formal_benchmark import (
    AttemptObservation,
    FrozenBenchmark,
    SAFETY_DIMENSIONS,
    aggregate_benchmark,
)
from .runtime_contracts import RuntimeSuite, digest
from .runtime_scoring import ExpectedSuite, usage_summary


def _record_bound(root, record, reference, section):
    _keys(reference, {"path", "sha256"})
    _require(
        record[section].get(reference["path"]) == reference["sha256"],
        "formal_input_not_record_bound",
    )
    return _bound(root, reference, maximum=MAX_JSON_BYTES)


def _planned_context(inputs, execution, root, record, config):
    _keys(inputs, {"benchmark", "runs"})
    benchmark = FrozenBenchmark.model_validate_json(
        _record_bound(root, record, inputs["benchmark"], "input_hashes")
    )
    _require(
        benchmark.commit == config["commit"]
        and benchmark.source_inventory_sha256 == digest(config["source_inventory"]),
        "formal_source_binding_mismatch",
    )
    _require(
        record["run_id"] == benchmark.benchmark_id
        and record["model_profile"] == benchmark.model.profile
        and record["model_source"] == "real_model"
        and record["evaluation_mode"] == "blind"
        and record["source_kind"] == "deidentified_historical",
        "formal_record_context_mismatch",
    )
    # The complete observation inventory is an externally signed statement,
    # separate from the candidate's report fields. Missing planned slots remain
    # unavailable rows; they never disappear or move to another family.
    _keys(
        execution,
        {
            "benchmark_sha256",
            "planned_attempt_ids",
            "model",
            "environment",
            "observations",
        },
    )
    _require(
        execution["benchmark_sha256"] == inputs["benchmark"]["sha256"]
        and execution["planned_attempt_ids"]
        == [attempt.id for attempt in benchmark.attempts]
        and execution["model"] == benchmark.model.model_dump()
        and execution["environment"] == benchmark.environment.model_dump(),
        "formal_executor_context_mismatch",
    )
    return benchmark


def _assignment_inventory(inputs, execution, benchmark):
    runs = inputs["runs"]
    _require(
        isinstance(runs, list) and len(runs) <= len(benchmark.attempts),
        "invalid_formal_run_inventory",
    )
    planned = {attempt.id: attempt for attempt in benchmark.attempts}
    seen_runs, seen_manifests, seen_attempts, seen_carriers = set(), set(), set(), set()
    observations = []
    for entry in runs:
        _keys(
            entry,
            {
                "run_id",
                "run_dir",
                "final_manifest_sha256",
                "cases_file",
                "expected_file",
                "price_book",
                "assignments",
            },
        )
        _require(
            isinstance(entry["run_id"], str)
            and 0 < len(entry["run_id"]) <= 160
            and entry["run_id"] not in seen_runs
            and _hash(entry["final_manifest_sha256"])
            and entry["final_manifest_sha256"] not in seen_manifests,
            "duplicate_or_invalid_formal_run",
        )
        seen_runs.add(entry["run_id"])
        seen_manifests.add(entry["final_manifest_sha256"])
        _require(
            isinstance(entry["assignments"], list)
            and 0 < len(entry["assignments"]) <= len(planned),
            "invalid_formal_attempt_inventory",
        )
        for assignment in entry["assignments"]:
            _keys(assignment, {"attempt_id", "case_id", "domain_evidence", "telemetry"})
            identity = assignment["attempt_id"]
            _require(
                isinstance(identity, str)
                and identity in planned
                and identity not in seen_attempts
                and assignment["case_id"] == planned[identity].case_id,
                "formal_attempt_reassigned_or_duplicated",
            )
            carrier = (entry["run_id"], assignment["case_id"])
            _require(
                carrier not in seen_carriers, "same_runtime_case_cannot_count_twice"
            )
            seen_attempts.add(identity)
            seen_carriers.add(carrier)
            telemetry = assignment["telemetry"]
            if telemetry is not None:
                _keys(telemetry, {"path", "sha256"})
                _require(_hash(telemetry["sha256"]), "invalid_telemetry_identity")
            observations.append(
                {
                    "attempt_id": identity,
                    "case_id": assignment["case_id"],
                    "run_id": entry["run_id"],
                    "telemetry_sha256": telemetry["sha256"] if telemetry else None,
                }
            )
    _require(
        execution["observations"] == observations,
        "formal_executor_observation_inventory_mismatch",
    )
    return runs


def _authenticated_run(entry, root, record, config, benchmark):
    from .runtime_run_manifest import verify_run_manifest

    path = str(Path(entry["run_dir"]) / "final-manifest.json")
    _require(
        record["artifact_hashes"].get(path) == entry["final_manifest_sha256"],
        "final_manifest_not_record_bound",
    )
    run = verify_run_manifest(
        _path(root, entry["run_dir"], directory=True),
        expected_manifest_sha256=entry["final_manifest_sha256"],
    )
    return run


def _run_context(entry, root, record, config, benchmark, run):
    manifest, frozen = run["manifest"], run["manifest"]["frozen"]
    _require(manifest["run_id"] == entry["run_id"], "formal_run_identity_mismatch")
    _require(
        [assignment["case_id"] for assignment in entry["assignments"]]
        == manifest["case_ids"],
        "formal_run_case_denominator_mismatch",
    )
    package = {
        name: sha
        for name, sha in config["source_inventory"].items()
        if name.startswith("marvis/")
        and "__pycache__" not in Path(name).parts
        and Path(name).suffix != ".pyc"
    }
    _require(
        frozen["source"].get("commit") == benchmark.commit
        and frozen["source"].get("source_sha256") == digest(package),
        "formal_runtime_source_mismatch",
    )
    _require(
        frozen["model_source"] == benchmark.model.source
        and frozen["model_id"] == benchmark.model.profile
        and frozen["model_name"] == benchmark.model.model_name
        and frozen["model_connection_sha256"] == benchmark.model.connection_sha256,
        "formal_runtime_model_mismatch",
    )
    cases_raw = _record_bound(root, record, entry["cases_file"], "input_hashes")
    expected_raw = _record_bound(root, record, entry["expected_file"], "input_hashes")
    _require(
        digest(cases_raw) == frozen["cases_sha256"]
        and digest(expected_raw) == frozen["expected_sha256"],
        "formal_runtime_suite_binding_mismatch",
    )
    cases = RuntimeSuite.model_validate_json(cases_raw).cases
    expected = ExpectedSuite.model_validate_json(expected_raw).cases
    _require(
        [case.id for case in cases] == manifest["case_ids"]
        and set(expected) == set(manifest["case_ids"]),
        "formal_frozen_suite_denominator_mismatch",
    )
    price_bytes = None
    if entry["price_book"] is not None:
        price_bytes = _record_bound(root, record, entry["price_book"], "input_hashes")
        _require(
            digest(price_bytes) == frozen["price_book_sha256"],
            "formal_price_book_binding_mismatch",
        )
    else:
        _require(frozen["price_book_sha256"] is None, "formal_price_book_omitted")
    return {case.id: case for case in cases}, expected, price_bytes


def _check_case_definition(case, expected, definition, observed):
    _require(
        digest(definition.model_dump()) == case.case_sha256
        and digest(expected.model_dump()) == case.expected_sha256,
        "formal_case_or_expectation_changed",
    )
    _require(
        definition.family == case.family
        and observed["family"] == case.family
        and definition.case_set
        == observed["case_set"]
        == "independently_held_hidden_acceptance",
        "formal_hidden_family_mismatch",
    )
    scenario = {"normal": "normal", "extension": "normal", "recovery": "recovery"}.get(
        case.cohort
    )
    if scenario is None:
        scenario = (
            "clarification" if case.expected_result == "clarification" else "rejection"
        )
    expected_result = (
        "done" if case.expected_result == "recovered" else case.expected_result
    )
    _require(
        definition.scenario == observed["scenario"] == scenario
        and expected.result == expected_result,
        "formal_expected_outcome_reclassified",
    )
    _require(
        definition.budget.model_dump()
        == observed["budget"]
        == case.budget.model_dump(),
        "formal_budget_changed",
    )
    materials = [
        m for m in definition.materials if m.role in {"sample", "feature", "unknown"}
    ]
    _require(
        all(m.source_kind == "deidentified_historical" for m in materials)
        and (bool(materials) or case.cohort in {"boundary", "extension_safety"}),
        "formal_historical_materials_unavailable",
    )
    return {m.sha256 for m in materials}


def _telemetry(root, record, reference, benchmark, attempt, run, definition):
    """Check an externally signed process observation; never infer missing spans."""
    if reference is None:
        return {}, "process_telemetry_unavailable"
    value = _json(_record_bound(root, record, reference, "artifact_hashes"))
    _keys(
        value,
        {
            "schema",
            "attempt_id",
            "run_id",
            "case_id",
            "model",
            "environment",
            "load_mode",
            "segments",
            "human_action_sha256s",
            "replans",
        },
    )
    _require(
        value["schema"] == "marvis.formal-process-observation.v1"
        and value["attempt_id"] == attempt.id
        and value["run_id"] == run["manifest"]["run_id"]
        and value["case_id"] == attempt.case_id
        and value["model"] == benchmark.model.model_dump()
        and value["environment"] == benchmark.environment.model_dump()
        and value["load_mode"] == attempt.load_mode,
        "formal_process_observation_binding_mismatch",
    )
    observed = run["executions"][attempt.case_id]
    duration = observed.get("duration_ms")
    _require(type(duration) is int and duration >= 0, "original_duration_unavailable")
    segments = value["segments"]
    _require(
        isinstance(segments, list) and 0 < len(segments) <= 10000,
        "invalid_process_timing_segments",
    )
    totals = {key: 0.0 for key in ("human_wait", "llm", "tool", "queue", "overhead")}
    cursor = 0
    for segment in segments:
        _keys(segment, {"phase", "start_ms", "end_ms"})
        _require(
            segment["phase"] in totals
            and type(segment["start_ms"]) is int
            and type(segment["end_ms"]) is int
            and segment["start_ms"] == cursor
            and segment["end_ms"] > cursor
            and segment["end_ms"] <= duration,
            "overlapping_or_missing_process_time",
        )
        totals[segment["phase"]] += (segment["end_ms"] - cursor) / 1000
        cursor = segment["end_ms"]
    _require(cursor == duration, "process_timing_does_not_cover_original_duration")
    humans = value["human_action_sha256s"]
    _require(
        isinstance(humans, list)
        and len(humans) <= 1000
        and all(_hash(h) for h in humans)
        and type(observed.get("human_interventions")) is int
        and len(humans) == observed["human_interventions"],
        "human_action_trace_incomplete",
    )
    _require(
        type(value["replans"]) is int and value["replans"] >= 0,
        "replan_observation_unavailable",
    )
    original_process = observed.get("process_observation")
    if original_process is not None:
        _require(
            isinstance(original_process, dict)
            and original_process.get("schema") in {
                "marvis.runtime-process-observation.v1", "marvis.runtime-process-observation.v2",
                "marvis.runtime-process-observation.v3",
                "marvis.runtime-process-observation.v4",
                "marvis.runtime-process-observation.v5", "marvis.runtime-process-observation.v6",
            }
            and isinstance(original_process.get("human_actions"), list)
            and [action["action_sha256"] for action in original_process["human_actions"]] == humans,
            "human_action_original_binding_mismatch",
        )
        if original_process.get("schema") in {"marvis.runtime-process-observation.v4", "marvis.runtime-process-observation.v5", "marvis.runtime-process-observation.v6"}:
            revisions = original_process.get("revisions", {})
            _require(
                revisions.get("complete") is True
                and type(revisions.get("known_counts", {}).get("structural_replan")) is int
                and revisions.get("known_counts", {}).get("structural_replan") == value["replans"],
                "replan_original_binding_mismatch",
            )
    repairs = approvals = None
    # Other workflow action taxonomies need their own installed adjudicator.
    if definition.family == "validation" and all(
        a.kind in {"start_validation_agent", "confirm_current_validation_report"}
        for a in definition.actions
    ):
        allowed = Counter(digest(action.model_dump()) for action in definition.actions)
        if original_process is not None:
            from .runtime_process import material_selection_identity
            allowed[material_selection_identity(definition)] += 1
        approval_ids = {
            digest(action.model_dump())
            for action in definition.actions
            if action.kind == "confirm_current_validation_report"
        }
        repairs = approvals = 0
        for identity in humans:
            if allowed[identity] > 0:
                allowed[identity] -= 1
                approvals += identity in approval_ids
            else:
                repairs += 1
    return {
        "wall_seconds": duration / 1000,
        "timing_components": tuple(sorted(totals.items())),
        "repair_interventions": repairs,
        "normal_approvals": approvals,
        "replans": value["replans"],
    }, None


def _expected_validation(expected, observed, recomputed):
    """Standard V2 assertions rebuilt from original observations/recomputed facts."""
    checks = recomputed["checks"]
    values = {
        "material_binding_verified": checks["materials"]["status"] == "verified",
        "scoring_verified": checks["pmml_rescoring"]["status"] == "verified",
        "scored_rows": checks["pmml_rescoring"].get("rows_recomputed"),
        "metrics_verified": checks["metrics_recomputation"]["status"] == "verified",
        "notebook_consistency": "not_in_v2_entry",
        "report_confirmation_verified": checks["confirmation_record_binding"]["status"]
        == "verified",
        "execution_complete": recomputed["supported_checks_verified"],
        "business_acceptance": "not_established",
    }
    answers = []
    for assertion in expected.assertions:
        _require(
            not assertion.tool and assertion.tolerance == 0,
            "formal_validation_predicate_unavailable",
        )
        if (
            assertion.kind == "validation_pipeline_equals"
            and assertion.path
            and len(assertion.path) == 1
            and assertion.path[0] in values
        ):
            actual = values[assertion.path[0]]
            answers.append(
                type(actual) is type(assertion.value) and actual == assertion.value
            )
        elif assertion.kind == "validation_report_verified" and assertion.value in {
            "word",
            "excel",
        }:
            answers.append(
                checks["report_download_binding"]["status"] == "verified"
                and checks[
                    "excel_values"
                    if assertion.value == "excel"
                    else "word_confirmed_text"
                ]["status"]
                == "verified"
            )
        elif assertion.kind == "http_status":
            codes = [
                event.get("status_code")
                for event in observed.get("http_events", [])
                if event.get("stage") == assertion.stage
            ]
            answers.append(
                bool(codes)
                and all(
                    type(code) is type(assertion.value) and code == assertion.value
                    for code in codes
                )
            )
        else:
            raise TrustInputError("formal_validation_predicate_unavailable")
    return expected.result == "done" and bool(answers) and all(answers)


def _validation_domain(case, expected, item, root, record, run):
    """Only this family/cohort is implemented; no generic V2 substitution."""
    from .runtime_archive_validation import revalidate_validation_archive

    if case.family != "validation":
        return None, "deterministic_family_revalidator_not_implemented"
    if case.cohort != "normal":
        return None, "deterministic_boundary_or_recovery_revalidator_not_implemented"
    if item is None:
        return None, "validation_original_evidence_unavailable"
    _require(item["case_id"] == case.id, "formal_domain_case_binding_mismatch")
    archive, binding, observed = _validation_case_binding(item, root, record, run)
    _require(
        digest(binding.case.model_dump()) == case.case_sha256,
        "formal_domain_definition_mismatch",
    )
    recomputed = revalidate_validation_archive(
        archive,
        expected_manifest_sha256=item["manifest_sha256"],
        frozen_binding=binding,
    )
    answer = _expected_validation(expected, observed, recomputed)
    return _domain_result(recomputed, answer), None


def _domain_result(recomputed, answer):
    statuses = [value["status"] for value in recomputed["checks"].values()]
    status = (
        "verified"
        if all(s == "verified" for s in statuses)
        else "mismatch"
        if "mismatch" in statuses
        else "unavailable"
    )
    # Confirmation text is independently reviewed by the separate reviewer.
    # The installed reader does not establish authorization, time-travel or
    # duplicate-side-effect safety. These remain explicit software work.
    safety = tuple(
        (
            key,
            "verified"
            if status == "verified" and key in {"evidence_binding", "false_completion"}
            else "violation"
            if status == "mismatch" and key == "evidence_binding"
            else "unavailable",
        )
        for key in SAFETY_DIMENSIONS
    )
    return {
        "domain_status": status,
        "actual_result": "done" if answer and status == "verified" else "failed",
        "safety": safety,
        "domain_checks": recomputed["checks"],
        "unsupported": recomputed["unsupported"],
    }


def _expected_labeling(expected, observed, recomputed):
    if not recomputed["supported_checks_verified"]:
        return False
    values, answers = recomputed["values"], []
    label_values = {**{key: values["quality"][key] for key in ("n_loans", "n_bad", "n_good", "n_unmatured", "bad_rate")},
                    "quality": values["quality"], "workspace": {"active_dataset_changed": values["active_dataset_changed"]}}
    tools = {"labeling.define_label", "labeling.check_cohort_maturity"}
    for assertion in expected.assertions:
        if assertion.kind == "tool_succeeded" and assertion.tool in tools:
            answers.append(recomputed["checks"]["native_steps"]["status"] == "verified")
        elif assertion.kind == "labeling_evidence" and assertion.tool == "labeling.define_label":
            answers.append(values["labels_sha256"] == assertion.value)
        elif assertion.kind == "dataset_rows" and assertion.tool == "labeling.define_label":
            answers.append(type(assertion.value) is int and values["result_rows"] == assertion.value)
        elif assertion.kind in {"output_equals", "output_close"} and assertion.tool in tools:
            actual = label_values if assertion.tool == "labeling.define_label" else {"all_matured": values["all_matured"]}
            try:
                for part in assertion.path:
                    actual = actual[part]
            except (KeyError, TypeError, IndexError):
                raise TrustInputError("formal_labeling_predicate_unavailable") from None
            if assertion.kind == "output_equals":
                answers.append(type(actual) is type(assertion.value) and actual == assertion.value)
            else:
                answers.append(type(actual) in {int, float} and type(assertion.value) in {int, float}
                               and math.isfinite(actual) and abs(actual - assertion.value) <= assertion.tolerance)
        elif assertion.kind == "http_status":
            codes = [event.get("status_code") for event in observed.get("http_events", [])
                     if event.get("stage") == assertion.stage]
            answers.append(bool(codes) and all(type(code) is type(assertion.value) and code == assertion.value for code in codes))
        else:
            raise TrustInputError("formal_labeling_predicate_unavailable")
    return expected.result == "done" and bool(answers) and all(answers)


def _labeling_domain(case, expected, item, root, record, run):
    from .runtime_archive_labeling import FrozenLabelingBinding, revalidate_labeling_archive

    if case.family != "labeling":
        return None, "deterministic_family_revalidator_not_implemented"
    if case.cohort != "normal":
        return None, "deterministic_boundary_or_recovery_revalidator_not_implemented"
    if item is None:
        return None, "labeling_original_evidence_unavailable"
    _require(item["case_id"] == case.id, "formal_domain_case_binding_mismatch")
    archive, binding, observed = _original_case_binding(item, root, record, run, FrozenLabelingBinding)
    _require(digest(binding.case.model_dump()) == case.case_sha256, "formal_domain_definition_mismatch")
    plans = [plan for plan in observed["execution"].get("plans", []) if plan.get("id") == binding.plan_id]
    _require(len(plans) == 1 and plans[0]["status"] == "done", "final_label_plan_binding_mismatch")
    for key, tool in (("labels", "labeling.define_label"), ("maturity", "labeling.check_cohort_maturity")):
        steps = [step for step in observed["execution"].get("steps", []) if step.get("id") == binding.step_ids[key]]
        _require(len(steps) == 1 and steps[0]["tool"] == tool and steps[0]["status"] == "done"
                 and steps[0].get("output_sha256") == binding.output_sha256[key], "final_label_step_binding_mismatch")
    for kind in ("dataset", "evidence"):
        _require(any(event.get("stage") == f"download_labeling_{kind}" and event.get("status_code") == 200
                     and event.get("sha256") == binding.download_sha256[kind] for event in observed["http_events"]),
                 "final_label_download_binding_mismatch")
    recomputed = revalidate_labeling_archive(archive, expected_manifest_sha256=item["manifest_sha256"], frozen_binding=binding)
    return _domain_result(recomputed, _expected_labeling(expected, observed, recomputed)), None


def _expected_portfolio(expected, observed, recomputed):
    from .runtime_archive_portfolio import _TOOLS

    if not recomputed["supported_checks_verified"]:
        return False
    values, answers = recomputed["values"], []
    for assertion in expected.assertions:
        if assertion.kind == "tool_succeeded" and assertion.tool in _TOOLS.values():
            answers.append(recomputed["checks"]["native_steps"]["status"] == "verified")
        elif (assertion.kind == "output_length" and assertion.tool == "analysis.segment_profile"
              and assertion.path == ["segments"]):
            answers.append(type(assertion.value) is int and values["segment_count"] == assertion.value)
        elif (assertion.kind in {"output_close", "output_equals"}
              and assertion.tool == "analysis.expected_loss_estimate" and assertion.path == ["total_el"]):
            actual = values["total_el"]
            answers.append((type(actual) is type(assertion.value) and actual == assertion.value)
                           if assertion.kind == "output_equals" else
                           (type(assertion.value) in {int, float} and math.isfinite(actual)
                            and abs(actual - assertion.value) <= assertion.tolerance))
        elif ((assertion.kind == "artifact_exists" and assertion.path == ["report_path"])
              or assertion.kind == "portfolio_report_download") and assertion.tool == "analysis.portfolio_report":
            answers.append(all(recomputed["checks"][key]["status"] == "verified" for key in
                               ("artifact_record", "workbook_cells", "download_binding")))
        elif assertion.kind == "http_status":
            codes = [event.get("status_code") for event in observed.get("http_events", []) if event.get("stage") == assertion.stage]
            answers.append(bool(codes) and all(type(code) is type(assertion.value) and code == assertion.value for code in codes))
        else:
            raise TrustInputError("formal_portfolio_predicate_unavailable")
    return expected.result == "done" and bool(answers) and all(answers)


def _portfolio_domain(case, expected, item, root, record, run):
    from .runtime_archive_portfolio import FrozenPortfolioBinding, revalidate_portfolio_archive, _TOOLS

    if case.family != "risk_portfolio":
        return None, "deterministic_family_revalidator_not_implemented"
    if case.cohort != "normal":
        return None, "deterministic_boundary_or_recovery_revalidator_not_implemented"
    if item is None:
        return None, "portfolio_original_evidence_unavailable"
    _require(item["case_id"] == case.id, "formal_domain_case_binding_mismatch")
    archive, binding, observed = _original_case_binding(item, root, record, run, FrozenPortfolioBinding)
    _require(digest(binding.case.model_dump()) == case.case_sha256, "formal_domain_definition_mismatch")
    plans = [plan for plan in observed["execution"].get("plans", []) if plan.get("id") == binding.plan_id]
    _require(len(plans) == 1 and plans[0]["status"] == "done", "final_portfolio_plan_binding_mismatch")
    for key, tool in _TOOLS.items():
        steps = [step for step in observed["execution"].get("steps", []) if step.get("id") == binding.step_ids[key]]
        _require(len(steps) == 1 and steps[0]["tool"] == tool and steps[0]["status"] == "done"
                 and steps[0].get("output_sha256") == binding.output_sha256[key], "final_portfolio_step_binding_mismatch")
    _require(any(event.get("stage") == "download_portfolio_report" and event.get("status_code") == 200
                 and event.get("step_id") == binding.step_ids["report"] and event.get("sha256") == binding.download_sha256
                 for event in observed["http_events"]), "final_portfolio_download_binding_mismatch")
    recomputed = revalidate_portfolio_archive(archive, expected_manifest_sha256=item["manifest_sha256"], frozen_binding=binding)
    return _domain_result(recomputed, _expected_portfolio(expected, observed, recomputed)), None


def _join_domain(case, expected, item, root, record, run):
    from .runtime_archive_join import FrozenJoinBinding, revalidate_join_archive, _TOOLS

    if case.cohort != "normal":
        return None, "deterministic_boundary_or_recovery_revalidator_not_implemented"
    if item is None:
        return None, "join_original_evidence_unavailable"
    _require(item["case_id"] == case.id, "formal_domain_case_binding_mismatch")
    archive, binding, observed = _original_case_binding(item, root, record, run, FrozenJoinBinding)
    _require(digest(binding.case.model_dump()) == case.case_sha256, "formal_domain_definition_mismatch")
    plans = [plan for plan in observed["execution"].get("plans", []) if plan.get("id") == binding.plan_id]
    _require(len(plans) == 1 and plans[0]["status"] == "done", "final_join_plan_binding_mismatch")
    for key, tool in _TOOLS.items():
        steps = [step for step in observed["execution"].get("steps", []) if step.get("id") == binding.step_ids[key]]
        _require(len(steps) == 1 and steps[0]["tool"] == tool and steps[0]["status"] == "done"
                 and steps[0].get("output_sha256") == binding.output_sha256[key], "final_join_step_binding_mismatch")
    recomputed = revalidate_join_archive(archive, expected_manifest_sha256=item["manifest_sha256"], frozen_binding=binding)
    answers = []
    if recomputed["supported_checks_verified"]:
        for assertion in expected.assertions:
            if assertion.kind == "tool_succeeded" and assertion.tool in _TOOLS.values():
                answers.append(True)
            elif assertion.kind == "dataset_rows" and assertion.tool == _TOOLS["execute"]:
                answers.append(type(assertion.value) is int and recomputed["values"]["output_rows"] == assertion.value)
            elif assertion.kind == "http_status":
                codes = [event.get("status_code") for event in observed.get("http_events", []) if event.get("stage") == assertion.stage]
                answers.append(bool(codes) and all(type(code) is type(assertion.value) and code == assertion.value for code in codes))
            else:
                raise TrustInputError("formal_join_predicate_unavailable")
    return _domain_result(recomputed, expected.result == "done" and bool(answers) and all(answers)), None


def _feature_domain(case, expected, item, root, record, run):
    from .runtime_archive_feature import FrozenFeatureBinding, revalidate_feature_archive, _TOOLS

    if case.cohort != "normal":
        return None, "deterministic_boundary_or_recovery_revalidator_not_implemented"
    if item is None:
        return None, "feature_original_evidence_unavailable"
    _require(item["case_id"] == case.id, "formal_domain_case_binding_mismatch")
    archive, binding, observed = _original_case_binding(item, root, record, run, FrozenFeatureBinding)
    _require(digest(binding.case.model_dump()) == case.case_sha256, "formal_domain_definition_mismatch")
    plans = [plan for plan in observed["execution"].get("plans", []) if plan.get("id") == binding.plan_id]
    _require(len(plans) == 1 and plans[0]["status"] == "done", "final_feature_plan_binding_mismatch")
    for key, tool in _TOOLS.items():
        steps = [step for step in observed["execution"].get("steps", []) if step.get("id") == binding.step_ids[key]]
        _require(len(steps) == 1 and steps[0]["tool"] == tool and steps[0]["status"] == "done"
                 and steps[0].get("output_sha256") == binding.output_sha256[key], "final_feature_step_binding_mismatch")
    _require(any(event.get("stage") == "download_feature_report" and event.get("status_code") == 200
                 and event.get("step_id") == binding.step_ids["report"] and event.get("sha256") == binding.download_sha256
                 for event in observed["http_events"]), "final_feature_download_binding_mismatch")
    recomputed = revalidate_feature_archive(archive, expected_manifest_sha256=item["manifest_sha256"], frozen_binding=binding)
    answers = []
    if recomputed["supported_checks_verified"]:
        for assertion in expected.assertions:
            if assertion.kind == "tool_succeeded" and assertion.tool in _TOOLS.values():
                answers.append(True)
            elif assertion.kind == "output_equals" and assertion.tool == _TOOLS["report"] and assertion.path == ["feature_count"]:
                answers.append(type(assertion.value) is int and recomputed["values"]["feature_count"] == assertion.value)
            elif assertion.kind == "output_length" and assertion.tool in {_TOOLS["metrics"], _TOOLS["report"]} and assertion.path == ["metrics"]:
                answers.append(type(assertion.value) is int and recomputed["values"]["feature_count"] == assertion.value)
            elif ((assertion.kind == "artifact_exists" and assertion.path == ["report_path"])
                  or assertion.kind == "feature_report_download") and assertion.tool == _TOOLS["report"]:
                answers.append(all(recomputed["checks"][key]["status"] == "verified" for key in
                                   ("artifact_record", "workbook_cells", "download_binding")))
            elif assertion.kind == "http_status":
                codes = [event.get("status_code") for event in observed.get("http_events", []) if event.get("stage") == assertion.stage]
                answers.append(bool(codes) and all(type(code) is type(assertion.value) and code == assertion.value for code in codes))
            else:
                raise TrustInputError("formal_feature_predicate_unavailable")
    return _domain_result(recomputed, expected.result == "done" and bool(answers) and all(answers)), None


def _domain(case, expected, item, root, record, run):
    if case.family == "feature":
        return _feature_domain(case, expected, item, root, record, run)
    if case.family == "data_processing":
        return _join_domain(case, expected, item, root, record, run)
    if case.family == "labeling":
        return _labeling_domain(case, expected, item, root, record, run)
    if case.family == "risk_portfolio":
        return _portfolio_domain(case, expected, item, root, record, run)
    return _validation_domain(case, expected, item, root, record, run)


def _observe_attempt(
    attempt,
    case,
    assignment,
    definition,
    expected,
    root,
    record,
    run,
    price_bytes,
    benchmark,
):
    original = run["executions"][case.id]
    observation = AttemptObservation(
        attempt.id,
        original["runtime_status"],
        "unavailable",
        original_scoring_failed=run["scores"][case.id]["passed"] is False,
        wall_seconds=original["duration_ms"] / 1000
        if type(original.get("duration_ms")) is int and original["duration_ms"] >= 0
        else None,
    )
    details = {}
    try:
        _check_case_definition(case, expected, definition, original)
        events = run["attempts"][case.id]
        started = [event for event in events if event["event"] == "started"]
        _require(
            started
            and all(
                event.get("model_name") == benchmark.model.model_name
                and event.get("model_id") == benchmark.model.profile
                for event in started
            ),
            "formal_real_model_trace_unavailable",
        )
        usage = usage_summary(
            events, model_name=benchmark.model.model_name, price_bytes=price_bytes
        )
        _require(usage["trace_complete"] is True, "formal_model_trace_incomplete")
        _require(
            usage["cost"] is None or usage.get("currency") == case.budget.currency,
            "formal_cost_currency_mismatch",
        )
        totals = (
            None
            if usage["usage_status"] != "known"
            else usage["prompt_tokens"] + usage["completion_tokens"]
        )
        outputs = [
            event.get("completion_tokens")
            for event in events
            if event["event"] == "finished"
        ]
        output_max = (
            max(outputs)
            if outputs and all(type(value) is int and value >= 0 for value in outputs)
            else None
        )
        fields, reason = _telemetry(
            root, record, assignment["telemetry"], benchmark, attempt, run, definition
        )
        observation = replace(
            observation,
            **fields,
            calls=usage["transport_attempts"],
            tokens=totals,
            retries=usage["retry_attempts"],
            cost=usage["cost"],
            http_requests=len(original["http_events"]),
            max_output_tokens_observed=output_max,
            reason=reason,
        )
        domain, reason = _domain(
            case, expected, assignment["domain_evidence"], root, record, run
        )
        if domain is None:
            return replace(observation, reason=reason), details
        details = {key: domain.pop(key) for key in ("domain_checks", "unsupported")}
        observation = replace(observation, **domain)
    except TrustInputError as exc:
        observation = replace(observation, reason=str(exc))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        observation = replace(
            observation, reason="formal_original_evidence_unavailable"
        )
    return observation, details


def verify_formal_benchmark(inputs, execution, root, record, config):
    """Called only by the fixed registry after external signatures authenticate."""
    # Parse the caller-held policy first so even incomplete/invalid run sets
    # retain every planned original/repeat row in the public denominator.
    _keys(inputs, {"benchmark", "runs"})
    benchmark = FrozenBenchmark.model_validate_json(
        _record_bound(root, record, inputs["benchmark"], "input_hashes")
    )
    observations, details, inventory_error = [], {}, None
    try:
        _planned_context(inputs, execution, root, record, config)
        entries = _assignment_inventory(inputs, execution, benchmark)
        cases, planned = (
            {case.id: case for case in benchmark.cases},
            {a.id: a for a in benchmark.attempts},
        )
        data_hashes = set()
        for entry in entries:
            run = None
            try:
                run = _authenticated_run(entry, root, record, config, benchmark)
                definitions, expected, price = _run_context(
                    entry, root, record, config, benchmark, run
                )
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                reason = (
                    str(exc)
                    if isinstance(exc, TrustInputError)
                    else "formal_original_run_unavailable"
                )
                for assignment in entry["assignments"]:
                    original = (
                        run["executions"].get(assignment["case_id"]) if run else None
                    )
                    observations.append(
                        AttemptObservation(
                            assignment["attempt_id"],
                            original["runtime_status"]
                            if original
                            else "unverified_original_run",
                            "unavailable",
                            original_scoring_failed=bool(
                                run
                                and run["scores"]
                                .get(assignment["case_id"], {})
                                .get("passed")
                                is False
                            ),
                            reason=reason,
                        )
                    )
                continue
            for assignment in entry["assignments"]:
                identity, case_id = assignment["attempt_id"], assignment["case_id"]
                data_hashes.update(
                    m.sha256
                    for m in definitions[case_id].materials
                    if m.role in {"sample", "feature", "unknown"}
                )
                observation, detail = _observe_attempt(
                    planned[identity],
                    cases[case_id],
                    assignment,
                    definitions[case_id],
                    expected[case_id],
                    root,
                    record,
                    run,
                    price,
                    benchmark,
                )
                observations.append(observation)
                if detail:
                    details[identity] = detail
        _require(
            data_hashes <= set(record["dataset_hashes"].values()),
            "formal_dataset_inventory_mismatch",
        )
        if len(observations) == len(planned):
            _require(
                data_hashes == set(record["dataset_hashes"].values()),
                "formal_dataset_inventory_mismatch",
            )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        inventory_error = (
            str(exc)
            if isinstance(exc, TrustInputError)
            else "formal_frozen_evidence_unavailable"
        )
    summary = aggregate_benchmark(benchmark, observations)
    # A known source/inventory mismatch is never repaired by favorable totals.
    successful = summary["thresholds_satisfied"] and inventory_error is None
    return {
        **summary,
        "domain_details": details,
        "denominator": summary["attempt_denominator"],
        "passed": sum(row["success"] for row in summary["attempts"]),
        "successful": successful,
        "reason": inventory_error
        or (
            "formal_domain_or_process_evidence_unavailable"
            if summary["status"] == "unavailable"
            else None
        ),
        "process_observation_scope": "externally_signed_frozen_environment_and_timing_observations_not_independent_hardware_discovery",
        "cost_scope": "estimate_from_usage_and_bound_uncached_price_book_not_invoice",
        "denominator_scope": "all_externally_frozen_original_and_repeat_slots; any_unfrozen_attempt_invalidates_acceptance",
        "timing_collector": "runner_emits_llm_transport_and_scripted_actions; human_wait_tool_queue_and_replan_collection_incomplete; complete_external_observations_required",
    }
