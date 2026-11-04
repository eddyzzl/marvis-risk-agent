"""Original monitoring slots exercise their own confirmation and score worker."""
import copy
import json
import shutil
import sqlite3

import pytest
from pydantic import ValidationError

from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest


def _cases(root):
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite

    paths = write_synthetic_suite(root / "suite", normal_monitoring_only=True)
    template = json.loads(paths["cases"].read_bytes())["cases"][0]
    expectation = json.loads(paths["expected"].read_bytes())["cases"][template["id"]]
    cases, expected = [], {}
    for kind in ("stale_monitoring_revision_once", "tool_timeout_once"):
        case = copy.deepcopy(template)
        case.update(id="actual_monitoring_" + kind, scenario="recovery")
        case["budget"].update(max_total_tokens=100000, max_cost=1.0, currency="USD")
        case["actions"][-1]["fault"] = {"kind": kind}
        case["actions"].append({"kind": "recover_monitoring_execution", "fault_source_action": len(case["actions"]),
            "content": "显式重读同一监控计划，通过现有接口尝试恢复；不能绕过未知副作用。"})
        cases.append(case)
        expected[case["id"]] = expectation
    paths["cases"].write_text(json.dumps({"schema_version": 1, "cases": cases}))
    paths["expected"].write_text(json.dumps({"schema_version": 1, "cases": expected}))
    return paths, cases


@pytest.fixture(scope="module")
def monitoring_fault_archives(tmp_path_factory):
    from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
    from test_runtime_agent_benchmark import fixture_model
    from test_runtime_domain_archives import _protocol

    root = tmp_path_factory.mktemp("native-monitoring-fault")
    paths, _ = _cases(root)
    price = root / "price.json"
    price.write_text(json.dumps({"model_name": "fixture", "provider": "local-protocol-substitute", "currency": "USD",
        "source": "Synthetic local rates; no provider billing", "effective_date": "2026-11-04",
        "prompt_per_million": 1.0, "completion_per_million": 2.0}))
    with fixture_model(answer_factory=_protocol) as (model, calls):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"], dataset_root=paths["dataset_root"],
            output_dir=root / "public", evidence_custody_dir=root / "custody", model=model,
            model_source="fixture_model", price_book_path=price)
    assert calls
    result = {}
    for row in report["cases"]:
        archive = root / "custody" / report["run_id"] / row["case_id"]
        private = json.loads((archive / "private.json").read_bytes())
        assert private.get("runtime_faults"), (row, private.keys())
        fault = private["runtime_faults"][0]
        assert fault.get("fault_executed") and "recovery" in fault, (row, fault)
        result[fault["kind"]] = archive, row, fault
    return result


@pytest.mark.e2e
def test_native_monitoring_fault_slots_are_not_modeling_prerequisites(monitoring_fault_archives):
    for kind, (_, row, fault) in monitoring_fault_archives.items():
        assert fault["plan_before"]["template_id"] == "monitoring_run"
        assert fault["tool"] == "modeling.score_dataset"
        assert fault["action_index"] == 8 and fault["recovery"]["action_index"] == 9
        if kind == "stale_monitoring_revision_once":
            assert fault["http_status"] == 409
            assert fault["before"]["runs"] == fault["after_fault"]["runs"] == []
            assert fault["before"]["dataset_ids"] == fault["after_fault"]["dataset_ids"]
            assert fault["recovery"]["completed"], (row, fault)
        else:
            assert fault["ready"]["kind"] == "tool_timeout_once"
            assert fault["after_fault"]["runs"][0]["tool_ref"] == "modeling.score_dataset"
            assert fault["recovery"]["http_status"] == 409 and not fault["recovery"]["completed"]


def _safety(archive):
    from marvis.orchestrator.eval.runtime_safety_contracts import FrozenSafetyBinding

    common = json.loads((archive / "bindings.json").read_bytes())
    manifest = json.loads((archive / "manifest.json").read_bytes())
    fault = json.loads((archive / "private.json").read_bytes())["runtime_faults"][0]
    return FrozenSafetyBinding(case=RuntimeCase.model_validate_json((archive / "case.json").read_bytes()),
        task_id=fault["task_id"], recovery_action_indices=[9], before_manifest_sha256=manifest["safety_before"]["manifest_sha256"],
        **{key: common[key] for key in ("run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")})


def _business(archive, case):
    from test_runtime_domain_archives import _binding, _model_binding
    from marvis.orchestrator.eval.runtime_archive_monitoring import FrozenMonitoringBinding, DEFAULT_DECLARED_THRESHOLDS

    model = _model_binding(archive, case.model_dump())
    values, outputs, tables = _binding(archive, case.model_dump(), "monitoring_run")
    score = outputs["modeling.score_dataset"]
    inputs = json.loads(next(r for r in tables["plan_step_runs"] if r["tool_ref"] == "modeling.score_dataset")["input_json"])
    source = next(r for r in tables["datasets"] if r["id"] == inputs["dataset_id"])
    child = next(r for r in tables["datasets"] if r["id"] == score["result_dataset_id"])
    values.update(modeling=model.model_dump(), dataset_id=source["id"], dataset_sha256=source["content_hash"],
        scored_dataset_id=child["id"], scored_dataset_sha256=child["content_hash"],
        experiment_id=model.selected_experiment_id, artifact_id=model.final_artifact_id,
        monitoring_binding_sha256=digest(inputs["monitoring_binding"]), thresholds=DEFAULT_DECLARED_THRESHOLDS)
    return FrozenMonitoringBinding.model_validate(values)


@pytest.mark.e2e
@pytest.mark.parametrize("kind", ["stale_monitoring_revision_once", "tool_timeout_once"])
def test_installed_monitoring_recovery_reader_and_formal_preserve_actual_outcome(monitoring_fault_archives, tmp_path, kind):
    from marvis.orchestrator.eval.runtime_archive_monitoring_recovery import revalidate_monitoring_recovery_archive
    from marvis.orchestrator.eval.formal_benchmark import FormalCase
    from marvis.orchestrator.eval.formal_safety_adapters import future_fault_contract, retained_safety
    from marvis.orchestrator.eval.runtime_run_manifest import verify_run_manifest
    from marvis.orchestrator.eval.runtime_scoring import ExpectedSuite
    from test_acceptance_trust import put
    from test_formal_benchmark import frozen_fixture

    archive, _, _ = monitoring_fault_archives[kind]
    safety = _safety(archive)
    sha = digest((archive / "manifest.json").read_bytes())
    result = revalidate_monitoring_recovery_archive(archive, expected_manifest_sha256=sha, frozen_binding=safety)
    assert result["status"] == "verified", result
    assert result["values"]["fault_executed"]
    assert result["values"]["recovery_completed"] is (kind == "stale_monitoring_revision_once")
    shutil.copytree(archive, tmp_path / "archive")
    safety_ref = put(tmp_path, "safety.json", safety.model_dump())
    refs = [safety_ref]
    business_ref = None
    if kind == "stale_monitoring_revision_once":
        business_ref = put(tmp_path, "business.json", _business(archive, safety.case).model_dump())
        refs.append(business_ref)
    public = archive.parents[2] / "public" / safety.run_id
    run = verify_run_manifest(public, expected_manifest_sha256=digest((public / "final-manifest.json").read_bytes()))
    expected = ExpectedSuite.model_validate_json((archive.parents[2] / "suite/private/expected.json").read_bytes()).cases[safety.case.id]
    template = next(c for c in frozen_fixture().cases if c.family == "monitoring" and c.cohort == "recovery")
    formal = FormalCase.model_validate({**template.model_dump(), "id": safety.case.id, "budget": safety.case.budget.model_dump(),
        "case_sha256": digest(safety.case.model_dump()), "expected_sha256": digest(expected.model_dump()),
        "fault_contract_sha256": digest(future_fault_contract(safety.case))})
    item = {"case_id": safety.case.id, "archive": "archive", "manifest_sha256": sha, "binding": safety_ref}
    record = {"input_hashes": {r["path"]: r["sha256"] for r in refs}}
    value = retained_safety(formal, expected, item, tmp_path, record, run,
        domain_item=({**item, "binding": business_ref} if business_ref else None))
    assert value["domain"]["domain_status"] == "verified", value
    assert value["domain"]["actual_result"] == ("recovered" if kind == "stale_monitoring_revision_once" else "failed"), value
    assert value["domain"]["fault_contract_status"] == "verified"


@pytest.mark.e2e
@pytest.mark.parametrize("damage", ["wrong_plan", "wrong_dataset", "valid_cas", "fake_recovered", "http_method", "repeated_retry", "worker_parent", "fake_duration", "fake_phase", "reverse_job", "rejected_retry_reversed", "rejected_retry_before_failure", "preimage_score_input", "preimage_score_policy", "preimage_monitor_input", "preimage_dependencies", "preimage_tool"])
def test_monitoring_raw_and_private_repin_cannot_forge_native_recovery(monitoring_fault_archives, tmp_path, damage):
    from marvis.orchestrator.eval.runtime_archive_monitoring_recovery import revalidate_monitoring_recovery_archive
    from test_runtime_archive_validation import _rewrite_manifest

    kind = "tool_timeout_once" if damage in {"fake_recovered", "worker_parent", "fake_duration", "fake_phase", "rejected_retry_reversed", "rejected_retry_before_failure"} else "stale_monitoring_revision_once"
    archive, _, _ = monitoring_fault_archives[kind]
    safety = _safety(archive)
    copied = tmp_path / "archive"
    shutil.copytree(archive, copied)
    private = json.loads((copied / "private.json").read_bytes())
    fault = private["runtime_faults"][0]
    if damage == "wrong_plan":
        fault["plan_before"]["template_id"] = "modeling"
    elif damage == "wrong_dataset":
        fault["submission"]["request"]["dataset_id"] = "foreign"
    elif damage == "valid_cas":
        fault["dispatched_request"] = fault["original_request"]
    elif damage.startswith("preimage_"):
        from marvis.orchestrator.eval.runtime_archive_safety import _PLAN_FIELDS, _STEP_FIELDS

        for plan in (fault["plan_before"], fault["recovery"]["plan_before"]):
            score = next(s for s in plan["steps"] if s["tool_ref"]["tool"] == "score_dataset")
            monitor = next(s for s in plan["steps"] if s["tool_ref"]["tool"] == "monitor_run")
            if damage == "preimage_score_input":
                score["inputs"]["__not_the_native_input__"] = True
            elif damage == "preimage_score_policy":
                score["policy"]["human_decision_gate"] = "required"
            elif damage == "preimage_monitor_input":
                monitor["inputs"]["__not_the_native_input__"] = True
            elif damage == "preimage_dependencies":
                monitor["depends_on"].append("foreign-step")
            else:
                monitor["tool_ref"]["version"] = "foreign-version"
            canonical = {k: plan[k] for k in _PLAN_FIELDS}
            canonical["steps"] = [{k: s[k] for k in _STEP_FIELDS} for s in plan["steps"]]
            plan["confirmation_snapshot"]["expected_plan_fingerprint"] = digest(canonical)
        fault["original_request"]["expected_plan_fingerprint"] = fault["plan_before"]["confirmation_snapshot"]["expected_plan_fingerprint"]
        fault["recovery"]["request"]["expected_plan_fingerprint"] = fault["recovery"]["plan_before"]["confirmation_snapshot"]["expected_plan_fingerprint"]
    elif damage == "fake_recovered":
        fault["recovery"]["completed"] = True
    elif damage in {"http_method", "repeated_retry"}:
        path = copied / "bindings.json"
        value = json.loads(path.read_bytes())
        events = value["execution_before_scoring"]["http_events"]
        event = next(e for e in events if e["ordinal"] == fault["recovery"]["http_ordinal"])
        if damage == "http_method":
            event["method"] = "GET"
        else:
            events.append({**event, "ordinal": max(e["ordinal"] for e in events) + 1})
        path.write_text(json.dumps(value))
    elif damage in {"worker_parent", "fake_phase"}:
        path = copied / "workspace/.runtime-eval-fault/launch/ready.json"
        ready = json.loads(path.read_bytes())
        if damage == "worker_parent":
            ready["worker"]["ppid"] += 1
        else:
            ready["phase"] = "after_business_commit_before_protocol"
        fault["ready"] = ready
        path.write_text(json.dumps(ready))
    elif damage == "fake_duration":
        path = copied / "workspace/.runtime-eval-fault/launch/terminal.json"
        value = json.loads(path.read_bytes())
        value["finished_elapsed_ns"] = value["started_elapsed_ns"] + 1
        path.write_text(json.dumps(value))
    elif damage in {"rejected_retry_reversed", "rejected_retry_before_failure"}:
        job = next(j for j in fault["recovery"]["after"]["jobs"] if j["kind"] == "plan" and j["started_at"] is None)
        job["finished_at"] = "2000-01-01T00:00:00+00:00"
        with sqlite3.connect(copied / "workspace/marvis.sqlite") as conn:
            conn.execute("UPDATE jobs SET finished_at=? WHERE id=?", (job["finished_at"], job["id"]))
            if damage == "rejected_retry_before_failure":
                conn.execute("UPDATE jobs SET created_at=? WHERE id=?", (job["finished_at"], job["id"]))
    else:
        failed = next(j for j in fault["after_fault"]["jobs"] if j["status"] == "failed")
        identity = failed["id"]
        with sqlite3.connect(copied / "workspace/marvis.sqlite") as conn:
            conn.execute("UPDATE jobs SET finished_at=? WHERE id=?", ("2000-01-01T00:00:00+00:00", identity))
        for section in (fault["after_fault"], fault["recovery"]["before"], fault["recovery"]["after"]):
            next(j for j in section["jobs"] if j["id"] == identity)["finished_at"] = "2000-01-01T00:00:00+00:00"
    originals = copied / "workspace/.runtime-eval-fault/monitoring"
    (originals / "fault.json").write_text(json.dumps({k: v for k, v in fault.items() if k != "recovery"}))
    (originals / "recovery.json").write_text(json.dumps(fault["recovery"]))
    (copied / "private.json").write_text(json.dumps(private))
    result = revalidate_monitoring_recovery_archive(copied, expected_manifest_sha256=_rewrite_manifest(copied), frozen_binding=safety)
    assert result["status"] == "mismatch", result


@pytest.mark.parametrize("damage", ["prerequisite", "normal", "wrong_source", "omit_retry", "foreign_family", "wrong_action", "manual_entry"])
def test_monitoring_recovery_contract_rejects_other_scope(tmp_path, damage):
    _, cases = _cases(tmp_path)
    case = cases[0]
    if damage == "prerequisite":
        case["actions"][0]["fault"] = case["actions"][-2].pop("fault")
    elif damage == "normal":
        case["scenario"] = "normal"
    elif damage == "wrong_source":
        case["actions"][-1]["fault_source_action"] = 1
    elif damage == "omit_retry":
        case["actions"].pop()
    elif damage == "foreign_family":
        case["family"] = "modeling"
    elif damage == "manual_entry":
        case["actions"][-3]["kind"] = "start_model_monitoring_workflow"
    else:
        case["actions"][-2]["kind"] = "approve_step"
        case["actions"][-2]["tool"] = "modeling.score_dataset"
    with pytest.raises(ValidationError):
        RuntimeCase.model_validate(case)
