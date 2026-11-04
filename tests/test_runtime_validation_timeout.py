"""Real native V2 PMML transport failure and explicit API retry originals."""
import copy
import json
import shutil
import sqlite3

import pytest
from pydantic import ValidationError

from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest


def _case(root):
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite

    paths = write_synthetic_suite(root / "suite", normal_validation_agent_only=True)
    case = json.loads(paths["cases"].read_bytes())["cases"][0]
    case["scenario"] = "recovery"
    case["budget"].update(max_total_tokens=100000, max_cost=1.0, currency="USD")
    case["actions"][0]["fault"] = {"kind": "tool_timeout_once"}
    case["actions"].insert(1, {"kind": "recover_validation_execution", "fault_source_action": 1,
        "content": "显式通过原生 Agent 启动接口重试同一任务；保留首次超时失败。"})
    paths["cases"].write_text(json.dumps({"schema_version": 1, "cases": [case]}))
    return paths, RuntimeCase.model_validate(case)


@pytest.fixture(scope="module")
def timeout_archive(tmp_path_factory):
    from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
    from test_runtime_agent_benchmark import fixture_model
    from test_runtime_validation_agent import _v2_protocol

    root = tmp_path_factory.mktemp("native-validation-timeout")
    paths, case = _case(root)
    price = root / "price.json"
    price.write_text(json.dumps({"model_name": "fixture", "provider": "local-protocol-substitute",
        "currency": "USD", "source": "Synthetic local rates; no provider billing",
        "effective_date": "2026-11-04", "prompt_per_million": 1.0, "completion_per_million": 2.0}))
    with fixture_model(answer_factory=_v2_protocol) as (model, calls):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
            model=model, model_source="fixture_model", price_book_path=price)
    assert calls
    row = report["cases"][0]
    archive = root / "custody" / report["run_id"] / row["case_id"]
    private = json.loads((archive / "private.json").read_bytes())
    assert private.get("validation_faults"), row
    fault = private["validation_faults"][0]
    assert fault.get("fault_executed") is True, (row, fault)
    return archive, report, case


@pytest.mark.pmml_runtime
def test_actual_jvm_timeout_keeps_failed_job_and_explicit_native_retry(timeout_archive):
    archive, report, case = timeout_archive
    terminal = json.loads((archive / "workspace/.runtime-eval-fault/validation-timeout/terminal.json").read_bytes())
    private = json.loads((archive / "private.json").read_bytes())
    fault = private["validation_faults"][0]
    assert terminal["outcome"] == "native_exception"
    assert terminal["socket_timeouts"] and terminal["socket_timeouts"][0]["socket_timeout_seconds"] == 2
    assert terminal["jvm"]["ppid"] == terminal["host"]["pid"]
    assert terminal["jvm"] == terminal["resumed_identity"]
    assert terminal["finished_elapsed_ns"] - terminal["started_elapsed_ns"] >= 2_000_000_000
    assert terminal["resume_signal"] == "SIGCONT"
    failed = [j for j in fault["after_fault"]["jobs"] if j["id"] == terminal["job_id"]]
    assert len(failed) == 1 and failed[0]["status"] == "failed", fault
    recovery = fault["recovery"]
    assert recovery["http_status"] == 202
    assert recovery["action_index"] == 2 and case.actions[1].fault_source_action == 1
    assert len(recovery["after"]["jobs"]) == len(fault["after_fault"]["jobs"]) + 1
    assert not report["all_passed"]  # Failed original job is never rewritten as success.


def _binding(archive, case):
    from marvis.orchestrator.eval.runtime_safety_contracts import FrozenSafetyBinding

    bindings = json.loads((archive / "bindings.json").read_bytes())
    private = json.loads((archive / "private.json").read_bytes())
    return FrozenSafetyBinding(case=case, task_id=private["validation_faults"][0]["task_id"],
        recovery_action_indices=[2], **{key: bindings[key] for key in (
            "run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")})


@pytest.mark.pmml_runtime
def test_independent_timeout_reader_and_original_math_verify_real_recovery(timeout_archive, tmp_path):
    from marvis.orchestrator.eval.runtime_archive_validation_timeout import revalidate_validation_timeout_archive
    from marvis.orchestrator.eval.runtime_archive_validation import FrozenValidationBinding, revalidate_validation_archive

    archive, report, case = timeout_archive
    sha = digest((archive / "manifest.json").read_bytes())
    binding = _binding(archive, case)
    result = revalidate_validation_timeout_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "verified", result
    assert result["values"]["fault_executed"] and result["values"]["recovery_completed"]
    pipeline = report["cases"][0]["execution"]["validation_pipeline"]
    business = FrozenValidationBinding(**{key: getattr(binding, key) for key in (
        "case", "run_id", "task_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")},
        input_contract_sha256=pipeline["input_contract_sha256"],
        confirmed_draft_sha256=pipeline["confirmed_draft_sha256"], report_revision=pipeline["report_revision"],
        report_sha256={f["kind"]: f["sha256"] for f in pipeline["report_files"]}, bin_count=10)
    math = revalidate_validation_archive(archive, expected_manifest_sha256=sha, frozen_binding=business)
    assert math["supported_checks_verified"], math
    # These are the exact joins required by the installed formal recovery branch.
    assert result["values"]["confirmation_request"]["revision"] + 1 == business.report_revision
    assert digest(result["values"]["confirmation_request"]["text_values"]) == business.confirmed_draft_sha256
    assert {d["kind"]: d["sha256"] for d in result["values"]["downloads"]} == business.report_sha256
    from marvis.orchestrator.eval.formal_benchmark import FormalCase
    from marvis.orchestrator.eval.formal_safety_adapters import future_fault_contract, retained_safety
    from marvis.orchestrator.eval.runtime_run_manifest import verify_run_manifest
    from marvis.orchestrator.eval.runtime_scoring import ExpectedSuite
    from test_acceptance_trust import put
    from test_formal_benchmark import frozen_fixture

    shutil.copytree(archive, tmp_path / "archive")
    safety_ref = put(tmp_path, "safety.json", binding.model_dump())
    business_ref = put(tmp_path, "business.json", business.model_dump())
    public = archive.parents[2] / "public" / binding.run_id
    run = verify_run_manifest(public, expected_manifest_sha256=digest((public / "final-manifest.json").read_bytes()))
    expected = ExpectedSuite.model_validate_json((archive.parents[2] / "suite/private/expected.json").read_bytes()).cases[case.id]
    template = next(c for c in frozen_fixture().cases if c.family == "validation" and c.cohort == "recovery")
    formal = FormalCase.model_validate({**template.model_dump(), "id": case.id, "budget": case.budget.model_dump(),
        "case_sha256": digest(case.model_dump()), "expected_sha256": digest(expected.model_dump()),
        "fault_contract_sha256": digest(future_fault_contract(case))})
    item = {"case_id": case.id, "archive": "archive", "manifest_sha256": sha, "binding": safety_ref}
    record = {"input_hashes": {r["path"]: r["sha256"] for r in (safety_ref, business_ref)}}
    value = retained_safety(formal, expected, item, tmp_path, record, run, domain_item={**item, "binding": business_ref})
    assert value["domain"]["domain_status"] == "verified", value
    assert value["domain"]["actual_result"] == "recovered", value
    assert value["domain"]["fault_contract_status"] == "verified"
    assert value["domain"]["domain_checks"]["original_business_math"]["status"] == "verified"


@pytest.mark.pmml_runtime
@pytest.mark.parametrize("damage", ["wrong_task", "wrong_parent", "reused_pid", "fake_deadline", "wrong_port", "no_socket_timeout", "repeated_retry", "wrong_job", "old_draft", "changed_report"])
def test_repinned_raw_and_private_fault_declarations_cannot_override_native_originals(timeout_archive, tmp_path, damage):
    from marvis.orchestrator.eval.runtime_archive_validation_timeout import revalidate_validation_timeout_archive
    from test_runtime_archive_validation import _rewrite_manifest

    archive, _, case = timeout_archive
    binding = _binding(archive, case)
    copied = tmp_path / "archive"
    shutil.copytree(archive, copied)
    directory = copied / "workspace/.runtime-eval-fault/validation-timeout"
    private = json.loads((copied / "private.json").read_bytes())
    fault = private["validation_faults"][0]
    terminal = json.loads((directory / "terminal.json").read_bytes())
    if damage == "wrong_task":
        terminal["task_id"] = "other"
    elif damage == "wrong_parent":
        terminal["jvm"]["ppid"] += 1
        terminal["resumed_identity"] = terminal["jvm"]
    elif damage == "reused_pid":
        terminal["resumed_identity"]["create_time"] += 1
    elif damage == "fake_deadline":
        terminal["finished_elapsed_ns"] = terminal["started_elapsed_ns"] + 1
    elif damage == "wrong_port":
        terminal["gateway_port"] += 1
    elif damage == "no_socket_timeout":
        terminal["socket_timeouts"] = []
    elif damage == "repeated_retry":
        fault["recovery"]["after"]["jobs"].append(dict(fault["recovery"]["after"]["jobs"][-1]))
        (directory / "after-retry.json").write_text(json.dumps(fault["recovery"]))
    elif damage == "wrong_job":
        terminal["job_id"] = fault["recovery"]["after"]["jobs"][-1]["id"]
    elif damage == "old_draft":
        fault["confirmation"]["request"]["draft_message_id"] = "other"
        (directory / "confirmation.json").write_text(json.dumps(fault["confirmation"]))
    else:
        report = copied / "workspace/tasks" / binding.task_id / "outputs/validation_report.docx"
        report.write_bytes(report.read_bytes() + b"changed")
    # Repin even the matching launcher declaration and private digest. The
    # independent reader must still reject conflicting native/transport facts.
    claimed = json.loads((directory / "claimed.json").read_bytes())
    claimed = {key: terminal[key] for key in claimed}
    claimed["socket_timeouts"] = []
    (directory / "claimed.json").write_text(json.dumps(claimed))
    (directory / "terminal.json").write_text(json.dumps(terminal))
    fault["terminal_sha256"] = digest(terminal)
    (directory / "after-fault.json").write_text(json.dumps({k: v for k, v in fault.items() if k not in {"recovery", "confirmation"}}))
    (copied / "private.json").write_text(json.dumps(private))
    result = revalidate_validation_timeout_archive(copied, expected_manifest_sha256=_rewrite_manifest(copied), frozen_binding=binding)
    assert result["status"] == "mismatch", result


@pytest.mark.pmml_runtime
@pytest.mark.parametrize("damage", ["download_post", "failed_reversed", "retry_reversed", "report_reversed", "report_before_retry"])
def test_independent_review_time_and_download_method_contradictions_fail_closed(timeout_archive, tmp_path, damage):
    from marvis.orchestrator.eval.runtime_archive_validation_timeout import revalidate_validation_timeout_archive
    from test_runtime_archive_validation import _rewrite_manifest

    archive, _, case = timeout_archive
    binding = _binding(archive, case)
    copied = tmp_path / "archive"
    shutil.copytree(archive, copied)
    if damage == "download_post":
        path = copied / "bindings.json"
        value = json.loads(path.read_bytes())
        next(e for e in value["execution_before_scoring"]["http_events"] if e["stage"] == "download_validation_word")["method"] = "POST"
        path.write_text(json.dumps(value))
    else:
        with sqlite3.connect(copied / "workspace/marvis.sqlite") as conn:
            if damage == "failed_reversed":
                job = conn.execute("SELECT id FROM jobs WHERE status='failed'").fetchone()[0]
            elif damage == "retry_reversed":
                job = conn.execute("SELECT id FROM jobs WHERE kind='agent' AND status='succeeded'").fetchone()[0]
            else:
                job = conn.execute("SELECT id FROM jobs WHERE kind='report'").fetchone()[0]
            field = "started_at" if damage == "report_before_retry" else "finished_at"
            value = "2000-01-01T00:00:00+00:00"
            conn.execute(f"UPDATE jobs SET {field}=? WHERE id=?", (value, job))
        def mutate(value):
            if isinstance(value, dict):
                if value.get("id") == job and "kind" in value and field in value:
                    value[field] = "2000-01-01T00:00:00+00:00"
                for child in value.values():
                    mutate(child)
            elif isinstance(value, list):
                for child in value:
                    mutate(child)
        originals = copied / "workspace/.runtime-eval-fault/validation-timeout"
        for path in [copied / "private.json", *originals.glob("*.json")]:
            data = json.loads(path.read_bytes())
            mutate(data)
            path.write_text(json.dumps(data))
        terminal = json.loads((originals / "terminal.json").read_bytes())
        for path in (copied / "private.json", originals / "after-fault.json"):
            data = json.loads(path.read_bytes())
            fault = data["validation_faults"][0] if path.name == "private.json" else data
            fault["terminal_sha256"] = digest(terminal)
            path.write_text(json.dumps(data))
    result = revalidate_validation_timeout_archive(copied, expected_manifest_sha256=_rewrite_manifest(copied), frozen_binding=binding)
    assert result["status"] == "mismatch", result


@pytest.mark.parametrize("damage", ["without_retry", "wrong_source", "report_fault", "normal", "v1", "tool"])
def test_validation_timeout_contract_rejects_semantic_substitutions(tmp_path, damage):
    _, valid = _case(tmp_path)
    case = copy.deepcopy(valid.model_dump())
    if damage == "without_retry":
        case["actions"].pop(1)
    elif damage == "wrong_source":
        case["actions"][1]["fault_source_action"] = 2
    elif damage == "report_fault":
        case["actions"][0]["fault"] = None
        case["actions"][2]["fault"] = {"kind": "tool_timeout_once"}
    elif damage == "normal":
        case["scenario"] = "normal"
    elif damage == "v1":
        case["actions"][0]["kind"] = "start_validation_workflow"
    else:
        case["actions"][1]["tool"] = "v1_compat.render_reports"
    with pytest.raises(ValidationError):
        RuntimeCase.model_validate(case)
