"""Actual Agent report-confirmation loss and read-only recovery, with retained originals."""
import copy
import json
import shutil

import pytest

from marvis.orchestrator.eval.runtime_archive_validation import FrozenValidationBinding
from marvis.orchestrator.eval.runtime_archive_validation_recovery import revalidate_validation_recovery_archive
from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from marvis.orchestrator.eval.runtime_safety_contracts import FrozenSafetyBinding
from test_runtime_agent_benchmark import fixture_model
from test_runtime_validation_agent import _v2_protocol


@pytest.fixture(scope="module")
def validation_recovery_originals(tmp_path_factory):
    root = tmp_path_factory.mktemp("validation-recovery")
    paths = write_synthetic_suite(root / "suite", normal_validation_agent_only=True)
    suite = json.loads(paths["cases"].read_text())
    raw = suite["cases"][0]
    raw["budget"].update(max_total_tokens=100000, max_cost=1.0, currency="USD")
    raw["scenario"] = "recovery"
    raw["actions"][1]["fault"] = {"kind": "discard_success_response_once"}
    raw["actions"].append({"kind": "recover_validation_report", "fault_source_action": 2,
                           "content": "读取同一确认的任务状态和报告，不重复确认、不重新评分。"})
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    expected = json.loads(paths["expected"].read_text())
    for assertion in expected["cases"][raw["id"]]["assertions"]:
        if assertion.get("stage") == "human_validation_report_confirmation":
            assertion["stage"] = "validation_discard_report_response"
    paths["expected"].write_text(json.dumps(expected, ensure_ascii=False))
    case = RuntimeCase.model_validate(raw)
    price = root / "synthetic-price.json"
    price.write_text(json.dumps({"model_name": "fixture", "provider": "local-protocol-substitute",
        "currency": "USD", "source": "Synthetic fixture rates; no provider billing",
        "effective_date": "2026-11-04", "prompt_per_million": 1.0, "completion_per_million": 2.0}))
    with fixture_model(answer_factory=_v2_protocol) as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
            model=model, model_source="fixture_model", price_book_path=price)
    assert report["all_passed"], [(c.get("error_code"), c.get("runtime_status"), c.get("score")) for c in report["cases"]]
    observed = report["cases"][0]
    archive = root / "custody" / report["run_id"] / case.id
    common = dict(case=case, run_id=report["run_id"], task_id=observed["execution"]["task_id"],
        cases_sha256=report["cases_sha256"], expected_sha256=report["expected_sha256"], source=report["source"],
        model_connection_sha256=report["model_connection_sha256"])
    safety = FrozenSafetyBinding(**common, recovery_action_indices=[3])
    pipeline = observed["execution"]["validation_pipeline"]
    business = FrozenValidationBinding(**common, input_contract_sha256=pipeline["input_contract_sha256"],
        confirmed_draft_sha256=pipeline["confirmed_draft_sha256"], report_revision=pipeline["report_revision"],
        report_sha256={r["kind"]: r["sha256"] for r in pipeline["report_files"]}, bin_count=10)
    return archive, observed["evidence_custody"]["manifest_sha256"], safety, business, paths


@pytest.mark.pmml_runtime
def test_real_validation_response_loss_recovers_without_reexecuting(validation_recovery_originals):
    archive, sha, safety, _, _ = validation_recovery_originals
    result = revalidate_validation_recovery_archive(archive, expected_manifest_sha256=sha, frozen_binding=safety)
    assert result["supported_checks_verified"], result
    assert result["values"]["confirmation_count"] == result["values"]["new_jobs"] == 1
    assert result["values"]["metrics_unchanged"]


@pytest.mark.pmml_runtime
@pytest.mark.parametrize("math_status", [None, "mismatch", "unavailable"])
def test_formal_validation_recovery_combines_original_fault_and_independent_math(
    validation_recovery_originals, tmp_path, monkeypatch, math_status,
):
    from marvis.orchestrator.eval.formal_benchmark import FormalCase
    from marvis.orchestrator.eval.formal_safety_adapters import future_fault_contract, retained_safety
    from marvis.orchestrator.eval.runtime_run_manifest import verify_run_manifest
    from marvis.orchestrator.eval.runtime_scoring import ExpectedSuite
    from test_acceptance_trust import put
    from test_formal_benchmark import frozen_fixture

    archive, sha, safety, business, paths = validation_recovery_originals
    shutil.copytree(archive, tmp_path / "archive")
    safety_ref = put(tmp_path, "safety.json", safety.model_dump())
    business_ref = put(tmp_path, "business.json", business.model_dump())
    public = archive.parents[2] / "public" / safety.run_id
    run = verify_run_manifest(public, expected_manifest_sha256=digest((public / "final-manifest.json").read_bytes()))
    expected = ExpectedSuite.model_validate_json(paths["expected"].read_bytes()).cases[safety.case.id]
    template = next(c for c in frozen_fixture().cases if c.family == "validation" and c.cohort == "recovery")
    case = FormalCase.model_validate({**template.model_dump(), "id": safety.case.id,
        "budget": safety.case.budget.model_dump(),
        "case_sha256": digest(safety.case.model_dump()), "expected_sha256": digest(expected.model_dump()),
        "fault_contract_sha256": digest(future_fault_contract(safety.case))})
    item = {"case_id": case.id, "archive": "archive", "manifest_sha256": sha, "binding": safety_ref}
    record = {"input_hashes": {r["path"]: r["sha256"] for r in (safety_ref, business_ref)}}
    if math_status is not None:
        # A composition fault: the real native recovery remains verified, but
        # the independent math reader's failure must remain visible. The None
        # case above/below runs the actual math reader without substitution.
        monkeypatch.setattr("marvis.orchestrator.eval.formal_acceptance._validation_domain", lambda *_args: ({
            "domain_status": math_status, "actual_result": "failed",
            "domain_checks": {"word_confirmed_text": {"status": math_status}},
            "unsupported": {"injected_reader_failure": "composition-test-only"},
        }, None))
    value = retained_safety(case, expected, item, tmp_path, record, run, domain_item={**item, "binding": business_ref})
    if math_status is not None:
        assert value["domain"]["domain_status"] == math_status
        assert value["domain"]["actual_result"] == "failed"
        assert value["domain"]["domain_checks"]["original_business_math"]["checks"]["word_confirmed_text"]["status"] == math_status
        assert value["domain"]["unsupported"]["original_business_math"]["injected_reader_failure"] == "composition-test-only"
        assert value["domain"]["fault_contract_status"] == "verified"
        return
    assert value["domain"]["domain_status"] == "verified", value
    assert value["domain"]["actual_result"] == "recovered"
    assert value["domain"]["fault_contract_status"] == "verified"
    assert value["safety"]["authorization"] == "unavailable"


@pytest.mark.pmml_runtime
@pytest.mark.parametrize("damage", [
    "missing", "metric", "repeated_post", "duplicate_confirmation", "before_other_task",
    "before_wrong_revision", "response_missing", "response_wrong_task", "wrong_confirmed_draft",
    "wrong_draft_edit_revision", "response_wrong_confirmation", "readback_before_confirmation",
    "download_differs_from_file",
])
def test_validation_recovery_rejects_changed_originals(validation_recovery_originals, tmp_path, damage):
    from test_runtime_archive_validation import _rewrite_manifest
    from marvis.orchestrator.eval.runtime_native_json import decode_native, encode_native

    archive, _, safety, _, _ = validation_recovery_originals
    target = tmp_path / "archive"
    shutil.copytree(archive, target)
    private_path = target / "private.json"
    private = decode_native(json.loads(private_path.read_text()))
    fault = private["validation_faults"][0]
    if damage == "missing":
        fault.pop("recovery")
    elif damage == "metric":
        fault["before"]["state"]["metric_hashes"]["validation_results.json"] = "f" * 64
    elif damage == "duplicate_confirmation":
        fault["before"]["state"]["confirmation_audits"] = copy.deepcopy(fault["after_response"]["state"]["confirmation_audits"])
    elif damage == "before_other_task":
        fault["before"]["state"]["task"]["id"] = "other-task"
    elif damage == "before_wrong_revision":
        fault["before"]["state"]["task"]["report_values_revision"] += 1
    elif damage == "response_missing":
        fault.pop("received_response")
    elif damage == "response_wrong_task":
        fault["received_response"]["task_id"] = "other-task"
    elif damage == "wrong_confirmed_draft":
        fault["request"]["draft_message_id"] = "nonexistent-draft"
    elif damage == "wrong_draft_edit_revision":
        fault["request"]["draft_edit_revision"] += 1
    elif damage == "response_wrong_confirmation":
        message = next(m for m in fault["received_response"]["messages"] if m["stage"] == "word_conclusion_confirmed")
        message["id"] = "nonexistent-confirmation"
    else:
        path = target / "bindings.json"
        value = json.loads(path.read_text())
        events = value["execution_before_scoring"]["http_events"]
        if damage == "readback_before_confirmation":
            for event in events:
                if event["ordinal"] in fault["recovery"]["http_ordinals"]:
                    event.update(started_elapsed_ns=0, finished_elapsed_ns=1)
        elif damage == "download_differs_from_file":
            next(d for d in fault["recovery"]["downloads"] if d["kind"] == "word")["sha256"] = "f" * 64
            next(e for e in events if e["stage"] == "download_validation_word")["sha256"] = "f" * 64
        else:
            ordinal = fault["recovery"]["http_ordinals"][0]
            next(e for e in events if e["ordinal"] == ordinal)["method"] = "POST"
        path.write_text(json.dumps(value))
    private_path.write_text(json.dumps(encode_native(private)))
    result = revalidate_validation_recovery_archive(target, expected_manifest_sha256=_rewrite_manifest(target), frozen_binding=safety)
    assert result["checks"]["archive_integrity"]["status"] == "verified", result
    assert not result["supported_checks_verified"], result


def test_validation_fault_contract_requires_exact_confirmation_then_readback(tmp_path):
    paths = write_synthetic_suite(tmp_path / "suite", normal_validation_agent_only=True)
    raw = json.loads(paths["cases"].read_text())["cases"][0]
    raw["actions"][1]["fault"] = {"kind": "discard_success_response_once"}
    with pytest.raises(ValueError):
        RuntimeCase.model_validate(raw)
    raw["actions"].append({"kind": "recover_validation_report", "content": "readback", "fault_source_action": 1})
    with pytest.raises(ValueError):
        RuntimeCase.model_validate(raw)
