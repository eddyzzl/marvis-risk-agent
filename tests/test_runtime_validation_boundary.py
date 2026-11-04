"""Actual report-revision rejection, independent originals and formal dispatch."""
import copy
import json
import shutil

import pytest

from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeAction, RuntimeCase, digest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from marvis.orchestrator.eval.runtime_safety_contracts import FrozenSafetyBinding
from marvis.orchestrator.eval.runtime_validation_boundary import (
    STAGE, boundary_observation, revalidate_validation_boundary_archive,
)
from test_runtime_agent_benchmark import fixture_model
from test_runtime_validation_agent import _v2_protocol


@pytest.fixture(scope="module")
def validation_boundary_originals(tmp_path_factory):
    root = tmp_path_factory.mktemp("validation-stale-report")
    paths = write_synthetic_suite(root / "suite", normal_validation_agent_only=True)
    suite = json.loads(paths["cases"].read_text())
    raw = suite["cases"][0]
    raw["budget"].update(max_total_tokens=100000, max_cost=1.0, currency="USD")
    raw["scenario"] = "rejection"
    raw["actions"][1]["fault"] = {"kind": "stale_report_revision_once"}
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    expected = {"schema_version": 1, "cases": {raw["id"]: {"result": "rejected", "assertions": [
        {"kind": "http_status", "stage": STAGE, "value": 409},
        *[{"kind": "validation_boundary_equals", "path": [key], "value": value} for key, value in
          (("confirmation_count", 0), ("new_jobs", 0), ("metrics_unchanged", True), ("revision_unchanged", True))]],
    }}}
    paths["expected"].write_text(json.dumps(expected))
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
    case = RuntimeCase.model_validate(raw)
    archive = root / "custody" / report["run_id"] / case.id
    manifest = json.loads((archive / "manifest.json").read_bytes())
    binding = FrozenSafetyBinding(case=case, run_id=report["run_id"], task_id=observed["execution"]["task_id"],
        cases_sha256=report["cases_sha256"], expected_sha256=report["expected_sha256"], source=report["source"],
        model_connection_sha256=report["model_connection_sha256"], boundary_action_indices=[2],
        before_manifest_sha256=manifest["safety_before"]["manifest_sha256"])
    return archive, observed["evidence_custody"]["manifest_sha256"], binding, paths


@pytest.mark.pmml_runtime
def test_report_stale_revision_409_has_no_confirmation_job_or_metric_mutation(validation_boundary_originals):
    archive, sha, binding, _ = validation_boundary_originals
    result = revalidate_validation_boundary_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "verified", result
    assert result["values"]["confirmation_count"] == result["values"]["new_jobs"] == 0
    assert result["values"]["http_status"] == 409
    assert result["values"]["metrics_unchanged"] and result["values"]["revision_unchanged"]
    assert result["acceptance_claim"] == "not_established"


@pytest.mark.pmml_runtime
def test_formal_boundary_dispatch_uses_original_zero_mutation_evidence(validation_boundary_originals, tmp_path):
    from marvis.orchestrator.eval.formal_benchmark import FormalCase
    from marvis.orchestrator.eval.formal_safety_adapters import retained_safety
    from marvis.orchestrator.eval.runtime_run_manifest import verify_run_manifest
    from marvis.orchestrator.eval.runtime_scoring import ExpectedSuite
    from test_acceptance_trust import put
    from test_formal_benchmark import frozen_fixture

    archive, sha, binding, paths = validation_boundary_originals
    shutil.copytree(archive, tmp_path / "archive")
    reference = put(tmp_path, "binding.json", binding.model_dump())
    public = archive.parents[2] / "public" / binding.run_id
    run = verify_run_manifest(public, expected_manifest_sha256=digest((public / "final-manifest.json").read_bytes()))
    expected = ExpectedSuite.model_validate_json(paths["expected"].read_bytes()).cases[binding.case.id]
    template = next(c for c in frozen_fixture().cases if c.family == "validation" and c.cohort == "boundary")
    case = FormalCase.model_validate({**template.model_dump(), "id": binding.case.id,
        "expected_result": "rejected", "budget": binding.case.budget.model_dump(),
        "case_sha256": digest(binding.case.model_dump()), "expected_sha256": digest(expected.model_dump())})
    item = {"case_id": case.id, "archive": "archive", "manifest_sha256": sha, "binding": reference}
    record = {"input_hashes": {reference["path"]: reference["sha256"]}}
    result = retained_safety(case, expected, item, tmp_path, record, run)
    assert result["domain"]["domain_status"] == "verified", result
    assert result["domain"]["actual_result"] == "rejected"
    assert result["safety"]["authorization"] == "unavailable"
    assert result["safety"]["point_in_time"] == "unavailable"


@pytest.mark.pmml_runtime
@pytest.mark.parametrize("damage", ["missing", "metric", "job", "revision", "draft", "http", "later_confirmation"])
def test_changed_boundary_originals_fail_closed(validation_boundary_originals, tmp_path, damage):
    from marvis.orchestrator.eval.runtime_native_json import decode_native, encode_native
    from test_runtime_archive_validation import _rewrite_manifest

    archive, _, binding, _ = validation_boundary_originals
    target = tmp_path / "archive"
    shutil.copytree(archive, target)
    path = target / "private.json"
    private = decode_native(json.loads(path.read_text()))
    fault = private["validation_faults"][0]
    if damage == "missing":
        private.pop("validation_faults")
    elif damage == "metric":
        fault["before"]["state"]["metric_hashes"]["validation_results.json"] = "f" * 64
    elif damage == "job":
        fault["after_response"]["state"]["jobs"].append({"id": "forged", "kind": "report"})
    elif damage == "revision":
        fault["dispatched_request"]["revision"] = fault["original_request"]["revision"]
    elif damage == "draft":
        fault["original_request"]["text_values"]["TEXT:final_validation_conclusion"] = "forged"
        fault["dispatched_request"]["text_values"] = copy.deepcopy(fault["original_request"]["text_values"])
    else:
        p = target / "bindings.json"
        data = json.loads(p.read_text())
        events = data["execution_before_scoring"]["http_events"]
        if damage == "http":
            next(e for e in events if e.get("ordinal") == fault["http_ordinal"])["status_code"] = 202
        else:
            # A later explicit confirmation belongs to recovery adjudication;
            # its presence must invalidate a whole-journey zero-effect claim.
            last = events[-1]
            events.append({**last, "ordinal": last["ordinal"] + 1,
                "method": "POST", "stage": "human_validation_report_confirmation", "status_code": 202,
                "started_elapsed_ns": last["finished_elapsed_ns"] + 1,
                "finished_elapsed_ns": last["finished_elapsed_ns"] + 2})
        p.write_text(json.dumps(data))
    path.write_text(json.dumps(encode_native(private)))
    result = revalidate_validation_boundary_archive(target, expected_manifest_sha256=_rewrite_manifest(target), frozen_binding=binding)
    if damage == "later_confirmation":
        assert result["checks"]["archive_integrity"]["status"] == "verified", result
    assert result["status"] == ("unavailable" if damage == "missing" else "mismatch"), result


def test_boundary_fault_is_only_allowed_in_explicit_report_rejection_or_recovery(tmp_path):
    paths = write_synthetic_suite(tmp_path / "suite", normal_validation_agent_only=True)
    raw = json.loads(paths["cases"].read_text())["cases"][0]
    original = RuntimeCase.model_validate(raw).model_dump()
    assert "fault" not in original["actions"][1]
    raw["actions"][1]["fault"] = {"kind": "stale_report_revision_once"}
    with pytest.raises(ValueError):
        RuntimeCase.model_validate(raw)
    raw["scenario"] = "rejection"
    assert RuntimeCase.model_validate(raw).actions[1].fault.kind == "stale_report_revision_once"
    with pytest.raises(ValueError):
        RuntimeAction(kind="approve_step", tool="feature.analyze_feature_bins", content="confirm", fault={"kind": "stale_report_revision_once"})
    raw["scenario"] = "recovery"
    raw["actions"].append({"kind": "recover_validation_report", "content": "confirm the current report", "fault_source_action": 2})
    recovery = RuntimeCase.model_validate(raw)
    assert recovery.actions[2].kind == "recover_validation_report"
    assert recovery.actions[2].fault_source_action == 2
    for source in (1, 3):
        changed = copy.deepcopy(raw)
        changed["actions"][2]["fault_source_action"] = source
        with pytest.raises(ValueError):
            RuntimeCase.model_validate(changed)
    changed = copy.deepcopy(raw)
    del changed["actions"][1]["fault"]
    with pytest.raises(ValueError):
        RuntimeCase.model_validate(changed)
    assert boundary_observation({"runtime_entry": "standard_validation_agent_v2"}, {}) is None
