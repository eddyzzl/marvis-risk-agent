"""Real isolated app + local protocol substitute; no paid model or claimed recovery."""
from contextlib import closing
import json
import shutil
import sqlite3

import pytest

from marvis.orchestrator.eval.runtime_archive_safety import revalidate_recovery_archive
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from marvis.orchestrator.eval.runtime_safety_contracts import FrozenSafetyBinding
from tests.test_runtime_safety_archives import repin_private


@pytest.fixture(scope="module")
def originals(tmp_path_factory):
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
    from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
    from tests.test_runtime_agent_benchmark import fixture_model

    root = tmp_path_factory.mktemp("actual-process-fault-archives")
    paths = write_synthetic_suite(root / "suite")
    template = json.loads(paths["cases"].read_bytes())["cases"][0]
    old_expected = json.loads(paths["expected"].read_bytes())["cases"][template["id"]]
    cases, expected = [], {}
    for kind in ("worker_crash_after_commit_once", "tool_timeout_once", "cancel_then_resume_once", "service_restart_mid_step_once"):
        name = "actual_" + kind
        approval = {**template["actions"][0], "fault": {"kind": kind}}
        followup = {"kind": "recover_governed_decision", "tool": approval["tool"],
                    "content": "Try native recovery without bypassing unknown effects", "fault_source_action": 1}
        cases.append({**template, "id": name, "family": "data_processing", "scenario": "recovery",
                      "revision": "actual-process-fault.v1", "actions": [approval, followup],
                      "budget": {**template["budget"], "max_total_tokens": 100000, "max_cost": 1.0, "currency": "USD"}})
        expected[name] = old_expected
    cases_path, expected_path = root / "cases.json", root / "expected.json"
    cases_path.write_text(json.dumps({"schema_version": 1, "cases": cases}))
    expected_path.write_text(json.dumps({"schema_version": 1, "cases": expected}))
    price = root / "price.json"
    price.write_text(json.dumps({"model_name": "fixture", "provider": "local-protocol-substitute", "currency": "USD",
        "source": "synthetic local rates only; no actual provider billing", "effective_date": "2026-10-10",
        "prompt_per_million": 1.0, "completion_per_million": 2.0}))
    with fixture_model() as (model, calls):
        report = run_runtime_suite(cases_path=cases_path, expected_path=expected_path, dataset_root=paths["dataset_root"],
            output_dir=root / "public", evidence_custody_dir=root / "custody", model=model,
            model_source="fixture_model", price_book_path=price)
    assert calls
    result = {}
    for row in report["cases"]:
        archive = root / "custody" / report["run_id"] / row["case_id"]
        assert archive.is_dir(), row
        manifest = json.loads((archive / "manifest.json").read_bytes())
        bindings = json.loads((archive / "bindings.json").read_bytes())
        private = json.loads((archive / "private.json").read_bytes())
        fault = private["runtime_faults"][0]
        assert fault.get("fault_executed") is True, (row, fault)
        assert "recovery" in fault, (row, fault)
        with closing(sqlite3.connect(archive / "workspace/marvis.sqlite")) as conn:
            conn.row_factory = sqlite3.Row
            native = dict(conn.execute("SELECT * FROM decision_records WHERE id=?",
                                       (fault["received_response"]["decision_id"],)).fetchone())
        frozen = FrozenSafetyBinding(case=RuntimeCase.model_validate_json((archive / "case.json").read_bytes()),
            task_id=row["execution"]["task_id"], recovery_action_indices=[2],
            governed_actions=[{"action_index": 1, "plan_id": native["plan_id"], "step_id": native["step_id"],
                               "decision_id": native["id"], "decision_record_sha256": digest(native)}],
            before_manifest_sha256=manifest["safety_before"]["manifest_sha256"],
            **{key: bindings[key] for key in ("run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")})
        result[fault["kind"]] = (archive, digest((archive / "manifest.json").read_bytes()), frozen)
    return result


@pytest.mark.e2e
def test_fault_execution_and_unsatisfied_recovery_are_independently_verified(originals):
    for archive, sha, binding in originals.values():
        result = revalidate_recovery_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
        assert result["status"] == "verified", result
        assert result["values"]["fault_executed"] is True
        assert result["values"]["recovery_completed"] is False


@pytest.mark.e2e
@pytest.mark.parametrize("mutation", ["wrong_pid", "missing_commit", "fake_recovered", "wrong_http", "fake_timeout"])
def test_new_manifest_pin_does_not_bless_forged_process_fault(originals, tmp_path, mutation):
    kind = "tool_timeout_once" if mutation == "fake_timeout" else "worker_crash_after_commit_once"
    archive, _, binding = originals[kind]
    copied = tmp_path / "archive"
    shutil.copytree(archive, copied)
    private = json.loads((copied / "private.json").read_bytes())
    row = private["runtime_faults"][0]
    if mutation == "wrong_pid":
        row["ready"]["worker"]["pid"] = row["ready"]["worker"]["ppid"]
    elif mutation == "missing_commit":
        row["ready"]["commit_sha256"] = "0" * 64
    elif mutation == "fake_recovered":
        row["recovery"]["outcome"] = "done"
    elif mutation == "wrong_http":
        row["http_ordinal"] += 1
    else:
        row["ready"]["phase"] = "after_business_commit_before_protocol"
    sha = repin_private(copied, private)
    result = revalidate_recovery_archive(copied, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "mismatch", result
