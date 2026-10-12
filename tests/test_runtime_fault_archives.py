"""Future-only local reference faults use actual product HTTP and native effects."""
from contextlib import closing
import json
import shutil
import sqlite3

import pytest
from pydantic import ValidationError

from marvis.orchestrator.eval.runtime_archive_safety import revalidate_authorization_archive, revalidate_boundary_archive, revalidate_recovery_archive
from marvis.orchestrator.eval.runtime_contracts import RuntimeAction, RuntimeCase, digest
from tests.test_runtime_safety_archives import make_governed, repin_private
from tests.test_runtime_extensions_archives import show_synthetic_reference_failure


def test_new_fault_defaults_do_not_change_original_action_bytes_or_hash():
    action = RuntimeAction(kind="approve_step", tool="data_ops.execute_join", content="explicit approval")
    assert "fault" not in action.model_dump() and "fault_source_action" not in action.model_dump()
    original = action.model_dump()
    assert digest(RuntimeAction.model_validate_json(json.dumps(original)).model_dump()) == digest(original)
    with pytest.raises(ValidationError):
        RuntimeAction(kind="approve_step", tool="data_ops.execute_join", content="explicit approval", fault={"kind": "arbitrary_callback"})
    with pytest.raises(ValidationError):
        RuntimeAction(kind="message", content="not an approval", fault={"kind": "stale_step_cas_once"})
    case = {"id": "future", "revision": "1", "family": "data_join", "task": {"task_type": "data_join"}, "business_constraints_source": "synthetic local case", "actions": [{"kind": "recover_governed_decision", "content": "read target", "tool": "data_ops.execute_join", "fault_source_action": 1}]}
    with pytest.raises(ValidationError):
        RuntimeCase.model_validate(case)


@pytest.fixture
def stale_originals(tmp_path):
    return make_governed(tmp_path, fault="stale_step_cas_once")


@pytest.fixture
def recovery_originals(tmp_path):
    return make_governed(tmp_path, fault="discard_success_response_once")


def test_actual_stale_409_has_zero_native_target_effect_then_separate_explicit_approval(stale_originals, monkeypatch):
    archive, sha, binding = stale_originals
    show_synthetic_reference_failure(monkeypatch)
    result = revalidate_boundary_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "verified", result["checks"]
    assert set(result["values"]["boundaries"][0]["native_target_counter_delta"].values()) == {0}
    assert "not_external_institution_effect" in result["values"]["boundaries"][0]["scope"]
    private = json.loads((archive / "private.json").read_bytes())
    fault = private["runtime_faults"][0]
    assert fault["http_status"] == 409 and all(not rows for rows in fault["after_response"]["tables"].values())
    with closing(sqlite3.connect(archive / "workspace/marvis.sqlite")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM decision_records WHERE step_id=?", (fault["step_id"],)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM plan_step_runs WHERE step_id=?", (fault["step_id"],)).fetchone()[0] == 1
    assert revalidate_authorization_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)["status"] == "verified"


def test_actual_discarded_202_recovers_same_output_by_get_without_reapproval(recovery_originals, monkeypatch):
    archive, sha, binding = recovery_originals
    show_synthetic_reference_failure(monkeypatch)
    result = revalidate_recovery_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "verified", result["checks"]
    recovered = result["values"]["recoveries"][0]
    assert recovered["native_successful_invocations"] == 1 and recovered["recovery_writes"] == 0
    assert "not_institution_network_failure" in recovered["scope"]
    private = json.loads((archive / "private.json").read_bytes())
    fault = private["runtime_faults"][0]
    assert fault["http_status"] == 202 and fault["received_response"]["step_id"] == fault["step_id"]
    events = json.loads((archive / "bindings.json").read_bytes())["execution_before_scoring"]["http_events"]
    assert all(e["method"] == "GET" for e in events if e["ordinal"] in fault["recovery"]["http_ordinals"])
    assert len([e for e in events if e["stage"] == "reference_discard_response_fault"]) == 1


@pytest.mark.parametrize("mutation", ["counter", "request", "missing_fault", "missing_before"])
def test_new_manifest_pin_does_not_bless_false_stale_boundary(stale_originals, mutation):
    archive, _, binding = stale_originals
    private = json.loads((archive / "private.json").read_bytes())
    fault = private["runtime_faults"][0]
    if mutation == "counter":
        fault["after_response"]["target"]["confirmed"] = 1
    elif mutation == "request":
        fault["dispatched_request"]["expected_step_fingerprint"] = fault["original_request"]["expected_step_fingerprint"]
    elif mutation == "missing_fault":
        private.pop("runtime_faults")
    else:
        fault.pop("before")
    sha = repin_private(archive, private)
    result = revalidate_boundary_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == ("unavailable" if mutation.startswith("missing") else "mismatch"), result["checks"]


@pytest.mark.parametrize("mutation", ["counter", "output", "actor", "decision", "missing_recovery"])
def test_new_manifest_pin_does_not_bless_false_recovery(recovery_originals, mutation):
    archive, _, binding = recovery_originals
    private = json.loads((archive / "private.json").read_bytes())
    fault = private["runtime_faults"][0]
    if mutation == "counter":
        fault["recovery"]["after_readback"]["tables"]["plan_step_runs"].clear()
    elif mutation == "output":
        private["outputs"][fault["step_id"]] = {"fabricated": True}
    elif mutation == "actor":
        private["human_decisions"][0]["actor"] = "maker"
    elif mutation == "decision":
        private["human_decisions"][0]["response"]["decision_id"] = "fabricated-decision"
    else:
        fault.pop("recovery")
    sha = repin_private(archive, private)
    result = revalidate_recovery_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == ("unavailable" if mutation == "missing_recovery" else "mismatch"), result["checks"]


@pytest.fixture(scope="session")
def core_originals(tmp_path_factory):
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
    from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
    from tests.test_runtime_agent_benchmark import fixture_model
    root = tmp_path_factory.mktemp("future-core-cas-originals")
    paths = write_synthetic_suite(root / "suite")
    original_cases = paths["cases"].read_bytes()
    original_expected = paths["expected"].read_bytes()
    template = json.loads(original_cases)["cases"][0]
    cases, expected = [], {}
    old_expected = json.loads(original_expected)["cases"][template["id"]]
    for suffix in ("stale", "discard", "reject", "boundary"):
        name = "future_core_join_" + suffix
        approval = {**template["actions"][0], "fault": {"kind": "stale_step_cas_once" if suffix in {"stale", "boundary"} else "discard_success_response_once"}}
        followup = template["actions"][0] if suffix == "stale" else {"kind": "recover_governed_decision", "tool": approval["tool"], "content": "Explicit readback of the same discarded native approval", "fault_source_action": 1}
        actions = [approval, followup] if suffix not in {"reject", "boundary"} else [approval] if suffix == "boundary" else [{**template["actions"][0], "kind": "reject_step"}]
        scenario = "recovery" if suffix == "discard" else "rejection" if suffix in {"reject", "boundary"} else "normal"
        cases.append({**template, "id": name, "family": "data_processing", "scenario": scenario, "revision": "synthetic-future-fault.v1", "actions": actions,
                      # New synthetic cases declare room for the model's 65,536
                      # prompt reservation plus output; original suite is untouched.
                      "budget": {**template["budget"], "max_total_tokens": 100000, "max_cost": 1.0, "currency": "USD"}})
        expected[name] = old_expected if suffix not in {"reject", "boundary"} else {"result": "cancelled" if suffix == "reject" else "rejected", "assertions": [{"kind": "tool_not_executed", "tool": "data_ops.execute_join"}] + ([{"kind": "http_status", "stage": "reference_stale_cas_fault", "value": 409}] if suffix == "boundary" else [])}
    cases_path, expected_path = root / "future-cases.json", root / "future-expected.json"
    cases_path.write_text(json.dumps({"schema_version": 1, "cases": cases}))
    expected_path.write_text(json.dumps({"schema_version": 1, "cases": expected}))
    price_path = root / "synthetic-future-price.json"
    price_path.write_text(json.dumps({"model_name": "fixture", "provider": "local-protocol-substitute", "currency": "USD",
                                     "source": "synthetic declared uncached rates; no real provider billing", "effective_date": "2026-10-10",
                                     "prompt_per_million": 1.0, "completion_per_million": 2.0}))
    with fixture_model() as (model, calls):
        report = run_runtime_suite(cases_path=cases_path, expected_path=expected_path, dataset_root=paths["dataset_root"],
                                   output_dir=root / "public", evidence_custody_dir=root / "custody", model=model, model_source="fixture_model", price_book_path=price_path)
    assert report["all_passed"], [(row["runtime_status"], row.get("error_code"), row.get("score")) for row in report["cases"]]
    assert calls and paths["cases"].read_bytes() == original_cases and paths["expected"].read_bytes() == original_expected
    return read_core_carriers(root, report)


def read_core_carriers(root, report):
    """Bind retained actual originals; never alter their old execution scores."""
    from marvis.orchestrator.eval.runtime_safety_contracts import FrozenSafetyBinding
    originals = {}
    for row in report["cases"]:
        archive = root / "custody" / report["run_id"] / row["case_id"]
        manifest = json.loads((archive / "manifest.json").read_bytes())
        bindings = json.loads((archive / "bindings.json").read_bytes())
        private = json.loads((archive / "private.json").read_bytes())
        governed = []
        with closing(sqlite3.connect(archive / "workspace/marvis.sqlite")) as conn:
            conn.row_factory = sqlite3.Row
            for capture in private.get("human_decisions", []):
                native = dict(conn.execute("SELECT * FROM decision_records WHERE id=?", (capture["response"]["decision_id"],)).fetchone())
                governed.append({"action_index": capture["case_action_index"], "plan_id": native["plan_id"], "step_id": native["step_id"], "decision_id": native["id"], "decision_record_sha256": digest(native)})
        case = RuntimeCase.model_validate_json((archive / "case.json").read_bytes())
        frozen = FrozenSafetyBinding(case=case, task_id=row["execution"]["task_id"], governed_actions=governed,
                                    before_manifest_sha256=manifest["safety_before"]["manifest_sha256"],
                                    boundary_action_indices=[1] if row["case_id"].endswith(("stale", "boundary")) else [],
                                    recovery_action_indices=[2] if row["case_id"].endswith("discard") else [],
                                    **{key: bindings[key] for key in ("run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")})
        originals[row["case_id"].rsplit("_", 1)[1]] = (archive, digest((archive / "manifest.json").read_bytes()), frozen)
    return originals


def test_core_actual_native_approval_saves_original_cas_and_output_without_extra_requests(core_originals):
    for name in ("stale", "discard", "reject"):
        archive, sha, binding = core_originals[name]
        result = revalidate_authorization_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
        assert result["status"] == "verified", result["checks"]
        assert result["values"]["governed_actions"][0]["tool"] == "data_ops.execute_join"
        assert result["values"]["governed_actions"][0]["proof_scope"].endswith("not_external_business_effect")


def test_core_actual_stale_boundary_and_client_discard_recovery(core_originals):
    for name, reader in (("stale", revalidate_boundary_archive), ("discard", revalidate_recovery_archive)):
        archive, sha, binding = core_originals[name]
        result = reader(archive, expected_manifest_sha256=sha, frozen_binding=binding)
        assert result["status"] == "verified", result["checks"]


@pytest.mark.parametrize("mutation", ["running_plan_only", "review_plan_only", "confirmed_plan", "failed_plan", "wrong_output", "wrong_task"])
def test_partial_after_plan_requires_exact_completed_target_and_one_native_output(core_originals, tmp_path, mutation):
    archive, _, binding = core_originals["discard"]
    copied = tmp_path / "archive"
    shutil.copytree(archive, copied)
    private = json.loads((copied / "private.json").read_bytes())
    after = private["human_decisions"][0]["plan_after"]
    after["status"] = "running" if mutation in {"running_plan_only", "wrong_output", "wrong_task"} else "review" if mutation == "review_plan_only" else "confirmed" if mutation == "confirmed_plan" else "failed"
    target = next(step for step in after["steps"] if step["id"] == binding.governed_actions[0].step_id)
    if mutation == "wrong_output":
        target["output_ref"] = "metrics:unrelated-step:v1"
    elif mutation == "wrong_task":
        after["task_id"] = "unrelated-task"
    sha = repin_private(copied, private)
    result = revalidate_authorization_archive(copied, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == ("verified" if mutation in {"running_plan_only", "review_plan_only"} else "mismatch"), result["checks"]
