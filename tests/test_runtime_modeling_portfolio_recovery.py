"""Exact modeling/portfolio client response loss on their actual native gates."""
from contextlib import closing
import json
import shutil
import sqlite3

import pytest

from marvis.orchestrator.eval.runtime_archive_safety import revalidate_recovery_archive
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from marvis.orchestrator.eval.runtime_safety_contracts import FrozenSafetyBinding
from tests.test_runtime_safety_archives import repin_private


def _run_case(root, family, fault_kind="discard_success_response_once", downstream=None, *, keep_followups=False):
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
    from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
    from tests.test_runtime_agent_benchmark import fixture_model
    from tests.test_runtime_normal_modeling import _modeling_protocol
    from tests.test_runtime_workflow_families import _business_protocol

    paths = write_synthetic_suite(root / "suite", **({"normal_modeling_only": True} if family == "modeling"
                                                    else {"include_workflow_families": True}))
    old_name = "synthetic_normal_modeling" if family == "modeling" else "synthetic_portfolio"
    case = next(c for c in json.loads(paths["cases"].read_bytes())["cases"] if c["id"] == old_name)
    old_expected = json.loads(paths["expected"].read_bytes())["cases"][old_name]
    case.update(id=f"actual_{family}_{fault_kind}", family=family, scenario="recovery")
    tool = "modeling.configure_tuning" if family == "modeling" else "analysis.portfolio_gate_summary"
    index = next(i for i,a in enumerate(case["actions"]) if a.get("tool") == tool)
    case["actions"][index]["fault"] = {"kind": fault_kind}
    if downstream:
        case["actions"][index]["fault"]["downstream_tool"] = downstream
    if not keep_followups:
        case["actions"] = case["actions"][:index+1]
    case["actions"].insert(index+1, {"kind": "recover_governed_decision", "tool": tool,
        "content": "明确读取本任务原审批的实际结果；不重放写入，不绕过未知影响。", "fault_source_action": index+1})
    case["budget"].update(max_total_tokens=250000, max_cost=2.0, currency="USD", wall_seconds=300)
    paths["cases"].write_text(json.dumps({"schema_version": 1, "cases": [case]}))
    paths["expected"].write_text(json.dumps({"schema_version": 1, "cases": {case["id"]: old_expected}}))
    price = root / "price.json"
    price.write_text(json.dumps({"model_name": "fixture", "provider": "local-protocol-substitute", "currency": "USD",
        "source": "synthetic local rates only; no actual provider billing", "effective_date": "2026-10-10",
        "prompt_per_million": 1.0, "completion_per_million": 2.0}))
    protocol = _modeling_protocol if family == "modeling" else _business_protocol
    with fixture_model(answer_factory=protocol) as (model, calls):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"], dataset_root=paths["dataset_root"],
            output_dir=root / "public", evidence_custody_dir=root / "custody", model=model,
            model_source="fixture_model", price_book_path=price)
    assert calls
    if keep_followups:
        assert report["all_passed"], report["cases"]
    observed = report["cases"][0]
    archive = root / "custody" / report["run_id"] / case["id"]
    assert archive.is_dir(), observed
    private = json.loads((archive / "private.json").read_bytes())
    fault = private["runtime_faults"][0]
    assert "recovery" in fault, (observed, fault)
    raw = json.loads((archive / "bindings.json").read_bytes())
    manifest = json.loads((archive / "manifest.json").read_bytes())
    with closing(sqlite3.connect(archive / "workspace/marvis.sqlite")) as conn:
        conn.row_factory = sqlite3.Row
        decision = dict(conn.execute("SELECT * FROM decision_records WHERE id=?", (fault["received_response"]["decision_id"],)).fetchone())
    binding = FrozenSafetyBinding(case=RuntimeCase.model_validate(case), task_id=observed["execution"]["task_id"],
        recovery_action_indices=[index+2], governed_actions=[{"action_index": index+1, "plan_id": decision["plan_id"],
        "step_id": decision["step_id"], "decision_id": decision["id"], "decision_record_sha256": digest(decision)}],
        before_manifest_sha256=manifest["safety_before"]["manifest_sha256"],
        **{key: raw[key] for key in ("run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")})
    return archive, digest((archive / "manifest.json").read_bytes()), binding


@pytest.fixture(scope="module", params=["modeling", "risk_portfolio"])
def response_loss_originals(tmp_path_factory, request):
    return _run_case(tmp_path_factory.mktemp(request.param + "-response-loss"), request.param)


def test_exact_family_response_loss_then_readback_has_one_native_write(response_loss_originals):
    archive, sha, binding = response_loss_originals
    result = revalidate_recovery_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "verified", result
    assert len(result["values"]["recoveries"]) == 1
    proof = result["values"]["recoveries"][0]
    assert proof["native_successful_invocations"] == 1
    assert proof["recovery_writes"] == 0


@pytest.mark.parametrize("damage", ["wrong_step", "fake_readback"])
def test_exact_family_response_loss_repin_does_not_authorize_another_operation(response_loss_originals, tmp_path, damage):
    original, _, binding = response_loss_originals
    archive = tmp_path / "archive"
    shutil.copytree(original, archive)
    private = json.loads((archive / "private.json").read_bytes())
    fault = private["runtime_faults"][0]
    if damage == "wrong_step":
        fault["step_id"] = "another-native-step"
    else:
        fault["recovery"]["http_ordinals"] = [fault["http_ordinal"]]
    result = revalidate_recovery_archive(archive, expected_manifest_sha256=repin_private(archive, private), frozen_binding=binding)
    assert result["status"] == "mismatch", result
