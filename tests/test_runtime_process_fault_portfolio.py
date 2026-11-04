"""Real portfolio report commit after its explicit gate, then worker SIGKILL."""
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3

import pytest
from pydantic import ValidationError

from marvis.orchestrator.eval.runtime_archive_safety import revalidate_recovery_archive
from marvis.orchestrator.eval.runtime_contracts import RuntimeAction, RuntimeCase, digest
from marvis.orchestrator.eval.runtime_fault_inputs import RuntimeFault
from marvis.orchestrator.eval.runtime_safety_contracts import FrozenSafetyBinding


def test_downstream_fault_preserves_old_wire_and_rejects_arbitrary_targets():
    assert RuntimeFault(kind="worker_crash_after_commit_once").model_dump() == {"kind": "worker_crash_after_commit_once"}
    for definition in ({"kind": "tool_timeout_once", "downstream_tool": "analysis.portfolio_report"},
                       {"kind": "worker_crash_after_commit_once", "downstream_tool": "data_ops.execute_join"}):
        with pytest.raises(ValidationError):
            RuntimeFault.model_validate(definition)
    action = {"kind": "approve_step", "tool": "analysis.portfolio_gate_summary", "content": "explicit native approval",
              "fault": {"kind": "worker_crash_after_commit_once", "downstream_tool": "analysis.portfolio_report"}}
    assert RuntimeAction.model_validate(action).model_dump()["fault"] == action["fault"]
    with pytest.raises(ValidationError):
        RuntimeAction.model_validate(action | {"tool": "modeling.train_model"})


@pytest.fixture(scope="module")
def portfolio_commit_originals(tmp_path_factory):
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
    from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
    from tests.test_runtime_agent_benchmark import fixture_model
    from tests.test_runtime_workflow_families import _business_protocol

    root = tmp_path_factory.mktemp("native-portfolio-after-commit")
    paths = write_synthetic_suite(root / "suite", include_workflow_families=True)
    suite = json.loads(paths["cases"].read_bytes())
    case = next(c for c in suite["cases"] if c["id"] == "synthetic_portfolio")
    case.update(id="actual_portfolio_report_after_commit", family="risk_portfolio", scenario="recovery")
    case["actions"][-1]["fault"] = {"kind": "worker_crash_after_commit_once", "downstream_tool": "analysis.portfolio_report"}
    case["actions"].append({"kind": "recover_governed_decision", "tool": "analysis.portfolio_gate_summary",
                            "content": "Explicitly request native report recovery, retaining any unknown effect.", "fault_source_action": 4})
    case["budget"].update(max_total_tokens=100000, max_cost=1.0, currency="USD", wall_seconds=90)
    paths["cases"].write_text(json.dumps({"schema_version": 1, "cases": [case]}))
    old = json.loads(paths["expected"].read_bytes())["cases"]["synthetic_portfolio"]
    paths["expected"].write_text(json.dumps({"schema_version": 1, "cases": {case["id"]: old}}))
    price = root / "price.json"
    price.write_text(json.dumps({"model_name": "fixture", "provider": "local-protocol-substitute", "currency": "USD",
        "source": "synthetic local rates only; no actual provider billing", "effective_date": "2026-10-10",
        "prompt_per_million": 1.0, "completion_per_million": 2.0}))
    with fixture_model(answer_factory=_business_protocol) as (model, calls):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"], dataset_root=paths["dataset_root"],
            output_dir=root / "public", evidence_custody_dir=root / "custody", model=model,
            model_source="fixture_model", price_book_path=price)
    assert calls
    observed = report["cases"][0]
    archive = root / "custody" / report["run_id"] / case["id"]
    assert archive.is_dir(), observed
    raw = json.loads((archive / "bindings.json").read_bytes())
    private = json.loads((archive / "private.json").read_bytes())
    fault = private["runtime_faults"][0]
    assert fault["fault_executed"] is True, (observed, fault)
    assert fault["tool"] == "analysis.portfolio_report"
    assert fault["approval_tool"] == "analysis.portfolio_gate_summary"
    assert fault["step_id"] != fault["approval_step_id"]
    with closing(sqlite3.connect(archive / "workspace/marvis.sqlite")) as conn:
        conn.row_factory = sqlite3.Row
        decision = dict(conn.execute("SELECT * FROM decision_records WHERE id=?", (fault["received_response"]["decision_id"],)).fetchone())
    manifest = json.loads((archive / "manifest.json").read_bytes())
    binding = FrozenSafetyBinding(case=RuntimeCase.model_validate(case), task_id=observed["execution"]["task_id"],
        recovery_action_indices=[5], governed_actions=[{"action_index": 4, "plan_id": decision["plan_id"],
        "step_id": decision["step_id"], "decision_id": decision["id"], "decision_record_sha256": digest(decision)}],
        before_manifest_sha256=manifest["safety_before"]["manifest_sha256"],
        **{key: raw[key] for key in ("run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")})
    assert not Path(manifest["original_workspace"]).exists()
    return archive, digest((archive / "manifest.json").read_bytes()), binding


@pytest.mark.e2e
def test_actual_portfolio_report_commit_and_kill_are_independently_verified(portfolio_commit_originals):
    archive, sha, binding = portfolio_commit_originals
    result = revalidate_recovery_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["status"] == "verified", result
    assert result["values"]["fault_executed"] is True
    assert result["values"]["recovery_completed"] is False
    prefix = archive / "workspace/.runtime-eval-fault/launch"
    commit = json.loads((prefix / "commit.json").read_bytes())
    assert commit["table"] == "task_artifacts"
    assert commit["row"]["kind"] == "portfolio_report_xlsx"
    assert commit["audit"]["kind"] == "analysis.portfolio.report"
    assert len(commit["files"]) == 1
    data = (archive / "workspace" / commit["files"][0]["relative_path"]).read_bytes()
    assert data.startswith(b"PK") and digest(data) == commit["row"]["content_hash"]
    terminal = json.loads((prefix / "terminal.json").read_bytes())
    assert terminal["returncode"] == -9


def _repin(archive):
    path = archive / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["files"] = {p.relative_to(archive).as_posix(): {"sha256": digest(p.read_bytes()), "size_bytes": p.stat().st_size}
                         for p in archive.rglob("*") if p.is_file() and p != path}
    manifest["bindings_sha256"] = manifest["files"]["bindings.json"]["sha256"]
    path.write_text(json.dumps(manifest))
    return digest(path.read_bytes())


@pytest.mark.e2e
@pytest.mark.parametrize("damage", ["file", "dependency", "approval", "foreign_path", "producer", "input_hash", "audit"])
def test_repin_cannot_bless_wrong_portfolio_native_commit(portfolio_commit_originals, tmp_path, damage):
    original, _, binding = portfolio_commit_originals
    archive = tmp_path / "archive"
    shutil.copytree(original, archive)
    private_path = archive / "private.json"
    private = json.loads(private_path.read_bytes())
    fault = private["runtime_faults"][0]
    prefix = archive / "workspace/.runtime-eval-fault/launch"
    commit = json.loads((prefix / "commit.json").read_bytes())
    with closing(sqlite3.connect(archive / "workspace/marvis.sqlite")) as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        if damage == "file":
            (archive / "workspace" / commit["files"][0]["relative_path"]).write_bytes(b"changed workbook")
        elif damage == "dependency":
            conn.execute("UPDATE plan_steps SET depends_on_json='[]' WHERE id=?", (fault["step_id"],))
        elif damage == "approval":
            fault["approval_step_id"] = fault["step_id"]
        elif damage == "foreign_path":
            commit["row"]["path"] = commit["row"]["path"].replace('/portfolio/', '/another-task/')
            conn.execute("UPDATE task_artifacts SET path=? WHERE id=?", (commit["row"]["path"], commit["row"]["id"]))
        elif damage == "producer":
            commit["row"]["origin_tool"] = "data_ops.execute_join"
            conn.execute("UPDATE task_artifacts SET origin_tool=? WHERE id=?", (commit["row"]["origin_tool"], commit["row"]["id"]))
        elif damage == "input_hash":
            provenance = json.loads(commit["row"]["provenance_json"])
            provenance["input_hash"] = '0' * 64
            commit["row"]["provenance_json"] = json.dumps(provenance)
            conn.execute("UPDATE task_artifacts SET provenance_json=? WHERE id=?", (commit["row"]["provenance_json"], commit["row"]["id"]))
        else:
            conn.execute("DELETE FROM audit WHERE id=?", (commit["audit"]["id"],))
        conn.commit()
    # Re-pin producer claims too. The independent reader still requires the
    # actual same-task report provenance, graph, atomic audit and original bytes.
    (prefix / "commit.json").write_text(json.dumps(commit))
    fault["ready"]["commit_sha256"] = digest(commit)
    (prefix / "ready.json").write_text(json.dumps(fault["ready"]))
    signal = json.loads((prefix / "signal.json").read_bytes())
    signal["commit_sha256"] = digest(commit)
    (prefix / "signal.json").write_text(json.dumps(signal))
    private_path.write_text(json.dumps(private))
    result = revalidate_recovery_archive(archive, expected_manifest_sha256=_repin(archive), frozen_binding=binding)
    assert result["status"] == "mismatch", result
