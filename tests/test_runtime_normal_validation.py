"""Actual compatibility validation HTTP, kernel, PMML, tools and report carriers."""

import copy
import json
import time

import httpx
import pytest
from pydantic import ValidationError

from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import Material, RuntimeAction, RuntimeCase
from marvis.orchestrator.eval.runtime_runner import (
    Journey, RuntimeJourneyError, _validation_report_files, run_runtime_suite,
)
from marvis.orchestrator.eval.runtime_scoring import runtime_task_coverage
from test_runtime_agent_benchmark import fixture_model


def _validation_protocol(request, answer, payload):
    # The model is the only fixture: no app, kernel, scoring or report mocking.
    if "output_summary" in request:
        return {"passed": True, "reasons": []}
    if "business_acceptance" in request:
        return {"summary": "公开合成验证完成，真实业务验收未成立。", "open_items": [],
                "goal_doubt": False, "goal_met": None}
    return answer


def test_compatibility_workflow_executes_notebook_pmml_and_real_reports(tmp_path):
    paths = write_synthetic_suite(tmp_path / "suite", normal_validation_only=True)
    with fixture_model(answer_factory=_validation_protocol) as (model, calls):
        report = run_runtime_suite(
            cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=tmp_path / "runs",
            model=model, model_source="fixture_model",
        )
    assert report["all_passed"], json.dumps(report, ensure_ascii=False, indent=2)
    assert calls
    case = report["cases"][0]
    assert case["runtime_entry"] == "manual_compatibility_workflow"
    assert case["human_interventions"] == 3
    assert report["denominator"] == 1
    assert report["a_evidence_case_count"] == 0
    assert report["acceptance_claim"] == "not_established"
    coverage = report["runtime_task_coverage"]
    assert coverage["cells"]["validation"]["normal"]["denominator"] == 0
    assert coverage["manual_compatibility_case_ids"] == [case["case_id"]]


def _case(tmp_path):
    paths = write_synthetic_suite(tmp_path / "suite", normal_validation_only=True)
    raw = json.loads(paths["cases"].read_text())["cases"][0]
    return paths, raw


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_role", "wrong_start", "wrong_gate", "algorithm", "initial"])
def test_validation_contract_rejects_incomplete_or_ambiguous_journeys(tmp_path, mutation):
    _, raw = _case(tmp_path)
    if mutation == "missing":
        raw["materials"].pop()
    elif mutation == "duplicate":
        raw["materials"][1] = copy.deepcopy(raw["materials"][0])
    elif mutation == "wrong_role":
        raw["materials"][1]["role"] = "unknown"
    elif mutation == "wrong_start":
        raw["actions"][0]["kind"] = "message"
    elif mutation == "wrong_gate":
        raw["actions"][1]["tool"] = "strategy.adopt_strategy"
    elif mutation == "algorithm":
        del raw["task"]["algorithm"]
    else:
        raw["initial_message"] = "bypass"
    with pytest.raises(ValidationError):
        RuntimeCase.model_validate(raw)


@pytest.mark.parametrize("extra", [{"route": "/bypass"}, {"callback": "code"}, {"tool": "v1_compat.run_notebook"}, {"content": ""}, {"plan_id": "invented"}])
def test_validation_start_has_no_route_code_or_identifier_injection(extra):
    with pytest.raises(ValidationError):
        RuntimeAction.model_validate({"kind": "start_validation_workflow", "content": "确认本地验证", **extra})


@pytest.mark.parametrize("role,path", [("notebook", "model.py"), ("pmml", "model.pkl"), ("sample", "sample.ipynb"), ("dictionary", "../dict.csv"), ("pmml", "c:\\model.pmml")])
def test_material_role_extension_and_path_are_strict(role, path):
    with pytest.raises(ValidationError):
        Material(path=path, role=role, sha256="a" * 64)


def test_changed_material_is_rejected_before_any_http_upload(tmp_path):
    paths, raw = _case(tmp_path)
    (paths["dataset_root"] / "model.pmml").write_text("changed")
    class Client:
        def request(self, *args, **kwargs):
            pytest.fail("changed materials must fail before upload")
    journey = Journey(Client(), RuntimeCase.model_validate(raw), time.monotonic() + 10)
    with pytest.raises(RuntimeJourneyError, match="material_identity_mismatch"):
        journey.prepare_validation_task(paths["dataset_root"], tmp_path / "workspace")


@pytest.mark.parametrize("mutation", ["template", "foreign_task", "tool", "input", "stale"])
def test_start_revalidates_native_plan_and_does_not_retry_stale_confirmation(tmp_path, mutation):
    _, raw = _case(tmp_path)
    tools = ["scan_materials", "run_notebook", "compute_validation_metrics", "render_reports"]
    plan = {"id": "native", "task_id": "task", "template_id": "model_validation", "status": "validated",
            "steps": [{"tool_ref": "v1_compat." + t, "inputs": {"task_id": "task"}} for t in tools],
            "confirmation_snapshot": {"expected_plan_status": "validated", "expected_plan_revision": 1, "expected_plan_fingerprint": "current"}}
    if mutation == "template":
        plan["template_id"] = "other"
    elif mutation == "foreign_task":
        plan["task_id"] = "other"
    elif mutation == "tool":
        plan["steps"][0]["tool_ref"] = "strategy.adopt_strategy"
    elif mutation == "input":
        plan["steps"][0]["inputs"] = {"task_id": "other"}
    class Client:
        calls = []
        def request(self, method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            if path == "/api/tasks/task/plans" and method == "GET":
                return httpx.Response(200, json={"plans": []})
            if path.endswith("/confirm"):
                return httpx.Response(409, json={"detail": "stale snapshot"})
            return httpx.Response(200, json={"plan": plan})
    client = Client()
    journey = Journey(client, RuntimeCase.model_validate(raw), time.monotonic() + 10)
    journey.task_id = "task"
    with pytest.raises(RuntimeJourneyError):
        journey.start_validation_workflow(journey.case.actions[0])
    assert not any(path.endswith("/run") for _, path, _ in client.calls)
    assert sum(path.endswith("/confirm") for _, path, _ in client.calls) == int(mutation == "stale")


@pytest.mark.parametrize("mutation", ["escape", "wrong_format", "symlink", "duplicate"])
def test_report_carrier_rejects_untrusted_path_or_non_report(tmp_path, mutation):
    root = tmp_path / "workspace"
    path = root / "tasks/task/outputs/validation.xlsx"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a workbook")
    artifact = {"kind": "excel", "path": "tasks/task/outputs/validation.xlsx"}
    if mutation == "escape":
        artifact["path"] = "../validation.xlsx"
    elif mutation == "symlink":
        other = tmp_path / "other.xlsx"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    artifacts = [artifact] * (2 if mutation == "duplicate" else 1)
    assert _validation_report_files(root, "task", {"artifacts": artifacts}) == []


def test_manual_compatibility_does_not_fill_standard_validation_agent_coverage():
    record = {"case_id": "compat", "task_type": "validation", "scenario": "normal",
              "runtime_entry": "manual_compatibility_workflow", "score": {"passed": True}}
    assert runtime_task_coverage([record])["cells"]["validation"]["normal"]["denominator"] == 0


def test_real_notebook_consistency_rejects_different_pmml_model(tmp_path):
    from marvis.domain import TaskCreate
    from marvis.plugins.manifest import ToolRef
    from marvis.repositories.tasks import TaskRepository
    from test_v1_compat_pack import _runner

    paths, _ = _case(tmp_path)
    pmml = paths["dataset_root"] / "model.pmml"
    pmml.write_text(pmml.read_text().replace('coefficient="1.0"', 'coefficient="0.1"'))
    runner, _ = _runner(tmp_path)
    repo = TaskRepository(tmp_path / "workspace/marvis.sqlite")
    task = repo.create_task(TaskCreate(
        model_name="Public synthetic mismatch", model_version="1", validator="benchmark",
        algorithm="lr", source_dir=str(paths["dataset_root"]), notebook_path="model.ipynb",
        sample_path="sample.csv", pmml_path="model.pmml", dictionary_path="dictionary.csv",
        feature_columns=["x1", "x2"],
    ))
    result = runner.invoke(ToolRef("v1_compat", "run_notebook"), {"task_id": task.id}, task_id=task.id)
    assert result.ok, result.error
    evidence = json.loads((tmp_path / "workspace/tasks" / task.id / "outputs/reproducibility_result.json").read_text())
    assert evidence["summary"]["status"] == "fail"
    assert evidence["summary"]["match_count"] < evidence["sample_size"]
