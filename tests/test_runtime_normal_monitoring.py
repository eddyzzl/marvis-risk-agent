"""Actual Agent monitoring plan and worker execution; model protocol alone is fake."""
import copy
import json

import pytest
from pydantic import ValidationError

from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeAction, RuntimeSuite
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from test_runtime_agent_benchmark import fixture_model
from test_runtime_normal_modeling import _modeling_protocol


@pytest.mark.parametrize("manual", [False, True])
def test_normal_monitoring_real_training_intake_gate_scoring_and_psi(tmp_path, monkeypatch, manual):
    import marvis.orchestrator.eval.runtime_monitoring as monitoring

    errors = []
    original = monitoring.monitoring_receipt
    def observed(*args):
        try:
            return original(*args)
        except Exception as exc:
            errors.append(str(exc))
            raise
    monkeypatch.setattr(monitoring, "monitoring_receipt", observed)
    paths = write_synthetic_suite(tmp_path / "suite", **{
        "normal_monitoring_workflow_only" if manual else "normal_monitoring_only": True,
    })
    with fixture_model(answer_factory=_modeling_protocol) as (model, calls):
        report = run_runtime_suite(
            cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=tmp_path / "runs",
            model=model, model_source="fixture_model",
        )
    (tmp_path / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    assert report["all_passed"], (errors, json.dumps(report, ensure_ascii=False, indent=2))
    record = report["cases"][0]
    evidence = record["execution"]["monitoring"]
    entry = "manual_monitoring_workflow" if manual else "standard_model_monitoring_agent"
    assert evidence["verified"] and evidence["actual_rows"] == 180
    assert evidence["score_psi"] > 0.25 and evidence["runtime_entry"] == entry
    assert evidence["label_maturity_assurance"] == "unknown"
    assert evidence["business_acceptance"] == "not_established"
    assert record["human_interventions"] == 10
    assert report["a_evidence_case_count"] == 0 and report["acceptance_claim"] == "not_established"
    assert calls and len(calls) == record["score"]["usage"]["transport_attempts"]
    assert record["execution"]["plans"][-1]["template_id"] == "monitoring_run"
    if not manual:
        assert evidence["binding_sha256"]
        assert not any(e["stage"] == "create_manual_monitoring_workflow" for e in record["http_events"])


@pytest.mark.parametrize("extra", [
    {"experiment_id": "foreign"}, {"dataset_id": "foreign"}, {"expected_content_hash": "a" * 64},
    {"workspace_revision": 0}, {"route": "/bypass"}, {"expected_psi": 0.0}, {"target_col": False},
])
def test_monitoring_case_cannot_inject_runtime_identity_or_expected_results(extra):
    with pytest.raises(ValidationError):
        RuntimeAction.model_validate({
            "kind": "submit_model_monitoring_request", "content": "监控已选模型",
            "monitoring_request": {"material_path": "new.parquet", **extra},
        })


@pytest.mark.parametrize("mutation", ["wrong_task", "same_material", "missing_confirmation", "early_proposal"])
def test_monitoring_material_cannot_replace_training_or_skip_current_confirmation(tmp_path, mutation):
    paths = write_synthetic_suite(tmp_path / "suite", normal_monitoring_only=True)
    suite = json.loads(paths["cases"].read_text())
    case = suite["cases"][0]
    if mutation == "wrong_task":
        case["task"]["task_type"] = "data_join"
    elif mutation == "same_material":
        case["actions"][-2]["monitoring_request"]["material_path"] = case["materials"][0]["path"]
    elif mutation == "missing_confirmation":
        case["actions"].pop()
    else:
        case["actions"] = case["actions"][-2:]
    with pytest.raises(ValidationError):
        RuntimeSuite.model_validate(suite)


def test_monitoring_score_requires_bound_actual_evidence_and_human_gate():
    from marvis.orchestrator.eval.runtime_scoring import Assertion, _assertion

    assertion = Assertion(kind="monitoring_evidence", tool="modeling.monitor_run", value="standard_model_monitoring_agent")
    record = {"execution": {"monitoring": {"verified": True, "runtime_entry": assertion.value}},
              "http_events": [{"stage": "human_monitoring_plan_start", "status_code": 202}]}
    assert _assertion(assertion, record, {})
    for key in ("verified", "runtime_entry"):
        wrong = copy.deepcopy(record)
        wrong["execution"]["monitoring"][key] = False
        assert not _assertion(assertion, wrong, {})
    record["http_events"] = []
    assert not _assertion(assertion, record, {})
