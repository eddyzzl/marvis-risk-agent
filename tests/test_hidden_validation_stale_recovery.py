"""A fixed validation recovery slot can execute CAS rejection without fake recovery."""
import json

from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from marvis.orchestrator.eval.runtime_validation_boundary import STAGE
from test_runtime_agent_benchmark import fixture_model
from test_runtime_validation_agent import _v2_protocol


def test_validation_recovery_stale_cas_executes_native_rejection_and_remains_unsatisfied(tmp_path):
    paths = write_synthetic_suite(tmp_path / "suite", normal_validation_agent_only=True)
    suite = json.loads(paths["cases"].read_bytes())
    case = suite["cases"][0]
    case.update(id="hb-validation-recovery-01", family="validation", scenario="recovery")
    case["actions"][1]["fault"] = {"kind": "stale_report_revision_once"}
    paths["cases"].write_text(json.dumps(suite))
    # The fixed recovery objective stays done: CAS rejection alone does not meet it.
    paths["expected"].write_text(json.dumps({"schema_version": 1, "cases": {case["id"]: {
        "result": "done", "assertions": [{"kind": "validation_report_verified", "value": "word"}]}}}))
    with fixture_model(answer_factory=_v2_protocol) as (model, calls):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=tmp_path / "public", model=model, model_source="fixture_model")
    row = report["cases"][0]
    assert calls and row["runtime_status"] == "completed", row.get("error_code")
    assert any(e["stage"] == STAGE and e["status_code"] == 409 for e in row["http_events"])
    assert not row["score"]["passed"] and not report["all_passed"]
    assert report["denominator"] == 1 and report["acceptance_claim"] == "not_established"
