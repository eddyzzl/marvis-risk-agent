"""Actual HTTP/Agent/worker label journey; only model transport is a fixture."""

import copy
import json

import pytest
from pydantic import ValidationError

from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeAction, RuntimeSuite
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from marvis.orchestrator.eval.runtime_scoring import Assertion, _assertion
from test_runtime_agent_benchmark import fixture_model
from test_runtime_workflow_families import _business_protocol


def test_normal_labeling_real_http_agent_tools_and_actual_downloads(
    tmp_path, monkeypatch
):
    import marvis.orchestrator.eval.runtime_labeling as labeling

    original_receipt = labeling.labeling_receipt
    errors = []

    def observed_receipt(*args):
        try:
            return original_receipt(*args)
        except Exception as exc:
            errors.append(str(exc))
            raise

    monkeypatch.setattr(labeling, "labeling_receipt", observed_receipt)
    paths = write_synthetic_suite(tmp_path / "suite", normal_labeling_only=True)
    with fixture_model(answer_factory=_business_protocol) as (model, calls):
        report = run_runtime_suite(
            cases_path=paths["cases"],
            expected_path=paths["expected"],
            dataset_root=paths["dataset_root"],
            output_dir=tmp_path / "runs",
            model=model,
            model_source="fixture_model",
        )
    assert report["all_passed"], (
        errors,
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    case = report["cases"][0]
    assert report["denominator"] == 1 and report["a_evidence_case_count"] == 0
    assert report["acceptance_claim"] == "not_established"
    assert case["human_interventions"] == 6
    assert case["execution"]["plans"][0]["template_id"] == "label_construction"
    assert case["execution"]["plans"][0]["replan_count"] == 0
    revisions = case["process_observation"]["revisions"]
    assert revisions["complete"] and revisions["plan_denominator"] == 1
    assert revisions["plans"][0]["initial_revision"] == revisions["plans"][0]["final_revision"] == 0
    assert sum(revisions["known_counts"].values()) == 0
    assert case["process_observation"]["backend"]["scopes"]["plugin"]["measured_intervals"] > 0
    waits = [case["process_observation"]["backend"]["scopes"][key]
             for key in ("plan_confirmation_wait", "workflow_confirmation_wait")]
    assert sum(row["measured_intervals"] for row in waits) >= 1
    assert all(row["complete"] and row["right_censored_intervals"] == 0 for row in waits)
    label = next(
        s for s in case["execution"]["steps"] if s["tool"] == "labeling.define_label"
    )
    assert label["labeling"]["verified"] is True
    assert label["labeling"]["result_rows"] == 4
    assert label["labeling"]["rows_excluded_after_as_of"] == 2
    assert calls and case["score"]["usage"]["transport_attempts"] == len(calls)
    expected = json.loads(paths["expected"].read_text())["cases"][case["case_id"]]
    label_hash = next(
        a["value"] for a in expected["assertions"] if a["kind"] == "labeling_evidence"
    )
    assert label_hash not in json.dumps(calls)


@pytest.mark.parametrize(
    "extra",
    [
        {"dataset_id": "foreign"},
        {"expected_content_hash": "a" * 64},
        {"workspace_revision": 0},
        {"analysis_generation": 0},
        {"route": "/bypass"},
        {"expected_labels": [1, 0]},
        {"at_mob": True},
    ],
)
def test_business_case_cannot_supply_runtime_identity_or_answers(tmp_path, extra):
    paths = write_synthetic_suite(tmp_path / "suite", normal_labeling_only=True)
    action = json.loads(paths["cases"].read_text())["cases"][0]["actions"][0]
    action["labeling_request"].update(extra)
    with pytest.raises(ValidationError):
        RuntimeAction.model_validate(action)


def test_labeling_entry_cannot_be_silently_inserted_into_another_workflow(tmp_path):
    paths = write_synthetic_suite(tmp_path / "suite", normal_labeling_only=True)
    suite = json.loads(paths["cases"].read_text())
    for change in ("wrong_task", "two_samples", "late_proposal", "initial_message"):
        altered = copy.deepcopy(suite)
        case = altered["cases"][0]
        if change == "wrong_task":
            case["task"]["task_type"] = "strategy"
        elif change == "two_samples":
            case["materials"] *= 2
        elif change == "late_proposal":
            case["actions"].insert(0, {"kind": "message", "content": "开始"})
        else:
            case["initial_message"] = "开始"
        with pytest.raises(ValidationError):
            RuntimeSuite.model_validate(altered)


@pytest.mark.parametrize(
    "mutation",
    [
        "unbound",
        "wrong_rows",
        "no_download",
        "wrong_download",
        "wrong_tool",
        "empty_file",
        "wrong_invocation",
        "failed_invocation",
    ],
)
def test_label_score_cannot_pass_from_output_counters_or_forged_downloads(mutation):
    assertion = Assertion(
        kind="labeling_evidence", tool="labeling.define_label", value="a" * 64
    )
    step = {
        "id": "step",
        "tool": assertion.tool,
        "status": "done",
        "binding_verified": True,
        "producer_invocation_id": "native-run",
        "runs": [{"invocation_id": "native-run", "status": "succeeded"}],
        "labeling": {
            "verified": True,
            "labels_sha256": assertion.value,
            "dataset_download_sha256": "b" * 64,
            "evidence_download_sha256": "c" * 64,
        },
    }
    events = [
        {
            "stage": f"download_labeling_{kind}",
            "status_code": 200,
            "size_bytes": 10,
            "sha256": sha * 64,
        }
        for kind, sha in (("dataset", "b"), ("evidence", "c"))
    ]
    record = {"execution": {"steps": [step]}, "http_events": events}
    assert _assertion(assertion, record, {})
    if mutation == "unbound":
        step["binding_verified"] = False
    elif mutation == "wrong_rows":
        step["labeling"]["labels_sha256"] = "d" * 64
    elif mutation == "no_download":
        events.clear()
    elif mutation == "wrong_download":
        events[0]["sha256"] = "d" * 64
    elif mutation == "wrong_tool":
        step["tool"] = "feature.compute_feature_metrics"
    elif mutation == "wrong_invocation":
        step["producer_invocation_id"] = "another-run"
    elif mutation == "failed_invocation":
        step["runs"][0]["status"] = "failed"
    else:
        events[0]["size_bytes"] = 0
    assert not _assertion(
        assertion, record, {"outputs": {"step": {"n_loans": 4, "n_bad": 1}}}
    )


def test_old_action_digest_does_not_gain_absent_labeling_fields():
    legacy = {"kind": "message", "content": "确认", "tool": ""}
    assert RuntimeAction.model_validate(legacy).model_dump() == legacy


@pytest.mark.parametrize(
    "mutation", ["counter", "source", "csv", "evidence", "result", "foreign_artifact"]
)
def test_actual_label_receipt_rejects_changed_native_data_or_identity(
    tmp_path, mutation
):
    from pathlib import Path
    from marvis.data.errors import DatasetContentDriftError
    from marvis.orchestrator.eval.runtime_labeling import labeling_receipt
    from marvis.packs.labeling.tools import tool_define_label
    from marvis.repositories.task_artifacts import TaskArtifactRepository
    from test_labeling_workflow import _labeling_runtime

    settings, registry, _, _, task, ctx, request = _labeling_runtime(tmp_path)
    output = tool_define_label(
        {
            **request.to_dict(),
            "proposal_hash": request.contract_hash,
            "confirm_immature_cohorts": False,
        },
        ctx,
    )
    assert labeling_receipt(settings.workspace, task.id, output)["verified"] is True
    if mutation == "counter":
        output["n_bad"] += 1
    elif mutation == "source":
        output["source_content_hash"] = "0" * 64
    elif mutation in {"csv", "evidence"}:
        key = "dataset_artifact_id" if mutation == "csv" else "evidence_artifact_id"
        artifact = TaskArtifactRepository(settings.db_path).get_for_task(
            task.id, output[key]
        )
        Path(artifact["path"]).write_text("forged result")
    elif mutation == "result":
        path = registry.resolve_verified_path(output["result_dataset_id"])
        Path(path).write_text("forged labels")
    else:
        output["evidence_artifact_id"] = "foreign"
    with pytest.raises((ValueError, KeyError, DatasetContentDriftError)):
        labeling_receipt(settings.workspace, task.id, output)


def test_actual_runtime_waits_without_creating_a_plan_before_human_confirmation(
    tmp_path,
):
    paths = write_synthetic_suite(tmp_path / "suite", normal_labeling_only=True)
    suite = json.loads(paths["cases"].read_text())
    case = suite["cases"][0]
    case.update(id="synthetic_labeling_waits_for_human", scenario="clarification")
    case["actions"] = case["actions"][:1]
    case["business_constraints_source"] += " Explicit human confirmation is withheld."
    paths["cases"].write_text(json.dumps(suite))
    paths["expected"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": {
                    case["id"]: {
                        "result": "clarification",
                        "assertions": [
                            {
                                "kind": "latest_assistant_metadata",
                                "path": ["kind"],
                                "value": "labeling_preplan_confirmation",
                            },
                            {
                                "kind": "latest_assistant_metadata",
                                "path": [
                                    "labeling_proposal",
                                    "requires_human_confirmation",
                                ],
                                "value": True,
                            },
                        ],
                    }
                },
            }
        )
    )
    with fixture_model(answer_factory=_business_protocol) as (model, calls):
        report = run_runtime_suite(
            cases_path=paths["cases"],
            expected_path=paths["expected"],
            dataset_root=paths["dataset_root"],
            output_dir=tmp_path / "runs",
            model=model,
            model_source="fixture_model",
        )
    assert report["all_passed"], json.dumps(report, ensure_ascii=False, indent=2)
    observed = report["cases"][0]
    assert observed["execution"]["plans"] == observed["execution"]["steps"] == []
    assert observed["human_interventions"] == 2 and calls == []
    assert observed["score"]["usage"]["transport_attempts"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_task",
        "none",
        "multiple",
        "foreign",
        "role",
        "revision",
        "conflict",
        "changed_binding",
    ],
)
def test_labeling_source_binding_rejects_drift_without_sending_agent_request(
    tmp_path, mutation
):
    from marvis.orchestrator.eval.runtime_labeling import submit_labeling_request
    from marvis.orchestrator.eval.runtime_runner import RuntimeJourneyError
    from test_runtime_normal_strategy import _binding_journey

    paths = write_synthetic_suite(tmp_path / "suite", normal_labeling_only=True)
    action = RuntimeAction.model_validate(
        json.loads(paths["cases"].read_text())["cases"][0]["actions"][0]
    )
    journey, client, _ = _binding_journey()
    journey.case.task = journey.case.task.model_copy(update={"task_type": "data_join"})
    if mutation == "wrong_task":
        journey.case.task = journey.case.task.model_copy(
            update={"task_type": "modeling"}
        )
    elif mutation == "none":
        journey.uploaded_samples = []
    elif mutation == "multiple":
        journey.uploaded_samples *= 2
    elif mutation == "foreign":
        journey.uploaded_samples[0]["task_id"] = "other"
    elif mutation == "role":
        journey.uploaded_samples[0]["role"] = "feature"
    elif mutation == "revision":
        client.snapshot["revision"] = True
    elif mutation == "conflict":
        client.conflict_at = "0"
    else:
        import httpx

        original = client.request

        def changed(method, path, **kwargs):
            response = original(method, path, **kwargs)
            if method == "PUT":
                body = response.json()
                body["active_dataset_content_hash"] = "b" * 64
                return httpx.Response(200, json=body)
            return response

        client.request = changed
    with pytest.raises((RuntimeJourneyError, ValidationError)):
        submit_labeling_request(journey, action)
    assert all(call[1] == "/api/tasks/task/data-workspace" for call in client.calls)
