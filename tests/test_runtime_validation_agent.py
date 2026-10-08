"""Standard V2 Agent carrier; never create a compatibility plan or fake receipts."""
import json
import copy
import time

import httpx
import pytest
from pydantic import ValidationError

from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, RuntimeAction, digest
from marvis.orchestrator.eval.runtime_runner import Journey, RuntimeJourneyError, run_runtime_suite
from marvis.orchestrator.eval.runtime_validation_adapter import confirm_current_report, pipeline_receipt
from marvis.orchestrator.eval.runtime_scoring import runtime_task_coverage
from test_runtime_agent_benchmark import fixture_model


def _v2_protocol(request, answer, payload):
    if request.get("stage") == "word_conclusion_draft":
        values = {
            "TEXT:pressure_test_summary": "公开合成样本的压力风险应按已展示的确定性分组结果复核。",
            "TEXT:pressure_impact_recommendation": "建议结合各特征类别的实际压力结果持续监测；本次没有真实资金表现证据。",
            "TEXT:final_validation_conclusion": "合成样本指标只能用于技术验证。业务代表性、成熟表现和真实经营收益均未建立，不能据此批准上线。PMML部署可用。",
            "TEXT:model_training_description": "本次采用公开合成固定逻辑回归模型，不代表实际信贷训练方案。",
        }
        requested = request.get("requested_fields")
        return {key: value for key, value in values.items() if requested is None or key in requested}
    return {"summary": "仅根据实际结构化证据解释公开合成样本，不代表业务验收。"}


def test_standard_v2_agent_executes_pmml_stages_draft_confirmation_and_downloads(tmp_path, monkeypatch):
    import marvis.orchestrator.eval.runtime_runner as runner_module

    native_receipts = runner_module._receipts
    tamper_checks = []

    def inspect_real_settled_carrier(workspace, task_id, **kwargs):
        # The app and its actual stages have already stopped. Corrupt only the
        # public synthetic test copy to prove the parent verifier fails closed;
        # never inject these files/results into the application or its execution.
        result = native_receipts(workspace, task_id, **kwargs)
        evidence, private = result
        if not evidence.get("validation_pipeline", {}).get("execution_complete"):
            return result
        outputs = workspace / "tasks" / task_id / "outputs"
        journey = kwargs["journey"]
        def receipt(**extra):
            return pipeline_receipt(workspace, task_id, kwargs["case"], private["messages"],
                confirmation=extra.get("confirmation", journey.validation_confirmation),
                downloads=extra.get("downloads", journey.validation_downloads))
        for filename, mutation, failed in (
            ("pmml_scores.parquet", lambda raw: raw + b"changed", "pmml_scoring"),
            ("validation_results.json", lambda raw: json.dumps({**json.loads(raw), "algorithm": "lgb"}).encode(), "metrics"),
        ):
            path = outputs / filename
            original = path.read_bytes()
            try:
                path.write_bytes(mutation(original))
                bad = receipt()
                assert not bad["execution_complete"] and bad["failed_check"] == failed
                tamper_checks.append(filename)
            finally:
                path.write_bytes(original)
        stale = {**journey.validation_confirmation, "draft_edit_revision": 999}
        assert receipt(confirmation=stale)["failed_check"] == "report_confirmation"
        assert receipt(downloads=[])["failed_check"] == "report_files"
        tamper_checks.extend(["stale_draft", "unbound_download"])
        assert receipt()["execution_complete"]
        return result

    monkeypatch.setattr(runner_module, "_receipts", inspect_real_settled_carrier)
    paths = write_synthetic_suite(tmp_path / "suite", normal_validation_agent_only=True)
    with fixture_model(answer_factory=_v2_protocol) as (model, calls):
        report = run_runtime_suite(
            cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=tmp_path / "runs",
            model=model, model_source="fixture_model",
        )
    assert report["all_passed"], json.dumps(report, ensure_ascii=False, indent=2)
    assert calls
    case = report["cases"][0]
    assert case["runtime_entry"] == "standard_validation_agent_v2"
    assert case["execution"]["plans"] == []
    assert case["execution"]["steps"] == []
    assert case["execution"]["validation_pipeline"]["notebook_consistency"] == "not_in_v2_entry"
    assert case["human_interventions"] == 3
    process = case["process_observation"]
    assert process["llm_transport"]["complete"] is True
    assert process["llm_transport"]["measured_attempts"] == len(calls)
    assert process["llm_transport"]["known_busy_duration_ns"] > 0
    assert process["formal_timing_complete"] is False
    assert process["backend"]["journal_complete"] is True
    for scope in ("validation_scan", "validation_pmml", "validation_metrics", "validation_report", "queue"):
        timing = process["backend"]["scopes"][scope]
        assert timing["complete"] is True, (scope, timing)
        assert timing["measured_intervals"] >= 1, (scope, timing)
        assert timing["known_busy_duration_ns"] > 0
    assert process["overlapping_scopes"] is True
    assert process["revisions"]["complete"] is True
    assert process["revisions"]["plan_denominator"] == 0
    assert process["revisions"]["known_counts"] == {
        "structural_replan": 0, "explore_append": 0, "upstream_revision": 0,
    }
    waiting = process["backend"]["scopes"]["report_confirmation_wait"]
    assert waiting["complete"] and waiting["measured_intervals"] == 1
    assert waiting["intervals"][0]["outcome"] == "resumed"
    assert waiting["known_busy_duration_ns"] > 0
    assert [a["kind"] for a in process["human_actions"]] == [
        "validation_material_selection", "start_validation_agent", "confirm_current_validation_report",
    ]
    from marvis.orchestrator.eval.runtime_process import material_selection_identity
    definition = RuntimeCase.model_validate(json.loads(paths["cases"].read_text())["cases"][0])
    assert [a["action_sha256"] for a in process["human_actions"]] == [
        material_selection_identity(definition), *(digest(a.model_dump()) for a in definition.actions),
    ]
    assert report["runtime_task_coverage"]["cells"]["validation"]["normal"]["passed"] == 1
    assert report["a_evidence_case_count"] == 0
    assert report["acceptance_claim"] == "not_established"
    assert len(tamper_checks) == 4


def _draft_journey(tmp_path):
    paths = write_synthetic_suite(tmp_path / "suite", normal_validation_agent_only=True)
    case = RuntimeCase.model_validate(json.loads(paths["cases"].read_text())["cases"][0])
    values = _v2_protocol({"stage": "word_conclusion_draft"}, {}, {})
    class Client:
        def __init__(self):
            self.calls = []
            self.messages = [{"id": "draft", "task_id": "task", "role": "assistant", "stage": "word_conclusion_draft",
                "metadata": {"draft_values": values, "report_revision": 3, "draft_edit_revision": 2, "fallback": False}}]
        def request(self, method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            if method == "GET":
                return httpx.Response(200, json={"messages": self.messages})
            return httpx.Response(409, json={"detail": "current draft was changed"})
    client = Client()
    journey = Journey(client, case, time.monotonic() + 20)
    journey.task_id = "task"
    return journey, client


def test_current_draft_uses_actual_values_and_all_revisions_then_fails_stale_once(tmp_path):
    journey, client = _draft_journey(tmp_path)
    with pytest.raises(RuntimeJourneyError, match="http_409"):
        confirm_current_report(journey, journey.case.actions[1])
    assert len(client.calls) == 2
    body = client.calls[1][2]["json"]
    assert body == {"revision": 3, "draft_message_id": "draft", "draft_edit_revision": 2,
                    "text_values": client.messages[0]["metadata"]["draft_values"]}
    assert journey.validation_confirmation["values_sha256"] == digest(body["text_values"])
    assert journey.interventions == 1


@pytest.mark.parametrize("mutation", ["fallback", "unknown_fallback", "streaming", "foreign", "new_empty", "confirmed", "edit_revision", "unconfirmable", "duplicate"])
def test_draft_selection_does_not_invent_or_fall_back_to_older_text(tmp_path, mutation):
    journey, client = _draft_journey(tmp_path)
    metadata = client.messages[0]["metadata"]
    if mutation == "fallback":
        metadata["fallback"] = True
    elif mutation == "unknown_fallback":
        del metadata["fallback"]
    elif mutation == "streaming":
        metadata["streaming"] = True
    elif mutation == "foreign":
        client.messages[0]["task_id"] = "other"
    elif mutation == "new_empty":
        client.messages.append({**copy.deepcopy(client.messages[0]), "id": "newer", "metadata": {"draft_values": {}, "report_revision": 3}})
    elif mutation == "confirmed":
        client.messages.append({"id": "confirmed", "role": "assistant", "stage": "word_conclusion_confirmed"})
    elif mutation == "edit_revision":
        metadata["draft_edit_revision"] = True
    elif mutation == "unconfirmable":
        metadata["confirmable"] = False
    else:
        client.messages *= 2
    with pytest.raises(RuntimeJourneyError, match="draft_not_confirmable"):
        confirm_current_report(journey, journey.case.actions[1])
    assert [call[0] for call in client.calls] == ["GET"]
    assert journey.interventions == 0


@pytest.mark.parametrize("extra", [{"tool": "v1_compat.render_reports"}, {"revision": 3}, {"draft_message_id": "injected"}, {"text_values": {}}, {"route": "/bypass"}, {"content": ""}])
def test_public_confirmation_can_only_choose_current_displayed_draft(extra):
    with pytest.raises(ValidationError):
        RuntimeAction.model_validate({"kind": "confirm_current_validation_report", "content": "人工确认当前展示版本", **extra})


def test_v2_contract_is_distinct_and_requires_native_static_notebook_material(tmp_path):
    paths = write_synthetic_suite(tmp_path / "suite", normal_validation_agent_only=True)
    raw = json.loads(paths["cases"].read_text())["cases"][0]
    for mutation in ("no_notebook", "compat_gate", "repeat_start"):
        case = copy.deepcopy(raw)
        if mutation == "no_notebook":
            case["materials"] = [m for m in case["materials"] if m["role"] != "notebook"]
        elif mutation == "compat_gate":
            case["actions"][1] = {"kind": "approve_step", "tool": "v1_compat.render_reports", "content": "错误入口"}
        else:
            case["actions"].insert(1, copy.deepcopy(case["actions"][0]))
        with pytest.raises(ValidationError):
            RuntimeCase.model_validate(case)


def test_unidentified_validation_entry_never_fills_standard_agent_coverage():
    record = {"case_id": "unknown", "task_type": "validation", "scenario": "normal", "score": {"passed": True}}
    coverage = runtime_task_coverage([record])
    assert coverage["cells"]["validation"]["normal"]["denominator"] == 0
    assert coverage["unclassified_case_ids"] == ["unknown"]


def test_missing_importance_stops_actual_agent_before_scoring_and_confirmation(tmp_path):
    paths = write_synthetic_suite(tmp_path / "suite", normal_validation_agent_only=True)
    dictionary = paths["dataset_root"] / "dictionary.csv"
    dictionary.write_text("特征名,类别\nx1,synthetic_internal\nx2,synthetic_external\n")
    suite = json.loads(paths["cases"].read_text())
    for material in suite["cases"][0]["materials"]:
        if material["role"] == "dictionary":
            material["sha256"] = digest(dictionary.read_bytes())
    # Separate deliberately invalid scenario; do not rewrite any frozen corpus.
    suite["cases"][0]["revision"] = "missing-importance-regression"
    paths["cases"].write_text(json.dumps(suite))
    with fixture_model(answer_factory=_v2_protocol) as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=tmp_path / "runs", model=model, model_source="fixture_model")
    assert not report["all_passed"]
    case = report["cases"][0]
    pipeline = case["execution"]["validation_pipeline"]
    assert not pipeline["material_binding_verified"]
    assert not pipeline["scoring_verified"]
    assert not pipeline["report_confirmation_verified"]
    assert all(e["stage"] != "human_validation_report_confirmation" for e in case["http_events"])
    assert all(j["kind"] != "report" for j in pipeline["jobs"])
