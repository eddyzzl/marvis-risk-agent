"""Normal Strategy through actual HTTP, native ToolRunner and governance."""

import json

import pytest
from pydantic import ValidationError
from marvis.orchestrator.eval.runtime_contracts import RuntimeAction, RuntimeCase
from marvis.orchestrator.eval.runtime_runner import RuntimeJourneyError

from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from test_runtime_agent_benchmark import fixture_model
from test_runtime_workflow_families import _business_protocol


def _sample_request():
    return {
        "request_kind": "standard_workflow",
        "workflow": "strategy_sample_design_v2",
        "workflow_inputs": {
            "target_bad_value": 1,
            "drop_nan_labels": False,
            "relationship": "nested_same_cohort",
            "approval_population": {"inclusion": None, "exclusion": None},
            "risk_population": {"inclusion": None, "exclusion": None},
            "partitioning": {
                "method": "time_ranges",
                "column": "apply_date",
                "ranges": {
                    "development": {"start": "2026-01-01", "end": "2026-01-31"},
                    "validation": {"start": "2026-02-01", "end": "2026-02-28"},
                    "oot": {"start": "2026-03-01", "end": "2026-03-31"},
                },
            },
            "maturity": {
                "status": "confirmed_matured",
                "performance_window_days": 30,
                "cutoff_date": "2026-04-30",
                "reason": None,
            },
            "performance_window": {"status": "provided", "days": 30},
            "observation_window": {
                "status": "provided",
                "start": "2026-01-01",
                "end": "2026-04-30",
            },
            "field_bindings": {
                "entity_field": "synthetic_id",
                "time_field": "apply_date",
                "group_field": None,
                "month_field": "apply_month",
                "weight_field": None,
                "loan_amount_field": None,
                "overdue_amount_field": None,
            },
            "historical_score": {
                "status": "available",
                "column": "credit_score",
                "direction": "lower_is_riskier",
                "reason": None,
            },
        },
    }


def _strategy_protocol(request, answer, payload):
    user = payload["messages"][-1]["content"]
    if "【用户策略请求】" in user:
        utterance = user.split("【用户策略请求】\n", 1)[1].split("\n", 1)[0]
        if "固化 V2 策略样本设计" in utterance:
            return _sample_request()
        return {
            "operation": "develop",
            "strategy_type": "approval",
            "objective": "max_approval",
            "max_bad_rate": 0.25,
            "min_approval_rate": 0.4,
        }
    answer = _business_protocol(request, answer, payload)
    if "allowed_intents" in request and (
        "固化 V2 策略样本设计" in request["instruction"]
        or "做完整审批策略开发" in request["instruction"]
    ):
        answer["intent"] = "strategy_workflow"
    return answer


def test_normal_strategy_crosses_real_http_native_sample_backtest_and_adoption(
    tmp_path,
):
    paths = write_synthetic_suite(tmp_path / "suite", normal_strategy_only=True)
    with fixture_model(answer_factory=_strategy_protocol) as (model, calls):
        report = run_runtime_suite(
            cases_path=paths["cases"],
            expected_path=paths["expected"],
            dataset_root=paths["dataset_root"],
            output_dir=tmp_path / "runs",
            model=model,
            model_source="fixture_model",
        )
    assert report["all_passed"], json.dumps(report, ensure_ascii=False, indent=2)
    assert report["denominator"] == 1
    assert report["acceptance_claim"] == "not_established"
    assert report["a_evidence_case_count"] == 0
    assert calls


def _binding_journey():
    import time
    import httpx
    from marvis.orchestrator.eval.runtime_contracts import RuntimeAction, RuntimeCase
    from marvis.orchestrator.eval.runtime_runner import Journey

    class Client:
        def __init__(self):
            self.calls = []
            self.conflict_at = None
            self.snapshot = {
                "schema_version": "data-workspace.v1",
                "task_id": "task",
                "revision": 0,
                "active_dataset_id": None,
                "active_dataset_content_hash": None,
                "analysis_generation": 0,
                "page": "overview",
                "selected_field": None,
                "semantic_mapping": {
                    "target_col": None,
                    "field_roles": {},
                    "business_names": {},
                },
                "updated_at": "2026-01-01T00:00:00Z",
            }

        def request(self, method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            if method == "PUT":
                if kwargs["headers"]["If-Match"] == self.conflict_at:
                    return httpx.Response(412, json={"detail": "stale revision"})
                self.snapshot.update(kwargs["json"])
                self.snapshot["revision"] += 1
            return httpx.Response(200, json=self.snapshot)

    case = RuntimeCase(
        id="binding",
        revision="1",
        family="strategy",
        task={"task_type": "strategy"},
        business_constraints_source="Public synthetic only",
    )
    client = Client()
    journey = Journey(client, case, time.monotonic() + 20)
    journey.task_id = "task"
    journey.uploaded_samples = [
        {
            "id": "uploaded-only",
            "content_hash": "a" * 64,
            "task_id": "task",
            "role": "sample",
        }
    ]
    action = RuntimeAction(
        kind="bind_single_strategy_sample",
        content="人工绑定样本与目标字段",
        semantic_mapping={
            "target_col": "y",
            "field_roles": {"y": "target"},
            "business_names": {},
        },
    )
    return journey, client, action


def test_human_binding_uses_exact_upload_and_each_returned_cas_revision():
    journey, client, action = _binding_journey()
    journey.bind_single_strategy_sample(action)
    assert [x[:2] for x in client.calls] == [
        ("GET", "/api/tasks/task/data-workspace"),
        ("PUT", "/api/tasks/task/data-workspace"),
        ("PUT", "/api/tasks/task/data-workspace"),
    ]
    first, second = [x[2] for x in client.calls[1:]]
    assert first["headers"] == {"If-Match": "0"}
    assert second["headers"] == {"If-Match": "1"}
    assert first["json"]["semantic_mapping"] == {
        "target_col": None,
        "field_roles": {},
        "business_names": {},
    }
    assert second["json"]["semantic_mapping"] == action.semantic_mapping.model_dump()
    assert all(
        x["json"]["active_dataset_id"] == "uploaded-only"
        and x["json"]["active_dataset_content_hash"] == "a" * 64
        for x in (first, second)
    )
    assert journey.interventions == 1
    assert [x["stage"] for x in journey.events] == [
        "read_strategy_workspace",
        "human_strategy_dataset_selection",
        "human_strategy_sample_binding",
    ]


def test_existing_exact_binding_does_not_reset_workspace():
    journey, client, action = _binding_journey()
    client.snapshot.update(
        active_dataset_id="uploaded-only",
        active_dataset_content_hash="a" * 64,
        revision=7,
    )
    journey.bind_single_strategy_sample(action)
    assert len(client.calls) == 2
    assert client.calls[1][2]["headers"] == {"If-Match": "7"}


@pytest.mark.parametrize("revision", ["0", "1"])
def test_stale_binding_fails_without_retry_or_plan_execution(revision):
    journey, client, action = _binding_journey()
    client.conflict_at = revision
    with pytest.raises(RuntimeJourneyError, match="http_412"):
        journey.bind_single_strategy_sample(action)
    assert len(client.calls) == 2 + int(revision)
    assert journey.events[-1]["status_code"] == 412
    assert all(x[1] == "/api/tasks/task/data-workspace" for x in client.calls)


@pytest.mark.parametrize(
    "mutation",
    ["wrong_task", "none", "multiple", "foreign", "role", "target", "revision"],
)
def test_binding_rejects_ambiguous_or_foreign_upload_before_mutation(mutation):
    journey, client, action = _binding_journey()
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
    elif mutation == "target":
        action = action.model_copy(update={"semantic_mapping": None})
    elif mutation == "revision":
        client.snapshot["revision"] = True
    with pytest.raises((RuntimeJourneyError, ValidationError)):
        journey.bind_single_strategy_sample(action)
    assert all(x[0] == "GET" for x in client.calls)
    assert journey.interventions == 0


@pytest.mark.parametrize(
    "extra",
    [
        {"dataset_id": "arbitrary"},
        {"hash": "a" * 64},
        {"route": "/bypass"},
        {"callback": "code"},
        {"tool": "strategy.adopt_strategy"},
        {"content": ""},
        {"kind": "message"},
        {
            "semantic_mapping": {
                "target_col": "y",
                "field_roles": {},
                "business_names": {},
                "dataset_id": "arbitrary",
            }
        },
    ],
)
def test_binding_contract_has_no_arbitrary_route_callback_or_identity(extra):
    _, _, action = _binding_journey()
    with pytest.raises(ValidationError):
        RuntimeAction.model_validate({**action.model_dump(), **extra})


def test_binding_case_requires_unique_strategy_sample(tmp_path):
    paths = write_synthetic_suite(tmp_path / "normal", normal_strategy_only=True)
    raw = json.loads(paths["cases"].read_text())["cases"][0]
    assert RuntimeCase.model_validate(raw).task.task_type == "strategy"
    import copy

    for mutation in ("task", "none", "multiple", "role"):
        case = copy.deepcopy(raw)
        if mutation == "task":
            case["task"]["task_type"] = "modeling"
        elif mutation == "none":
            case["materials"] = []
        elif mutation == "multiple":
            case["materials"] *= 2
        else:
            case["materials"][0]["role"] = "feature"
        with pytest.raises(ValidationError):
            RuntimeCase.model_validate(case)
    for other in ({"normal_modeling_only": True}, {"include_workflow_families": True}):
        with pytest.raises(ValueError):
            write_synthetic_suite(
                tmp_path / "invalid", normal_strategy_only=True, **other
            )
        assert not (tmp_path / "invalid").exists()


def test_open_ended_native_result_hash_does_not_weaken_case_hashes():
    from marvis.orchestrator.eval.runtime_contracts import digest
    from marvis.orchestrator.eval.runtime_runner import _output_identity
    from marvis.orchestrator.evidence import payload_hash

    finite = {"band_edges": [0.0, 1.0]}
    assert _output_identity(finite, {}, "strategy.design_cutoff_bands") == {
        "output_sha256": digest(finite)
    }
    native = {"band_edges": [float("-inf"), 0.5, float("inf")]}
    with pytest.raises(ValueError):
        digest(native)
    with pytest.raises(ValueError, match="differs"):
        _output_identity(
            native,
            {"output_hash": "sha256:" + "0" * 64},
            "strategy.design_cutoff_bands",
        )
    result = _output_identity(
        native, {"output_hash": payload_hash(native)}, "strategy.design_cutoff_bands"
    )
    assert result == {
        "output_sha256": payload_hash(native).removeprefix("sha256:"),
        "output_hash_contract": "marvis.orchestrator.evidence.payload_hash",
    }


@pytest.mark.parametrize(
    "output,tool",
    [
        ({"band_edges": [0.0, float("inf"), 1.0]}, "strategy.design_cutoff_bands"),
        (
            {"band_edges": [float("-inf"), float("inf")], "profit": float("nan")},
            "strategy.design_cutoff_bands",
        ),
        ({"profit": float("inf")}, "strategy.backtest_strategy"),
    ],
)
def test_native_hash_never_accepts_nonfinite_metrics_or_other_tools(output, tool):
    from marvis.orchestrator.eval.runtime_runner import _output_identity
    from marvis.orchestrator.evidence import payload_hash

    with pytest.raises(ValueError):
        _output_identity(output, {"output_hash": payload_hash(output)}, tool)


def test_rejecting_local_adoption_never_runs_effect_or_delivery(tmp_path):
    paths = write_synthetic_suite(tmp_path / "suite", normal_strategy_only=True)
    suite = json.loads(paths["cases"].read_text())
    case = suite["cases"][0]
    case["actions"][-1] = {
        "kind": "reject_step",
        "tool": "strategy.adopt_strategy",
        "content": "拒绝本地采纳当前合成候选，停止后续策略文档交付。",
    }
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    paths["expected"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": {
                    case["id"]: {
                        "result": "cancelled",
                        "assertions": [
                            {
                                "kind": "http_status",
                                "stage": "human_rejection",
                                "value": 202,
                            },
                            {
                                "kind": "tool_succeeded",
                                "tool": "strategy.backtest_strategy",
                            },
                            {
                                "kind": "tool_not_executed",
                                "tool": "strategy.adopt_strategy",
                            },
                            {
                                "kind": "tool_not_executed",
                                "tool": "strategy.render_strategy_doc",
                            },
                        ],
                    }
                },
            }
        )
    )
    with fixture_model(answer_factory=_strategy_protocol) as (model, calls):
        report = run_runtime_suite(
            cases_path=paths["cases"],
            expected_path=paths["expected"],
            dataset_root=paths["dataset_root"],
            output_dir=tmp_path / "runs",
            model=model,
            model_source="fixture_model",
        )
    assert report["all_passed"], json.dumps(report, ensure_ascii=False, indent=2)
