"""Native portfolio download evidence, with adversarial identity checks."""

import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeAction, RuntimeSuite, digest
from marvis.orchestrator.eval.runtime_portfolio import download_portfolio_report
from marvis.orchestrator.eval.runtime_runner import RuntimeJourneyError, run_runtime_suite
from marvis.orchestrator.eval.runtime_scoring import Assertion, _assertion
from test_runtime_agent_benchmark import fixture_model
from test_runtime_workflow_families import _business_protocol


def _case(paths):
    suite = json.loads(paths["cases"].read_text())
    case = next(c for c in suite["cases"] if c["id"] == "synthetic_portfolio")
    case["actions"].append({"kind": "download_portfolio_report", "content": "下载组合报告。"})
    suite["cases"] = [case]
    return suite


def test_native_portfolio_download_and_retained_originals(tmp_path):
    paths = write_synthetic_suite(tmp_path / "suite", include_workflow_families=True)
    paths["cases"].write_text(json.dumps(_case(paths)))
    expected = json.loads(paths["expected"].read_text())
    expected["cases"]["synthetic_portfolio"]["assertions"].append({
        "kind": "portfolio_report_download", "tool": "analysis.portfolio_report",
    })
    paths["expected"].write_text(json.dumps(expected))
    with fixture_model(answer_factory=_business_protocol) as (model, calls):
        report = run_runtime_suite(
            cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=tmp_path / "runs",
            evidence_custody_dir=tmp_path / "custody",
            model=model, model_source="fixture_model",
        )
    assert report["all_passed"], json.dumps(report, ensure_ascii=False, indent=2)
    assert report["denominator"] == 1 and report["a_evidence_case_count"] == 0
    assert report["acceptance_claim"] == "not_established"
    case = report["cases"][0]
    assert len(case["execution"]["steps"]) == 6
    step = next(s for s in case["execution"]["steps"] if s["tool"] == "analysis.portfolio_report")
    event = next(e for e in case["http_events"] if e["stage"] == "download_portfolio_report")
    assert event["sha256"] == step["output_files"][0]["sha256"]
    assert event["step_id"] == step["id"] and event["size_bytes"] > 0
    assert calls
    # Recompute from retained original source and frozen human declarations.
    # Native signatures, workbook cells and complete chain custody still require
    # archive-reader verification; this is arithmetic regression evidence.
    import pandas as pd
    from marvis.orchestrator.eval.runtime_portfolio_reference import transition_reference
    original = next((tmp_path / "custody").glob("*/*/private.json"))
    native = json.loads(original.read_text())["outputs"]
    frozen = json.loads((original.parent / "case.json").read_text())
    request = frozen["actions"][0]["portfolio_request"]
    fields = {key: request[key] for key in (
        "id_col", "snapshot_col", "bucket_col", "balance_col", "loss_state", "lgd", "horizon_months",
    )}
    reference = transition_reference(
        pd.read_parquet(original.parent / "inputs/000.parquet").to_dict("records"),
        states=frozen["actions"][1]["content"].split(","), **fields,
    )
    for name, tool in (("flow", "flow_rate"), ("migration", "bucket_migration"),
                       ("expected_loss", "expected_loss_estimate")):
        step_id = next(s["id"] for s in case["execution"]["steps"] if s["tool"] == f"analysis.{tool}")
        actual = copy.deepcopy(native[step_id])
        for flag in actual["red_flags"]:
            flag.pop("message", None)
        _numeric_equal(actual, reference[name])
    # The new evidence predicate must fail even if the rest of the run succeeds.
    assertion = Assertion(kind="portfolio_report_download", tool="analysis.portfolio_report")
    private = {"outputs": {step["id"]: {
        "artifact_id": event["artifact_id"], "artifact_content_hash": event["sha256"],
    }}}
    assert _assertion(assertion, case, private)
    for field, replacement in (
        ("step_id", "foreign"), ("artifact_id", "foreign"), ("sha256", "0" * 64),
        ("size_bytes", 0), ("status_code", 404), ("stage", "not_a_download"),
    ):
        forged = copy.deepcopy(case)
        next(e for e in forged["http_events"] if e["stage"] == "download_portfolio_report")[field] = replacement
        assert not _assertion(assertion, forged, private), field


def _numeric_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _numeric_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _numeric_equal(left, right)
    elif type(expected) in {int, float}:
        assert type(actual) in {int, float} and actual == pytest.approx(expected, rel=1e-10, abs=1e-10)
    else:
        assert type(actual) is type(expected) and actual == expected


@pytest.mark.parametrize("mutation", ["wrong_task", "no_declaration", "early", "extra_sample"])
def test_portfolio_download_requires_declared_task_and_input(tmp_path, mutation):
    paths = write_synthetic_suite(tmp_path / "suite", include_workflow_families=True)
    suite = _case(paths)
    case = suite["cases"][0]
    if mutation == "wrong_task":
        case["task"]["task_type"] = "data_join"
    elif mutation == "no_declaration":
        case["actions"][0].pop("portfolio_request")
    elif mutation == "early":
        case["actions"].insert(0, case["actions"].pop())
    else:
        case["materials"].append(copy.deepcopy(case["materials"][0]))
    with pytest.raises(ValidationError):
        RuntimeSuite.model_validate(suite)


@pytest.mark.parametrize("extra", [{"tool": "arbitrary"}, {"content": ""}, {"url": "/bypass"}])
def test_download_cannot_supply_tool_or_url(extra):
    with pytest.raises(ValidationError):
        RuntimeAction.model_validate({"kind": "download_portfolio_report", "content": "下载", **extra})


@pytest.mark.parametrize("failure", ["missing", "duplicate", "hash", "status"])
def test_download_does_not_record_success_on_invalid_report(failure):
    output = {"artifact_id": "report", "artifact_content_hash": digest(b"original")}
    step = {"id": "step", "tool_ref": "analysis.portfolio_report", "status": "done"}
    requests = []
    events = []

    def request(method, route, **kwargs):
        requests.append(route)
        events.append({"stage": kwargs["label"]})
        return SimpleNamespace(status_code=404 if failure == "status" else 200,
                               content=b"changed" if failure == "hash" else b"original")

    journey = SimpleNamespace(
        task_id="task", plans=lambda: [{"steps": [] if failure == "missing" else [step] * (2 if failure == "duplicate" else 1)}],
        json_request=lambda *args, **kwargs: output,
        record_human_action=lambda action: None, request=request, events=events,
    )
    with pytest.raises(RuntimeJourneyError):
        download_portfolio_report(journey, SimpleNamespace())
    assert not any("sha256" in event for event in events)
    assert len(requests) == (0 if failure in {"missing", "duplicate"} else 1)
