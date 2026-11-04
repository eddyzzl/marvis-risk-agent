"""Real warmup/measurement originals, using only the local model fixture."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_runtime_agent_benchmark import (
    base_case,
    fixture_model,
    materials,
    write_suite,
)

from marvis.orchestrator.eval.formal_acceptance import _telemetry
from marvis.orchestrator.eval.process_timing_collector import build_formal_observation
from marvis.orchestrator.eval.runtime_contracts import digest
from marvis.orchestrator.eval.runtime_run_manifest import verify_run_manifest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from marvis.orchestrator.eval.runtime_warmup import (
    LoadBindingError,
    measurement_window,
    validate_load_binding,
)


@pytest.fixture(scope="module")
def actual_warm_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("warm-original")
    data = root / "data"
    material = materials(data)
    case = base_case("warm_join", "data_join", [material["sample.parquet"], material["features.parquet"]])
    case["actions"] = [{"kind": "approve_step", "tool": "data_ops.execute_join", "content": "已检查合成材料，批准拼接。"}]
    cases, expected = write_suite(root, cases=[case], expected={case["id"]: {"result": "done", "assertions": [
        {"kind": "tool_succeeded", "tool": "data_ops.execute_join"},
        {"kind": "dataset_rows", "tool": "data_ops.execute_join", "value": 80},
    ]}})
    warmup = {"endpoints": ["/api/branding", "/api/tasks"], "requests": 3, "minimum_seconds": 0.02}
    load_spec = {"profile_load": {"concurrent_users": 2}, "load_policy": {"concurrent_workers": 2, "warmup": warmup}}
    binding = {"load_mode": "steady", "warmup": warmup, "load_spec": load_spec, "load_sha256": digest(load_spec)}
    with fixture_model() as (model, calls):
        report = run_runtime_suite(cases_path=cases, expected_path=expected, dataset_root=data,
                                   output_dir=root / "runs", model=model, model_source="fixture_model",
                                   load_binding=binding)
    assert calls and report["all_passed"]
    run = verify_run_manifest(Path(report["report_path"]).parent, expected_manifest_sha256=report["final_manifest_sha256"])
    return root, binding, report, run


@pytest.mark.e2e
def test_warmup_executes_fixed_reads_on_same_process_before_actual_journey(actual_warm_run):
    _root, binding, report, run = actual_warm_run
    execution = report["cases"][0]
    originals = execution["load_originals"]
    events = originals["events"]
    assert [row["event"] for row in events] == ["process_ready", "warmup_started", "warmup_response", "warmup_response", "warmup_response", "warmup_completed", "journey_started", "journey_finished", "process_stopped"]
    assert [row["endpoint"] for row in events if row["event"] == "warmup_response"] == ["/api/branding", "/api/tasks", "/api/branding"]
    assert len({row["pid"] for row in events}) == 1
    assert events[0]["workspace_sha256"] == originals["child_identity"]["workspace_sha256"]
    assert execution["runtime_status"] == "completed" and execution["isolated_workspace_removed"]
    window = measurement_window(execution, load_mode="steady", frozen_load_sha256=binding["load_sha256"])
    assert window["start_ns"] == events[-4]["elapsed_ns"] > 0
    assert window["end_ns"] == execution["duration_ms"] * 1_000_000
    assert run["report"]["load_binding"] == binding


@pytest.mark.e2e
def test_warm_window_round_trips_fixed_collector_and_formal_consumer(actual_warm_run, tmp_path):
    _root, binding, report, run = actual_warm_run
    execution = report["cases"][0]
    model, environment = {"profile": "fixture-only"}, {"load_sha256": binding["load_sha256"]}
    observation, coverage = build_formal_observation(execution, attempt_id="warm-r0", case_id="warm_join", run_id=report["run_id"],
                                                     load_mode="steady", model=model, environment=environment)
    assert observation["schema"] == "marvis.formal-process-observation.v2"
    assert coverage["duration_ms"] < coverage["original_duration_ms"]
    raw = json.dumps(observation).encode()
    (tmp_path / "observation.json").write_bytes(raw)
    reference = {"path": "observation.json", "sha256": digest(raw)}
    facts, reason = _telemetry(tmp_path, {"artifact_hashes": {"observation.json": digest(raw)}}, reference,
                              SimpleNamespace(model=SimpleNamespace(model_dump=lambda: model), environment=SimpleNamespace(model_dump=lambda: environment)),
                              SimpleNamespace(id="warm-r0", case_id="warm_join", load_mode="steady"), run,
                              SimpleNamespace(family="data_processing", actions=[]))
    assert reason is None and facts["wall_seconds"] == coverage["duration_ms"] / 1000
    # A producer-supplied offset cannot shorten the fixed runner-derived window.
    observation["measurement_window"]["start_ns"] += 1_000_000
    raw = json.dumps(observation).encode()
    (tmp_path / "observation.json").write_bytes(raw)
    reference["sha256"] = digest(raw)
    with pytest.raises(ValueError, match="formal_measurement_window_or_segments_changed"):
        _telemetry(tmp_path, {"artifact_hashes": {"observation.json": digest(raw)}}, reference,
                   SimpleNamespace(model=SimpleNamespace(model_dump=lambda: model), environment=SimpleNamespace(model_dump=lambda: environment)),
                   SimpleNamespace(id="warm-r0", case_id="warm_join", load_mode="steady"), run,
                   SimpleNamespace(family="data_processing", actions=[]))


@pytest.mark.e2e
@pytest.mark.parametrize("mutation", ["pid", "workspace", "nonce", "missing_warmup", "failed_read", "duration", "frozen_policy"])
def test_load_originals_reject_missing_or_forged_boundaries(actual_warm_run, mutation):
    _root, binding, report, _run = actual_warm_run
    execution = deepcopy(report["cases"][0])
    originals = execution["load_originals"]
    if mutation == "pid":
        originals["events"][-3]["pid"] += 1
    elif mutation == "workspace":
        originals["events"][0]["workspace_sha256"] = "f" * 64
    elif mutation == "nonce":
        originals["events"][0]["nonce"] = "f" * 32
    elif mutation == "missing_warmup":
        del originals["events"][1]
    elif mutation == "failed_read":
        originals["events"][2]["status_code"] = 500
    elif mutation == "duration":
        originals["events"][-4]["elapsed_ns"] = originals["events"][1]["elapsed_ns"]
    else:
        originals["binding"]["warmup"]["requests"] += 1
    with pytest.raises(LoadBindingError):
        measurement_window(execution, load_mode="steady", frozen_load_sha256=binding["load_sha256"])


def test_warmup_rejects_mutating_or_external_endpoints_before_execution():
    policy = {"endpoints": ["/api/tasks/create"], "requests": 1, "minimum_seconds": 0}
    specification = {"profile_load": {}, "load_policy": {"warmup": policy}}
    with pytest.raises(LoadBindingError, match="fixed_readonly"):
        validate_load_binding({"load_mode": "steady", "warmup": policy, "load_spec": specification, "load_sha256": digest(specification)})


@pytest.mark.e2e
def test_actual_concurrent_journeys_have_overlapping_distinct_process_originals(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from marvis.orchestrator.eval.hidden_formal_runner import _bind_concurrency
    data = tmp_path / "data"
    material = materials(data)
    case = base_case("concurrent_join", "data_join", [material["sample.parquet"], material["features.parquet"]])
    case["actions"] = [{"kind": "approve_step", "tool": "data_ops.execute_join", "content": "批准本地合成材料拼接。"}]
    cases, expected = write_suite(tmp_path, cases=[case], expected={case["id"]: {"result": "done", "assertions": [
        {"kind": "dataset_rows", "tool": "data_ops.execute_join", "value": 80}]}})
    load_spec = {"profile_load": {"concurrent_users": 2}, "load_policy": {"concurrent_workers": 2, "warmup": None}}
    binding = {"load_mode": "concurrent", "warmup": None, "load_spec": load_spec, "load_sha256": digest(load_spec)}
    with fixture_model(delay=.3) as (model, calls):
        def run(_):
            return run_runtime_suite(cases_path=cases, expected_path=expected, dataset_root=data,
                                     output_dir=tmp_path / "runs", model=model, model_source="fixture_model", load_binding=binding)
        with ThreadPoolExecutor(max_workers=2) as pool:
            reports = list(pool.map(run, range(2)))
    assert calls and all(report["all_passed"] for report in reports)
    rows = []
    for index, report in enumerate(reports):
        record = report["cases"][0]
        originals = record["load_originals"]
        events = {e["event"]: e for e in originals["events"]}
        rows.append({"attempt_id": str(index), "load_mode": "concurrent", "status": "completed", "journey_interval": {
            "pid": originals["child_identity"]["pid"], "workspace_sha256": originals["child_identity"]["workspace_sha256"],
            "start_ns": originals["origin_ns"] + events["journey_started"]["elapsed_ns"],
            "end_ns": originals["origin_ns"] + events["journey_finished"]["elapsed_ns"]}})
    _bind_concurrency(rows)
    assert all(row["concurrency"]["observed_overlap"] and row["status"] == "completed" for row in rows)


@pytest.mark.e2e
def test_restarted_process_cannot_reuse_steady_warmup_window(actual_warm_run):
    _root, binding, report, _run = actual_warm_run
    execution = deepcopy(report["cases"][0])
    execution["process_epochs"] = [{"ordinal": 0}, {"ordinal": 1}]
    with pytest.raises(LoadBindingError, match="restarted_process_requires_epoch_timing"):
        measurement_window(execution, load_mode="steady", frozen_load_sha256=binding["load_sha256"])
