"""Actual local HTTP carriers; no paid calls or independent signatures."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from marvis.orchestrator.eval import runtime_concurrency as concurrency
from marvis.orchestrator.eval.formal_acceptance import _telemetry
from marvis.orchestrator.eval.process_timing_collector import build_formal_observation
from marvis.orchestrator.eval.runtime_contracts import digest
from marvis.orchestrator.eval.runtime_run_manifest import verify_run_manifest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from test_runtime_agent_benchmark import base_case, fixture_model, materials, write_suite
from test_runtime_run_manifest import _rewrite_inventory


def _actual_pair(root, required_workers=2):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    material = materials(data)
    case = base_case("concurrent_join", "data_join", [material["sample.parquet"], material["features.parquet"]])
    case["actions"] = [{"kind": "approve_step", "tool": "data_ops.execute_join", "content": "批准本地合成材料拼接。"}]
    cases, expected = write_suite(root, cases=[case], expected={case["id"]: {"result": "done", "assertions": [
        {"kind": "dataset_rows", "tool": "data_ops.execute_join", "value": 80}]}})
    specification = {"profile_load": {"concurrent_users": required_workers},
                     "load_policy": {"concurrent_workers": required_workers, "warmup": None}}
    load = {"load_mode": "concurrent", "warmup": None, "load_spec": specification, "load_sha256": digest(specification)}
    with fixture_model(delay=.3) as (model, calls):
        def execute(_):
            return run_runtime_suite(cases_path=cases, expected_path=expected, dataset_root=data,
                output_dir=root / "runs", model=model, model_source="fixture_model", load_binding=load)
        with ThreadPoolExecutor(max_workers=2) as pool:
            reports = list(pool.map(execute, range(2)))
    assert calls and all(report["all_passed"] for report in reports)
    peers = [{"attempt_id": f"parallel-{index}", "case_id": case["id"], "run_id": report["run_id"],
              "run_dir": str(Path(report["report_path"]).parent.relative_to(root)),
              "final_manifest_sha256": report["final_manifest_sha256"]} for index, report in enumerate(reports)]
    run = verify_run_manifest(root / peers[0]["run_dir"], expected_manifest_sha256=peers[0]["final_manifest_sha256"])
    frozen = run["manifest"]["frozen"]
    model_fields = {"profile": frozen["model_id"], "model_name": frozen["model_name"],
                    "connection_sha256": frozen["model_connection_sha256"]}
    environment = {"load_sha256": load["load_sha256"], "runtime_identity_sha256": digest({
        "python": frozen["source"]["python"], "platform": frozen["source"]["platform"],
        "package_source_sha256": frozen["source"]["source_sha256"]})}
    attempts = [SimpleNamespace(id=p["attempt_id"], case_id=p["case_id"], load_mode="concurrent") for p in peers]
    benchmark = SimpleNamespace(attempts=attempts, commit=frozen["source"]["commit"],
        cases=[SimpleNamespace(id=case["id"], case_sha256=run["executions"][case["id"]]["case_sha256"])],
        source_inventory_sha256=frozen["source"]["source_sha256"],
        model=SimpleNamespace(**model_fields, model_dump=lambda: model_fields),
        environment=SimpleNamespace(**environment, model_dump=lambda: environment))
    proof = concurrency.build_concurrency_evidence(root, attempt_id=attempts[0].id, peers=peers)
    hashes = {str(Path(p["run_dir"]) / "final-manifest.json"): p["final_manifest_sha256"] for p in peers}
    return root, proof, hashes, benchmark, attempts[0], run


@pytest.fixture(scope="module")
def actual_pair(tmp_path_factory):
    return _actual_pair(tmp_path_factory.mktemp("concurrent-proof"))


def _verify(fixture):
    root, proof, hashes, benchmark, attempt, run = fixture
    return concurrency.verify_concurrency_evidence(root, proof, artifact_hashes=hashes,
                                                  benchmark=benchmark, attempt=attempt, run=run)


@pytest.mark.e2e
def test_actual_hash_bound_carrier_slots_recompute_two_simultaneous_journeys(actual_pair):
    result = _verify(actual_pair)
    assert result["required_workers"] == result["observed_simultaneous_journeys"] == 2
    assert "overlap" not in actual_pair[1]  # the producer supplies carrier references only


def test_two_real_processes_do_not_prove_three_frozen_workers(tmp_path):
    fixture = _actual_pair(tmp_path / "three-required", required_workers=3)
    with pytest.raises(ValueError, match="worker_overlap_not_observed"):
        _verify(fixture)


@pytest.mark.e2e
def test_concurrent_telemetry_requires_separate_originals_and_round_trips(actual_pair):
    root, proof, hashes, benchmark, attempt, run = actual_pair
    execution = run["executions"][attempt.case_id]
    observation, _ = build_formal_observation(execution, attempt_id=attempt.id, case_id=attempt.case_id,
        run_id=run["manifest"]["run_id"], load_mode="concurrent", model=benchmark.model.model_dump(),
        environment=benchmark.environment.model_dump())
    observation_raw, proof_raw = json.dumps(observation).encode(), json.dumps(proof).encode()
    (root / "observation.json").write_bytes(observation_raw)
    (root / "concurrency.json").write_bytes(proof_raw)
    reference = {"path": "observation.json", "sha256": digest(observation_raw)}
    record = {"artifact_hashes": {**hashes, "observation.json": digest(observation_raw), "concurrency.json": digest(proof_raw)}}
    definition = SimpleNamespace(family="data_processing", actions=[])
    facts, reason = _telemetry(root, record, reference, benchmark, attempt, run, definition)
    assert facts == {} and reason == "concurrent_originals_unavailable"
    reference["concurrency"] = {"path": "concurrency.json", "sha256": digest(proof_raw)}
    facts, reason = _telemetry(root, record, reference, benchmark, attempt, run, definition)
    assert reason is None and facts["wall_seconds"] > 0


@pytest.mark.e2e
@pytest.mark.parametrize("mutation,reason", [
    ("unbound", "not_record_bound"), ("duplicate", "peer_not_planned"),
    ("reclassified", "peer_not_planned"), ("escape", "not_record_bound"),
])
def test_peer_inventory_cannot_be_substituted(actual_pair, mutation, reason):
    root, original, hashes, benchmark, attempt, run = actual_pair
    proof, hashes = deepcopy(original), dict(hashes)
    if mutation == "unbound":
        hashes.clear()
    elif mutation == "duplicate":
        proof["peers"].append(deepcopy(proof["peers"][0]))
    elif mutation == "reclassified":
        proof["peers"][1]["attempt_id"] = "never-planned"
    else:
        proof["peers"][1]["run_dir"] = "../outside"
    with pytest.raises(ValueError, match=reason):
        _verify((root, proof, hashes, benchmark, attempt, run))


@pytest.mark.e2e
def test_peer_original_tampering_is_rejected_by_manifest_verifier(actual_pair, tmp_path):
    root, proof, hashes, benchmark, attempt, run = actual_pair
    copied = tmp_path / "carriers"
    shutil.copytree(root / "runs", copied / "runs")
    peer = proof["peers"][1]
    path = copied / peer["run_dir"] / "report.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="original_bytes_changed"):
        _verify((copied.resolve(), proof, hashes, benchmark, attempt, run))


@pytest.mark.e2e
def test_rehashed_peer_on_another_boot_clock_is_not_simultaneous_evidence(actual_pair, tmp_path):
    root, original, original_hashes, benchmark, attempt, run = actual_pair
    copied = tmp_path / "carriers"
    shutil.copytree(root / "runs", copied / "runs")
    proof, hashes = deepcopy(original), dict(original_hashes)
    peer = proof["peers"][1]
    path = copied / peer["run_dir"]
    for name in ("manifest.json", "report.json"):
        item = json.loads((path / name).read_bytes())
        item["clock_identity"]["boot_sha256"] = "f" * 64
        (path / name).write_text(json.dumps(item))
    peer["final_manifest_sha256"] = _rewrite_inventory(path)
    hashes[str(Path(peer["run_dir"]) / "final-manifest.json")] = peer["final_manifest_sha256"]
    with pytest.raises(ValueError, match="different_host_clocks"):
        _verify((copied.resolve(), proof, hashes, benchmark, attempt, run))


def test_host_clock_is_stable_for_the_current_boot():
    assert concurrency.host_clock_identity() == concurrency.host_clock_identity()


def test_legacy_timing_labels_cannot_claim_actual_steady_warmup(tmp_path):
    raw = json.dumps({"schema": "marvis.formal-process-observation.v1"}).encode()
    (tmp_path / "legacy.json").write_bytes(raw)
    facts, reason = _telemetry(tmp_path, {"artifact_hashes": {"legacy.json": digest(raw)}},
        {"path": "legacy.json", "sha256": digest(raw)}, None,
        SimpleNamespace(load_mode="steady"), None, None)
    assert facts == {} and reason == "steady_warmup_originals_unavailable"
