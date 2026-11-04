"""Recompute simultaneous journeys from signature-bound runtime manifests.

The producer lists carriers only. It cannot supply an overlap, count, process
interval or successful verdict. Those facts are reconstructed by the consumer.
"""
from __future__ import annotations

from pathlib import Path
import platform
import re
import subprocess
import time

from .acceptance_trust import _hash, _keys, _path, _require
from .runtime_contracts import digest
from .runtime_run_manifest import verify_run_manifest
from .runtime_warmup import measurement_window

SCHEMA = "marvis.runtime-concurrency-originals.v1"
_PEER_KEYS = {"attempt_id", "case_id", "run_id", "run_dir", "final_manifest_sha256"}


def host_clock_identity():
    """Bind the host's actual boot session and shared monotonic clock."""
    system = platform.system()
    if system == "Darwin":
        boot = subprocess.run(["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
                              check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    elif system == "Linux":
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    else:
        raise ValueError("concurrent_host_clock_unavailable")
    _require(re.fullmatch(r"[0-9a-fA-F-]{36}", boot) is not None, "concurrent_boot_identity_unavailable")
    info = time.get_clock_info("monotonic")
    _require(info.monotonic and not info.adjustable, "concurrent_clock_not_monotonic")
    return {"schema": "marvis.host-monotonic-identity.v1", "clock": "host_monotonic_ns",
            "boot_sha256": digest({"system": system, "boot_session": boot.lower()}),
            "implementation": info.implementation}


def _clock(value):
    _keys(value, {"schema", "clock", "boot_sha256", "implementation"})
    _require(value["schema"] == "marvis.host-monotonic-identity.v1" and value["clock"] == "host_monotonic_ns"
             and _hash(value["boot_sha256"]) and isinstance(value["implementation"], str)
             and bool(value["implementation"]), "concurrent_clock_identity_unavailable")
    return value


def build_concurrency_evidence(root: Path, *, attempt_id: str, peers: list[dict]):
    """Produce references to actual originals, with no self-reported verdict."""
    _require(isinstance(peers, list) and 1 <= len(peers) <= 193, "concurrent_peer_inventory_invalid")
    for peer in peers:
        _keys(peer, _PEER_KEYS)
        run = verify_run_manifest(_path(Path(root).resolve(), peer["run_dir"], directory=True),
                                  expected_manifest_sha256=peer["final_manifest_sha256"])
        _require(run["manifest"]["run_id"] == peer["run_id"] and peer["case_id"] in run["executions"],
                 "concurrent_peer_run_mismatch")
    return {"schema": SCHEMA, "attempt_id": attempt_id, "peers": peers}


def verify_concurrency_evidence(root, evidence, *, artifact_hashes, benchmark, attempt, run):
    """Return measured count only after every participating carrier is checked."""
    root = Path(root).resolve()
    _keys(evidence, {"schema", "attempt_id", "peers"})
    _require(evidence["schema"] == SCHEMA and evidence["attempt_id"] == attempt.id
             and attempt.load_mode == "concurrent", "concurrent_attempt_binding_mismatch")
    peers = evidence["peers"]
    _require(isinstance(peers, list) and 1 <= len(peers) <= 193, "concurrent_peer_inventory_invalid")
    planned = {item.id: item for item in benchmark.attempts}
    planned_cases = {item.id: item for item in benchmark.cases}
    original = run["executions"][attempt.case_id]
    load = original["load_originals"]["binding"]
    policy = load["load_spec"]["load_policy"]
    required = policy["concurrent_workers"]
    _require(type(required) is int and 2 <= required <= 64
             and load["load_spec"]["profile_load"]["concurrent_users"] == required,
             "concurrent_frozen_worker_count_invalid")
    clock = _clock(run["report"].get("clock_identity"))
    base = run["manifest"]["frozen"]
    runtime_identity = {"python": base["source"]["python"], "platform": base["source"]["platform"],
                        "package_source_sha256": base["source"]["source_sha256"]}
    _require(base["source"]["commit"] == benchmark.commit
             and digest(runtime_identity) == benchmark.environment.runtime_identity_sha256
             and base["model_connection_sha256"] == benchmark.model.connection_sha256,
             "concurrent_frozen_source_or_model_mismatch")
    rows, identities, carriers, processes, workspaces = [], set(), set(), set(), set()
    current = None
    for peer in peers:
        _keys(peer, _PEER_KEYS)
        identity = peer["attempt_id"]
        _require(identity in planned and identity not in identities
                 and planned[identity].case_id == peer["case_id"] and planned[identity].load_mode == "concurrent",
                 "concurrent_peer_not_planned")
        identities.add(identity)
        manifest_path = str(Path(peer["run_dir"]) / "final-manifest.json")
        _require(_hash(peer["final_manifest_sha256"])
                 and artifact_hashes.get(manifest_path) == peer["final_manifest_sha256"],
                 "concurrent_peer_manifest_not_record_bound")
        other = verify_run_manifest(_path(root, peer["run_dir"], directory=True),
                                    expected_manifest_sha256=peer["final_manifest_sha256"])
        manifest = other["manifest"]
        carrier = (manifest["run_id"], peer["case_id"])
        _require(carrier not in carriers and manifest["run_id"] == peer["run_id"]
                 and peer["case_id"] in other["executions"], "concurrent_peer_carrier_reused")
        carriers.add(carrier)
        frozen = manifest["frozen"]
        _require(all(frozen[field] == base[field] for field in (
            "source", "model_connection_sha256", "model_source", "model_id", "model_name", "price_book_sha256")),
            "concurrent_peer_source_or_model_changed")
        _require(_clock(other["report"].get("clock_identity")) == clock,
                 "concurrent_peers_use_different_host_clocks")
        execution = other["executions"][peer["case_id"]]
        _require(execution["case_sha256"] == planned_cases[peer["case_id"]].case_sha256,
                 "concurrent_peer_case_definition_changed")
        originals = execution["load_originals"]
        _require(originals["binding"] == load == other["report"].get("load_binding"),
                 "concurrent_peer_load_changed")
        measurement_window(execution, load_mode="concurrent", frozen_load_sha256=benchmark.environment.load_sha256)
        child = originals["child_identity"]
        _require(child["pid"] not in processes and child["workspace_sha256"] not in workspaces,
                 "concurrent_peer_process_reused")
        processes.add(child["pid"])
        workspaces.add(child["workspace_sha256"])
        events = {item["event"]: item["elapsed_ns"] for item in originals["events"]}
        start = originals["origin_ns"] + events["journey_started"]
        end = originals["origin_ns"] + events["journey_finished"]
        _require(end > start, "concurrent_peer_empty_journey")
        rows.append((start, end))
        if identity == attempt.id:
            _require(manifest == run["manifest"], "concurrent_current_run_reassigned")
            current = (start, end)
    _require(current is not None, "concurrent_current_run_missing")
    points = {}
    for start, end in rows:
        start, end = max(start, current[0]), min(end, current[1])
        if start < end:
            points[start] = points.get(start, 0) + 1
            points[end] = points.get(end, 0) - 1
    active = peak = 0
    for point in sorted(points):
        active += points[point]
        peak = max(peak, active)
    _require(peak >= required, "concurrent_frozen_worker_overlap_not_observed")
    return {"required_workers": required, "observed_simultaneous_journeys": peak,
            "scope": "verified_runtime_manifests_same_host_clock"}
